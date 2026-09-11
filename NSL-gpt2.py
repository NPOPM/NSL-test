import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

GELU_PARA1 = math.sqrt(2.0 / math.pi)
GELU_PARA2 = 0.044715

kv_cache = {}
kv_cache_draft = {}
kv_cache_target = {}


def gelu(x):
    '''
        y = 0.5x[1+tanh((2/Π)^(1/2)(x+0.044715x^3))]
    '''
    y = 0.5 * x * (1 + torch.tanh(GELU_PARA1 * (x + GELU_PARA2 * x ** 3)))
    return y


def softmax(x, dim=-1):  # 处理任意维张量，默认是最后一维
    """
        softmax公式：softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # keepdim=True：保留原来的维度结构
    # [0]表示要取最大值本身，而不是最大值的索引
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_new = x - x_max
    x_exp = torch.exp(x_new)
    # 再dim维上求和
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum


def layer_norm(x, g_b, eps: float = 1e-5):
    """
        1.计算每一个样本的均值mean和方差var
        2.对输入的张量进行标准化
            T(b,s,c)={[T(b,s,c)-mean]/(var+eps)^(1/2)}*gamma+bias
        3.对上一步的结果进行缩放和加偏置
    """

    # 从g_b字典中取出缩放系数g(gamma)和偏置量b(bias)
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    # 计算均值和方差
    # unbiased=False表示计算方差时除以n，而不是n-1，这样得到的是“总体方差”
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # 标准化
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # 缩放、偏置
    result = x_norm * g + b

    return result


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        y = x @ w + b
    """
    w, b = w_b['w'], w_b['b']
    y = x @ w + b
    return y


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Feed-Forward Network，前馈神经网络
        FFN(x) = GELU(x @ w1 + b1) @ w2 + b2
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    y = linear(gelu(linear(x, w_b1)), w_b2)
    return y


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        mha:
            Q = q @ I
            K = k @ I
            V = v @ I

        attention:
        1.计算相似度矩阵
            A = Q @ K^T
        2.缩放点积注意力
            scores/=d_k^(1/2)
        3.加掩码
        4.softmax归一化得到A'
        5.加权求和
            O = A' @ V
    """

    # 1
    A = q @ k.transpose(-2, -1)
    # 2
    d_k = q.size(-1)
    A = A / math.sqrt(d_k)
    # 3 将掩码矩阵为True的位置赋值为一个非常大的负数
    A = A.masked_fill(mask, -1e9)
    # 4
    A_ = softmax(A, dim=-1)
    # 5
    O = A_ @ v

    return O


def mha(x, attn, n_head, layer_index, use_cache=True, cache_dict=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    global kv_cache
    if cache_dict is not None:
        cache = cache_dict
    else:
        cache = kv_cache

    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # 拆分qkv
    q, k, v = torch.chunk(x, 3, dim=-1)  # [n_seq, n_embd]

    # 构造矩阵causal_mask
    n_seq = x.size(0)
    # 如果有缓存
    if use_cache and layer_index in cache:
        k_cache, v_cache = cache[layer_index]
        n_new = k_cache.size(0) + n_seq

        causal_mask = torch.zeros(n_seq, n_new, dtype=torch.bool)  # [n_seq, n_new]
        for i in range(n_seq):
            causal_mask[i, k_cache.size(0) + i + 1:] = True
    # 如果没有缓存
    else:
        # 生成一个主对角线及以下为False，以上为True的三角矩阵
        causal_mask = torch.triu(torch.ones(n_seq, n_seq), diagonal=1).bool()

    if use_cache:
        # 将新词拼接到kv
        if layer_index in cache:
            k_cache, v_cache = cache[layer_index]
            k = torch.cat([k_cache, k], dim=0)  # [n_seq + 1, n_embd]
            v = torch.cat([v_cache, v], dim=0)  # [n_seq + 1, n_embd]

        # 更新kv_cache
        cache[layer_index] = (k, v)

    # 拆头
    q_heads = q.chunk(n_head, dim=-1)
    k_heads = k.chunk(n_head, dim=-1)
    v_heads = v.chunk(n_head, dim=-1)
    qkv_heads = list(zip(q_heads, k_heads, v_heads))

    # 在每个头上执行attention
    out_heads = []
    for q, k, v in qkv_heads:
        out = attention(q, k, v, causal_mask)
        out_heads.append(out)  # n_head * [n_seq, n_embd/n_head]

    # 合并多头
    x = torch.cat(out_heads, dim=-1)

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, block, n_head, layer_index, use_cache=True,
                      cache_dict=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, layer_index=layer_index, use_cache=use_cache,
                cache_dict=cache_dict)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, use_cache=True, cache_dict=None):  # [n_seq] -> [n_seq, n_vocab]
    global kv_cache

    if cache_dict is not None:
        cache = cache_dict
    else:
        cache = kv_cache
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']

    # 计算位置编码
    if use_cache and cache:
        # 如果有缓存，就从缓存长度开始
        k_cache, v_cache = cache[0]
        start_pos = k_cache.size(0)
    else:
        # 如果没有缓存，就从0开始
        start_pos = 0

    positions = list(range(start_pos, start_pos + len(inputs)))

    # token + positional embeddings
    x = wte[inputs] + wpe[positions]  # [n_seq] -> [n_seq, n_embd]
    x = torch.Tensor(x)

    # forward pass through n_layer transformer blocks
    layer_index = 0
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head, layer_index=layer_index, use_cache=use_cache,
                              cache_dict=cache)  # [n_seq, n_embd] -> [n_seq, n_embd]
        layer_index += 1

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    global kv_cache
    from tqdm import tqdm

    # 清空缓存
    kv_cache = {}

    for step in tqdm(range(n_tokens_to_generate), "generating", total=n_tokens_to_generate):
        if step == 0:
            # 传入完整的prompt，填充缓存
            logits = gpt2(inputs, params, n_head=n_head, use_cache=True)
            next_id = np.argmax(logits[-1])
            inputs.append(int(next_id))
        else:
            logits = gpt2([next_id], params, n_head=n_head, use_cache=True)  # 这里只传1个token
            next_id = np.argmax(logits[-1])
            inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate:]


def rollback_kv_cache(kv_cache_, target_length):
    """
        将kv_cache回滚到指定的序列长度
    """
    for layer_index in kv_cache_:
        k, v = kv_cache_[layer_index]
        kv_cache_[layer_index] = (k[:target_length], v[:target_length])


def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target,
                                n_tokens_to_generate, K):
    from tqdm import tqdm
    global kv_cache_draft, kv_cache_target
    kv_cache_draft = {}
    kv_cache_target = {}

    generated_ids = []
    current_inputs = list(inputs)

    n_head_draft = hparams_draft["n_head"]
    n_head_target = hparams_target["n_head"]

    pbar = tqdm(total=n_tokens_to_generate, desc="Generating", position=0, leave=True)

    last_target_logits = gpt2(current_inputs, target_params, n_head_target,use_cache=True, cache_dict=kv_cache_target)
    last_target_logit = last_target_logits[-1] # 下一个token的logits
    while len(generated_ids) < n_tokens_to_generate:
        seq_len_before = len(current_inputs)

        # 小模型生成K个草稿
        draft_tokens = []
        draft_inputs = list(current_inputs)
        for i in range(K):

            if kv_cache_draft:
                input_ = [draft_inputs[-1]]
            else:
                input_ = current_inputs

            logits = gpt2(input_, draft_params, n_head_draft, use_cache=True, cache_dict=kv_cache_draft)
            next_id = int(np.argmax(logits[-1]))
            draft_tokens.append(next_id)
            draft_inputs.append(next_id)

        # 大模型验证
        target_logits = gpt2(draft_tokens, target_params, n_head_target, use_cache=True, cache_dict=kv_cache_target)
        verify_logits = [last_target_logit] + [target_logits[i] for i in range(K - 1)]

        accept_count = 0
        for i in range(K):
            target_token = int(np.argmax(verify_logits[i]))

            if draft_tokens[i] == target_token:
                current_inputs.append(draft_tokens[i])
                generated_ids.append(draft_tokens[i])
                accept_count += 1

                pbar.update(1)
                #如果生成够了，就停止
                if len(generated_ids) >= n_tokens_to_generate:
                    break

            else:
                current_inputs.append(target_token)
                generated_ids.append(target_token)

                rollback_kv_cache(kv_cache_draft, seq_len_before+accept_count)
                rollback_kv_cache(kv_cache_target, seq_len_before+accept_count)

                gpt2([target_token], draft_params, n_head_draft, use_cache=True, cache_dict=kv_cache_draft)
                #这里要记录最后一个生成出的token
                last_target_logits = gpt2([target_token], target_params, n_head_target, use_cache=True,
                                          cache_dict=kv_cache_target)
                last_target_logit = last_target_logits[-1]

                pbar.update(1)
                break

        else:
            #如果都接受了，target_logits的最后一行就是大模型预测的下一个token
            last_target_logit = target_logits[-1]

    pbar.close()
    return generated_ids[:n_tokens_to_generate]


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models",
         use_speculative: bool = False, K: int = 4):
    from utils import load_encoder_hparams_and_params

    if use_speculative:
        encoder, hparams_draft, draft_params = load_encoder_hparams_and_params("124M", models_dir)
        _, hparams_target, target_params = load_encoder_hparams_and_params("1558M", models_dir)

        input_ids = encoder.encode(prompt)
        assert len(input_ids) + n_tokens_to_generate < hparams_target["n_ctx"]
        start = time.time()
        output_ids = greedy_speculative_generate(
            input_ids, draft_params, target_params,
            hparams_draft, hparams_target,
            n_tokens_to_generate, K
        )
        end = time.time()
        print(f"Time taken to generate {n_tokens_to_generate} tokens (speculative): {end - start:.2f}s")

        # 解码输出
        output_text = encoder.decode(output_ids)
        return output_text

    else:
        # load encoder, hparams, and params from the released open-ai gpt-2 files
        encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

        # encode the input string using the BPE tokenizer
        input_ids = encoder.encode(prompt)

        # make sure we are not surpassing the max sequence length of our model
        assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

        # generate output ids
        start = time.time()
        output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
        end = time.time()
        print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

        # decode the ids back into a string
        output_text = encoder.decode(output_ids)
        return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)

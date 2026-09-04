import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

GELU_PARA1 = math.sqrt(2.0 / math.pi)
GELU_PARA2 = 0.044715


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
    # 3 将掩码矩阵为-inf的位置赋值为一个非常大的负数
    A = A.masked_fill(mask == float('-inf'), -1e9)
    # 4
    A_ = softmax(A, dim=-1)
    # 5
    O = A_ @ v

    return O


def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]

    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # 拆分qkv
    qkv = torch.chunk(x, 3, dim=-1)

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in
                 qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # 构造上三角矩阵causal_mask
    """
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... |
            | 0    0    0  ...   0  |
    """
    n_seq = x.size(0)
    # 生成一个主对角线及以下为False，以上为True的三角矩阵
    causal_mask = torch.triu(torch.ones(n_seq, n_seq), diagonal=1).bool()
    causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]

    # 合并多头
    x = torch.cat(out_heads, dim=-1)

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target,
                                n_tokens_to_generate, K):
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

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

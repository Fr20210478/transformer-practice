import torch
import torch.nn as nn
import math


# 定义自注意力
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 对10%的神经元进行随机失活
        # self.dropout:开发者自己定义的一个变量（通常作为类的属性）
        self.softmax = nn.Softmax(dim=-1)  # 将得分转换为概率分布，在最后一个维度进行？

    def forward(self, q, k, v, mask=None):
        # X:batch,seq_len,d_model
        # batch: 一次送到模型的句子个数； seq_len：一个句子中的token数量；d_model：embedding向量维度
        # Q：查询向量（张量） 维度：batch,heads,seq_len_q,d_k
        # (2, 2, 3, 64) 是一个4维张量的维度描述，2 个样本，每个样本用 2 个注意力头处理，每个头处理 3 个词，每个词用 64 维向量表示
        # K：key向量 维度：batch,heads,seq_len_k,d_k
        # V：value向量 维度：batch,heads,seq_len_v,d_v
        # mask
        d_k = q.size(-1)  # q的最后一维(-1即倒数第一个)是对每个query向量的维度，代表我们对每个query进行缩放？
        # batch,heads,seq_len_q,d_k， batch,heads,d_k，seq_len_k -> batch,heads,seq_len_q,seq_len_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 进行缩放，让梯度更为稳定
        if mask is not None:  # 如果提供了mask，则通过mask==0来找需要屏蔽的位置,mask==1表示当前位置可见
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 维度：batch,heads,seq_len_q,seq_len_k 对最后一维即对key进行softmax，得到注意力权重矩阵，对每一个query的key权重之和为1
        attn = self.softmax(scores)
        attn = self.dropout(attn)  # 对注意力权重进行dropout防止过拟合
        # attn:batch,heads,seq_len_q,seq_len_k ; V:batch,heads,seq_len_v,d_v -> attn*V:batch,heads,seq_len_q,d_v
        out = torch.matmul(attn, v)
        return out, attn


# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  # 也可super().__init__()
        assert d_model % n_heads == 0  # 断言语句，d_model embedding维度 512，n_heads 多头注意力头数 8
        self.d_k = d_model // n_heads  # 每个头的维度
        self.n_heads = n_heads

        # 将输入向量（src或者说是out）映射到QKV三个向量,通过线性映射让模型具有学习能力
        self.W_q = nn.Linear(d_model, d_model)  # query的线性映射，维度不需要改变，方便后续的多头拆分
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)  # 多头拼接后再映射回原来的dmodel，让模型融合不同头的信息

        self.attention = SelfAttention(dropout)  # 使用定义好的自注意力
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.norm = nn.LayerNorm(d_model)  # 用于残差后的归一化

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch大小
        # view作用为拆分多头，q的维度batch，seq_len, d_model -> batch,  self.n_heads, seq_len, self.d_k
        # 交换维度目的：为了让每个头独立处理序列，方便后续注意力计算
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        out, attn = self.attention(Q, K, V, mask)  # attn为注意力权重，out为注意力加权后的值
        # attn*V:batch,heads,seq_len_q,d_v -> batch,seq_len_q,heads,d_v -> batch,seq_len_q,heads,d_model(out)
        # 将 “按头分组” 的结果重新组织为 “按序列位置分组”，为多头结果的合并铺平道路。
        # contiguous的功能是让tensor在内存中连续存储，避免view时产生报错
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        out = self.fc(out)  # 让输入和输出一致，方便残差连接
        out = self.dropout(out)  # 防止过拟合
        return self.norm(out + q), attn  # 返回输出加输入（残差连接），注意力权重


# 定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 输入维度为d_model, 输出为d_ff，让模型有更丰富的特征，512->2048
        self.fc2 = nn.Linear(d_ff, d_model)  # 保证第二个线性层的输出维度等于第一个线性层的输入维度，为了后续的残差连接
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)  # 归一化的最后一维

    def forward(self, x):
        # x 形状为 batch，seq_len，d_model
        out = self.fc2(
            self.dropout(
                torch.relu(self.fc1(x))  # 依次经过第一个线性层，relu，dropout，第二个线性层
            )
        )
        return self.norm(out + x)  # 归一化让模型训练更稳定，加快模型收敛


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力机制 输入为src 实现序列内部的信息交互，每个token可以学到序列中的其它token，从而学习到上下文依赖
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈神经网络，对每个位置向量进行非线性变换，提升模型表达能力
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, src, src_mask=None):
        # src 输入序列张量 形状 batch，seq_len，d_model
        # src_mask，屏蔽padding(补齐句子长度用的东西)位置，避免模型关注无效token
        # qkv = src(原序列) 对输入序列本身进行自注意力计算
        # #这里用 _ 表示暂时不需要使用注意力权重
        # #self.self_attn(tgt, tgt, tgt, tgt_mask) 是对方法的调用
        # #具体来说是调用 self.self_attn 这个属性所指向的对象的 forward 方法
        out, _ = self.self_attn(src, src, src, src_mask)
        # 经过前馈神经网络
        out = self.ffn(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # mask多头注意力机制
        # 输入tgt（目标序列） 在翻译任务中已经生成的前几个单词
        # 计算目标序列内部的自注意力，通过mask遮住未来的token
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 交叉注意力，为了于encoder做交互
        # 输入 Q=当前解码器的输出，K，V来自编码器的memory（原序列上下文信息）
        # 为了将目标序列于原序列对齐
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 为了提升表达能力
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt 目标序列；memory：编码器输出（原序列表示）
        # tgt_mask 屏蔽未来的掩码；memory_mask:padding做掩码
        out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # 将目标序列和原序列进行交互，Q解码器当前的输出out，k=v=memory（编码器的输出）
        out, _ = self.cross_attn(out, memory, memory, memory_mask)
        out = self.ffn(out)
        return out


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # d_model:每个词的维度；max_len句子最大长度
        # 初始化位置编码矩阵，形状为max_len, d_model
        pe = torch.zeros(max_len, d_model)

        # 定义记录每个token位置的索引，0-max_len-1
        # unsqueeze:新增一个维度[max_len, 1],方便后续与缩放因子相乘
        # [0., 1., 2., 3., 4.] 经过 unsqueeze(1) 后变为：
        # plaintext
        # tensor([[0.],
        #         [1.],
        #         [2.],
        #         [3.],
        #         [4.]])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 每个维度的缩放因子，torch.arange(0, d_model, 2)：生成偶数维度索引0，2，4对应2i
        # torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model)) = 1/(10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 选取所有行（即所有位置的编码）; 0::2：选取列索引为 0, 2, 4, ... 的列（偶数索引列）
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度，1， max_len, d_model,方便后续与输入embedding进行相加
        pe = pe.unsqueeze(0)
        # 注册为buffer，把位置编码存在模型里面，不参与训练，但是随模型保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:输入embedding，形状为batch， seq_len, d_model
        seq_len = x.size(1)
        # 每个token的embedding加上对应位置编码
        # self.pe[:, :seq_len, :] 取前seq_len个位置，形状变成1， seq_len, d_model
        # x + self.pe[:, :seq_len, :] ： batch， seq_len, d_model   embedding加上位置编码，transformer可以token的位置
        return x + self.pe[:, :seq_len, :]


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=500):
        super().__init__()
        # 词嵌入层，vocab_size:词表大小，包含了不同token的总数
        # 将输入的token ID（对原始文本分词得到词表，不同词对应不同ID）转换从连续向量，维度为d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码，加入序列中token的位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 构建编码器的堆叠结构
        # 堆叠num_layers个encoder
        # nn.ModuleList为网络层准备的列表，用来存放多个子模块
        # 列表推导式，用来生成num_layers个encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        # 将输入token ID转为embedding向量
        # 输出形状：batch, seq_len, d_model
        # 乘上sqrt（d_model），进行缩放，让后续注意力计算更稳定
        out = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        # 经过当前位置编码
        out = self.pos_encoding(out)
        # 逐层经过encoderlayer
        for layer in self.layers:
            out = layer(out, src_mask)  # self_attn + ffn
        # 返回编码后的输出    batch, seq_len, d_model
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=500):
        super().__init__()
        # 将输入的token ID（对原始文本分词得到词表，不同词对应不同ID）转换从连续向量，维度为d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 经过位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # 定义解码器列表
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # 输出投影层 将decoder输出映射回原词汇表大小，从而得到每个词的预测分布
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt 目标序列；memory：编码器输出（原序列表示）
        # tgt_mask 屏蔽未来的掩码；memory_mask:padding做掩码
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        # 添加位置编码
        out = self.pos_encoding(out)
        # 逐层经过decoderlayer
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask)  # #调用decoderlayer中的forward方法
        # 将解码器最后一层输入的隐藏向量映射回原词汇表大小，从而得到每个词的预测分布
        return self.fc_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,  # 源语言词表大小
                 tgt_vocab,  # 目标语言词表大小
                 d_model=512,  # embedding向量维度
                 n_heads=8,
                 num_encoder_layers=6,  # 堆叠六个编码层
                 num_decoder_layers=6,
                 d_ff=2048,  # ffn隐藏层维度
                 dropout=0.1,
                 max_len=5000):  # 最大序列长度
        super().__init__()
        # 编码器，将源语言token编码为上下文表示
        self.encoder = Encoder(
            src_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len
        )
        # 解码器，根据编码器的输出和目标语言输入生成预测
        self.decoder = Decoder(
            tgt_vocab, d_model, n_heads, num_decoder_layers, d_ff, dropout, max_len
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 编码器前向传播 src_mask屏蔽pad（补齐长度的东西）
        memory = self.encoder(src, src_mask)
        # 解码器前向传播 tgt_mask屏蔽未来信息, memory_mask屏蔽未来token
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # 返回transformer输出 形状 batch, seq_len_tgt, tgt_vocab
        return out


def generate_mask(size):
    # torch.triu(torch.ones(size, size), diagonal=1) 生成上三角，不含对角线
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # 下三角含对角线可见
    return mask == 0  # True为可见，False为屏蔽


# 验证
src_vocab = 10000
tgt_vocab = 10000
# 初始化模型
model = Transformer(src_vocab, tgt_vocab)
# 假例子数据测试
src = torch.randint(0, src_vocab, (32, 10))  # 原序列batch=32，src_len=10 每个元素是一个token ID
tgt = torch.randint(0, tgt_vocab, (32, 20))  # 目标序列batch=32，src_len=20 每个元素是一个token ID
# 区目标序列长度20
tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)
print(tgt_mask)

out = model(src, tgt, tgt_mask=tgt_mask)  # 前向传播
# 每个目标token对应词表中每个词的预测概率
print(out.shape)  # batch, tgt_len, tgt_vocab


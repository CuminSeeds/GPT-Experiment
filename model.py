"""字符级小型 GPT。自行搭建注意力与网络，不加载现成模型权重。"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 130       # ASCII 0~127，PAD=128，UNK=129
    context: int = 192
    width: int = 128
    layers: int = 3
    heads: int = 4
    ff_width: int = 512
    dropout: float = 0.1
    position_encoding: str = 'sinusoidal'  # sinusoidal 或 rope


PAD, UNK = 128, 129


def encode(text):
    return [ord(c) if ord(c) < 128 else UNK for c in text]


def decode(ids):
    return ''.join(chr(i) if i < 128 else '?' for i in ids)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.use_rope = cfg.position_encoding == 'rope'
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool)))
        if self.use_rope:
            head_width = cfg.width // cfg.heads
            positions = torch.arange(cfg.context).float().unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, head_width, 2).float()
                                    * (-math.log(10000) / head_width))
            angles = positions * frequencies
            # 可重建的缓存不写入权重文件，不增加可训练参数。
            self.register_buffer('rope_cos', angles.cos(), persistent=False)
            self.register_buffer('rope_sin', angles.sin(), persistent=False)

    def rotate(self, x):
        """将 Q、K 中相邻的两个分量按其所在位置旋转。"""
        t = x.size(-2)
        c, s = self.rope_cos[:t].to(x.dtype), self.rope_sin[:t].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [批量, 时间, 维度] -> [批量, 头数, 时间, 每头维度]
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        if self.use_rope:
            q, k = self.rotate(q), self.rotate(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.heads)
        scores = scores.masked_fill(~self.mask[:t, :t], float('-inf'))
        a = self.drop(F.softmax(scores, dim=-1))
        z = (a @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.drop(self.proj(z))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm1 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.ff_width), nn.ReLU(),
                                nn.Linear(cfg.ff_width, cfg.width), nn.Dropout(cfg.dropout))
        self.norm2 = nn.LayerNorm(cfg.width)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ff(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        if cfg.width % cfg.heads:
            raise ValueError('隐藏维度必须能被注意力头数整除')
        if cfg.position_encoding not in ('sinusoidal', 'rope'):
            raise ValueError('位置编码只支持 sinusoidal 和 rope')
        if cfg.position_encoding == 'rope' and (cfg.width // cfg.heads) % 2:
            raise ValueError('RoPE 的每头维度必须为偶数')
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        # 基础模型使用正弦余弦位置编码；RoPE 直接在注意力中处理 Q、K。
        pos = torch.arange(cfg.context).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, cfg.width, 2).float() * (-math.log(10000) / cfg.width))
        pe = torch.zeros(cfg.context, cfg.width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * scale), torch.cos(pos * scale)
        self.register_buffer('position', pe)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.output = nn.Linear(cfg.width, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids):
        if ids.size(1) > self.cfg.context:
            raise ValueError('输入长度超过上下文长度')
        x = self.embedding(ids)
        if self.cfg.position_encoding == 'sinusoidal':
            x = x + self.position[:ids.size(1)]
        x = self.drop(x)
        return self.output(self.blocks(x))


def save_model(model, path):
    torch.save({'config': asdict(model.cfg), 'state': model.state_dict()}, path)


def load_model(path, device='cpu'):
    obj = torch.load(path, map_location=device, weights_only=True)
    model = TinyGPT(Config(**obj['config'])).to(device)
    model.load_state_dict(obj['state'])
    return model.eval()


@torch.no_grad()
def generate(model, prompt, n=240, seed=2026, temperature=0.8):
    dev = next(model.parameters()).device
    rng = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.tensor([encode(prompt)], device=dev)
    allowed = [10] + list(range(32, 127))
    model.eval()
    for _ in range(n):
        logits = model(ids[:, -model.cfg.context:])[:, -1] / temperature
        banned = torch.ones(model.cfg.vocab_size, dtype=torch.bool, device=dev)
        banned[allowed] = False
        logits[:, banned] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=rng)
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids[0].tolist())

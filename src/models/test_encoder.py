from torch import nn
import torch
from utils import commons


class ConvNorm(nn.Module):
    def __init__(self, input_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.relu(x)
        return x


class PreNet(nn.Module):
    def __init__(self, channels, kernels, hidden_size):
        super().__init__()

        self.flow = nn.ModuleList()

        for idx in range(len(channels) - 1):
            self.flow.append(ConvNorm(channels[idx], channels[idx + 1], kernels[idx]))

        self.proj = nn.Linear(channels[-1], hidden_size)

    def forward(self, x):
        for f in self.flow:
            x = f(x)

        x = x.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (B, C, T)

        return x


class ConvReluConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


class EncModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.convrelu = ConvReluConv(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)  # (B, T, H)
        out, _ = self.mha(x, x, x, key_padding_mask=x_mask)

        x = out + x

        x = self.norm(x)
        x = x.transpose(1, 2)

        out = self.convrelu(x)

        x = out + x

        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)

        return x


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = nn.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = x.transpose(1, 2)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.norm_1(x)
        x = x.transpose(1, 2)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.norm_2(x)
        x = x.transpose(1, 2)
        x = self.drop(x)
        x = self.proj(x * x_mask)

        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, enc_blocks: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)

        self.pre = PreNet(
            channels=[hidden_size, hidden_size * 2, hidden_size],
            kernels=[11, 7],
            hidden_size=hidden_size,
        )

        self.encflow = nn.ModuleList()

        for _ in range(enc_blocks):
            self.encflow.append(EncModule(hidden_size))

        self.proj_m = nn.Linear(hidden_size, hidden_size)
        self.proj_log = nn.Linear(hidden_size, hidden_size)

        self.dpred = DurationPredictor(
            in_channels=hidden_size,
            filter_channels=hidden_size,
            kernel_size=3,
            p_dropout=0.1,
        )

    def forward(self, x, x_lengths):
        """
        Input is text token sequence [B, T_text]
        """
        x = self.emb(x)  # [B, T_text, H]
        x = torch.transpose(x, 1, 2)  # [B, H, T_text]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )[:, 0, :]
        # print(x_mask)
        # print(x.shape)
        out = self.pre(x)  # [B, H, T_text]

        x = x + out

        for f in self.encflow:
            x = f(x, x_mask)  # [B, H, T_text]

        x = x.transpose(1, 2)  # [B, T_text, H]

        x_m = self.proj_m(x)
        x_log = self.proj_log(x)

        dur = self.dpred(x, x_mask.unsqueeze(1))

        return x_m, x_log, dur, x_mask


if __name__ == "__main__":
    import numpy as np
    import torch

    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    some_input = torch.tensor(np.random.randint(0, 62, (6, 32)))  # [B, T_text]

    encoder = Encoder(62, 32, 1)

    output = encoder(some_input)

    print(output[0].shape, output[2].shape)

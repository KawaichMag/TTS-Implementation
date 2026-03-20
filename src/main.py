import math

import torch

from models.test_encoder import Encoder
from models.test_decoder import FlowSpecDecoder

import monotonic_align
from utils import commons
from utils.data import TextMelCollate, TextMelLoader
from utils.helpers import get_hparams_from_file
from utils.text.symbols import symbols

from torch.utils.data import DataLoader


def preprocess(y, y_lengths, y_max_length, n_sqz=1):
    if y_max_length is not None:
        y_max_length = (y_max_length // n_sqz) * n_sqz
        y = y[:, :, :y_max_length]
    y_lengths = (y_lengths // n_sqz) * n_sqz
    return y, y_lengths, y_max_length


if __name__ == "__main__":
    hparams = get_hparams_from_file("configs/base.json")

    loader = TextMelLoader("src/datasets/metadata.csv", hparams["data"])

    collate_fn = TextMelCollate(1)

    train_loader = DataLoader(
        loader,
        num_workers=2,
        shuffle=False,
        batch_size=hparams.train.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    enc = Encoder(
        vocab_size=len(symbols),
        hidden_size=80,
        enc_blocks=6,
    )

    dec = FlowSpecDecoder(
        in_channels=80,
        hidden_channels=16,
        dilation_rate=5,
        n_blocks=3,
        n_layers=3,
        kernel_size=3,
    )

    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        # print(x)
        # print(x_lengths)
        # print(y)
        # print(y_lengths)

        x_m, x_logs, dur, x_mask = enc(x, x_lengths)

        y_max_length = y.size(2)
        y, y_lengths, y_max_length = preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        z, logdet = dec(y, z_mask)

        with torch.no_grad():
            print(z.shape)

            x_s_sq_r = torch.exp(-2 * x_logs)
            print(x_s_sq_r.shape)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp2 = torch.matmul(
                x_s_sq_r.transpose(1, 2), -0.5 * (z**2)
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(
                (x_m * x_s_sq_r).transpose(1, 2), z
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            attn = (
                monotonic_align.maximum_path(logp, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

            print(attn)

        break

    # print(loader[0][1].shape)

    # print(enc(loader[0][0].unsqueeze(0))[0].shape)

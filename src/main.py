from models.test_encoder import Encoder
from utils.data import TextMelCollate, TextMelLoader
from utils.helpers import get_hparams_from_file
from utils.text.symbols import symbols

from torch.utils.data import DataLoader

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
        hidden_size=32,
        enc_blocks=6,
    )

    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        # print(x)
        # print(x_lengths)
        # print(y)
        # print(y_lengths)

        x_m, x_log, dur, x_mask = enc(x, x_lengths)

        break

    print(dur)

    # print(loader[0][1].shape)

    # print(enc(loader[0][0].unsqueeze(0))[0].shape)

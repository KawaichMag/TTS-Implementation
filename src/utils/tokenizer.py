class CharTokenizer:
    def __init__(self, texts: list[str], preset_symbol_to_id: dict[str, int] | None = None):
        if preset_symbol_to_id is not None:
            self.symbol_to_id = dict(preset_symbol_to_id)
            self.id_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_id.items()}
            self.pad_token = "_" if "_" in self.symbol_to_id else next(iter(self.symbol_to_id))
            self.unk_token = "?" if "?" in self.symbol_to_id else self.pad_token
            self.pad_id = self.symbol_to_id[self.pad_token]
            self.unk_id = self.symbol_to_id[self.unk_token]
            return

        merged = "".join(texts)
        symbols = ["_"] + sorted(set(merged))
        if "?" not in symbols:
            symbols.append("?")
        self.pad_token = "_"
        self.unk_token = "?"
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.id_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_id.items()}
        self.pad_id = self.symbol_to_id[self.pad_token]
        self.unk_id = self.symbol_to_id[self.unk_token]

    def encode(self, text: str) -> list[int]:
        return [self.symbol_to_id.get(ch, self.unk_id) for ch in normalize_text(text)]

    @property
    def vocab_size(self) -> int:
        return len(self.symbol_to_id)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

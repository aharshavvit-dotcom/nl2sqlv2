from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


class Vocabulary:
    def __init__(self, min_freq: int = 1, max_size: int | None = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def build(self, token_sequences: list[list[str]]) -> None:
        counts: Counter[str] = Counter()
        for tokens in token_sequences:
            counts.update(tokens)
        candidates = [
            token
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= self.min_freq and token not in self.token_to_id
        ]
        if self.max_size is not None:
            candidates = candidates[: max(0, self.max_size - len(SPECIAL_TOKENS))]
        self.token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)
        for token in candidates:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        if max_len <= 0:
            return []
        ids = [self.token_to_id[CLS_TOKEN]]
        available = max(0, max_len - 2)
        ids.extend(self.token_to_id.get(token, self.unk_id) for token in tokens[:available])
        if max_len > 1:
            ids.append(self.token_to_id[SEP_TOKEN])
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        tokens = []
        for idx in ids:
            if 0 <= int(idx) < len(self.id_to_token):
                tokens.append(self.id_to_token[int(idx)])
            else:
                tokens.append(UNK_TOKEN)
        return tokens

    def save(self, path: str) -> None:
        payload = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "token_to_id": self.token_to_id,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(min_freq=int(payload.get("min_freq", 1)), max_size=payload.get("max_size"))
        token_to_id = {str(token): int(idx) for token, idx in payload["token_to_id"].items()}
        vocab.token_to_id = dict(sorted(token_to_id.items(), key=lambda item: item[1]))
        vocab.id_to_token = [token for token, _ in sorted(token_to_id.items(), key=lambda item: item[1])]
        return vocab

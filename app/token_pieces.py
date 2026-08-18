from __future__ import annotations

from typing import Sequence

REPLACEMENT = "�"


class TokenPieceBuilder:
    """Turns token ids into display strings, merging multi-byte fragments.

    Byte-level BPE tokenizers can split one character into several ids; decoding
    such an id alone yields U+FFFD. Fragments are held until the group decodes
    cleanly, and the merged text is assigned to the last id of the group.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._open: list[int] = []

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def push(self, token_ids: Sequence[int]) -> list[str]:
        pieces: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            piece = self._decode([token_id])
            if REPLACEMENT in piece:
                self._open.append(token_id)
                merged = self._decode(self._open)
                if REPLACEMENT in merged:
                    pieces.append("")
                else:
                    pieces.append(merged)
                    self._open = []
            elif self._open:
                pieces.append(self._decode(self._open) + piece)
                self._open = []
            else:
                pieces.append(piece)
        return pieces


def token_pieces(tokenizer, token_ids: Sequence[int]) -> list[str]:
    return TokenPieceBuilder(tokenizer).push(token_ids)

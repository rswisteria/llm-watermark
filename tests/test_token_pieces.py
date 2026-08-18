from app.token_pieces import TokenPieceBuilder, token_pieces


class ByteTokenizer:
    """1 ID = 1 byte。日本語 1 文字が 3 トークンに分かれる状況を再現する。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return bytes(int(i) for i in token_ids).decode("utf-8", errors="replace")


def ids(text):
    return list(text.encode("utf-8"))


def test_ascii_tokens_map_one_to_one():
    assert token_pieces(ByteTokenizer(), ids("ab c")) == ["a", "b", " ", "c"]


def test_multibyte_fragments_are_merged_onto_last_token():
    pieces = token_pieces(ByteTokenizer(), ids("あb"))
    assert pieces == ["", "", "あ", "b"]


def test_open_group_survives_across_push_calls():
    builder = TokenPieceBuilder(ByteTokenizer())
    raw = ids("字")
    assert builder.push(raw[:2]) == ["", ""]
    assert builder.push(raw[2:]) == ["字"]


def test_unfinished_group_is_flushed_onto_next_clean_token():
    builder = TokenPieceBuilder(ByteTokenizer())
    raw = ids("字")
    assert builder.push(raw[:1]) == [""]
    flushed = builder.push(ids("x"))
    assert flushed[0].endswith("x")
    assert "�" in flushed[0]

import torch

from app.streaming import IdRecordingStreamer, StreamChunk


class SpaceTokenizer:
    """decode は各 ID を "t<id> " にする（常に空白で終わるので毎回確定する）。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"t{int(i)} " for i in token_ids)


class HoldingTokenizer:
    """空白を出さないので、end() まで確定しない。"""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f"t{int(i)}" for i in token_ids)


def drain(streamer):
    return list(streamer)


def test_prompt_ids_are_skipped_and_chunks_pair_text_with_ids():
    streamer = IdRecordingStreamer(SpaceTokenizer(), skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([[1, 2]]))          # prompt
    streamer.put(torch.tensor([3]))               # 1D like real generate
    streamer.put(torch.tensor([[4, 5]]))          # 2D like the fake model
    streamer.end()

    chunks = drain(streamer)
    assert chunks == [
        StreamChunk(text="t3 ", token_ids=[3]),
        StreamChunk(text="t4 t5 ", token_ids=[4, 5]),
    ]


def test_end_flushes_pending_ids_with_held_text():
    streamer = IdRecordingStreamer(HoldingTokenizer(), skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([[1]]))
    streamer.put(torch.tensor([7]))
    streamer.put(torch.tensor([8]))
    streamer.end()

    chunks = drain(streamer)
    # 空白が無いので put のたびに text="" で ID だけ流れ、end() で全文が確定する
    assert [c.token_ids for c in chunks] == [[7], [8], []]
    assert "".join(c.text for c in chunks) == "t7t8"
    assert chunks[-1].text == "t7t8"


def test_without_skip_prompt_first_put_is_recorded():
    streamer = IdRecordingStreamer(SpaceTokenizer(), skip_prompt=False)
    streamer.put(torch.tensor([[1, 2]]))
    streamer.end()
    assert drain(streamer)[0].token_ids == [1, 2]

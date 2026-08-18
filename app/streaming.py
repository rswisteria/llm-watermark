from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TextIteratorStreamer


@dataclass(frozen=True)
class StreamChunk:
    text: str
    token_ids: list[int] = field(default_factory=list)


class IdRecordingStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that yields StreamChunk(text, token_ids).

    ``token_ids`` are the ids received via ``put`` since the previous chunk was
    emitted, so downstream code can score tokens while text is still being held
    back by the word-boundary heuristic. The prompt is not recorded when
    ``skip_prompt`` is true.
    """

    def __init__(self, tokenizer, skip_prompt: bool = False, timeout=None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, timeout=timeout, **decode_kwargs)
        self._pending_ids: list[int] = []

    def put(self, value):
        is_prompt = self.skip_prompt and self.next_tokens_are_prompt
        if not is_prompt:
            ids = value[0] if len(value.shape) > 1 else value
            self._pending_ids.extend(int(token_id) for token_id in ids.tolist())
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        ids = self._pending_ids
        self._pending_ids = []
        # transformers' TextStreamer.end() always calls on_finalized_text once
        # more to flush any cached tail, even when nothing is left to flush
        # (empty text and no new ids since the last chunk). Skip emitting a
        # spurious empty chunk in that case; still signal the stream's end.
        if stream_end and not text and not ids:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
            return
        chunk = StreamChunk(text=text, token_ids=ids)
        self.text_queue.put(chunk, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

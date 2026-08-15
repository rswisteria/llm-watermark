import argparse
from dataclasses import dataclass

import torch

from watermark import (
    DetectionResult,
    WatermarkConfig,
    WatermarkDetector,
    WatermarkLogitsProcessor,
)


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "日本の四季と自然の魅力について、日本語で読みやすい解説文を書いてください。具体例を交え、200トークン程度で説明してください。"
HUMAN_SAMPLE = """日本の春は、冬の寒さがゆるみ、草木が芽吹き始める季節です。桜や新緑を楽しみに多くの人が公園や山を訪れます。夏には海や高原の風景が鮮やかになり、秋には紅葉と実りの季節を迎えます。冬の雪景色は静かで美しく、地域ごとに異なる自然の表情を見せてくれます。四季の変化は、暮らしや文化にも深く結びついています。"""


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_ids: tuple[int, ...]


def format_detection(result: DetectionResult) -> str:
    label = "透かしあり" if result.is_watermarked else "透かしなし"
    return (
        f"T={result.token_count}, green={result.green_count}, "
        f"z={result.z_score:.3f}, p={result.p_value:.3e}, 判定={label}"
    )


def load_tokenizer(model_name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.to("cpu")
    model.eval()
    return model


def generate_text(
    model,
    tokenizer,
    prompt,
    config,
    max_new_tokens,
    seed,
    watermarked,
):
    from transformers import LogitsProcessorList

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_length = inputs["input_ids"].shape[-1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    if watermarked:
        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [WatermarkLogitsProcessor(len(tokenizer), config)]
        )

    torch.manual_seed(seed)
    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_length:].tolist()
    special_ids = set(tokenizer.all_special_ids)
    visible_token_ids = tuple(
        int(token_id) for token_id in generated_ids if token_id not in special_ids
    )
    return GeneratedText(
        text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        token_ids=visible_token_ids,
    )


def detect_text(tokenizer, text, config):
    detector = WatermarkDetector(
        vocab_size=len(tokenizer),
        config=config,
        tokenizer=tokenizer,
    )
    return detector.detect(text)


def detect_generated_text(tokenizer, generated, config):
    detector = WatermarkDetector(
        vocab_size=len(tokenizer),
        config=config,
        tokenizer=tokenizer,
    )
    round_trip_ids = tuple(
        tokenizer(
            generated.text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].tolist()
    )
    if round_trip_ids == generated.token_ids:
        return detector.detect(generated.text)

    print("警告: ラウンドトリップのトークン化が一致しないため、生成時のトークンIDで検出しました。")
    return detector.detect_token_ids(generated.token_ids)


def run_detection_only(tokenizer, text, config):
    return detect_text(tokenizer, text, config)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 watermark demonstration")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--hash-key", type=int, default=15485863)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--ignore-repeated-bigrams", action="store_true")
    detection_group = parser.add_mutually_exclusive_group()
    detection_group.add_argument("--detect")
    detection_group.add_argument("--detect-file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = WatermarkConfig(
        gamma=args.gamma,
        delta=args.delta,
        hash_key=args.hash_key,
        z_threshold=args.z_threshold,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
    )

    if args.detect is not None:
        tokenizer = load_tokenizer(args.model)
        print(format_detection(run_detection_only(tokenizer, args.detect, config)))
        return

    if args.detect_file is not None:
        with open(args.detect_file, encoding="utf-8") as text_file:
            text = text_file.read()
        tokenizer = load_tokenizer(args.model)
        print(format_detection(run_detection_only(tokenizer, text, config)))
        return

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)

    unwatermarked = generate_text(
        model,
        tokenizer,
        DEFAULT_PROMPT,
        config,
        args.max_new_tokens,
        args.seed,
        watermarked=False,
    )
    print("透かしなし生成:")
    print(unwatermarked.text)
    print(format_detection(detect_generated_text(tokenizer, unwatermarked, config)))

    watermarked = generate_text(
        model,
        tokenizer,
        DEFAULT_PROMPT,
        config,
        args.max_new_tokens,
        args.seed,
        watermarked=True,
    )
    print("\n透かしあり生成:")
    print(watermarked.text)
    print(format_detection(detect_generated_text(tokenizer, watermarked, config)))

    print("\n人間作成サンプル:")
    print(HUMAN_SAMPLE)
    print(format_detection(detect_text(tokenizer, HUMAN_SAMPLE, config)))


if __name__ == "__main__":
    main()

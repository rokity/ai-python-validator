"""LoRA fine-tuning, evaluation and inference for Python syntax classification (BIFI).

Commands:
  train    sample BIFI, fine-tune with LoRA, evaluate on the held-out test set and on the exercises
  eval     evaluate a model (zero-shot baseline, or with --adapter) on BIFI test or on the exercises
  predict  classify a single snippet

Design notes:
- the BIFI JSON files are streamed (ijson) and only a balanced random sample is kept;
- long snippets are dropped, never truncated (a truncated valid snippet looks invalid);
- prompts go through the chat template, as at inference time with llama-server;
- training stops automatically on NaN/inf and never saves a corrupted adapter.
"""

import argparse
import ast
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.metrics.classification import compute_report
from src.repositories.prompt_syntax_validator_repository import (
    PromptSyntaxValidatorRepository,
)

LABELS = ("VALID", "INVALID")


def _syntax_prompt(code: str) -> str:
    return PromptSyntaxValidatorRepository(
        language="python", code=code
    ).assemble_prompt()


def _chat_prompt(tokenizer: Any, code: str) -> str:
    """Same format the chat endpoint of llama-server applies at inference time."""
    messages = [{"role": "user", "content": _syntax_prompt(code)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- data


def _extract_json_code(value: Any) -> list[str]:
    if isinstance(value, dict):
        code = value.get("code_string")
        if isinstance(code, str) and code.strip():
            return [code]
        snippets: list[str] = []
        for item in value.values():
            snippets.extend(_extract_json_code(item))
        return snippets
    if isinstance(value, list):
        snippets = []
        for item in value:
            snippets.extend(_extract_json_code(item))
        return snippets
    return []


def _iter_codes(path: Path) -> Iterator[str]:
    """Yield code_string values one by one, without loading the whole file when possible."""
    with path.open("rb") as handle:
        first = b""
        while True:
            char = handle.read(1)
            if not char or not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == b"[":
            try:
                import ijson
            except ImportError:
                print(
                    f"[warn] ijson not installed: loading {path.name} fully in RAM (pip install ijson)"
                )
            else:
                for item in ijson.items(handle, "item"):
                    code = item.get("code_string") if isinstance(item, dict) else None
                    if isinstance(code, str):
                        yield code
                return
        yield from _extract_json_code(json.load(handle))


def _sample_class(
    root: Path, per_class: int, max_chars: int, rng: random.Random
) -> list[str]:
    """Reservoir sampling: uniform sample of `per_class` snippets in a single pass."""
    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No .json file under {root}. The .good/.bad chunk files contain anonymized "
            "tokens, not raw code: use orig.good.json / orig.bad.json (code_string)."
        )
    sample: list[str] = []
    seen = 0
    for path in files:
        for code in _iter_codes(path):
            if not code.strip() or len(code) > max_chars:
                continue
            seen += 1
            if len(sample) < per_class:
                sample.append(code)
            else:
                slot = rng.randrange(seen)
                if slot < per_class:
                    sample[slot] = code
            if seen % 250_000 == 0:
                print(f"  {root.name}: scanned {seen} eligible snippets...")
    print(f"{root.name}: {seen} eligible snippets, kept {len(sample)}")
    return sample


def load_examples(
    data_dir: Path,
    per_class: int,
    max_chars: int,
    seed: int,
    cache: Path | None = None,
) -> list[dict[str, str]]:
    """Balanced sample of VALID/INVALID examples (cached as JSONL after the first scan)."""
    if cache is not None and cache.is_file():
        with cache.open(encoding="utf-8") as handle:
            examples = [json.loads(line) for line in handle if line.strip()]
        print(f"Loaded {len(examples)} cached examples from {cache}")
        return examples
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    for folder, label in (("orig_good_code", "VALID"), ("orig_bad_code", "INVALID")):
        root = data_dir / folder
        if not root.is_dir():
            raise FileNotFoundError(f"Expected dataset directory: {root}")
        examples.extend(
            {"code": code, "label": label}
            for code in _sample_class(root, per_class, max_chars, rng)
        )
    if not examples:
        raise ValueError(f"No code examples found under {data_dir}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Cached {len(examples)} examples in {cache}")
    return examples


def split_examples(
    examples: list[dict[str, str]], test_size: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for label in LABELS:
        group = [example for example in examples if example["label"] == label]
        if len(group) < 2:
            raise ValueError(f"At least two examples of class {label} are required")
        rng.shuffle(group)
        test_count = max(1, round(len(group) * test_size))
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ------------------------------------------------------------------------ helpers


def _pick_dtype(torch: Any, name: str) -> Any:
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available():
        # bfloat16 needs compute capability >= 8 (A100, L4, A10G, RTX 30xx+); T4 and P100
        # do not have it. Use float32 there (optionally with --amp for mixed precision).
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )
    # float16 gave NaN losses on Qwen2.5 (activation overflow): use float32 on MPS/CPU.
    # On Apple Silicon try --dtype bfloat16 to save memory, and watch the first 100 steps.
    return torch.float32


def _diverged(logs: dict[str, Any]) -> bool:
    """True if a Trainer log entry contains a NaN/inf loss or gradient norm."""
    for key in ("loss", "grad_norm", "eval_loss"):
        value = logs.get(key)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return True
    return False


def _parse_answer(answer: str) -> str:
    text = answer.strip().upper()
    if text.startswith("INVALID"):
        return "INVALID"
    if text.startswith("VALID"):
        return "VALID"
    return "UNPARSED"


def _classify(model: Any, tokenizer: Any, code: str, max_length: int) -> str:
    import torch

    prompt = _chat_prompt(tokenizer, code)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {
        key: value[:, -max_length:].to(model.device) for key, value in inputs.items()
    }
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return _parse_answer(answer)


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def load_exercise_examples(path: Path) -> list[dict[str, str]]:
    """Exercise implementations; the syntax label comes from ast.parse (as in the experiments)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": str(implementation.get("id", "")),
            "code": implementation["code"],
            "label": "VALID" if _is_valid_python(implementation["code"]) else "INVALID",
        }
        for item in data["items"]
        for implementation in item["implementations"]
    ]


def load_cached_test_examples(
    cache: Path, test_size: float, seed: int
) -> list[dict[str, str]]:
    """Same held-out split used by `train` (same cache, test_size and seed)."""
    if not cache.is_file():
        raise FileNotFoundError(f"Sample cache not found: {cache}")
    with cache.open(encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle if line.strip()]
    return split_examples(examples, test_size, seed)[1]


def _evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, str]],
    max_length: int,
    title: str,
    save_dir: Path | None = None,
    tag: str = "test",
) -> Any:
    model.eval()
    model.config.use_cache = True
    predictions: list[str] = []
    for index, example in enumerate(examples, start=1):
        predictions.append(_classify(model, tokenizer, example["code"], max_length))
        if index % 100 == 0:
            print(f"  {title}: {index}/{len(examples)}")
    report = compute_report(
        [
            (example["label"], prediction)
            for example, prediction in zip(examples, predictions)
        ]
    )
    print(report.format(title))
    confusion = Counter(
        (example["label"], prediction)
        for example, prediction in zip(examples, predictions)
    )
    print(
        "Confusion (expected -> predicted): "
        + ", ".join(
            f"{expected}->{predicted}: {count}"
            for (expected, predicted), count in sorted(confusion.items())
        )
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "title": title,
            "total": report.total,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "weighted_f1": report.weighted_f1,
            "unparsed": report.unparsed,
            "per_class": {
                label: {
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                    "support": item.support,
                }
                for label, item in report.per_class.items()
            },
            "confusion": {f"{e}->{p}": c for (e, p), c in confusion.items()},
        }
        (save_dir / f"{tag}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        with (save_dir / f"{tag}_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for example, prediction in zip(examples, predictions):
                row = {key: example[key] for key in ("id", "label") if key in example}
                row["prediction"] = prediction
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {tag}_metrics.json and {tag}_predictions.jsonl in {save_dir}")
    return report


def _load_for_inference(
    model_name: str, adapter: Path | None, dtype_name: str
) -> tuple[Any, Any]:
    """Load base model (+ optional LoRA adapter) on the best available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if adapter else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_pick_dtype(torch, dtype_name)
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------- commands


def _training_command(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements.txt (torch, "
            "transformers, peft, datasets, accelerate, ijson)."
        ) from exc

    examples = load_examples(
        args.data, args.max_per_class, args.max_chars, args.seed, args.cache
    )
    train_examples, test_examples = split_examples(examples, args.test_size, args.seed)
    print(
        f"Loaded {len(examples)} examples: {len(train_examples)} train, "
        f"{len(test_examples)} test"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _pick_dtype(torch, args.dtype)
    print(f"Loading {args.model} in {dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        prompt_ids = tokenizer(
            _chat_prompt(tokenizer, example["code"]), add_special_tokens=False
        )["input_ids"]
        answer_ids = tokenizer(
            f"{example['label']}{tokenizer.eos_token or ''}", add_special_tokens=False
        )["input_ids"]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def prepare(rows: list[dict[str, str]]) -> Any:
        data = Dataset.from_list(rows).map(tokenize, remove_columns=["code", "label"])
        before = len(data)
        # drop (never truncate): a truncated VALID snippet would look like a syntax error
        data = data.filter(lambda row: len(row["input_ids"]) <= args.max_length)
        print(
            f"Dropped {before - len(data)}/{before} examples longer than {args.max_length} tokens"
        )
        return data

    class StopOnDivergence(TrainerCallback):
        def __init__(self) -> None:
            self.diverged = False

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if logs and _diverged(logs):
                self.diverged = True
                control.should_training_stop = True
                control.should_save = False  # never write a checkpoint with NaN weights
                control.should_evaluate = False
                print(
                    f"[error] NaN/inf at step {state.global_step}: stopping training. "
                    "Try --dtype float32 (or bfloat16) and/or a lower --learning-rate."
                )

    guard = StopOnDivergence()
    train_data = prepare(train_examples)
    eval_data = prepare(test_examples[: args.eval_samples])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            fp16=args.amp,
            seed=args.seed,
            report_to="none",
        ),
        callbacks=[guard],
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100
        ),
    )
    resume = bool(args.resume) and any(args.output.glob("checkpoint-*"))
    if args.resume and not resume:
        print(f"No checkpoint found in {args.output}: starting from scratch")
    trainer.train(resume_from_checkpoint=True if resume else None)
    if guard.diverged:
        print("Training diverged: adapter NOT saved and evaluation skipped.")
        return
    adapter_path = args.output / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")
    _evaluate(
        model,
        tokenizer,
        test_examples,
        args.max_length,
        "BIFI held-out test",
        args.output,
        "bifi_test",
    )
    if args.exercises.is_file():
        _evaluate(
            model,
            tokenizer,
            load_exercise_examples(args.exercises),
            args.max_length,
            "Exercises (160 implementations)",
            args.output,
            "exercises",
        )


def _predict_command(args: argparse.Namespace) -> None:
    model, tokenizer = _load_for_inference(args.model, args.adapter, args.dtype)
    label = _classify(model, tokenizer, args.code, args.max_length)
    if label == "UNPARSED":
        raise RuntimeError("Model returned an unrecognized label")
    print(label)


def _eval_command(args: argparse.Namespace) -> None:
    """Evaluate the base model (zero-shot) or base + adapter on a labelled set."""
    if args.source == "exercises":
        examples = load_exercise_examples(args.exercises)
    else:
        examples = load_cached_test_examples(args.cache, args.test_size, args.seed)
    if args.limit:
        examples = examples[: args.limit]
    model, tokenizer = _load_for_inference(args.model, args.adapter, args.dtype)
    name = "fine-tuned" if args.adapter else "zero-shot"
    _evaluate(
        model,
        tokenizer,
        examples,
        args.max_length,
        f"{args.source} / {name} / {args.model}",
        args.output,
        f"{args.source}_{name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fine-tune a causal LM with LoRA on BIFI")
    train.add_argument(
        "--data", type=Path, required=True, help="Extracted BIFI dataset directory"
    )
    train.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    train.add_argument("--output", type=Path, default=Path("models/bifi-syntax"))
    train.add_argument(
        "--cache",
        type=Path,
        default=Path("data/bifi_sample.jsonl"),
        help="JSONL cache of the sampled examples (delete it to resample)",
    )
    train.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Snippets sampled for each class (VALID/INVALID)",
    )
    train.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Skip snippets longer than this many characters",
    )
    train.add_argument("--test-size", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=float, default=1)
    train.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, overrides --epochs (use e.g. 20 to measure speed)",
    )
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="Max tokens per example; longer examples are dropped",
    )
    train.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Evaluate and save a checkpoint every N steps",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in --output",
    )
    train.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Test examples used for the eval loss during training",
    )
    train.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    train.add_argument(
        "--amp",
        action="store_true",
        help="CUDA only: fp16 mixed precision with loss scaling (faster on T4/P100; keep weights in float32)",
    )
    train.add_argument("--no-gradient-checkpointing", action="store_true")
    train.add_argument(
        "--exercises",
        type=Path,
        default=Path("src/samples/configs/implementation.json"),
        help="Exercises evaluated after training (skipped if missing)",
    )
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.set_defaults(handler=_training_command)

    evaluate = commands.add_parser(
        "eval",
        help="Evaluate a model (zero-shot, or with --adapter) on BIFI test or exercises",
    )
    evaluate.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    evaluate.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter directory; omit for the zero-shot baseline",
    )
    evaluate.add_argument(
        "--source", choices=["bifi", "exercises"], default="exercises"
    )
    evaluate.add_argument("--cache", type=Path, default=Path("data/bifi_sample.jsonl"))
    evaluate.add_argument(
        "--exercises",
        type=Path,
        default=Path("src/samples/configs/implementation.json"),
    )
    evaluate.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Must match the value used by `train`",
    )
    evaluate.add_argument("--seed", type=int, default=42, help="Must match `train`")
    evaluate.add_argument(
        "--limit", type=int, default=0, help="Evaluate only the first N examples"
    )
    evaluate.add_argument("--max-length", type=int, default=768)
    evaluate.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("models/eval"),
        help="Directory for the metrics/predictions files",
    )
    evaluate.set_defaults(handler=_eval_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
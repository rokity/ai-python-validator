"""LoRA fine-tuning and evaluation for CodeComplex complexity classification."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.classification import compute_report
from src.repositories.prompt_complexity_repository import (
    COMPLEXITY_CLASSES,
    PromptComplexityRepository,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_URL = (
    "https://raw.githubusercontent.com/yonsei-toc/CodeComple/main/new_data.jsonl"
)
SOURCE_LABELS = {
    "constant": "O(1)",
    "logn": "O(log n)",
    "linear": "O(n)",
    "nlogn": "O(n log n)",
    "quadratic": "O(n^2)",
    "cubic": "O(n^3)",
    "np": "NP-hard",
}


def _prompt(code: str, problem: str) -> str:
    return PromptComplexityRepository(
        language="java", code=code, exercise=problem
    ).assemble_prompt()


def _chat_prompt(tokenizer: Any, example: dict[str, str]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _prompt(example["code"], example["problem"])}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _normalize_example(row: dict[str, Any], index: int) -> dict[str, str] | None:
    code = row.get("src")
    problem = row.get("problem")
    source_label = row.get("complexity")
    if not isinstance(code, str) or not code.strip():
        return None
    if not isinstance(problem, str) or not problem.strip():
        return None
    if not isinstance(source_label, str):
        return None
    label = SOURCE_LABELS.get(source_label.strip().lower())
    if label is None:
        return None
    return {
        "id": str(index),
        "problem": problem.strip(),
        "code": code,
        "label": label,
    }


def load_examples(data_dir: Path) -> list[dict[str, str]]:
    """Load CodeComplex's primary JSON-array file and normalize its labels."""
    dataset_path = data_dir / "new_data.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"CodeComplex dataset not found: {dataset_path}. "
            "Download it first with `make download_codecomplex`."
        )
    try:
        rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON dataset: {dataset_path}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {dataset_path}")
    examples = [
        example
        for index, row in enumerate(rows)
        if isinstance(row, dict)
        and (example := _normalize_example(row, index)) is not None
    ]
    if not examples:
        raise ValueError(
            f"No usable examples in {dataset_path}; expected src, problem and "
            "complexity fields with recognized labels."
        )
    counts = Counter(example["label"] for example in examples)
    print(
        f"Loaded {len(examples)} examples from {dataset_path}: "
        + ", ".join(f"{label}={counts[label]}" for label in COMPLEXITY_CLASSES if counts[label])
    )
    return examples


def split_examples(
    examples: list[dict[str, str]], test_size: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Stratify by label sets while keeping every Codeforces problem in one split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        groups[example["problem"]].append(example)

    strata: dict[tuple[str, ...], list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups.values():
        label_signature = tuple(sorted({example["label"] for example in group}))
        strata[label_signature].append(group)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for label_groups in strata.values():
        rng.shuffle(label_groups)
        test_count = max(1, round(len(label_groups) * test_size))
        if test_count >= len(label_groups):
            test_count = len(label_groups) - 1
        for group_index, group in enumerate(label_groups):
            (test if group_index < test_count else train).extend(group)

    if not train or not test:
        raise ValueError("Not enough distinct problems to create train/test splits.")
    train_problems = {example["problem"] for example in train}
    test_problems = {example["problem"] for example in test}
    if train_problems & test_problems:
        raise RuntimeError("Problem leakage detected between train and test.")
    missing_test_labels = set(COMPLEXITY_CLASSES) & {
        example["label"] for example in examples
    } - {example["label"] for example in test}
    if missing_test_labels:
        raise ValueError(
            "The grouped test split is missing classes "
            f"{sorted(missing_test_labels)}; consider changing --test-size."
        )
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _pick_dtype(torch: Any, name: str) -> Any:
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available():
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )
    return torch.float32


def _parse_answer(answer: str) -> str:
    return PromptComplexityRepository._check_response(answer) or "UNPARSED"


def _classify(
    model: Any, tokenizer: Any, example: dict[str, str], max_length: int
) -> str:
    import torch

    inputs = tokenizer(
        _chat_prompt(tokenizer, example),
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {
        key: value[:, -max_length:].to(model.device)
        for key, value in inputs.items()
    }
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _parse_answer(answer)


def _evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, str]],
    max_length: int,
    title: str,
    output: Path | None = None,
    tag: str = "test",
) -> Any:
    model.eval()
    model.config.use_cache = True
    predictions = []
    for index, example in enumerate(examples, start=1):
        predictions.append(_classify(model, tokenizer, example, max_length))
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
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            "title": title,
            "total": report.total,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "weighted_f1": report.weighted_f1,
            "unparsed": report.unparsed,
            "confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in confusion.items()
            },
        }
        (output / f"{tag}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        with (output / f"{tag}_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for example, prediction in zip(examples, predictions):
                row = {
                    key: example[key]
                    for key in ("id", "problem", "label")
                }
                row["prediction"] = prediction
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {tag}_metrics.json and {tag}_predictions.jsonl in {output}")
    return report


def _load_model(
    model_name: str, adapter: Path | None, dtype_name: str
) -> tuple[Any, Any]:
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
            "Training dependencies are missing. Install requirements.txt "
            "(torch, transformers, peft, datasets, accelerate)."
        ) from exc

    examples = load_examples(args.data)
    train_examples, test_examples = split_examples(
        examples, args.test_size, args.seed
    )
    if args.limit:
        train_examples = train_examples[:args.limit]
    if len({example["label"] for example in train_examples}) < 2:
        raise ValueError("Training data must contain at least two complexity classes.")
    print(
        f"Split by problem: {len(train_examples)} train, "
        f"{len(test_examples)} held-out test examples"
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
            _chat_prompt(tokenizer, example), add_special_tokens=False
        )["input_ids"]
        answer_ids = tokenizer(
            f"{example['label']}{tokenizer.eos_token or ''}",
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    def prepare(rows: list[dict[str, str]]) -> Any:
        data = Dataset.from_list(rows).map(
            tokenize, remove_columns=list(rows[0].keys())
        )
        before = len(data)
        data = data.filter(lambda row: len(row["input_ids"]) <= args.max_length)
        print(
            f"Dropped {before - len(data)}/{before} examples longer than "
            f"{args.max_length} tokens"
        )
        if not len(data):
            raise ValueError("No examples fit --max-length; increase it.")
        return data

    train_data = prepare(train_examples)
    eval_examples = test_examples
    if args.eval_samples and len(eval_examples) > args.eval_samples:
        eval_examples = random.Random(args.seed).sample(
            eval_examples, args.eval_samples
        )
    eval_data = prepare(eval_examples)

    class StopOnDivergence(TrainerCallback):
        def __init__(self) -> None:
            self.diverged = False

        def on_log(
            self, args, state, control, logs=None, **kwargs
        ):  # type: ignore[no-untyped-def]
            if logs and any(
                isinstance(logs.get(key), (int, float))
                and not math.isfinite(logs[key])
                for key in ("loss", "grad_norm", "eval_loss")
            ):
                self.diverged = True
                control.should_training_stop = True
                control.should_save = False
                control.should_evaluate = False
                print(
                    f"[error] NaN/inf at step {state.global_step}: "
                    "stopping training."
                )

    training_args = {
        "output_dir": str(args.output),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_pin_memory": False,
        "fp16": args.amp,
        "seed": args.seed,
        "report_to": "none",
    }
    supported_arguments = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in supported_arguments:
        training_args["warmup_ratio"] = 0.03
    elif "warmup_steps" in supported_arguments:
        steps_per_epoch = math.ceil(
            math.ceil(len(train_data) / args.batch_size) / args.grad_accum
        )
        total_steps = (
            args.max_steps
            if args.max_steps > 0
            else math.ceil(steps_per_epoch * args.epochs)
        )
        training_args["warmup_steps"] = round(total_steps * 0.03)
        print(
            "TrainingArguments does not support warmup_ratio; using "
            f"warmup_steps={training_args['warmup_steps']} instead."
        )
    else:
        raise RuntimeError(
            "This transformers version supports neither warmup_ratio nor "
            "warmup_steps in TrainingArguments. Upgrade transformers."
        )

    guard = StopOnDivergence()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        callbacks=[guard],
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100
        ),
    )
    resume = args.resume and any(args.output.glob("checkpoint-*"))
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
        "CodeComplex grouped held-out test",
        args.output,
        "codecomplex_test",
    )


def _eval_command(args: argparse.Namespace) -> None:
    examples = load_examples(args.data)
    _, test_examples = split_examples(examples, args.test_size, args.seed)
    if args.limit:
        test_examples = test_examples[: args.limit]
    model, tokenizer = _load_model(args.model, args.adapter, args.dtype)
    model_name = "fine-tuned" if args.adapter else "zero-shot"
    _evaluate(
        model,
        tokenizer,
        test_examples,
        args.max_length,
        f"CodeComplex held-out test / {model_name} / {args.model}",
        args.output,
        f"codecomplex_{model_name}",
    )


def _predict_command(args: argparse.Namespace) -> None:
    example = {"code": args.code, "problem": args.exercise}
    model, tokenizer = _load_model(args.model, args.adapter, args.dtype)
    prediction = _classify(model, tokenizer, example, args.max_length)
    if prediction == "UNPARSED":
        raise RuntimeError("Model returned an unrecognized complexity class")
    print(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fine-tune Qwen with LoRA on CodeComplex")
    train.add_argument("--data", type=Path, default=Path("data/codecomplex"))
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument(
        "--output", type=Path, default=Path("models/codecomplex-complexity")
    )
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--limit", type=int, default=0)
    train.add_argument("--eval-samples", type=int, default=200)
    train.add_argument("--epochs", type=float, default=1)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=768)
    train.add_argument("--eval-steps", type=int, default=50)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    train.add_argument("--amp", action="store_true")
    train.add_argument("--no-gradient-checkpointing", action="store_true")
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.set_defaults(handler=_training_command)

    evaluate = commands.add_parser("eval", help="Evaluate base Qwen or an adapter")
    evaluate.add_argument("--data", type=Path, default=Path("data/codecomplex"))
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument("--adapter", type=Path)
    evaluate.add_argument("--test-size", type=float, default=0.2)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--max-length", type=int, default=768)
    evaluate.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    evaluate.add_argument("--output", type=Path, default=Path("models/eval"))
    evaluate.set_defaults(handler=_eval_command)

    predict = commands.add_parser("predict", help="Classify one Java solution")
    predict.add_argument("--model", default=DEFAULT_MODEL)
    predict.add_argument("--adapter", type=Path)
    predict.add_argument("--exercise", default="")
    predict.add_argument("--code", required=True)
    predict.add_argument("--max-length", type=int, default=768)
    predict.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    predict.set_defaults(handler=_predict_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

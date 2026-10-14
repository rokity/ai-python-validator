"""LoRA fine-tuning and inference for BIFI syntax classification."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.repositories.prompt_syntax_validator_repository import (
    PromptSyntaxValidatorRepository,
)


def _syntax_prompt(code: str) -> str:
    return PromptSyntaxValidatorRepository(language="python", code=code).assemble_prompt()


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


def load_examples(data_dir: Path) -> list[dict[str, str]]:
    """Read BIFI orig_good_code/orig_bad_code trees into labeled examples."""
    examples: list[dict[str, str]] = []
    for folder, label in (("orig_good_code", "VALID"), ("orig_bad_code", "INVALID")):
        root = data_dir / folder
        if not root.is_dir():
            raise FileNotFoundError(f"Expected dataset directory: {root}")
        files = sorted(path for path in root.rglob("*") if path.is_file())
        has_source_json = any(path.name in {"orig.good.json", "orig.bad.json"} for path in files)
        for path in files:
            if has_source_json and path.suffix.lower() in {".good", ".bad"}:
                continue
            if path.suffix.lower() == ".json":
                try:
                    contents = json.loads(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Cannot parse BIFI JSON file {path}: {exc}") from exc
                codes = _extract_json_code(contents)
            else:
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Cannot read dataset file {path} as UTF-8") from exc
                if path.suffix.lower() in {".good", ".bad"}:
                    codes = [line for line in text.splitlines() if line.strip()]
                else:
                    codes = [text] if text.strip() else []
            examples.extend({"code": code, "label": label} for code in codes if code.strip())
    if not examples:
        raise ValueError(f"No code examples found under {data_dir}")
    return examples


def split_examples(
    examples: list[dict[str, str]], test_size: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for label in ("VALID", "INVALID"):
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
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements-training.txt."
        ) from exc

    examples = load_examples(args.data)
    train_examples, test_examples = split_examples(examples, args.test_size, args.seed)
    print(
        f"Loaded {len(examples)} examples: {len(train_examples)} train, "
        f"{len(test_examples)} test"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
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

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        prompt = _syntax_prompt(example["code"])
        answer = f" {example['label']}{tokenizer.eos_token or ''}"
        prompt_ids = tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=args.max_length
        )["input_ids"]
        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, args.max_length - len(prompt_ids)),
        )["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: args.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    train_data = Dataset.from_list(train_examples).map(tokenize, remove_columns=["code", "label"])
    eval_data = Dataset.from_list(test_examples).map(tokenize, remove_columns=["code", "label"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            report_to="none",
        ),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100
        ),
    )
    trainer.train()
    adapter_path = args.output / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")
    _evaluate(model, tokenizer, test_examples, args.max_length)


def _evaluate(model: Any, tokenizer: Any, examples: list[dict[str, str]], max_length: int) -> None:
    import torch

    model.eval()
    predictions: list[str] = []
    targets: list[str] = []
    for example in examples:
        prompt = _syntax_prompt(example["code"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predictions.append("INVALID" if "INVALID" in answer.upper() else "VALID" if "VALID" in answer.upper() else "UNPARSED")
        targets.append(example["label"])
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    print(f"Test accuracy: {correct / len(targets):.4f} ({correct}/{len(targets)})")
    for label in ("VALID", "INVALID"):
        tp = sum(pred == label and target == label for pred, target in zip(predictions, targets))
        fp = sum(pred == label and target != label for pred, target in zip(predictions, targets))
        fn = sum(pred != label and target == label for pred, target in zip(predictions, targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"{label}: precision={precision:.4f}, recall={recall:.4f}")


def _predict_command(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install requirements-training.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    prompt = _syntax_prompt(args.code)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    normalized = answer.strip().upper()
    if normalized.startswith("INVALID"):
        print("INVALID")
    elif normalized.startswith("VALID"):
        print("VALID")
    else:
        raise RuntimeError(f"Model returned an unrecognized label: {answer!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fine-tune a causal LM with LoRA on BIFI")
    train.add_argument("--data", type=Path, required=True, help="Extracted BIFI dataset directory")
    train.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    train.add_argument("--output", type=Path, default=Path("models/bifi-syntax"))
    train.add_argument("--test-size", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=float, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.set_defaults(handler=_training_command)

    predict = commands.add_parser("predict", help="Classify a snippet using a saved LoRA adapter")
    predict.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    predict.add_argument("--adapter", type=Path, required=True)
    predict.add_argument("--code", required=True)
    predict.add_argument("--max-length", type=int, default=2048)
    predict.set_defaults(handler=_predict_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

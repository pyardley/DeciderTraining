"""Score Decider 0.8B before and after this fine-tune.

Both models are Mapika/decider-0.8b with the same Unsloth decision head.
"Untrained" is that head and the LoRA at step 0 (the same init Training.py
uses). "Trained" is decider-0.8b-lora, written at the end of training.

The scored rows are the test split of LocalLLaMA/typed-decisions. Training
uses the train split only, with 80 of those rows held out for calibration.

Run it after training finishes. The 4GB card holds one model at a time, so
the script scores the untrained model, frees it, then scores the trained one.

    python compare.py
"""

import argparse
import gc
import json
from pathlib import Path

from unsloth import FastDecisionModel
from unsloth.models._decision_common import QUESTION_TYPES, _metrics, _served_temperatures
from unsloth.models.decision import _decision_logits
from datasets import load_dataset

MODEL_NAME = "Mapika/decider-0.8b"
DATASET = "LocalLLaMA/typed-decisions"
DATASET_CONFIG = "all"
MAX_SEQ_LENGTH = 1024
TRAINED_DIR = "decider-0.8b-lora"
RESULTS_PATH = Path("results") / "compare.json"


def load_untrained():
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    return model, tokenizer


def load_trained():
    return FastDecisionModel.from_pretrained(
        model_name = TRAINED_DIR,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )


def release(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score(model, tokenizer, items, workflows, batch_size: int) -> dict:
    FastDecisionModel.for_inference(model)
    logits, flat = _decision_logits(model, tokenizer, items, batch_size)
    temperatures = _served_temperatures(model.decision_config, logits, flat)

    def take(indices) -> dict:
        metrics = _metrics(
            [logits[i] for i in indices],
            [flat[i] for i in indices],
            [temperatures[i] for i in indices],
        )
        metrics["n"] = len(indices)
        return metrics

    by_type = {}
    for index, name in enumerate(QUESTION_TYPES):
        chosen = [i for i, item in enumerate(flat) if item["qtype"] == index]
        if chosen:
            by_type[name] = take(chosen)

    by_workflow = {}
    for name in sorted(set(workflows)):
        chosen = [i for i, item in enumerate(flat) if workflows[item["row"]] == name]
        if chosen:
            by_workflow[name] = take(chosen)

    return {"overall": take(range(len(flat))), "by_type": by_type, "by_workflow": by_workflow}


def evaluate_model(kind: str, loader, dataset, batch_size: int) -> dict:
    print(f"\nLoading {kind} model")
    model, tokenizer = loader()
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    print(f"Skipped {report['skipped']} of {report['total']} decisions")
    if report.get("reason"):
        print(report["reason"])
    metrics = score(model, tokenizer, items, list(dataset["workflow"]), batch_size)
    release(model)
    return metrics


def fmt(value) -> str:
    return f"{value:.3f}"


def print_report(results: dict) -> None:
    untrained = results["untrained"]
    trained = results["trained"]
    rows = [("overall", untrained["overall"], trained["overall"])]
    for name in QUESTION_TYPES:
        if name in untrained["by_type"]:
            rows.append((name, untrained["by_type"][name], trained["by_type"][name]))
    for name in sorted(untrained["by_workflow"]):
        rows.append((name, untrained["by_workflow"][name], trained["by_workflow"][name]))

    print(f"\n{'':22} {'n':>6} {'untrained':>10} {'trained':>10} {'change':>10}")
    for label, before, after in rows:
        change = after["accuracy"] - before["accuracy"]
        print(
            f"{label:22} {after['n']:6d} {fmt(before['accuracy']):>10} "
            f"{fmt(after['accuracy']):>10} {change:+10.3f}"
        )
    print("\nAccuracy is the share of decisions whose top option matches the gold label.")
    print(
        f"Calibration error (ECE): untrained {fmt(untrained['overall']['ece'])}, "
        f"trained {fmt(trained['overall']['ece'])}."
    )
    print(
        f"Whole-case accuracy: untrained {fmt(untrained['overall']['record_accuracy'])}, "
        f"trained {fmt(trained['overall']['record_accuracy'])}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description = "Compare untrained and trained Decider 0.8B.")
    parser.add_argument("--limit", type = int, default = 0, help = "Score only the first N test cases.")
    parser.add_argument("--batch-size", type = int, default = 1)
    args = parser.parse_args()

    trained = Path(TRAINED_DIR)
    if not (trained / "joint_head.safetensors").is_file():
        raise SystemExit(
            f"{TRAINED_DIR} is not ready. Training.py writes it when the run finishes."
        )

    dataset = load_dataset(DATASET, DATASET_CONFIG, split = "test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"{len(dataset)} test cases")

    results = {
        "dataset": DATASET,
        "split": "test",
        "cases": len(dataset),
        "untrained": evaluate_model("untrained", load_untrained, dataset, args.batch_size),
        "trained": evaluate_model("trained", load_trained, dataset, args.batch_size),
    }
    print_report(results)
    RESULTS_PATH.parent.mkdir(exist_ok = True)
    RESULTS_PATH.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""Score Decider 0.8B before and after this fine-tune.

Both models are Mapika/decider-0.8b with the same Unsloth decision head.
"Untrained" is that head and the LoRA at step 0 (the same init Training.py
uses). "Trained" is decider-0.8b-lora, written at the end of training.

The scored rows are the test split of LocalLLaMA/typed-decisions. Training
uses the train split only, with 80 of those rows held out for calibration.

Run it after training finishes. The 4GB card holds one model at a time, so
the script scores the untrained model, frees it, then scores the trained one.
Each case is timed on that scoring pass. The median, with the first case
left out as warmup, is the local figure to set beside a warm Jev request.
The same run writes results/compare.html, a standalone page of the scores.

    python compare.py
"""

import argparse
import gc
import json
import time
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


def _percentile(sorted_values: list, fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _timing_summary(case_seconds: list, batch_size: int) -> dict:
    # The first forward pays for kernel warmup. Steady figures leave that case out
    # so they can be set beside a warm Jev request. One case per list entry.
    warmup_n = min(batch_size, len(case_seconds))
    warmup = case_seconds[:warmup_n]
    steady = case_seconds[warmup_n:] or case_seconds
    ordered = sorted(steady)
    return {
        "batch_size": batch_size,
        "cases": len(case_seconds),
        "score_seconds": sum(case_seconds),
        "warmup_seconds": sum(warmup),
        "warmup_excluded": len(case_seconds) > warmup_n,
        "steady_cases": len(steady),
        "mean_seconds": sum(steady) / len(steady),
        "median_seconds": _percentile(ordered, 0.5),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "p95_seconds": _percentile(ordered, 0.95),
        "case_seconds": case_seconds,
    }


def timed_logits(model, tokenizer, items: list, batch_size: int):
    """Same scores as DecisionModel evaluation, plus seconds for each case.

    The clock starts once the batch is on the GPU and stops after the logits
    are back on the CPU. Tokenizing happens earlier, in build_dataset, and is
    not included. With the default batch of 1, one entry is one request, which
    is the local figure to set beside a Jev call.
    """
    import torch
    from unsloth.models.clef import QUESTION_TYPES as CLEF_TYPES, question_options
    from unsloth.models.decision import ClefDataCollator, _clef_amp_dtype

    def _option_labels(question: dict) -> list:
        labels = []
        kind = question["type"]
        for key, value in question_options(question):
            if kind == "score":
                text = str(value).split(":", 1)[0].strip()
            elif kind == "noul":
                text = "yes" if key == "true" else "no"
            else:
                text = str(key).replace("_", " ")
            labels.append(text)
        return labels

    if not getattr(model, "is_clef", False):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits, flat = _decision_logits(model, tokenizer, items, batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        each = elapsed / max(len(items), 1)
        return logits, flat, [each] * len(items)

    pad_token_id = getattr(tokenizer, "tokenizer", tokenizer).pad_token_id
    device = next(model.parameters()).device
    amp_dtype = _clef_amp_dtype(model, device)
    collate = ClefDataCollator(pad_token_id)
    laya_type = {CLEF_TYPES[kind]: QUESTION_TYPES.index(kind) for kind in QUESTION_TYPES}
    was_training = model.training
    model.eval()
    logits, questions, case_seconds = [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate(chunk)
        batch.pop("target")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
            z, _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        z = z.float().cpu()
        each = (time.perf_counter() - started) / len(chunk)
        case_seconds.extend([each] * len(chunk))
        flat = []
        for item in chunk:
            source = (item.get("source") or {}).get("questions") or {}
            names = list(source)
            for index, (target, label, qtype) in enumerate(
                zip(item["targets"], item["labels"], item["qtypes"])
            ):
                name = names[index] if index < len(names) else None
                options = _option_labels(source[name]) if name is not None else None
                # A pick index is only meaningful when it addresses this option list.
                if options is None or len(options) != len(target):
                    name, options = None, None
                flat.append((target, label, qtype, item["row"], name, options))
        for row, (target, label, qtype, record_row, name, options) in enumerate(flat):
            logits.append(z[row, : len(target)])
            questions.append({
                "target": target,
                "label": label,
                "qtype": laya_type[qtype],
                "row": record_row,
                "question": name,
                "options": options,
            })
    model.train(was_training)
    return logits, questions, case_seconds


def _decision_points(logits, flat, temperatures, workflows) -> tuple:
    import torch

    points, catalog, seen = [], [], {}
    for z, item, temperature in zip(logits, flat, temperatures):
        log_p = torch.log_softmax(z / temperature, -1)
        pred = int(log_p.argmax())
        workflow = workflows[item["row"]]
        point = {
            "row": int(item["row"]),
            "qtype": QUESTION_TYPES[item["qtype"]],
            "workflow": workflow,
            "confidence": round(float(log_p.exp().max()), 5),
            "correct": bool(pred == item["label"]),
        }
        options = item.get("options")
        question = item.get("question")
        if question and options:
            key = (workflow, question)
            if key not in seen:
                seen[key] = len(catalog)
                catalog.append({
                    "workflow": workflow,
                    "question": question,
                    "qtype": point["qtype"],
                    "options": list(options),
                })
            point["question"] = question
            point["pick"] = pred
            point["gold"] = int(item["label"])
        points.append(point)
    return points, catalog


def score(model, tokenizer, items, workflows, batch_size: int) -> dict:
    FastDecisionModel.for_inference(model)
    logits, flat, case_seconds = timed_logits(model, tokenizer, items, batch_size)
    temperatures = _served_temperatures(model.decision_config, logits, flat)
    points, catalog = _decision_points(logits, flat, temperatures, workflows)

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

    return {
        "overall": take(range(len(flat))),
        "by_type": by_type,
        "by_workflow": by_workflow,
        "timing": _timing_summary(case_seconds, batch_size),
        "decisions": points,
        "catalog": catalog,
    }


def evaluate_model(kind: str, loader, dataset, batch_size: int) -> dict:
    import torch

    print(f"\nLoading {kind} model")
    started = time.perf_counter()
    model, tokenizer = loader()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    build_seconds = time.perf_counter() - started
    print(f"Skipped {report['skipped']} of {report['total']} decisions")
    if report.get("reason"):
        print(report["reason"])
    metrics = score(model, tokenizer, items, list(dataset["workflow"]), batch_size)
    metrics["timing"]["load_seconds"] = load_seconds
    metrics["timing"]["build_seconds"] = build_seconds
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
    before, after = untrained["timing"], trained["timing"]
    print(
        f"Median forward seconds per case, after warmup: untrained {fmt(before['median_seconds'])}, "
        f"trained {fmt(after['median_seconds'])}."
    )
    print(
        f"Mean forward seconds per case, after warmup: untrained {fmt(before['mean_seconds'])}, "
        f"trained {fmt(after['mean_seconds'])}."
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
    from report import report_gaps, write_report

    gaps = report_gaps(results)
    if gaps:
        raise SystemExit(
            "The results file is short of what the report page needs:\n- " + "\n- ".join(gaps)
        )
    print(f"Wrote {write_report(results)}")


if __name__ == "__main__":
    main()

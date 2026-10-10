"""Train Mapika/decider-0.8b on this machine.

Decider 0.8B is a Qwen3.5-0.8B language-model checkpoint. Unsloth attaches a
Clef decision head and fine-tunes it with 4-bit LoRA, so the model scores the
options you give it instead of generating text.

Defaults match a 4GB GTX 1650 Ti: 4-bit weights, LoRA rank 16, batch size 1,
sequence length 1024. Training data is LocalLLaMA/typed-decisions, the set
Unsloth's decision guide uses. Point DATASET at your own Hub dataset or a
local json/jsonl file with state, questions, and gold columns.

Set MAX_STEPS to an integer to stop early (a one-step memory check).
"""

import os

from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments

MODEL_NAME = "Mapika/decider-0.8b"
DATASET = "LocalLLaMA/typed-decisions"
DATASET_CONFIG = "all"
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "outputs"
SAVE_DIR = "decider-0.8b-lora"

max_steps = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else -1

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

dataset = load_dataset(DATASET, DATASET_CONFIG, split = "train")
items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")
if report.get("reason"):
    print(report["reason"])

train_items, eval_items = FastDecisionModel.split_holdout(items, seed = 3407)
print(f"{len(train_items)} training rows, {len(eval_items)} held out")

trainer = DecisionTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_items,
    eval_dataset = eval_items,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        num_train_epochs = 2,
        max_steps = max_steps,
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        weight_decay = 0.01,
        optim = "adamw_8bit",
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        gradient_checkpointing = True,
        eval_strategy = "no" if max_steps > 0 else "epoch",
        logging_steps = 5,
        save_strategy = "no" if max_steps > 0 else "steps",
        save_steps = 40,
        save_total_limit = 2,
        output_dir = OUTPUT_DIR,
        report_to = "none",
        seed = 3407,
        dataloader_num_workers = 0,
    ),
)

trainer_stats = trainer.train()
print(trainer_stats)

if max_steps > 0:
    raise SystemExit(0)

metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
print(metrics)

model.save_pretrained(SAVE_DIR)
print(f"Saved LoRA adapters and decision head to {SAVE_DIR}")

FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model,
    tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    },
)
for name, result in answers.items():
    print(name, result["answer"], {k: round(v, 3) for k, v in result["probabilities"].items()})

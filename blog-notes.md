# Blog source notes

Raw material for a post about installing Unsloth on Windows, training Decider 0.8B on a GTX 1650 Ti, and comparing it with the untrained model. Numbers below were measured in this project on 10 October 2026. Do not fill the accuracy section until `compare.py` has been run. The install steps in `unsloth_windows_setup_guide.md` came from Gemini and were not re-run here. What was checked is that the resulting environment trains.

Repo: https://github.com/pyardley/DeciderTraining

## What the post can claim

- Unsloth 2026.10.3 runs natively on this Windows 11 machine and trains a decision model.
- The GPU is a 4GB GTX 1650 Ti. It has no bfloat16, so this Qwen3.5 model trains in float32.
- A steady training step takes about 70 seconds. The trainer reported `train_runtime` of 73,420 seconds (20 hours 24 minutes) for 280 steps. That wall clock includes the machine sleeping: one step sat for about 99 minutes, and another for about 12.5 hours. The stepping itself is about five and a half hours, plus a short evaluation and calibration at the end.
- Accuracy before and after is still unmeasured. `compare.py` scores the typed-decisions test split, which training does not use.

## Machine

- OS: Windows 11, build 10.0.26200.
- GPU: NVIDIA GeForce GTX 1650 Ti, 4096 MiB, compute capability 7.5. No tensor cores, so no bfloat16. Driver 617.42.
- Unsloth printed `Bfloat16 = FALSE` and `FA [Xformers = None. FA2 = False]`.
- Conda env: `unsloth_win`, Python 3.10, interpreter `C:\Users\PaulYardley\.conda\envs\unsloth_win\python.exe`.
- `conda` is not on PATH in a normal PowerShell window. The env still runs if you call that interpreter directly.
- Torch 2.14.1+cu126. CUDA available. Transformers 5.17.0. Triton 3.8.0. CUDA toolkit reported by Unsloth: 12.6.
- Unsloth warns that Python 3.10 is below its recommended 3.11. Training proceeded anyway.

## Install notes, from the Gemini guide

The guide is `unsloth_windows_setup_guide.md`. Its sequence is:

1. Optional: `conda env remove --name unsloth_win`.
2. `conda create --name unsloth_win python=3.10`.
3. `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`.
4. Reinstall PyTorch from the cu126 index: `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`. The guide does this because a fresh Unsloth install can pull a torch build that is too old for `torchao` (`torch.int1`).
5. Verify with `python -c "import torch; import unsloth; ..."`.

The verification screenshot showed pip replacing torch 2.5.1+cu121 and torchvision 0.20.1+cu121 with torch 2.14.1+cu126, torchvision 0.29.1+cu126, torchaudio 2.11.0+cu126, and sympy 1.14.0. The check then printed:

```text
Torch version: 2.14.1+cu126
CUDA Available: True
GPU: NVIDIA GeForce GTX 1650 Ti
```

Also on that screen, before the version line: Unsloth's startup lines, a warning that Python 3.10 is below 3.11, and two torch warnings about `KernelPreference` and `ScaleCalculationMode` enums. Those warnings did not stop training.

## What the training does

The run teaches a small model to answer several fixed questions about one piece of state in a single forward pass, and to return a probability for every option instead of generating text.

It starts from `Mapika/decider-0.8b`, the published 0.8B Decider checkpoint. Those weights are a Qwen3.5-0.8B language model that Mapika already trained for decisions. Unsloth does not continue that original recipe. It treats the checkpoint as a backbone, attaches a new decision head (the same design as Cloudflare's Clef), and trains two things: a LoRA adapter on the backbone, and the new head from scratch. The head scores every question and option together. The model never writes a reply.

Each training row is one case. The case has a state (a JSON document or a short text) and five questions about that state. A question is one of three types: a choice among named options, a score on an ordered rubric, or a yes/no. The gold answer is a probability distribution over those options, not only the winning label. The loss is cross-entropy against that distribution. After the two epochs, temperatures are fitted on 80 rows held out of the training split so the reported confidence tracks how often the model is actually right. Those 80 rows are for calibration, not for the test score. The Laya video section below is the plain-language version of that step, including why a confidence gate needs it.

The result is a specialist on these four workflows. It has seen their question schemas in training. That is a different claim from a general model such as Jev, which is asked questions it was not fitted to. Accuracy against Jev on this test split would mix those two modes. Latency can still be compared, because both answer one case in one call.

## Settings

- 4-bit load, LoRA rank 16, alpha 16, dropout 0, Unsloth gradient checkpointing.
- Sequence length 1024. Unsloth reported that 89 of 1,200 training inputs were longer, so the end of those states was cut.
- Optimizer `adamw_8bit`, learning rate 2e-4, cosine schedule, 10 warmup steps, weight decay 0.01.
- Batch size 1, gradient accumulation 8. Effective batch 8.
- 2 epochs, 280 optimizer steps. Checkpoints every 40 steps in `outputs\`. Final adapters go to `decider-0.8b-lora`.
- Trainable parameters: 38,147,588 of 598,651,204 (6.37%).
- Unsloth rewrote the bad pad token `<|endoftext|>` to `<|vision_pad|>`.

Unsloth's own message during training: Qwen3.5 overflows under float16 autocast, so it trains with no autocast and float32 activations.

`causal-conv1d` is not installed. There is no Windows wheel, and a source build failed because `nvcc` is not installed. The depthwise convolution uses PyTorch's grouped `conv1d`. The gated-delta part of each layer is already on Unsloth's bundled fast kernel. Speeding up only the convolution looked like roughly 10–20% on the whole step, not a multiple. That estimate was not measured by installing the kernel.

## Data

The training data is the public Hugging Face dataset `LocalLLaMA/typed-decisions`, config `all`. It is an independent benchmark for the same style of typed decision as TypeSafe's Jev. It is not affiliated with TypeSafe, and it was not distilled from Jev. Dataset card: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

Each case is synthetic. A latent skeleton is drawn first (topic, severity, whether a constraint was violated, and similar factors). A state is then rendered from that skeleton: structured JSON where the artefact really is structured (invoices, agent traces), and model-written prose where it is text (customer threads, alert narratives). A teacher endpoint of roughly 4B-class capability labels every case, sampled three times at temperature 0.7. The gold distribution is the mean of those three samples, so an ambiguous decision stays soft instead of being collapsed to one label. The dataset card is explicit that a score on this set measures agreement with that teacher, not correctness against the world. Their reference points on the full 1,600-case set are a majority baseline of 0.520, a factor-ceiling of 0.704, and teacher self-agreement of 0.735.

There are four workflows, 300 training cases and 100 test cases each:

- `agent_trace_observability` — what to do with an agent run, including whether a human should review it.
- `customer_service` — what action to take from a customer thread and account state.
- `invoice_processing` — whether to pay, hold, or reject a vendor bill.
- `security_incidents` — whether to close, investigate, or contain an alert.

The `all` config is those four workflows together, not a fifth copy. This run confirmed 1,200 training cases and 400 test cases. The two splits were generated separately, at different seeds, and the package refuses to build if a case id or a state appears in both. Training uses the train split only. `compare.py` scores the test split.

- Train split: 1,200 cases, 6,000 decisions, none skipped. `split_holdout` kept 1,120 for training and 80 for calibration. Seed 3407.
- Each case is one state and five questions, a mix of choice, score, and yes/no.
- Test split: 400 cases. Not used in training.

## Timings on this GPU

One-step smoke test, before the real run: 463 seconds, loss 1.195. That step included kernel warmup and was limited to one optimizer step, so it is not the steady rate.

Full run, started 15:11 local time on 10 October 2026:

- Step 1 took 346 seconds.
- After that, steps settled. Across the latest 20 recorded intervals the average is 71.6 seconds, and the fastest interval in the whole log is 66 seconds.
- At step 35 the progress bar had been running for 46 minutes 18 seconds. Loss at the last logged point (every 5 steps) was 0.9405, down from 1.192 at step 5. Logged losses in order: 1.192, 1.181, 1.106, 1.078, 1.027, 1.012, 0.9405.
- The run finished at 11:40 local time on 11 October 2026, exit code 0. Trainer `train_runtime` was 73,420.0543 seconds. Final `train_loss` was 0.8968. The epoch-2 evaluation on the 80 holdout rows took 218.5 seconds, `eval_loss` 0.8481.
- Two steps account for most of the gap between 5.6 hours of stepping and the 20 hour 24 minute runtime. Step 168 took 5,936 seconds. Step 208 took 45,323 seconds. The steps on either side of both gaps were about 70 seconds, so those two intervals are the machine asleep, not a slower GPU.
- After training, calibration on those same 80 holdout rows fitted a temperature for choice, score, and yes/no. On that holdout it reported accuracy 0.805, ECE 0.029, and whole-case accuracy 0.388. That is not the test-split score. Adapters were saved to `decider-0.8b-lora`, including `joint_head.safetensors`.

## Framing from the Laya video

Useful introduction, not a result from this machine. Video: https://www.youtube.com/watch?v=ICtPrhMBUKA. The same experiment is written up by Daniel Walsh, The AI Automators, 7 October 2026: https://www.theaiautomators.com/laya-vs-jev-on-1000-real-support-chats/. The model is Laya, about 420 million parameters, open weights, one decision per forward pass. Jev is the cloud model he could not fine-tune. His numbers below are his, on his task. Do not paste them in as Decider results.

The task is one repeated choice: given a customer-support chat, pick the next tool from a fixed list of 30. The source is the Action-Based Conversations Dataset, about 10,000 conversations. He cut each example at the moment before a tool was used, which left 8,883 decisions. He split those before training:

- 5,900 training chats, the only rows that change weights.
- 983 validation chats, used to pick which checkpoint to keep.
- 1,000 calibration chats, used only to fit confidence.
- 1,000 test chats, the score that gets reported.

He ran two input formats, and both models saw the same format in each experiment. Basic state is the chat, the 30 tool names, and the procedures. Rich state adds a handbook of what each tool does, solved examples, manual sections, a rationale for the choice, and which tools were already used. A rich request was about 7,000 tokens.

On the 1,000 test chats:

| Approach | Input | Accuracy |
| --- | --- | --- |
| Laya, no task training | Basic | 7.5% |
| A non-LLM classifier trained on the examples | Basic | 77.3% |
| Jev | Basic | 67.3% |
| Laya after fine-tuning | Basic | 82.2% |
| Laya after fine-tuning | Rich | 86.1% |
| Jev | Rich | 87.4% |

Untrained Laya mostly picked two tools, Search Pricing and Validate Purchase. One pass over the training data on an RTX 5090 took under four minutes and, during that pass, passed both Jev's basic-state score and the classifier. The 82.2% figure is after two further passes: 75 correct out of 1,000, then 822. Richer context moved Jev from 67.3% to 87.4% without any training. Fine-tuned Laya on that same rich state reached 86.1%. With a lean request the specialist won. With a long request the general model edged it, by 13 decisions on this test. His point for a post: teaching the task, and putting more guidance in every request, are two different levers.

He treats three measures as separate:

- Accuracy is how often the chosen option is the right one, on the test set only. A correct answer at 99% and a correct answer at 60% both count as one.
- Confidence is the probability on the winning option. A gate on that number is the operating policy: above the gate, code runs the tool; below it, a person or a larger model takes the case.
- Calibration is whether a stated confidence matches the hit rate. Calls made at about 90% should be right about 90% of the time. If they are right 70% of the time, the model is overconfident. He compares it to a weather forecast.

He fitted that with temperature scaling on the 1,000 calibration chats, not on the test set. Raw scores are divided by a temperature above 1, which flattens the probabilities and does not change which tool wins. For this run the fitted temperature was 1.63. At a fixed 90% gate, automated mistakes fell from 97 to 36, and cases sent to a person rose from 160 to 326. Automatic decisions went from 840 to 674. The 36 mistakes are 36 of those 674, about 5.3%, not 36 of 1,000. Against rich-state Jev at the same gate, calibrated Laya let through 36 wrong automatic decisions and reviewed 326, while Jev let through 39 and reviewed 282. Fewer automatic mistakes cost more human review, including review of decisions that would have been right.

What carries over to this Decider run: the same three measures, and the same rule that the test split is not used to train or to fit confidence. This run's calibration is the temperature fit on the 80 held-out training rows. `compare.py` reports accuracy and expected calibration error, and writes `results\compare.html`. That page includes the radial chart from the Laya video: one dot per test case, sitting on the option the model chose, teal when that option is the gold label and red when it is not. The center number is accuracy for the selected workflow and question, and the gold arc is that same share of the circle. Untrained and trained are a switch, because this run does not have a score after every training step. The page also draws one dot per test question by confidence, with a movable gate. The gate table is the same shape as Walsh's 97-to-36 count. Here the untrained and trained models can pick different options, so the gate is an operating picture of each model, not a pure before-and-after of temperature scaling. Walsh also kept a simple classifier in the comparison, which this project does not have.

## Accuracy comparison, not run yet

`compare.py` loads the untrained model and the trained adapters one at a time, because both do not fit in 4GB together.

Untrained means the same starting point as `Training.py`: `Mapika/decider-0.8b`, a freshly initialized decision head, and LoRA at step 0, with `random_state` 3407. It does not mean Decider's original letter-logit readout.

Trained means `decider-0.8b-lora`. The script refuses to start until that folder contains `joint_head.safetensors`.

It reports accuracy, calibration error (ECE), and whole-case accuracy, plus accuracy by question type and by workflow. It also records load time, dataset build time, and per-case scoring time (mean, median, min, max, p95, with the first case separated as warmup). One case is one forward pass. With the default batch of 1, that median is the local number to set beside a warm Jev request. Tokenization is not included. Results are written to `results\compare.json`, and the same run writes `results\compare.html`, a standalone page in the style of the Laya video: a radial chart of which option the model chose, confidence dots, a gate slider, accuracy bars, and a calibration curve. Expect on the order of an hour for both models on this GPU.

Paste the `compare.py` table here after it runs, and only then put accuracy numbers in the post.

## Files

- `unsloth_windows_setup_guide.md` — Gemini's install steps.
- `Training.py` — the run described above.
- `compare.py` — held-out before/after score.
- `report.py` — turns `results\compare.json` into `results\compare.html`. `compare.py` calls it. `python report.py` rebuilds the page from a results file that is already there.
- `outputs\` — checkpoints, not in git.
- `decider-0.8b-lora` — final adapters, saved when training finished, not in git.

"""
Dry run for the baseline step (step 2) of the Ethics Corpus project.

Purpose: a rehearsal on the laptop. It checks that MY SCRIPT works.
Nothing printed here is a result, and the settings below are not the
locked evaluation settings.

What it checks:
  - packages import, and the script has no bugs
  - the Anthropic key works and the judge responds
  - logs are written and open in `inspect view`
  - token counts print (to sanity-check the judge cost estimate)

What it does NOT check:
  - tool calling for the 7-8B models, or any score

Usage:
    python dry_run.py           # Granite on the GPU, 4-bit (fits in 4 GB)
    python dry_run.py --cpu     # Granite on the CPU, full precision (slow)

Extra install for the GPU mode:
    uv pip install bitsandbytes

Before running (PowerShell):
    $env:ANTHROPIC_API_KEY = "..."

Status: tested on laptop
"""

import argparse
import os
import sys

import torch
from inspect_ai import eval
from inspect_evals.agentharm import agentharm, agentharm_benign

# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------

# The only candidate small enough for this laptop.
MODEL = "hf/ibm-granite/granite-4.0-micro"

# Judge used by AgentHarm's scorer.
JUDGE = "anthropic/claude-haiku-5-5"

# Samples per task. Two is enough to exercise every code path.
LIMIT = 2

# Dry-run-only caps, to keep the run short on small hardware.
MAX_TOKENS = 512        # longest single model reply
MESSAGE_LIMIT = 12      # stop a sample after this many messages

# AgentDojo has long prompts (tight in 4 GB) and needs no judge, so it is
# off by default. Turn on once the AgentHarm part runs cleanly.
INCLUDE_AGENTDOJO = False

LOG_DIR = "logs/dry_run"


# --------------------------------------------------------------------------
# 2. Where and how the model loads.
#    "Model arguments" are passed through to the hf provider (-M on the
#    command line). They are a third kind of argument, next to task
#    arguments (-T) and run options.
# --------------------------------------------------------------------------

def patched_chat_template():
    """Granite's chat template, with one line changed so it can read Inspect's data.

    The problem (found in the dry run): when a conversation already contains a
    tool call, Granite's template expects it in the standard shape
        {"function": {"name": ..., "arguments": ...}}
    but Inspect's hf provider supplies
        {"function": "<name>", "arguments": ...}
    so the template crashes on the second model turn.

    The patch only changes how the data is READ. The text shown to the model
    is identical to what the official template produces (checked by rendering
    both and comparing).
    """
    from transformers import AutoTokenizer

    anchor = "{%- if tool_call.function %}"
    patch = (
        "{%- if tool_call.function is string %}"
        "{%- set tool_call = {'name': tool_call.function, 'arguments': tool_call.arguments} %}"
        "{%- elif tool_call.function %}"
    )
    template = AutoTokenizer.from_pretrained(MODEL.removeprefix("hf/")).chat_template
    # Fail loudly if the official template ever changes, rather than patch blindly.
    if template.count(anchor) != 1:
        sys.exit("Granite's chat template has changed; the patch no longer applies.")
    return template.replace(anchor, patch)


def build_model_args(use_cpu):
    common = {
        # True greedy decoding (always take the most likely token). The hf
        # provider samples by default, which is why temperature=0 crashed.
        "do_sample": False,
        "chat_template": patched_chat_template(),
    }

    if use_cpu:
        # Full-precision model in RAM. Same weights as on Kaggle, just slow.
        return {"device": "cpu", **common}

    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU. Run with --cpu, or recheck the install.")

    # 4-bit quantisation: each weight is stored in 4 bits instead of 16.
    # Confirmed in the dry run: the hf provider forwards this to transformers.
    return {
        "device": "cuda:0",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
        },
        **common,
    }


# --------------------------------------------------------------------------
# 3. Tasks (validation split, so the test split stays untouched)
# --------------------------------------------------------------------------

def build_tasks():
    tasks = [
        agentharm(split="val", refusal_judge=JUDGE, semantic_judge=JUDGE),
        agentharm_benign(split="val", refusal_judge=JUDGE, semantic_judge=JUDGE),
    ]
    if INCLUDE_AGENTDOJO:
        # Imported here so a missing AgentDojo install does not stop the dry run.
        from inspect_evals.agentdojo import agentdojo

        tasks.append(agentdojo(with_injections=False))
    return tasks


# --------------------------------------------------------------------------
# 4. Reporting (same idea as in smoke_test.py)
# --------------------------------------------------------------------------

def summarise(log):
    print("=" * 70)
    print(f"task:   {log.eval.task}")
    print(f"status: {log.status}")
    print(f"log:    {log.location}")

    if log.status != "success":
        print(f"ERROR:  {log.error}")
        return

    samples = log.samples or []
    errors = [s.error for s in samples if s.error is not None]
    print(f"samples run: {len(samples)}   samples with errors: {len(errors)}")
    if errors:
        # Status can be "success" even when every sample failed, because
        # fail_on_error=False. Show the first error so it is not missed.
        print(f"  FIRST SAMPLE ERROR: {errors[0].message}")

    if log.results is not None:
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                print(f"  score  {score.name}/{name}: {metric.value}  (meaningless at n={LIMIT})")

    for model_name, usage in log.stats.model_usage.items():
        print(
            f"  tokens {model_name}: "
            f"input={usage.input_tokens} output={usage.output_tokens}"
        )


def report_gpu_memory():
    # Peak memory PyTorch used during the run: how close did we get to 4 GB?
    peak = torch.cuda.max_memory_allocated() / 1024**2
    _, total = torch.cuda.mem_get_info()
    print(f"GPU peak memory used by PyTorch: {peak:.0f} MiB of {total / 1024**2:.0f} MiB")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Laptop dry run")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU instead of the GPU")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; the AgentHarm judge needs it.")

    logs = eval(
        build_tasks(),
        model=MODEL,
        model_args=build_model_args(args.cpu),
        limit=LIMIT,
        # No temperature here: the AgentHarm task sets temperature=0 itself,
        # and do_sample=False (in the model arguments) makes that work.
        max_tokens=MAX_TOKENS,
        message_limit=MESSAGE_LIMIT,
        max_samples=1,          # one sample at a time: lowest memory use
        log_dir=LOG_DIR,
        fail_on_error=False,    # keep going, so one run shows every problem
    )

    for log in logs:
        summarise(log)

    print("=" * 70)
    if not args.cpu:
        report_gpu_memory()
    print("Next: run `inspect view --log-dir logs/dry_run` and read the transcripts.")


if __name__ == "__main__":
    main()

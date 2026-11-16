"""
Smoke test for the baseline step (step 2) of the Ethics Corpus project.
Runs on Kaggle (2 x T4 GPUs). One model per run.

Purpose: check that the REAL pipeline runs for each candidate model.
Nothing printed here is a result.
  - Does the model load, at the chosen precision, on this hardware?
  - Do tool calls work over several turns?
  - Do the judge and the scoring run?
  - How fast is it, and what does the judge cost per sample?

Usage:
    python smoke_test.py granite
    python smoke_test.py qwen25
    python smoke_test.py qwen3

What changed after the laptop dry run (all found there):
  - Greedy decoding is set with do_sample=False, a MODEL argument.
    AgentHarm sets temperature=0 itself; no override is needed.
  - Granite needs a one-line chat-template patch to survive a second turn.
  - AgentHarm sets max_connections=100, which the hf provider uses as its
    batch size. It is set explicitly here to protect GPU memory.
  - The summary now reports speed and malformed tool-call arguments.

Status: the Granite path ran on the laptop. NOT yet run: Qwen models,
AgentDojo, 16-bit loading, and anything on Kaggle. Marked UNVERIFIED below.
"""

import os
import sys

from inspect_ai import eval
from inspect_evals.agentharm import agentharm, agentharm_benign

# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------

# Judge used by AgentHarm's scorer (AgentDojo needs no judge).
JUDGE = "anthropic/claude-haiku-5-5"

# Samples per task. Inspect takes the FIRST n in dataset order. AgentHarm
# stores 4 variants per behaviour, so 5 samples cover only 2 behaviours.
LIMIT = 5

# How the weights are stored on the GPU. This changes the model slightly, so
# whichever is used for the baseline must be used for every arm later.
#   "16bit": full released precision. A 7-8B model (~16 GB) does not fit on
#            one 16 GB T4, so it is spread over BOTH GPUs: one model at a time.
#   "4bit":  quantised. Fits on one T4, so two models can run in parallel
#            (one per GPU, selected with CUDA_VISIBLE_DEVICES).
PRECISION = "16bit"

# Prompts generated together on the GPU, and samples in flight.
# Small on purpose: raise it only after checking memory with nvidia-smi.
BATCH = 4

INCLUDE_AGENTDOJO = True

# Per-model details. "spread" = needs both GPUs at 16-bit.
MODELS = {
    "granite": {
        "name": "hf/ibm-granite/granite-4.0-micro",
        "spread": False,
        "patch_template": True,
        "extra_args": {},
    },
    "qwen25": {
        "name": "hf/Qwen/Qwen2.5-7B-Instruct",
        "spread": True,
        "patch_template": False,
        "extra_args": {},
    },
    "qwen3": {
        "name": "hf/Qwen/Qwen3-8B",
        "spread": True,
        "patch_template": False,
        # Thinking mode off, and locked off (the hf provider passes this to
        # the chat template).
        "extra_args": {"enable_thinking": False},
    },
}


# --------------------------------------------------------------------------
# 2. Model arguments: how the provider loads and runs the model
# --------------------------------------------------------------------------

def patched_granite_template(model_name):
    """Granite's chat template with one line changed so it can read Inspect's data.

    Granite expects a past tool call as {"function": {"name", "arguments"}};
    Inspect's hf provider supplies {"function": "<name>", "arguments": ...}.
    The patch changes only how the data is read. The text shown to the model
    is identical to the official template's (checked by rendering both).
    """
    from transformers import AutoTokenizer

    anchor = "{%- if tool_call.function %}"
    patch = (
        "{%- if tool_call.function is string %}"
        "{%- set tool_call = {'name': tool_call.function, 'arguments': tool_call.arguments} %}"
        "{%- elif tool_call.function %}"
    )
    template = AutoTokenizer.from_pretrained(model_name.removeprefix("hf/")).chat_template
    if template.count(anchor) != 1:
        sys.exit("Granite's chat template has changed; the patch no longer applies.")
    return template.replace(anchor, patch)


def build_model_args(spec):
    args = {
        # True greedy decoding: always take the most likely token.
        "do_sample": False,
        **spec["extra_args"],
    }

    if spec["patch_template"]:
        args["chat_template"] = patched_granite_template(spec["name"])

    if PRECISION == "4bit":
        args["device"] = "cuda:0"
        args["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
        }
    elif PRECISION == "16bit":
        # UNVERIFIED: that transformers accepts the precision as the text
        # "float16" under this argument name. Without it the model may load
        # at 32-bit and need twice the memory.
        args["torch_dtype"] = "float16"
        # UNVERIFIED: "auto" asks transformers to spread the model over all
        # visible GPUs. The provider passes "device" straight through.
        args["device"] = "auto" if spec["spread"] else "cuda:0"
    else:
        sys.exit(f"Unknown PRECISION: {PRECISION}")

    return args


# --------------------------------------------------------------------------
# 3. Tasks (validation split, so the test split stays untouched)
# --------------------------------------------------------------------------

def build_tasks():
    tasks = [
        agentharm(split="val", refusal_judge=JUDGE, semantic_judge=JUDGE),
        agentharm_benign(split="val", refusal_judge=JUDGE, semantic_judge=JUDGE),
    ]
    if INCLUDE_AGENTDOJO:
        # UNVERIFIED: task name and argument, taken from the Inspect Evals docs.
        from inspect_evals.agentdojo import agentdojo

        tasks.append(agentdojo(with_injections=False))
    return tasks


# --------------------------------------------------------------------------
# 4. Reporting
# --------------------------------------------------------------------------

def looks_wrapped(arguments):
    """True if a tool argument is itself a JSON object written as text.

    Seen in the dry run on the first tool call of each sample, e.g.
    {"request": "{\\n  \\"request\\": \\"...\\"\\n}"}. It can make a tool
    return nothing and lower the score for reasons unrelated to ability.
    """
    return any(
        isinstance(value, str) and value.strip().startswith("{")
        for value in (arguments or {}).values()
    )


def summarise(log, model_name):
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

    # Tool-calling health and speed, from the transcripts.
    n_calls = n_wrapped = n_no_tools = 0
    out_tokens = seconds = 0.0
    for sample in samples:
        calls = [
            call
            for message in sample.messages
            if message.role == "assistant"
            for call in (message.tool_calls or [])
        ]
        n_calls += len(calls)
        n_wrapped += sum(looks_wrapped(call.arguments) for call in calls)
        n_no_tools += len(calls) == 0
        for event in sample.events:
            # Only the model under test, not the judge.
            if event.event == "model" and model_name.removeprefix("hf/") in event.model:
                if event.output.usage is not None and event.working_time:
                    out_tokens += event.output.usage.output_tokens
                    seconds += event.working_time

    print(f"tool calls: {n_calls}   with wrapped arguments: {n_wrapped}")
    print(f"samples with no tool call at all: {n_no_tools}")
    if seconds > 0:
        print(f"speed: {out_tokens / seconds:.1f} output tokens per second")

    if log.results is not None:
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                print(f"  score  {score.name}/{name}: {metric.value}  (meaningless at n={LIMIT})")

    # The judge appears as its own entry: divide by the number of samples
    # to rescale the cost estimate.
    for usage_name, usage in log.stats.model_usage.items():
        print(
            f"  tokens {usage_name}: "
            f"input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens}"
        )


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        sys.exit(f"Usage: python smoke_test.py [{' | '.join(MODELS)}]")
    key = sys.argv[1]
    spec = MODELS[key]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; the AgentHarm judge needs it.")

    logs = eval(
        build_tasks(),
        model=spec["name"],
        model_args=build_model_args(spec),
        limit=LIMIT,
        max_connections=BATCH,   # overrides AgentHarm's 100
        max_samples=BATCH,
        log_dir=f"logs/smoke/{key}_{PRECISION}",
        fail_on_error=False,     # keep going, so one run shows every problem
    )

    for log in logs:
        summarise(log, spec["name"])

    print("=" * 70)
    print("Next: download the logs folder, then read every transcript in `inspect view`.")


if __name__ == "__main__":
    main()

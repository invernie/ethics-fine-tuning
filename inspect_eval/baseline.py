"""
Baseline evaluation (step 2 of the Ethics Corpus project). FIRST DRAFT.
Runs on Kaggle (2 x T4 GPUs), one model per run, AFTER the smoke test passes.

Two stages, in this order:

  pilot     Validation split. Used to choose and lock the settings
            (decoding, epochs, judge, precision). Look at it as much as you like.

  baseline  Public test split, run ONCE with the locked settings. The selected
            model's run becomes arm A (no fine-tuning) of the experiment.

Usage:
    python baseline.py pilot granite              # greedy decoding, 1 epoch
    python baseline.py pilot granite --sampled    # temperature > 0, several epochs
    python baseline.py baseline granite           # refuses until settings are locked

Resuming: runs use eval_set(), which keeps track of what has finished. If a
Kaggle session ends mid-run, run the SAME command again with the same logs
folder present, and it continues instead of starting over.

Outputs, per run, in logs/<stage>/<model>_<precision>_<decoding>/ :
    *.eval              Inspect logs (open with `inspect view`)
    run_settings.json   every setting and package version used
    samples.csv         one row per sample and epoch, for the statistics

Status: NOT yet run. Built from the smoke-test script plus the project plan.
Open points are marked TODO or UNVERIFIED.
"""

import argparse
import csv
import json
import os
import sys
from importlib import metadata

from inspect_ai import eval_set
from inspect_ai.log import read_eval_log
from inspect_evals.agentharm import agentharm, agentharm_benign

# --------------------------------------------------------------------------
# 1. Settings. These are the things the project plan says to LOCK at baseline.
#    TODO: this block duplicates smoke_test.py. Once both scripts are stable,
#    move the shared parts into one file so they cannot drift apart.
# --------------------------------------------------------------------------

# Set to True only after the pilot, when every value in this section is final.
# The baseline stage refuses to run while this is False.
SETTINGS_LOCKED = False

# Judge for AgentHarm's scorer. TODO: decide Haiku vs Sonnet from the pilot,
# by comparing each against blind human labels. Its temperature cannot be set
# (Anthropic models no longer accept that parameter).
# TODO: set a low reasoning effort for the judge; I have not confirmed how.
JUDGE = "anthropic/claude-haiku-5-5"

# "16bit": released precision; a 7-8B model is spread over both GPUs.
# "4bit":  quantised; fits one GPU. Must be the same for every arm later.
PRECISION = "16bit"

# Prompts generated together on the GPU, and samples in flight.
# TODO: raise after the smoke test shows how much GPU memory is free.
BATCH = 4

# Decoding. "greedy" is AgentHarm's own default (temperature 0).
# "sampled" is the alternative under test in the pilot.
SAMPLED_TEMPERATURE = 0.7   # TODO: check each model card's recommended value
SAMPLED_EPOCHS = 5          # epochs = repeats of every sample

# Which decoding the baseline of record uses. Decide from the pilot.
BASELINE_DECODING = "greedy"

# AgentDojo with attacks. Leave off until the model clears your threshold for
# utility without attacks; below it, attack success cannot be interpreted.
AGENTDOJO_INJECTIONS = False

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
        "extra_args": {"enable_thinking": False},
    },
}

# Splits per stage: (AgentHarm split name, label used in file names).
SPLITS = {"pilot": "val", "baseline": "test_public"}


# --------------------------------------------------------------------------
# 2. Model arguments (same logic as smoke_test.py)
# --------------------------------------------------------------------------

def patched_granite_template(model_name):
    """Granite's chat template with one line changed so it can read Inspect's data.

    The text shown to the model is identical to the official template's.
    See the Inspect AI - FAQ tab for the explanation.
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


def build_model_args(spec, decoding):
    args = {
        # Greedy: always take the most likely token. Sampled: draw at random
        # in proportion to probability, scaled by the temperature.
        "do_sample": decoding == "sampled",
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
        args["torch_dtype"] = "float16"                          # UNVERIFIED
        args["device"] = "auto" if spec["spread"] else "cuda:0"  # UNVERIFIED
    else:
        sys.exit(f"Unknown PRECISION: {PRECISION}")

    return args


# --------------------------------------------------------------------------
# 3. Tasks
# --------------------------------------------------------------------------

def build_tasks(stage):
    split = SPLITS[stage]
    tasks = [
        agentharm(split=split, refusal_judge=JUDGE, semantic_judge=JUDGE),
        agentharm_benign(split=split, refusal_judge=JUDGE, semantic_judge=JUDGE),
    ]

    # UNVERIFIED: AgentDojo task name and arguments. It has no validation
    # split that I know of, so the pilot and the baseline would run the same
    # tasks. TODO: decide whether the pilot should use only a fixed subset.
    from inspect_evals.agentdojo import agentdojo

    tasks.append(agentdojo(with_injections=False))
    if stage == "baseline" and AGENTDOJO_INJECTIONS:
        tasks.append(agentdojo(with_injections=True))

    return tasks


# --------------------------------------------------------------------------
# 4. Record keeping
# --------------------------------------------------------------------------

def package_versions():
    versions = {}
    for name in ("inspect-ai", "inspect-evals", "transformers", "torch", "bitsandbytes"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_settings(log_dir, stage, key, decoding, epochs, model_args):
    """Save everything needed to reproduce this run next to its logs."""
    record = {
        "stage": stage,
        "split": SPLITS[stage],
        "model_key": key,
        "model": MODELS[key]["name"],
        "precision": PRECISION,
        "decoding": decoding,
        "temperature": SAMPLED_TEMPERATURE if decoding == "sampled" else 0,
        "epochs": epochs,
        "judge": JUDGE,
        "batch": BATCH,
        "agentdojo_injections": AGENTDOJO_INJECTIONS,
        "settings_locked": SETTINGS_LOCKED,
        # The chat template is long; record only whether it was patched.
        "model_args": {k: v for k, v in model_args.items() if k != "chat_template"},
        "chat_template_patched": "chat_template" in model_args,
        "packages": package_versions(),
        # TODO: record the dataset revisions (not available from here).
    }
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run_settings.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def sample_rows(log):
    """One row per sample and epoch, with every score as its own column."""
    rows = []
    for sample in log.samples or []:
        meta = sample.metadata or {}
        row = {
            "task": log.eval.task,
            "model": log.eval.model,
            "sample_id": sample.id,
            "epoch": sample.epoch,
            # The base behaviour. AgentHarm's 4 variants of one behaviour are
            # not independent, so the statistics must group by this column.
            "base_id": meta.get("id_original", ""),
            "category": meta.get("category", ""),
            "hint_included": meta.get("hint_included", ""),
            "detailed_prompt": meta.get("detailed_prompt", ""),
            "error": sample.error.message if sample.error is not None else "",
            # Set when a sample was stopped by a limit instead of finishing.
            "stopped_by_limit": sample.limit.type if sample.limit is not None else "",
        }
        for scorer_name, score in (sample.scores or {}).items():
            if isinstance(score.value, dict):
                # e.g. AgentHarm: {"score": 0.33, "refusal": 0.0}
                for part, value in score.value.items():
                    row[f"{scorer_name}.{part}"] = value
            else:
                row[scorer_name] = score.value
        rows.append(row)
    return rows


def write_samples_csv(log_dir, logs):
    rows = []
    for log in logs:
        # Re-read from disk so the samples are certainly loaded.
        rows.extend(sample_rows(read_eval_log(log.location)))
    if not rows:
        return None
    # Tasks have different score columns; use the union, in first-seen order.
    columns = list(dict.fromkeys(column for row in rows for column in row))
    path = os.path.join(log_dir, "samples.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarise(log):
    print("=" * 70)
    print(f"task:   {log.eval.task}")
    print(f"status: {log.status}")
    if log.status != "success":
        print(f"ERROR:  {log.error}")
        return
    if log.results is not None:
        print(f"samples: {log.results.completed_samples} of {log.results.total_samples} completed")
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                print(f"  {score.name}/{name}: {metric.value}")
    for usage_name, usage in log.stats.model_usage.items():
        print(f"  tokens {usage_name}: input={usage.input_tokens} output={usage.output_tokens}")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pilot and baseline evaluation")
    parser.add_argument("stage", choices=list(SPLITS))
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("--sampled", action="store_true",
                        help="pilot only: temperature > 0 with several epochs")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; the AgentHarm judge needs it.")

    if args.stage == "baseline":
        if not SETTINGS_LOCKED:
            sys.exit("Settings are not locked. Finish the pilot, fix every value in "
                     "section 1, set SETTINGS_LOCKED = True, then run the baseline.")
        if args.sampled:
            sys.exit("--sampled is a pilot option. The baseline uses BASELINE_DECODING.")
        decoding = BASELINE_DECODING
    else:
        decoding = "sampled" if args.sampled else "greedy"

    epochs = SAMPLED_EPOCHS if decoding == "sampled" else 1
    spec = MODELS[args.model]
    model_args = build_model_args(spec, decoding)
    log_dir = f"logs/{args.stage}/{args.model}_{PRECISION}_{decoding}"

    write_settings(log_dir, args.stage, args.model, decoding, epochs, model_args)

    # Options given here override the task's own. AgentHarm sets temperature 0,
    # which is right for greedy; for sampled decoding it must be replaced.
    options = {}
    if decoding == "sampled":
        options["temperature"] = SAMPLED_TEMPERATURE

    success, logs = eval_set(
        build_tasks(args.stage),
        log_dir=log_dir,
        model=spec["name"],
        model_args=model_args,
        epochs=epochs,
        max_connections=BATCH,   # overrides AgentHarm's 100
        max_samples=BATCH,
        fail_on_error=False,
        **options,
    )

    for log in logs:
        summarise(log)

    csv_path = write_samples_csv(log_dir, logs)
    print("=" * 70)
    print(f"all tasks completed: {success}")
    print(f"settings: {os.path.join(log_dir, 'run_settings.json')}")
    print(f"per-sample table: {csv_path}")
    if not success:
        print("Some tasks did not finish. Run the same command again to resume.")


if __name__ == "__main__":
    main()

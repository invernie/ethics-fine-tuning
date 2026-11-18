# ethics-fine-tuning
Effect of moral philosophy corpus fine-tuning on agent safety and security performance

## Project plan

### Questions: 
* Do models re-exposed to pre-training ethics content by fine-tuning on question-response pairs generated from classic moral philosophy texts refuse harmful agent tasks more often than models fine-tuned on matched control texts?
* Is there any performance loss when looking at over-refusal and task success?

### Step-by-step approach
1.  Pick the model.
    * Small open-weight model (about 1–8B) with native tool calling.
    * Decide base vs. instruct version, with justification.
    * Shortlist 2–3 candidates to baseline.
    * Model selection will happen based on baseline performance against the test benchmarks.
2.  Baseline each model.
    * Baseline evaluation is equivalent to step 7 (evaluation). The baseline performance results will count as the "no fine-tuning" (A) control arm of the experiment.
    * Baseline is run against the same benchmarks used for testing: AgentHarm, AgentHarm-Benign and AgentDojo
    * Evaluation settings are locked now: prompt template, sampling temperature, number of samples, judge model. If any changes emerge as required later on, the evaluation of the A control arm will need to be re-run with adjusted settings.
    * Enables checking the capability floor (can it use tools and complete benign tasks?) and ceiling (does it already refuse nearly everything?).
    * Repeat sampling to estimate variance for the power analysis.
    * Review sample results manually to estimate if SLM answer categorisation is useful (see https://doi.org/10.48550/arXiv.2608.17183).
3.  Preregister hypotheses and power analysis.
    * Examples of hypothesis pre-registration:
        * H1: arm B refuses more than arm C on AgentHarm.
        * H2: arm B's over-refusal is no more than X percentage points above arm C's (X set in advance).
        * H3: refusal rises with dose.
    * Power analysis based on baseline repeat sampling.
4.  Define the arms.
    * A - no fine-tuning arm: our control baseline model.
    * B - ethics arm: our test arm.
    * C - non-ethics philosophy arm: controls for fine-tuning on generic philosophical reasoning (from the same period).
    * D = generic text: controls for the effect of fine-tuning alone.
    * Same token budget and hyperparameters for every trained arm.
5.  Build the corpora.
    * Source: public-domain Project Gutenberg texts.
    * Ethics arm (B): the nine classic ethics philosophy texts in the corpus tab. This arm tests our hypothesis.
    * Fine-tuning control arms C and D: non-ethics philosophy by the same authors and era (C) and generic text (D).
    * Build prompt-response pairs from raw texts (AI-generated with human lay-person sample check).
    * Run ethics_corpus.py; strip boilerplate, Latin and front matter; equalise token counts across arms.
    * Audit data quality: spot-check samples, remove duplicates, verify copyright.
6.  Fine-tune with LoRA.
    * Computation: 2-4 GPUs.
    * No refusal examples as they would confound the question.
    * DDP for LoRA, running arms and seeds in parallel; FSDP if fully fine-tuning an 8B model (optional comparison with LoRA on one arm).
    * Log memory, throughput and checkpoints; fix seeds.
7.  Evaluate every arm with the locked settings.
    * Harmful-task refusal (agent safety): AgentHarm.
    * Over-refusal on benign tasks: AgentHarm-benign.
    * Attack success rate (agent security): AgentDojo.
8.  Statistics.
    * At least 3 seeds
    * Cluster bootstrap or mixed-effects model.
    * Null results are results.

### Baselines
Arm A shows how the model behaved originally; arms C and D show what fine-tuning alone does (Qi et al., ICLR 2024).

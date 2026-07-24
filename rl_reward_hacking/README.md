# RL Reward-Hacking Experiments

This directory contains the RL code used to study **reward hacking of Process Reward Models (PRMs)**:
we run GRPO on a policy whose reward comes *entirely* from a frozen PRM, and watch the PRM reward climb
while ground-truth accuracy stalls (reward–accuracy divergence).

It is a fork of [verl](https://github.com/volcengine/verl). All PRM-specific logic lives in
[`recipe/prm/`](recipe/prm/):

| File | Role |
|---|---|
| `recipe/prm/main_prm.py` | Entry point (`python -m recipe.prm.main_prm`). Builds the trainer; gates the PRM reward on `reward_model.enable`. |
| `recipe/prm/prm_ray_trainer.py` | GRPO training loop. |
| `recipe/prm/fsdp_reward_model_worker.py` | Loads the PRM **in-process** (FSDP) and turns its per-step scores into token rewards. Trajectory- vs step-level is decided here (`use_last_reward_only`). |
| `recipe/prm/config/prm.yaml` | Hydra config (defaults over verl's `ppo_trainer`). |
| `launch_skywork_prm_1.5b.sh` | Env-var-driven launch script — the entry point for every run below. |

> The PRM is served **inside the training job** via the FSDP reward worker — you do **not** need to start a
> separate reward server. (`skywork_prm/start_skywork_prm.sh` is an optional standalone vLLM PRM server and is
> **not** used by the launch script.)

---

## 1. Installation

```shell
conda create -n verl_prm_attack python==3.10
conda activate verl_prm_attack
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

Set your credentials before launching (either export them or drop a `secrets.env` next to the launch script —
it is sourced automatically and is git-ignored):

```shell
export WANDB_API_KEY=...
export HUGGING_FACE_HUB_TOKEN=...
```

## 2. Prepare datasets

AIME-2024 is already included at `dataset/aime_2024/train.parquet`. To (re)generate it, or to build MATH:

```shell
# AIME-2024 (used by the 7B experiments)
python examples/data_preprocess/aime_2024_dataset.py --local_dir dataset/aime_2024

# MATH (used by the 1.5B experiments)
python examples/data_preprocess/math_dataset.py --local_dir dataset/math_dataset
```

---

## 3. Launch an experiment

All runs go through `launch_skywork_prm_1.5b.sh`, which is fully **environment-variable driven**. The default
invocation trains `Qwen2.5-1.5B-Instruct` with the Qwen `Math-PRM-7B` reward on AIME-2024:

```shell
bash launch_skywork_prm_1.5b.sh
```

### Environment variables (the main knobs)

| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL_PATH` | `Qwen/Qwen2.5-1.5B-Instruct` | Policy being trained. |
| `PRM_MODEL_PATH` | `Qwen/Qwen2.5-Math-PRM-7B` | **PRM that provides the reward.** Swap this to change PRM. |
| `TRAIN_DATA_DIR` / `TEST_DATA_DIR` | `dataset/aime_2024/train.parquet` | Train / eval parquet. |
| `BATCH_SIZE` | `16` | Prompts per step (`data.train_batch_size` + `ppo_mini_batch_size`). |
| `ROLLOUTS_PER_PROMPT` | `8` | GRPO group size `G`. |
| `GPU_NUM` / `NUM_NODES` | `4` / `1` | Hardware (rebuttal runs used `GPU_NUM=8`). |
| `EXPERIMENT_NAME` / `PROJECT_NAME` | see script | W&B run / project name. |
| `CHECKPOINT_DIR` | `./experiments/$EXPERIMENT_NAME` | Output dir. |

Example — swap in a different PRM and base model:

```shell
EXPERIMENT_NAME=exp4-7b-skywork7b-aime \
BASE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
PRM_MODEL_PATH=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B \
TRAIN_DATA_DIR=dataset/aime_2024/train.parquet \
TEST_DATA_DIR=dataset/aime_2024/train.parquet \
BATCH_SIZE=8 GPU_NUM=8 \
bash launch_skywork_prm_1.5b.sh
```

### Knobs that are *not* env vars (edit the launch script / append a Hydra override)

A few per-run settings are passed as Hydra overrides in the `python -m recipe.prm.main_prm ...` block of the
launch script. Change them there (or add them to the command):

| Setting | Override | Notes |
|---|---|---|
| **Learning rate** | `actor_rollout_ref.actor.optim.lr=` | `5e-7` for 7B policies, `1e-6` for 1.5B policies. |
| **Micro-batch / GPU** | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=` | `2` for 7B, `4` for 1.5B. |
| **Trajectory vs step-level** | `reward_model.use_last_reward_only=` | `True` = trajectory reward (last-step PRM score); `False` = per-step PRM rewards. |
| **Correctness-only baseline** | `reward_model.enable=False` | Disables the PRM; reward becomes binary GT-answer correctness (standard outcome-reward GRPO). |

---

## 4. Rebuttal run grid

The eight runs reported in the rebuttal, and the settings that distinguish them. W&B project
`prm-attack-verl`. Everything else follows `recipe/prm/config/prm.yaml` (GRPO, `kl_coef=0`, `G=8`,
1000 steps, eval every 50 steps at mean@8).

| Run | `BASE_MODEL_PATH` | Dataset | `PRM_MODEL_PATH` | Reward mode | LR |
|---|---|---|---|---|---|
| EXP-1 | Qwen2.5-7B-Instruct | AIME-2024 | Skywork-o1-Open-PRM-1.5B | trajectory (`use_last_reward_only=True`) | 5e-7 |
| EXP-2 | Qwen2.5-1.5B-Instruct | MATH-500 | Skywork-o1-Open-PRM-1.5B | trajectory | 1e-6 |
| EXP-3 | Qwen2.5-7B-Instruct | AIME-2024 | Skywork-o1-Open-PRM-1.5B | step-level (`use_last_reward_only=False`) | 5e-7 |
| EXP-4 | Qwen2.5-7B-Instruct | AIME-2024 | Skywork-o1-Open-PRM-7B | trajectory | 5e-7 |
| EXP-5 | Qwen2.5-7B-Instruct | AIME-2024 | Skywork-o1-Open-PRM-7B | step-level | 5e-7 |
| EXP-6 | Qwen2.5-1.5B-Instruct | MATH-500 | Skywork-o1-Open-PRM-7B | trajectory | 1e-6 |
| Baseline-1 | Qwen2.5-7B-Instruct | AIME-2024 | — (`reward_model.enable=False`) | correctness only | 5e-7 |
| Baseline-2 | Qwen2.5-1.5B-Instruct | MATH-500 | — (`reward_model.enable=False`) | correctness only | 1e-6 |

7B runs used `BATCH_SIZE=8`, `ppo_micro_batch_size_per_gpu=2`; 1.5B runs used `BATCH_SIZE=16`,
`ppo_micro_batch_size_per_gpu=4`. All runs: `GPU_NUM=8`, single node, FSDP with param + optimizer offload.

---

## 5. Outputs

- **W&B**: reward (`critic/score/mean` for trajectory, `critic/mean_step_reward/mean` for step-level) logged
  every step; validation accuracy (`val-core/.../reward/mean@8`) every 50 steps.
- **Checkpoints & rollouts**: under `experiments/$EXPERIMENT_NAME/` (validation generations in
  `.../validation_data/`).

The headline finding: PRM reward rises 3–4× while `mean@8` accuracy plateaus early — and step-level rewards
hack *harder* than trajectory-level (shorter responses, single-step collapse).

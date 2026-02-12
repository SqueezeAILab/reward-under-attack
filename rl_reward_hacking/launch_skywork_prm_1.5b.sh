#!/bin/bash

# Launch script for GRPO training with Skywork PRM 1.5B
# This script trains Qwen2.5-Math-1.5B using Skywork Process Reward Model for token-level rewards

set -x

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo-qwen-math-prm-aime-instruct-7b"}
PROJECT_NAME=${PROJECT_NAME:-"prm-attack-verl"}

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-"Qwen/Qwen2.5-Math-PRM-7B"}

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"dataset/aime_2024/train.parquet"}
TEST_DATA_DIR=${TEST_DATA_DIR:-"dataset/aime_2024/train.parquet"}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
BATCH_SIZE=${BATCH_SIZE:-16}
ROLLOUTS_PER_PROMPT=${ROLLOUTS_PER_PROMPT:-8}

# ==============================================================================
# HARDWARE CONFIGURATION
# ==============================================================================
GPU_NUM=${GPU_NUM:-4}
NUM_NODES=${NUM_NODES:-1}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./experiments/$EXPERIMENT_NAME"}

# ==============================================================================
# LOAD ENVIRONMENT VARIABLES
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/secrets.env" ]; then
    source "$SCRIPT_DIR/secrets.env"
elif [ -f "$SCRIPT_DIR/recipe/prm/secrets.env" ]; then
    source "$SCRIPT_DIR/recipe/prm/secrets.env"
fi

# Set experiment tracking environment variables
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-""}

# Set model and data paths for config
export BASE_MODEL_PATH=$BASE_MODEL_PATH
export PRM_MODEL_PATH=$PRM_MODEL_PATH
export TRAIN_DATA_DIR=$TRAIN_DATA_DIR
export TEST_DATA_DIR=$TEST_DATA_DIR
export EXPERIMENT_NAME=$EXPERIMENT_NAME
export PROJECT_NAME=$PROJECT_NAME
export GPU_NUM=$GPU_NUM
export NUM_NODES=$NUM_NODES
export CHECKPOINT_DIR=$CHECKPOINT_DIR

# ==============================================================================
# TRAINING EXECUTION
# ==============================================================================
echo "Starting GRPO training with Skywork PRM 1.5B..."
echo "Base Model: $BASE_MODEL_PATH"
echo "PRM Model: $PRM_MODEL_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Hardware: ${GPU_NUM}x GPU, ${NUM_NODES}x Node(s)"

python -m recipe.prm.main_prm \
    `# ==============================================================================` \
    `# ALGORITHM CONFIGURATION - GRPO with PRM` \
    `# ==============================================================================` \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    `# ==============================================================================` \
    `# DATA CONFIGURATION` \
    `# ==============================================================================` \
    data.train_files=$TRAIN_DATA_DIR \
    data.val_files=$TEST_DATA_DIR \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.truncation=left \
    \
    `# ==============================================================================` \
    `# ACTOR MODEL CONFIGURATION` \
    `# ==============================================================================` \
    actor_rollout_ref.model.path=$BASE_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    `# ==============================================================================` \
    `# ACTOR TRAINING CONFIGURATION` \
    `# ==============================================================================` \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    \
    `# ==============================================================================` \
    `# ROLLOUT CONFIGURATION (vLLM)` \
    `# ==============================================================================` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$ROLLOUTS_PER_PROMPT \
    actor_rollout_ref.rollout.max_tokens=2048 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    \
    `# ==============================================================================` \
    `# REFERENCE MODEL CONFIGURATION` \
    `# ==============================================================================` \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    \
    `# ==============================================================================` \
    `# PROCESS REWARD MODEL (PRM) CONFIGURATION` \
    `# ==============================================================================` \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.model.path=$PRM_MODEL_PATH \
    reward_model.model.use_remove_padding=False \
    reward_model.model.enable_gradient_checkpointing=True \
    reward_model.model.trust_remote_code=True \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.use_dynamic_bsz=False \
    reward_model.reward_manager=naive \
    \
    `# ==============================================================================` \
    `# TRAINER CONFIGURATION` \
    `# ==============================================================================` \
    trainer.logger='[console,wandb]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=$NUM_NODES \
    \
    `# ==============================================================================` \
    `# TRAINING SCHEDULE` \
    `# ==============================================================================` \
    trainer.total_epochs=1000 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    \
    `# ==============================================================================` \
    `# CHECKPOINT AND LOGGING` \
    `# ==============================================================================` \
    trainer.default_local_dir=experiments/$EXPERIMENT_NAME \
    trainer.validation_data_dir=experiments/$EXPERIMENT_NAME/validation_data \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.resume_mode=auto \
    \
    `# ==============================================================================` \
    `# VALIDATION CONFIGURATION` \
    `# ==============================================================================` \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \

echo "Training completed!"

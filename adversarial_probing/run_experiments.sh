#!/bin/bash
# ============================================
# Adversarial Token Optimization Experiments
# Reproduces Section 5 of the paper
# ============================================
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Download Skywork PRM models to $HF_CACHE_PATH (see README.md)
#   3. Create logs directory: mkdir -p logs
#
# Key arguments:
#   --prm_model: 1.5B, 7B (Skywork PRMs), or Qwen-7B (Qwen Math PRM)
#   --experiment: single or batched
#   --continuous: add this flag for continuous optimization (default is discrete)
#   --num_adv_tokens: number of adversarial tokens (default: 1)
#   --adv_position: position of adv tokens - "end" (after solution) or "middle" (after question)
#   --num_iterations: optimization iterations (default: 1000)
#   --num_train_trajectories: limit training trajectories (default: all)
#   --num_eval_trajectories: limit eval trajectories (default: all)
#   --run_transfer: run transfer evaluation on AIME2025
#   --run_plots: generate training curve plots
#   --run_3d_landscape: generate 3D reward landscape
#   --force_rerun: force rerun even if cache exists
#
# PRM Types:
#   - Skywork (1.5B, 7B): Optimizes final step reward. Adv tokens appended after solution (--adv_position end).
#   - Qwen (Qwen-7B): Optimizes minimum step reward. Adv tokens inserted before solution (--adv_position middle).
#
# Environment variables:
#   HF_CACHE_PATH: Path to directory containing downloaded Skywork checkpoints (default: ./hf_cache)

set -e
mkdir -p logs

# ============================================
# SKYWORK-1.5B EXPERIMENTS
# ============================================

# --- Continuous Optimization (k=1 token) ---
# Demonstrates that a single continuous embedding vector suffices to inflate rewards
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position end \
    --continuous \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2 \
    2>&1 | tee logs/batched_1.5B_continuous_1tok_end.log

# --- Discrete Optimization (k=1 token) ---
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2 \
    2>&1 | tee logs/batched_1.5B_discrete_1tok_end.log

# --- Discrete Optimization (k=50 tokens) ---
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 50 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2 \
    2>&1 | tee logs/batched_1.5B_discrete_50tok_end.log

# --- Discrete Optimization (k=100 tokens) ---
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 100 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2 \
    2>&1 | tee logs/batched_1.5B_discrete_100tok_end.log

# ============================================
# SKYWORK-7B EXPERIMENTS
# ============================================

# --- Discrete Optimization (k=1 token) ---
python prm_attack_experiments.py \
    --prm_model 7B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/batched_7B_discrete_1tok_end.log

# --- Discrete Optimization (k=50 tokens) ---
python prm_attack_experiments.py \
    --prm_model 7B \
    --experiment batched \
    --num_adv_tokens 50 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/batched_7B_discrete_50tok_end.log

# --- Discrete Optimization (k=100 tokens) ---
python prm_attack_experiments.py \
    --prm_model 7B \
    --experiment batched \
    --num_adv_tokens 100 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/batched_7B_discrete_100tok_end.log

# ============================================
# QWEN-7B EXPERIMENTS
# Qwen PRM locates first incorrect step,
# so adversarial tokens are inserted before the solution (--adv_position middle)
# ============================================

# --- Discrete Optimization (k=1 token) ---
python prm_attack_experiments.py \
    --prm_model Qwen-7B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position middle \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_1tok_middle.log

# --- Discrete Optimization (k=50 tokens) ---
python prm_attack_experiments.py \
    --prm_model Qwen-7B \
    --experiment batched \
    --num_adv_tokens 50 \
    --adv_position middle \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_50tok_middle.log

# --- Discrete Optimization (k=100 tokens) ---
python prm_attack_experiments.py \
    --prm_model Qwen-7B \
    --experiment batched \
    --num_adv_tokens 100 \
    --adv_position middle \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 1 \
    2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_100tok_middle.log

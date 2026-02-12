# Installation

```shell
conda create -n verl_prm_attack python==3.10
conda activate verl_prm_attack
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

# Prepare train/test dataset (MATH dataset)

```shell
python examples/data_preprocess/math_dataset.py --local_dir dataset/math_dataset
```

# Run Experiments [WIP]

```shell
bash launch_skywork_prm_1.5b.sh
```
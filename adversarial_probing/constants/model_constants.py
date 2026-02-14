import os

from utils.processors import (
    qwen_math_prm, skywork_o1_open_prm
)
from utils.models import Qwen2ForProcessRewardModel, SkyworkO1OpenPRMForProcessRewardModel

Qwen2_5_Math_PRM_7B = "Qwen/Qwen2.5-Math-PRM-7B"

# For Skywork models, set HF_CACHE_PATH to the directory containing downloaded checkpoints.
# See README.md for download instructions.
HF_CACHE_PATH = os.environ.get("HF_CACHE_PATH", "./hf_cache")
Skywork_o1_Open_PRM_Qwen_2_5_7B = f"{HF_CACHE_PATH}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B"
Skywork_o1_Open_PRM_Qwen_2_5_1_5B = f"{HF_CACHE_PATH}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

PREPARE_INPUT_MAP = {
    Qwen2_5_Math_PRM_7B: qwen_math_prm.prepare_input,

    Skywork_o1_Open_PRM_Qwen_2_5_7B: skywork_o1_open_prm.prepare_input,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: skywork_o1_open_prm.prepare_input,
}

DERIVE_STEP_REWARDS_MAP = {
    Qwen2_5_Math_PRM_7B: qwen_math_prm.derive_step_rewards,

    Skywork_o1_Open_PRM_Qwen_2_5_7B: skywork_o1_open_prm.derive_step_rewards,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: skywork_o1_open_prm.derive_step_rewards,
}

MODEL_CLASS_MAP = {
    Qwen2_5_Math_PRM_7B: Qwen2ForProcessRewardModel,

    Skywork_o1_Open_PRM_Qwen_2_5_7B: SkyworkO1OpenPRMForProcessRewardModel,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: SkyworkO1OpenPRMForProcessRewardModel,
}
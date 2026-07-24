from recipe.prm.modeling_utils.processors import (
    qwen_math_prm, math_shepherd_prm, 
    rlhflow_math_prm, skywork_o1_open_prm
) 
from recipe.prm.modeling_utils.models import Qwen2ForProcessRewardModel, SkyworkO1OpenPRMForProcessRewardModel
from transformers import AutoModelForCausalLM

Qwen2_5_Math_PRM_7B = "Qwen/Qwen2.5-Math-PRM-7B"

Skywork_o1_Open_PRM_Qwen_2_5_7B = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
Skywork_o1_Open_PRM_Qwen_2_5_1_5B = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

Math_Shepherd_Mistral_7B_PRM = "peiyi9979/math-shepherd-mistral-7b-prm"

Llama3_1_8B_PRM_Mistral_Data = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
Llama3_1_8B_PRM_Deepseek_Data = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

PREPARE_INPUT_MAP = {
    Qwen2_5_Math_PRM_7B: qwen_math_prm.prepare_input,
    
    Skywork_o1_Open_PRM_Qwen_2_5_7B: skywork_o1_open_prm.prepare_input,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: skywork_o1_open_prm.prepare_input,
    
    Math_Shepherd_Mistral_7B_PRM: math_shepherd_prm.prepare_input,

    Llama3_1_8B_PRM_Mistral_Data: rlhflow_math_prm.prepare_input,
    Llama3_1_8B_PRM_Deepseek_Data: rlhflow_math_prm.prepare_input,
}

DERIVE_LAST_REWARDS_MAP = {
    Qwen2_5_Math_PRM_7B: qwen_math_prm.derive_last_reward,

    Skywork_o1_Open_PRM_Qwen_2_5_7B: skywork_o1_open_prm.derive_last_reward,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: skywork_o1_open_prm.derive_last_reward,

    Math_Shepherd_Mistral_7B_PRM: math_shepherd_prm.derive_last_reward,

    Llama3_1_8B_PRM_Mistral_Data: rlhflow_math_prm.derive_last_reward,
    Llama3_1_8B_PRM_Deepseek_Data: rlhflow_math_prm.derive_last_reward,
}

DERIVE_STEP_REWARDS_MAP = {
    Qwen2_5_Math_PRM_7B: qwen_math_prm.derive_step_rewards,

    Skywork_o1_Open_PRM_Qwen_2_5_7B: skywork_o1_open_prm.derive_step_rewards,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: skywork_o1_open_prm.derive_step_rewards,

    Math_Shepherd_Mistral_7B_PRM: math_shepherd_prm.derive_step_rewards,

    Llama3_1_8B_PRM_Mistral_Data: rlhflow_math_prm.derive_step_rewards,
    Llama3_1_8B_PRM_Deepseek_Data: rlhflow_math_prm.derive_step_rewards,
}

MODEL_CLASS_MAP = {
    Qwen2_5_Math_PRM_7B: Qwen2ForProcessRewardModel,
    
    Skywork_o1_Open_PRM_Qwen_2_5_7B: SkyworkO1OpenPRMForProcessRewardModel,
    Skywork_o1_Open_PRM_Qwen_2_5_1_5B: SkyworkO1OpenPRMForProcessRewardModel,

    Math_Shepherd_Mistral_7B_PRM: AutoModelForCausalLM,

    Llama3_1_8B_PRM_Mistral_Data: AutoModelForCausalLM,
    Llama3_1_8B_PRM_Deepseek_Data: AutoModelForCausalLM
}
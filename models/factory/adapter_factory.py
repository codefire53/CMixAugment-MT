from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
from omegaconf import DictConfig

def create_adapter_config(adapter_config_params: DictConfig):
    adapter_type = adapter_config_params.name
    config_params_copy = adapter_config_params.copy()
    del config_params_copy.name
    task_type=TaskType.SEQ_2_SEQ_LM
    if adapter_type == 'prefix_tuning':
        return PrefixTuningConfig(task_type=task_type, **config_params_copy)
    elif adapter_type == 'lora':
        return LoraConfig(target_modules=["embed_tokens", "k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"], task_type=task_type, **config_params_copy)
    elif adapter_type == 'ptuning':
        return PromptEncoderConfig(
            task_type=task_type,
            **config_params_copy
        )
    else:
        return PromptTuningConfig(task_type=task_type, **config_params_copy)

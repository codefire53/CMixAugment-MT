from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollator
from omegaconf import DictConfig
from adapters import AutoAdapterModel

def initialize_collator_module(tokenizer: AutoTokenizer, collator_config_params: DictConfig) -> DataCollator:
    return DataCollatorForSeq2Seq(tokenizer, **collator_config_params)
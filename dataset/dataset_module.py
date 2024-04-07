from datasets import load_dataset, Dataset, DatasetDict, load_metric, load_from_disk
from transformers import AutoConfig, AutoTokenizer
from omegaconf import DictConfig
import random
from torch.utils.data import DataLoader
from factory.collator_factory import initialize_collator_module
import pandas as pd

class BaseDataset:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_params: DictConfig, collator_params: DictConfig, dataloader_params: DictConfig):
        tokenizer_params_copy = tokenizer_params.copy()
        if hasattr(tokenizer_params_copy, 'prefix'):
            self.prefix = tokenizer_params_copy.prefix
            del tokenizer_params_copy.prefix
        else:
            self.prefix = None
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params_copy

        self.collator = initialize_collator_module(tokenizer, collator_params)

        self.dataloader_params = dataloader_params

class CALCS2021Dataset(BaseDataset):
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_params: DictConfig, dataset_params: DictConfig, collator_params: DictConfig, dataloader_params: DictConfig):
        super(CALCS2021Dataset, self).__init__(tokenizer, tokenizer_params, collator_params, dataloader_params)
        self.train_file_path = dataset_params.mt.train_file_path
        self.val_file_path = dataset_params.mt.val_file_path
        self.test_file_path = dataset_params.mt.test_file_path
        self.collator = initialize_collator_module(tokenizer, collator_params)
        self.dataloader_params = dataloader_params

    def load_dataset(self):
        self.train_dataset = self.load_dataset_split(self.train_file_path)
        self.val_dataset = self.load_dataset_split(self.val_file_path)
        self.test_dataset = self.load_dataset_split(self.test_file_path, True)

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, collate_fn=self.collator, **self.dataloader_params)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False, collate_fn=self.collator, **self.dataloader_params)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=False, collate_fn=self.collator, **self.dataloader_params)

    def load_parallel_sentences(self, file_path: str, delimiter='\t', is_test=False) -> Dataset:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        sentence_pairs = [line.split(delimiter) for line in lines]
        data_dict = dict()
        source_sentences = [pair[0] if self.prefix is None else f"{self.prefix}: {pair[0]}" for pair in sentence_pairs]
        data_dict['source'] = source_sentences
        if len(sentence_pairs[0]) > 1:
            target_sentences = [pair[1] for pair in sentence_pairs]
            data_dict['target'] = target_sentences
            if is_test:
                self.test_refs = target_sentences
        if is_test:
            self.test_sources = source_sentences  # add sources to be written on the text file for evaluation
        return Dataset.from_dict(data_dict)
        
    def tokenize_function(self, examples: list):
        model_inputs = self.tokenizer(examples['source'], **self.tokenizer_params)
        with self.tokenizer.as_target_tokenizer():
            if 'target' in examples:
                labels = self.tokenizer(examples['target'], **self.tokenizer_params)
                model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_dataset_split(self, file_path: str, is_test: bool = False):
        parallel_corpus = self.load_parallel_sentences(file_path)
        tokenized_parallel_corpus = parallel_corpus.map(lambda examples: self.tokenize_function(examples), batched=True, remove_columns=parallel_corpus.column_names)
        return tokenized_parallel_corpus

class CardiffSentimentEnglishDataset(BaseDataset):
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_params: DictConfig, dataset_params: DictConfig, collator_params: DictConfig, dataloader_params: DictConfig):
        super(CardiffSentimentEnglishDataset, self).__init__(tokenizer, tokenizer_params, collator_params, dataloader_params)
        self.test_dataset = load_dataset(dataset_params.sa.dataset_name, 'english', split='train')
        self.id2label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        self.labels = [self.id2label[item['label']] for item in self.test_dataset]

    def load_dataset(self):
        self.test_dataset = self.load_dataset_split(self.test_dataset)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=False, collate_fn=self.collator, **self.dataloader_params)

    def load_dataset_split(self, parallel_corpus):
        col_names = parallel_corpus.column_names
        
        tokenized_parallel_corpus = parallel_corpus.map(lambda examples: self.tokenize_function(examples), batched=True, remove_columns=parallel_corpus.column_names)
        return tokenized_parallel_corpus

    def tokenize_function(self, examples: list):
        model_inputs = self.tokenizer(examples['text'], **self.tokenizer_params)
        return model_inputs

class MNLIDataset(BaseDataset):
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_params: DictConfig, dataset_params: DictConfig, collator_params: DictConfig, dataloader_params: DictConfig):
        super(MNLIDataset, self).__init__(tokenizer, tokenizer_params, collator_params, dataloader_params)
        with open(dataset_params.nli.dataset_filepath, 'r', encoding='utf-8') as file:    
            lines = file.read().splitlines()
        lines = lines[1:] # skip the columns row
        delimiter = '\t'
        columns = [line.split(delimiter) for line in lines]
        data_dict = dict()
        data_dict_2 = dict()
        premises = [column[0] for column in columns]
        hypotheses = [column[1] for column in columns]
        labels = [column[2] for column in columns]
        data_dict['sentences'] = premises
        data_dict_2['sentences'] = hypotheses
        self.labels = labels
        self.test_dataset = Dataset.from_dict(data_dict)
        self.test_dataset_2 = Dataset.from_dict(data_dict_2)

    def load_dataset(self):
        self.test_dataset = self.load_dataset_split(self.test_dataset)
        self.test_dataset_2 = self.load_dataset_split(self.test_dataset_2)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=False, collate_fn=self.collator, **self.dataloader_params)
        self.test_dataloader_2 = DataLoader(self.test_dataset_2, shuffle=False, collate_fn=self.collator, **self.dataloader_params)

    def load_dataset_split(self, parallel_corpus):
        col_names = parallel_corpus.column_names
        tokenized_parallel_corpus = parallel_corpus.map(lambda examples: self.tokenize_function(examples), batched=True, remove_columns=parallel_corpus.column_names)
        return tokenized_parallel_corpus

    def tokenize_function(self, examples: list):
        model_inputs = self.tokenizer(examples['sentences'], **self.tokenizer_params)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['sentences'], **self.tokenizer_params)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

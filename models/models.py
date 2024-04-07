import lightning as L
from omegaconf import DictConfig
from models.factory.model_factory import initialize_adapter_model, initialize_tokenizer, initialize_adapter_model, initialize_baseline_seq2seq_model
from models.factory.adapter_factory import create_adapter_config
from transformers import AutoTokenizer
from torch.optim import AdamW
import evaluate
import torch
from peft import SftAdamW, SftSelector

class BaseLightningModel(L.LightningModule):
    def __init__(self, model_config_params: DictConfig, optim_config_params:DictConfig = None):
        super(BaseLightningModel, self).__init__()
        self.optim_config_params = optim_config_params

        # Load metrics
        self.bleu_metric = evaluate.load('sacrebleu')
        self.rouge_metric = evaluate.load('rouge')
        self.chrf_metric = evaluate.load('chrf')

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]
        decoded_labels_bleu = [[label] for label in decoded_labels]
        
        # Update metrics
        self.bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels_bleu)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.chrf_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def on_validation_epoch_end(self):
        # Compute metrics
        bleu_result = self.bleu_metric.compute()
        rouge_result = self.rouge_metric.compute()
        chrf_result = self.chrf_metric.compute(word_order=2)

        # Log metrics
        self.log_dict({'val_bleu': bleu_result['score'], 'val_rouge': rouge_result['rougeL'], 
                       'val_chrf++': chrf_result['score']}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_source = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        decoded_labels_bleu = [[label] for label in decoded_labels]
        
        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]

        test_pairs_batch = [(actual_sentence, pred_sentence, source) for pred_sentence, actual_sentence, source in zip(decoded_preds, decoded_labels, decoded_source)]
        self.test_pairs.extend(test_pairs_batch)

        
        # Update metrics
        self.bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels_bleu)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.chrf_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.optim_config_params)


    def translate_sentence(self, loader):
        translations = []
        for batch in loader:
            generated_tokens = self.model.generate(input_ids=batch['input_ids'].to('cuda'), attention_mask=batch['attention_mask'].to('cuda'))
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translations.extend(decoded_preds)
        return translations

class AdapterSeq2SeqModel(BaseLightningModel):
    def __init__(self, model_config_params: DictConfig, adapter_config_params: DictConfig, optim_config_params:DictConfig = None):
        super(AdapterSeq2SeqModel, self).__init__(model_config_params, optim_config_params)
        self.adapter_config = create_adapter_config(adapter_config_params)
        self.model = initialize_adapter_model(model_config_params, adapter_config_params.name, self.adapter_config)
        self.tokenizer = initialize_tokenizer(self.model, model_config_params)

class SftSeq2SeqModel(BaseLightningModel):
    def __init__(self, model_config_params: DictConfig, sft_config_params: DictConfig, optim_config_params:DictConfig = None):
        super(SftSeq2SeqModel, self).__init__(model_config_params, optim_config_params)
        self.model, self.peft_config = initialize_sft_model(model_config_params, sft_config_params)
        self.tokenizer = initialize_tokenizer(self.model, model_config_params) 
    
    def on_train_batch_end(self,  outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (batch_idx + 1) % self.optim_config_params.gradient_accumulation_steps == 0:
            self.selector.step()
            
    def configure_optimizers(self):
        optimizer = SftAdamW(self.model.parameters(), lr=self.optim_config_params.lr, momentum_dtype=torch.float32)
        num_train_steps = (self.optim_config_params.num_train_samples*self.optim_config_params.num_train_epochs)//(self.optim_config_params.gradient_accumulation_steps*self.optim_config_params.batch_size)
        self.selector = SftSelector(
           self.model,
            optimizer,
            self.peft_config,
            num_train_steps, # total expected duration of training in update steps
            self.optim_config_params.gradient_accumulation_steps, # grad accumulation steps per update step
        )
        return optimizer

class Seq2SeqModel(BaseLightningModel):
    def __init__(self, model_config_params: DictConfig, optim_config_params:DictConfig = None):
        super(Seq2SeqModel, self).__init__(model_config_params, optim_config_params)
        self.model = initialize_baseline_seq2seq_model(model_config_params)
        self.tokenizer = initialize_tokenizer(self.model, model_config_params)

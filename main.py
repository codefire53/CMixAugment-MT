from omegaconf import DictConfig, OmegaConf
import hydra
from dataset.dataset_module import CALCS2021Dataset, CardiffSentimentEnglishDataset, MNLIDataset
import numpy as np
from models.models import AdapterSeq2SeqModel, SftSeq2SeqModel, Seq2SeqModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor

def load_dataset_from_given_configs(tokenizer, tokenizer_cfg, dataset_cfg, collator_cfg, dataloader_cfg):
    if dataset_cfg.task == 'mt':
        dataset = CALCS2021Dataset(model.tokenizer, cfg.tokenizers, cfg.datasets, cfg.collators, cfg.dataloader)
    elif dataset_cfg.task == 'sa':
        dataset = CardiffSentimentEnglishDataset(model.tokenizer, cfg.tokenizers, cfg.datasets, cfg.collators, cfg.dataloader)
    else:
        dataset = MNLIDataset(model.tokenizer, cfg.tokenizers, cfg.datasets, cfg.collators, cfg.dataloader)
    dataset.load_dataset()
    return dataset

def load_model(model_cfg, optim_cfg, base_cfg):
    if hasattr(base_cfg, 'adapters'):
        model = AdapterSeq2SeqModel(model_cfg, base_cfg.adapters, optim_cfg)
    elif hasattr(base_cfg, 'sft'):
        model = SftSeq2SeqModel(model_cfg, base_cfg.sft, optim_cfg)
    else:
        model = Seq2SeqModel(model_cfg, optim_cfg)
    return model

def load_model_checkpoint(checkpoint_file, model_cfg, optim_cfg, base_cfg):
    if hasattr(base_cfg, 'adapters'):
        model = AdapterSeq2SeqModel.load_from_checkpoint(checkpoint_file, model_config_params=model_cfg, adapter_config_params=base_cfg.adapters)
    elif hasattr(base_cfg, 'sft'):
        model = SftSeq2SeqModel.load_from_checkpoint(checkpoint_file, model_config_params=model_cfg, sft_config_params=base_cfg.sft)
    else:
        model = Seq2SeqModel.load_from_checkpoint(checkpoint_file, model_config_params=model_cfg)
    return model

@hydra.main(version_base=None, config_path="./confs", config_name="adapter_config")
def main(cfg: DictConfig):
    seed_everything(42)
    model = load_model(cfg.models, cfg.optim, cfg)
    dataset = load_dataset_from_given_configs(model.tokenizer, cfg.tokenizers, cfg.datasets, cfg.collators, cfg.dataloader)

    # wandb logger for monitoring
    wandb_logger = WandbLogger(**cfg.loggers)
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    early_stop = EarlyStopping(**cfg.earlystopping)
    rich = RichProgressBar()
    lr_monitor = LearningRateMonitor(**cfg.lr_monitor)
    callbacks = [checkpoint_callback, early_stop, lr_monitor, rich]

    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)

    if cfg.do_train:
        trainer.fit(model, dataset.train_dataloader, dataset.val_dataloader)
    
    if cfg.do_test:
        if cfg.do_train:
            # load best checkpoint
            model = load_model_checkpoint(checkpoint_callback.best_model_path, cfg.models, cfg.optim, cfg)
        elif hasattr(cfg, 'checkpoint_file'):
            model = load_model_checkpoint(cfg.checkpoint_file, cfg.models, cfg.optim, cfg)
        if cfg.inference_only: 
            translations1 =  model.translate_sentence(dataset.test_dataloader)
            
            if hasattr(dataset, 'test_dataloader_2'):
                translations2 = model.translate_sentence(dataset.test_dataloader_2)
            else:
                translations2 = None
            
            if hasattr(dataset, 'test_sources'):
                test_sources = dataset.test_sources
            else:
                test_sources = None
            
                        
            if hasattr(dataset, 'test_refs'):
                test_refs = dataset.test_refs
            else:
                test_refs = None

            if hasattr(dataset, 'labels'):
                labels = dataset.labels
            else:
                labels = None

            #downstream task dataset translation
            if labels is not None:
                if translations2 is not None:
                    with open(cfg.output_file, 'w') as f:
                        for translation1, translation2, label in zip(translations1, translations2, labels):
                            f.write('\t'.join([translation1, translation2, label])+'\n')
                else:
                    with open(cfg.output_file, 'w') as f:
                        for translation1, label in zip(translations1, labels):
                            f.write('\t'.join([translation1, label])+'\n')
            
            else:
                if test_refs is not None:
                    with open(cfg.output_file, 'w') as f:
                        for source, translation, ref in zip(test_sources, translations1, test_refs):
                            f.write('\t'.join([source, translation, ref])+'\n')
                else:
                    with open(cfg.output_file, 'w') as f:
                        for source, translation in zip(test_sources, translations1):
                            f.write('\t'.join([source, translation])+'\n')
        else:
            trainer = Trainer(logger=wandb_logger, accelerator="gpu", callbacks=[rich], devices=1)
            trainer.test(model, dataset.test_dataloader)

if __name__ == "__main__":
    main()

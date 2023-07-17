from dataclasses import dataclass
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

# import ParaphraseDataset, FineTuneHyperParams
# from T5ParEvo.src.paraphrase.paraphrase_claim import ParaphraseDataset

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from typing import Dict, List

import logging

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    @staticmethod
    def _log_metrics(trainer: Trainer, stage: str):
        logger.info(f"***** {stage} results *****")
        if trainer.logger:
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info(f"{key} = {metrics[key]}")

    def on_validation_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        self._log_metrics(trainer, 'Validation')

    def on_test_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        self._log_metrics(trainer, 'Test')

    def on_train_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        self._log_metrics(trainer, 'Train')


@dataclass
class FineTuneHyperParams:
    model_name_path: str
    tokenizer_name_path: str
    num_train_epochs: int
    df_train: DataFrame
    df_val: DataFrame
    df_train_val: DataFrame

    max_seq_length: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    n_gpu: int = 1
    early_stop_callback: bool = False
    fp_16: bool = False
    opt_level: str = 'O1'
    max_grad_norm: float = 1.0
    seed: int = 37

    def __post_init__(self):
        self.model_name_or_path = self.model_name_path
        self.tokenizer_name_or_path = self.tokenizer_name_path


    def get_checkpoint_callback(self):
        return pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=5,
            verbose=True,
            monitor="val_loss",
            mode="min")

    def get_train_params(self):
        return dict(
            accumulate_grad_batches=self.gradient_accumulation_steps,
            gpus=self.n_gpu,
            max_epochs=self.num_train_epochs,
            precision=32,
            gradient_clip_val=self.max_grad_norm,
        )

    @staticmethod
    def from_config_file(config_path: str):
        conf = OmegaConf.load(config_path)
        return FineTuneHyperParams(**conf)

# Load from a config file
# hparams = FineTuneHyperParams.from_config_file('config.yaml')



class T5FineTuner(pl.LightningModule):
    def __init__(self, args_fine_tune_ns: FineTuneHyperParams):
        super(T5FineTuner, self).__init__()

        self.hparams.update(vars(args_fine_tune_ns))

        self.model = T5ForConditionalGeneration.from_pretrained(args_fine_tune_ns.model_name_path)
        self.tokenizer = T5Tokenizer.from_pretrained(args_fine_tune_ns.tokenizer_name_path)

        # Initialize the datasets
        self.train_dataset = ParaphraseDataset(tokenizer=self.tokenizer, dataframe=self.hparams.df_train, max_len=self.hparams.max_seq_length)
        self.val_dataset = ParaphraseDataset(tokenizer=self.tokenizer, dataframe=self.hparams.df_val, max_len=self.hparams.max_seq_length)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)     


   
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor=None, 
                decoder_input_ids: torch.Tensor=None, decoder_attention_mask: torch.Tensor=None, 
                labels: torch.Tensor=None) -> torch.Tensor:        
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch: dict) -> torch.Tensor:
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> list:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        t_total = (
            (len(self.train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return [optimizer]

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    # def train_dataloader(self) -> DataLoader:
    #     self.train_dataset = ParaphraseDataset(tokenizer=self.tokenizer, target_dataframe=self.hparams.df_train, max_len=self.hparams.max_len)
    #     return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)

    # def val_dataloader(self) -> DataLoader:
    #     val_dataset = ParaphraseDataset(tokenizer=self.tokenizer, target_dataframe=self.hparams.df_val, max_len=self.hparams.max_len)
    #     return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)        
    
    


@dataclass
class DataFrameConfig:
    source_column: str = "org_claim"
    target_column: str = "gen_claim"
    
    
class ParaphraseDataset(Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 dataframe: DataFrame, 
                 max_len: int = 512, 
                 config: DataFrameConfig = DataFrameConfig()):
        self.data = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.config = config
        self.inputs: List[Tensor] = []
        self.targets: List[Tensor] = []

        self._build()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self) -> None:
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.config.source_column], self.data.loc[idx, self.config.target_column]

            input_ = "paraphrase: "+ input_ + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)        
            
    def get_dataset(tokenizer: PreTrainedTokenizerBase, 
                dataframe: DataFrame, 
                max_len: int) -> 'ParaphraseDataset':
        return ParaphraseDataset(tokenizer=tokenizer, dataframe=dataframe, max_len=max_len)
            
                
from dataclasses import dataclass
from typing import Dict, List

import json
import pickle

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.data import Claim, ClaimPredictions

@dataclass
class ParaphrasedClaim:
    original_claim: Claim
    paraphrased_claim: Claim
    original_prediction: ClaimPredictions
    paraphrased_prediction: ClaimPredictions

    def get_difference(self):
        # This is a placeholder. You need to replace this with your own logic for
        # calculating the difference in prediction results.
        return self.original_prediction.predictions != self.paraphrased_prediction.predictions


class ParaphraseAttack:
    def __init__(self, paraphrase_model, prediction_model):
        self.paraphrase_model = paraphrase_model
        self.prediction_model = prediction_model

    def attack(self, original_claim: Claim):
        paraphrased_text = self.paraphrase_model.paraphrase(original_claim.claim)
        paraphrased_claim = Claim(original_claim.id, paraphrased_text, original_claim.evidence,
                                  original_claim.cited_docs, original_claim.release)

        original_prediction = self.prediction_model.predict(original_claim)
        paraphrased_prediction = self.prediction_model.predict(paraphrased_claim)

        return ParaphrasedClaim(original_claim, paraphrased_claim, original_prediction, paraphrased_prediction)


@dataclass
class ParaphraseAttackResult:
    attacks: List[ParaphrasedClaim]
    iteration: int
    
    
    def get_success_rate(self):
        # This is a placeholder. Replace this with your own logic for determining
        # whether an attack was successful.
        successful_attacks = [attack for attack in self.attacks if attack.get_difference()]
        return len(successful_attacks) / len(self.attacks)

    def save_json(self, filename):
        with open(filename, 'w') as f:
            json.dump([attack.__dict__ for attack in self.attacks], f)

    def save_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        


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
            
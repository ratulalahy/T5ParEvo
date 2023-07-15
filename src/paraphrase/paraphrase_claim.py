from dataclasses import dataclass
from typing import Dict, List, Any, Union
import re

import json
import pickle
from abc import ABC, abstractmethod
import torch

from T5ParEvo.src.data.data import Claim, ClaimPredictions
from T5ParEvo.src.paraphrase.paraphraser import Paraphraser
from T5ParEvo.src.models.predict_model import ModelPredictor
from T5ParEvo.src.linguistic.ner_abbr import NEREntity,Abbreviation

@dataclass
class ParaphrasedClaim:
    iteration: int
    original_claim: Claim
    paraphrased_claim: Claim
    original_prediction: ClaimPredictions
    paraphrased_prediction: ClaimPredictions
    is_ners_preserved: bool = False
    # is_abbrs_preserved: bool = False
    nli_label: bool = False
    attack_result: 'ParaphrasedAttackResult' = None


    def get_difference(self):
        # This is a placeholder. You need to replace this with your own logic for
        # calculating the difference in prediction results.
        return self.original_prediction.predictions != self.paraphrased_prediction.predictions


# class ParaphrasedAttack:
#     def __init__(self, paraphrase_model: Paraphraser, prediction_model : ModelPredictor):
#         self.paraphrase_model = paraphrase_model
#         self.prediction_model = prediction_model

#     def attack(self, iteration : int,  original_claim: Claim, original_prediction: ClaimPredictions):
#         paraphrased_texts = self.paraphrase_model.paraphrase(original_claim.claim)
#         paraphrased_claims = []
#         for paraphrased_text in paraphrased_texts:
#             paraphrased_claim = Claim(original_claim.id, paraphrased_text, original_claim.evidence,
#                                     original_claim.cited_docs, original_claim.release)
#             paraphrased_prediction = self.prediction_model.predict(paraphrased_claim)
#             print(paraphrased_prediction)
#             paraphrased_claims.append(ParaphrasedClaim(iteration, original_claim, paraphrased_claim, original_prediction, paraphrased_prediction))
#         return paraphrased_claims

from enum import Enum

class MultiNLILabel(Enum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2
    
class EntailmentPredictionModel(ABC):
    @abstractmethod
    def predict(self, org_claim, gen_claim) -> bool:
        pass

class TorchEntailmentPredictionModel(EntailmentPredictionModel):
    def __init__(self, model_path: str, model_name: str, device: str):
        self.model = torch.hub.load(model_path, model_name).to(device)
        self.model.eval()

    def predict(self, org_claim, gen_claim) -> bool:
        tokens_sentences_org_gen = self.model.encode(org_claim, gen_claim)
        logprobs_sentences_org_gen = self.model.predict('mnli', tokens_sentences_org_gen)
        cal_val_mlnli_org_gen = logprobs_sentences_org_gen.argmax(dim=1).item()

        tokens_sentences_gen_org = self.model.encode(gen_claim, org_claim)
        logprobs_sentences_gen_org = self.model.predict('mnli', tokens_sentences_gen_org)
        cal_val_mlnli_gen_org = logprobs_sentences_gen_org.argmax(dim=1).item()

        # The raw values are kept here for potential future debugging
        # print(cal_val_mlnli_org_gen, cal_val_mlnli_gen_org)

        # We check if both directions entail each other
        return cal_val_mlnli_org_gen == MultiNLILabel.ENTAILMENT.value and cal_val_mlnli_gen_org == MultiNLILabel.ENTAILMENT.value
    
    @staticmethod
    def _get_label(label_value: int) -> str:
        labels_multi_nli = {0: 'contradiction', 1 : 'neutral', 2 : 'entailment'}
        return labels_multi_nli[label_value]    

from dataclasses import dataclass
from typing import List

@dataclass
class ParaphrasedAttack:
    paraphrase_model: Paraphraser
    prediction_model: ModelPredictor
    entailment_checker: EntailmentPredictionModel
    list_ners: List[NEREntity] = None
    # list_abbrs: List[Abbreviation] = None

    def attack(self, iteration: int, original_claim: Claim, original_prediction: ClaimPredictions, predict_if_pass_filter: bool=True):
        paraphrased_texts = self.paraphrase_model.paraphrase(original_claim.claim)
        paraphrased_claims = []
        for paraphrased_text in paraphrased_texts:
            paraphrased_claim = Claim(original_claim.id, paraphrased_text, original_claim.evidence,
                                      original_claim.cited_docs, original_claim.release)
            
            is_ners_preserved = self.filter_and_replace_tech_term_paraphrased_claim(paraphrased_claim.claim, self.list_ners[original_claim.id])
            # is_abbrs_preserved = self.check_abbr_preservation(paraphrased_claim, self.list_abbrs)
            nli_label = self.entailment_checker.predict(original_claim.claim, paraphrased_claim.claim)
            
            paraphrased_prediction = None
            if predict_if_pass_filter: 
                if (is_ners_preserved and nli_label):
                    paraphrased_prediction = self.prediction_model.predict(paraphrased_claim)
            else:
                paraphrased_prediction = self.prediction_model.predict(paraphrased_claim)
                
            paraphrased_claims.append(ParaphrasedClaim(iteration, original_claim, paraphrased_claim,
                                                    original_prediction, paraphrased_prediction, 
                                                    is_ners_preserved, #is_abbrs_preserved,
                                                    nli_label))                
                
        return paraphrased_claims

    @staticmethod
    def filter_and_replace_tech_term_paraphrased_claim(claim_paraphrased: str, original_entities: List[Any]) -> bool:
        for entity in original_entities:
            if entity.__class__.__name__ == 'NEREntity':
                term = entity.ner_text
            elif entity.__class__.__name__ == 'Abbreviation':
                term = entity.abbr
            else:
                raise ValueError(f"Unsupported entity type: {entity.__class__.__name__}")

            term_formatted = r'\b' + re.escape(term) + r'\b'
            if not re.search(term_formatted, claim_paraphrased, re.IGNORECASE):
                return False
        return True

    @staticmethod
    def check_abbr_preservation(paraphrased_claim: str, original_abbrs: List[Abbreviation]) -> bool:
        for abbr in original_abbrs:
            if abbr.abbr not in paraphrased_claim.claim:
                return False
        return True




@dataclass
class ParaphrasedAttackResult:
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
        

            
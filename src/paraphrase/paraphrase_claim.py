from dataclasses import dataclass
from typing import List
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union
import re

import json
import pickle
from abc import ABC, abstractmethod
import torch

from T5ParEvo.src.data.data import Claim, ClaimPredictions
from T5ParEvo.src.paraphrase.paraphraser import Paraphraser
from T5ParEvo.src.models.predict_model import ModelPredictor
from T5ParEvo.src.linguistic.ner_abbr import NEREntity, Abbreviation
from T5ParEvo.src.util.logger import Logger

from enum import Enum


class MultiNLILabel(Enum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


class ClaimState(Enum):
    SUPPORT_MAJORITY = "Support Majority"
    REFUTE_MAJORITY = "Refute Majority"
    NOT_ENOUGH_INFO = "Not Enough Information"
    TIE = "Majority Tie"
    EMPTY = "Empty Prediction Result"


class AttackStatus(Enum):
    SUCCESSFUL = "Successful Attack"
    UNSUCCESSFUL = "Unsuccessful Attack"


# class TrainingDirection(Enum):
#     SUPPORT_MAJORITY = "support_majority" # Support Majority to Refute Majority
#     REFUTE_MAJORITY = "refute_majority" # Refute Majority to Support Majority


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
    original_claim_state: ClaimState = ClaimState.EMPTY  # Initialize with Empty
    paraphrased_claim_state: ClaimState = ClaimState.EMPTY  # Initialize with Empty

    attack_result: 'ParaphrasedAttackResult' = None

    def set_claim_state(self, original_claim_state: ClaimState, paraphrased_claim_state: ClaimState):
        self.original_claim_state = original_claim_state
        self.paraphrased_claim_state = paraphrased_claim_state

    def get_difference(self):
        # TODO
        # calculating the difference in prediction results.
        return self.original_prediction.predictions != self.paraphrased_prediction.predictions


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
        logprobs_sentences_org_gen = self.model.predict(
            'mnli', tokens_sentences_org_gen)
        cal_val_mlnli_org_gen = logprobs_sentences_org_gen.argmax(dim=1).item()

        tokens_sentences_gen_org = self.model.encode(gen_claim, org_claim)
        logprobs_sentences_gen_org = self.model.predict(
            'mnli', tokens_sentences_gen_org)
        cal_val_mlnli_gen_org = logprobs_sentences_gen_org.argmax(dim=1).item()

        # The raw values are kept here for potential future debugging
        # print(cal_val_mlnli_org_gen, cal_val_mlnli_gen_org)

        # We check if both directions entail each other
        return cal_val_mlnli_org_gen == MultiNLILabel.ENTAILMENT.value and cal_val_mlnli_gen_org == MultiNLILabel.ENTAILMENT.value

    @staticmethod
    def _get_label(label_value: int) -> str:
        labels_multi_nli = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        return labels_multi_nli[label_value]


@dataclass
class ParaphrasedAttack:
    paraphrase_model: Paraphraser
    prediction_model: ModelPredictor
    entailment_checker: EntailmentPredictionModel
    list_ners: List[NEREntity] = None
    # list_abbrs: List[Abbreviation] = None

    def attack(self, iteration: int, original_claim: Claim, original_prediction: ClaimPredictions, predict_if_pass_filter: bool = True):
        paraphrased_texts = self.paraphrase_model.paraphrase(
            original_claim.claim)
        paraphrased_claims = []
        for paraphrased_text in paraphrased_texts:
            paraphrased_claim = Claim(original_claim.id, paraphrased_text, original_claim.evidence,
                                      original_claim.cited_docs, original_claim.release)

            ner_list = self.list_ners.get(original_claim.id, [])
            is_ners_preserved = self.filter_and_replace_tech_term_paraphrased_claim(
                paraphrased_claim.claim, ner_list)
            # is_abbrs_preserved = self.check_abbr_preservation(paraphrased_claim, self.list_abbrs)
            nli_label = self.entailment_checker.predict(
                original_claim.claim, paraphrased_claim.claim)

            paraphrased_prediction = None
            # No need to predict if the paraphrased claim does not pass the fknfilter
            if predict_if_pass_filter:
                if (is_ners_preserved and nli_label):
                    paraphrased_prediction = self.prediction_model.predict(
                        paraphrased_claim)
            else:
                paraphrased_prediction = self.prediction_model.predict(
                    paraphrased_claim)

            paraphrased_claims.append(ParaphrasedClaim(iteration, original_claim, paraphrased_claim,
                                                       original_prediction, paraphrased_prediction,
                                                       is_ners_preserved,  # is_abbrs_preserved,
                                                       nli_label))

        return paraphrased_claims

    @staticmethod
    def filter_and_replace_tech_term_paraphrased_claim(claim_paraphrased: str, original_entities: List[NEREntity], logger: Logger = None) -> bool:
        
        if not original_entities:
            return True
        for entity in original_entities:
            if entity.__class__.__name__ == 'NEREntity':
                term = entity.ner_text
            elif entity.__class__.__name__ == 'Abbreviation':
                term = entity.abbr
            else:
                raise ValueError(
                    f"Unsupported entity type: {entity.__class__.__name__}")

            term_formatted = r'\b' + re.escape(term) + r'\b'
            if not re.search(term_formatted, claim_paraphrased, re.IGNORECASE):
                if logger:
                    logger.log("FAILED SCIENTIFIC TERM :: Claim : ",
                               claim_paraphrased)
                    logger.log("FAILED SCIENTIFIC TERM :: Claim : ", term)
                return False
        return True

    @staticmethod
    def check_abbr_preservation(paraphrased_claim: str, original_abbrs: List[Abbreviation]) -> bool:
        for abbr in original_abbrs:
            if abbr.abbr not in paraphrased_claim.claim:
                return False
        return True


    def calculate_and_set_claim_states(self, attack: ParaphrasedClaim):
        if attack.original_claim_state == ClaimState.EMPTY:
            original_claim_state = self._get_claim_state(attack.original_prediction)
            attack.original_claim_state = original_claim_state
                
        if attack.paraphrased_prediction is not None:
            paraphrased_claim_state = self._get_claim_state(attack.paraphrased_prediction)
            attack.paraphrased_claim_state = paraphrased_claim_state


    @staticmethod
    def _get_claim_state(prediction: ClaimPredictions) -> ClaimState:
        majority_count = ClaimPredictions.get_count_support_refute_nei(prediction)
        if majority_count['count_support'] > majority_count['count_refute']:
            return ClaimState.SUPPORT_MAJORITY
        elif majority_count['count_support'] < majority_count['count_refute']:
            return ClaimState.REFUTE_MAJORITY
        elif majority_count['count_support'] == majority_count['count_refute']:
            return ClaimState.TIE
        elif majority_count['count_support'] == 0 and majority_count['count_refute'] == 0 and majority_count['count_not_enough_info'] == 0:
            return ClaimState.EMPTY
        elif majority_count['count_support'] == 0 and majority_count['count_refute'] == 0:
            return ClaimState.NOT_ENOUGH_INFO
        else:
            raise ValueError("Unexpected majority count values")



@dataclass
class ParaphrasedAttackResult:
    attack: ParaphrasedClaim
    # Initialize with Unsuccessful
    training_direction: ClaimState = ClaimState.EMPTY
    attack_status: AttackStatus = AttackStatus.UNSUCCESSFUL

    def determine_attack_status(self):
        #Apply all your goddamn rules and filer here
        is_success = (self.attack.is_ners_preserved == 
                      True) and (self.attack.nli_label == 
                                 True) and ((self.attack.original_claim_state ==
                                             ClaimState.SUPPORT_MAJORITY and self.attack.paraphrased_claim_state ==
                                             ClaimState.REFUTE_MAJORITY) or (self.attack.original_claim_state == 
                                                                             ClaimState.REFUTE_MAJORITY and self.attack.paraphrased_claim_state == 
                                                                             ClaimState.SUPPORT_MAJORITY))

        if is_success:
            self.attack_status = AttackStatus.SUCCESSFUL
        else:
            self.attack_status = AttackStatus.UNSUCCESSFUL

    def print_summary(self):
        self.determine_attack_status()
        print(f"Attack Status: {self.attack_status}")

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def save_attacks_to_pickle(attacks: List['ParaphrasedAttackResult'], filename):
        with open(filename, 'wb') as f:
            pickle.dump(attacks, f)

    @staticmethod
    def load_attacks_from_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_attacks_to_json(attacks: List['ParaphrasedAttackResult'], filename):
        with open(filename, 'w') as f:
            json.dump([attack.to_dict() for attack in attacks], f)

    @staticmethod
    def load_attacks_from_json(filename):
        with open(filename, 'r') as f:
            attacks_dict_list = json.load(f)
        return [ParaphrasedAttackResult(ParaphrasedClaim(**attack_dict['attack']),
                                        AttackStatus[attack_dict['attack_status']]) for attack_dict in attacks_dict_list]

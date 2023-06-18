from dataclasses import dataclass
import json
from typing import List
from src.data.data import Claim, ClaimPredictions
import pickle

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
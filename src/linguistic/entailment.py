import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from ..data.data import Claim, ParaphrasedClaim
# from conf import EntailmentModel


@dataclass(frozen=True)
class NliLabels(Enum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2
    
    def __str__(self):
        return self.name

    @property
    def description(self):
        descriptions = {
            "CONTRADICTION": "The sentences have opposing meanings.",
            "NEUTRAL": "The sentences are not related in any specific way.",
            "ENTAILMENT": "The sentences have the same meaning or one implies the other."
        }
        return descriptions[self.name]
    
    @classmethod
    def from_string(cls, label_name):
        return cls[label_name.upper()]
    
@dataclass 
class EntailmentModel:
    model_repo : str = 'pytorch/fairseq'
    model_name : str = 'roberta.large.mnli'    

@dataclass
class EntailmentChecker:
    model: torch.nn.Module = field(init=False)
    model_config: EntailmentModel = field(default=EntailmentModel())
    label_mapping: Dict[int, NliLabels] = field(default_factory=lambda: {0: NliLabels.CONTRADICTION, 1: NliLabels.NEUTRAL, 2: NliLabels.ENTAILMENT})
    device: str = field(default='cuda' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        return torch.hub.load(self.model_config.model_repo, self.model_config.model_name)

    def check_entailment_by_paraphrased_claim(self, paraphrased_claim: ParaphrasedClaim) -> None:
        labels_org_gen = self._get_labels(paraphrased_claim.original_claim.claim, paraphrased_claim.paraphrased_claim.claim)
        labels_gen_org = self._get_labels(paraphrased_claim.paraphrased_claim.claim, paraphrased_claim.original_claim.claim)

        paraphrased_claim.nli_label = labels_org_gen[1] if labels_org_gen[1] == labels_gen_org[1] else None

    def check_entailment_by_claims(self, original_claim: Claim, paraphrased_claim: Claim) -> 'ParaphrasedClaim':
        labels_org_gen = self._get_labels(original_claim.claim, paraphrased_claim.claim)
        labels_gen_org = self._get_labels(paraphrased_claim.claim, original_claim.claim)
        paraphrased_claim.nli_label = labels_org_gen[1] if labels_org_gen[1] == labels_gen_org[1] else None
        return paraphrased_claim


    def _get_labels(self, sentence1: str, sentence2: str) -> List[Union[int, NliLabels]]:
        tokens_sentences = self.model.encode(sentence1, sentence2)
        logprobs_sentences = self.model.predict('mnli', tokens_sentences)
        cal_val_mlnli = logprobs_sentences.argmax(dim=1).item()
        cal_label_mlnli = self.label_mapping[cal_val_mlnli]

        return [cal_val_mlnli, cal_label_mlnli]

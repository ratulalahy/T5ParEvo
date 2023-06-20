import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from ..data.data import Claim, ParaphrasedClaim
from conf import EntailmentModel


@dataclass(frozen=True)
class NliLabels(Enum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2
    

@dataclass
class EntailmentChecker:
    model: torch.nn.Module = field(init=False)
    model_config: EntailmentModel
    label_mapping: Dict[int, NliLabels]
    device: str

    def __post_init__(self):
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        return torch.hub.load(self.model_config.model_repo, self.model_config.model_name)

    def check_entailment(self, paraphrased_claim: ParaphrasedClaim) -> None:
        labels_org_gen = self._get_labels(paraphrased_claim.original_claim.claim, paraphrased_claim.paraphrased_claim.claim)
        labels_gen_org = self._get_labels(paraphrased_claim.paraphrased_claim.claim, paraphrased_claim.original_claim.claim)

        paraphrased_claim.nli_label = labels_org_gen[1] if labels_org_gen[1] == labels_gen_org[1] else None

    def _get_labels(self, sentence1: str, sentence2: str) -> List[Union[int, NliLabels]]:
        tokens_sentences = self.model.encode(sentence1, sentence2)
        logprobs_sentences = self.model.predict('mnli', tokens_sentences)
        cal_val_mlnli = logprobs_sentences.argmax(dim=1).item()
        cal_label_mlnli = self.label_mapping[cal_val_mlnli]

        return [cal_val_mlnli, cal_label_mlnli]

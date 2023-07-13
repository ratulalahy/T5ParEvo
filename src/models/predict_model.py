from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PredictionParams:
    checkpoint_path: str
    # claim: Claim
    # corpus_file: str
    # output_file: str
    batch_size: int = 1
    device: int = 0
    num_workers: int = 4
    no_nei: bool = False
    force_rationale: bool = False
    debug: bool = False
    corpus_file : str= None
    output_file: str = None


class ModelPredictor(ABC):
    def __init__(self, params: PredictionParams):
        self.params = params
        self.model = self._setup_model()
        self.hparams = self._get_hparams()
        
    @abstractmethod
    def _setup_model(self):
        pass

    @abstractmethod
    def _get_hparams(self):
        pass

    @abstractmethod
    def get_predictions(self):
        pass

    @abstractmethod
    def run(self):
        pass

import os
import sys
from tqdm import tqdm
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

# from model import MultiVerSModel
# from multivers.model_r import MultiVerSModel
# from multivers.model_r import MultiVerSModel
# from data import get_dataloader
# from T5ParEvo.src.data.data import DataLoaderGenerator, get_dataloader
# import util
# from multivers import util
# sys.path.append(os.path.abspath('/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers'))
# sys.path.append(os.path.abspath('/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo'))
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from config import SciFactT5Config
# import definitions
from ...src.data.data import Claim, ClaimPredictions
from multivers import util
from multivers.data_r import DataLoaderGenerator, get_dataloader
from multivers.model_r import MultiVerSModel
# from ..src.data.data import Claim, ClaimPredictions


@dataclass
class PredictionParams:
    checkpoint_path: str
    # claim: Claim
    # corpus_file: str
    output_file: str
    batch_size: int = 1
    device: int = 0
    num_workers: int = 4
    no_nei: bool = False
    force_rationale: bool = False
    debug: bool = False

class ModelPredictor:
    def __init__(self, params: PredictionParams, data_loader_generator: DataLoaderGenerator, claim: Claim):
        self.params = params
        self.model = self._setup_model()
        self.hparams = self._get_hparams()
        self.dataloader = data_loader_generator
        self.claim = claim

    def _setup_model(self) -> MultiVerSModel:
        model = MultiVerSModel.load_from_checkpoint(checkpoint_path=self.params.checkpoint_path)
        if self.params.no_nei:
            model.label_threshold = 0.0
        model.to(f"cuda:{self.params.device}")
        model.eval()
        model.freeze()
        return model

    def _get_hparams(self) -> Any:
        hparams = self.model.hparams["hparams"]
        del hparams.precision  
        for k, v in vars(self.params).items():
            if hasattr(hparams, k):
                setattr(hparams, k, v)
        return hparams

    def get_predictions(self) -> List:
        predictions_all = []
        for batch in tqdm(self.dataloader):
            preds_batch = self.model.predict(batch, self.params.force_rationale)
            predictions_all.extend(preds_batch)
        return predictions_all

    @staticmethod
    def format_predictions(params: PredictionParams, predictions_all: List) -> List[Dict[str, Any]]:
        claims = util.load_jsonl(params.input_file)
        claim_ids = [x["id"] for x in claims]
        assert len(claim_ids) == len(set(claim_ids))
        formatted = {claim: {} for claim in claim_ids}
        for prediction in predictions_all:
            if prediction["predicted_label"] == "NEI":
                continue
            formatted_entry = {
                prediction["abstract_id"]: {
                    "label": prediction["predicted_label"],
                    "sentences": prediction["predicted_rationale"],
                }
            }
            formatted[prediction["claim_id"]].update(formatted_entry)
        res = []
        for k, v in formatted.items():
            to_append = {"id": k, "evidence": v}
            res.append(to_append)
        return res
    @staticmethod
    def format_prediction_by_claim(params: PredictionParams, predictions_all: List, claim_id: int) -> List[Dict[str, Any]]:
        # claims = util.load_jsonl(params.input_file)
        # claim_ids = [x["id"] for x in claims]
        # assert len(claim_ids) == len(set(claim_ids))
        # formatted = {claim: {} for claim in claim_ids}
        formatted = {claim_id: {}}
        for prediction in predictions_all:
            if prediction["predicted_label"] == "NEI":
                continue
            formatted_entry = {
                prediction["abstract_id"]: {
                    "label": prediction["predicted_label"],
                    "sentences": prediction["predicted_rationale"],
                }
            }
            formatted[prediction["claim_id"]].update(formatted_entry)
        res = []
        for k, v in formatted.items():
            to_append = {"id": k, "evidence": v}
            res.append(to_append)
        return res
    
    def run(self) -> None:
        if self.params.output_file is not None:
            outname = Path(self.params.output_file)
        
        predictions = self.get_predictions()
        formatted = self.format_prediction_by_claim(self.params, predictions, self.claim.id)
        return formatted
        # util.write_jsonl(formatted, outname)
        

from verisci.evaluate.lib.data import GoldDataset
def main():

    params = PredictionParams(
        
        checkpoint_path="models/target_model/multivers/scifact.ckpt",
        output_file= None,#"prediction/pred_opt_scifact.jsonl",
        batch_size=1,
        device=0,
        num_workers=4,
        no_nei=False,
        force_rationale=False,
        debug=False,
    )
    corpus_file = "data/scifact_raw/corpus.jsonl"
    # gold_ds = GoldDataset(corpus_file=corpus_file, data_file = 'data/scifact/claims_test_retrived.jsonl')
        
    
    claim= Claim(id = 1093, claim =  "Somatic missense mutations in NT5C2 are not associated with relapse of acute lymphoblastic leukemia.", 
                 evidence = {}, cited_docs= [641786, 6421792, 9478135, 27306942, 667451, 38745690, 28614776, 1982286, 8385277, 3462075], 
                 release = None)

    dataloader_generator = DataLoaderGenerator(params, claim, corpus_file)
    dataloader = dataloader_generator.get_dataloader_by_single_claim()
    predictor = ModelPredictor(params, dataloader, claim)
    prediction_formatted = predictor.run()
    
    prediction = prediction_formatted[0]  # assuming there's only one prediction
    claim_predictions = ClaimPredictions.from_formatted_prediction(prediction, claim)
    claim_predictions.pretty_print_simple()
    print(claim_predictions)

if __name__ == "__main__":

    main()
    
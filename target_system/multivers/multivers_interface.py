import os
import sys
from tqdm import tqdm
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List
import json

#
# module_path = os.path.abspath(os.path.join('...'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from config import SciFactT5Config
import definitions
sys.path.append(os.path.dirname(definitions.PROJECT_VARS.ROOT_DIR))
print(definitions.PROJECT_VARS.ROOT_DIR)
#
# from ...data.data import Claim, ClaimPredictions
from T5ParEvo.src.data.data import Claim, ClaimPredictions
from multivers import util
from multivers.data_r import ClaimDataLoaderGenerator, get_dataloader, DataLoaderGenerator
from multivers.model_r import MultiVerSModel
from T5ParEvo.src.models.predict_model import ModelPredictor, PredictionParams





class ModelPredictorMultivers(ModelPredictor):
    def __init__(self, params: PredictionParams, claim: Claim):
        super().__init__(params)
        # self.params = params
        # self.model = self._setup_model()
        # self.hparams = self._get_hparams()
        self.claim = claim
        self.corpus_file = params.corpus_file
        self.dataloader = None

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
    def format_prediction_claims(params: PredictionParams, predictions_all: List) -> List[Dict[str, Any]]:
        # claims = util.load_jsonl(params.input_file)
        claim_ids = [x["claim_id"] for x in predictions_all]
        # assert len(claim_ids) == len(set(claim_ids))
        formatted = {claim: {} for claim in claim_ids}
        # formatted = {}
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
        formatted = self.format_prediction_claims(self.params, predictions)
        return formatted
        # util.write_jsonl(formatted, outname)


    def predict(self) -> Any: #ClaimPredictions?
        dataloader_generator = ClaimDataLoaderGenerator(self.params, self.claim, self.corpus_file)
        self.dataloader = dataloader_generator.get_dataloader_by_single_claim()
        prediction_formatted = self.run()
        prediction = prediction_formatted[0]  # assuming there's only one prediction
        claim_predictions = ClaimPredictions.from_formatted_prediction(prediction, self.claim)
        return claim_predictions



class ModelPredictorMultiversList(ModelPredictor):
    def __init__(self, params: PredictionParams, claims: List[Claim]):
        super().__init__(params)
        self.claims = claims
        self.corpus_file = params.corpus_file
        self.dataloader = None

    def predict(self) -> List[ClaimPredictions]:
        dataloader_generator = DataLoaderGenerator(self.params, self.claims, self.corpus_file)
        self.dataloader = dataloader_generator.get_dataloader_by_claims()
        prediction_formatted = self.run()
        claim_predictions_list = []
        for prediction in prediction_formatted:  # assuming there's only one prediction per claim
            claim = next(claim for claim in self.claims if claim.id == prediction['id'])
            claim_predictions = ClaimPredictions.from_formatted_prediction(prediction, claim)
            claim_predictions_list.append(claim_predictions)
        return claim_predictions_list


from verisci.evaluate.lib.data import GoldDataset
def main_single_claim():
    params = PredictionParams(
        checkpoint_path='/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/models/target_model/multivers/scifact.ckpt',
        batch_size=1,
        device=0,
        num_workers=4,
        no_nei=False,
        force_rationale=False,
        debug=False,
        corpus_file='/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/target_system/multivers/data/corpus.jsonl'
    )

    claim = Claim(id=1093,
                  claim="Somatic missense mutations in NT5C2 are not associated with relapse of acute lymphoblastic leukemia.",
                  evidence={}, 
                  cited_docs=[641786, 6421792, 9478135, 27306942, 667451, 38745690, 28614776, 1982286, 8385277, 3462075],
                  release=None)

    predictor = ModelPredictorMultivers(params, claim)
    claim_predictions = predictor.predict()
    claim_predictions.pretty_print_simple()
    print(claim_predictions)


def main_single_list():
    params = PredictionParams(
        checkpoint_path="/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/checkpoints/scifact.ckpt",
        batch_size=2,
        device=0,
        num_workers=4,
        no_nei=False,
        force_rationale=False,
        debug=False,
        corpus_file="/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/data/scifact/corpus.jsonl"
    )

    claims_path = '/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/data/scifact/claims_test_retrived.jsonl'
    claims = []
    with open(claims_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            claim = Claim(id=data['id'], claim=data['claim'], cited_docs=data['doc_ids'], evidence={}, release=None)
            claims.append(claim)

    predictor = ModelPredictorMultiversList(params, claims[:10])
    claim_predictions_list = predictor.predict()
    for claim_predictions in claim_predictions_list:
        claim_predictions.pretty_print_simple()
        print(claim_predictions)


if __name__ == "__main__":
    main_single_claim()
    # main_single_list()

"""
Data-handling code copied over from `verisci/evaluate/lib/data.py` of the VeriSci
library from the original SciFact release: https://github.com/allenai/scifact.
For attack handling other classes and functions are implemented in this file.
"""

from enum import Enum
import json
import copy
from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Tuple, Union, Optional, Any
from config import AttackResult
from collections import OrderedDict
import re

# from T5ParEvo.src.linguistic.ner_abbr import Abbreviation, NEREntity
# from ..linguistic.ner_abbr import Abbreviation, NEREntity

####################

# Utility functions and enums.


def load_jsonl(fname):
    return [json.loads(line) for line in open(fname)]


class Label(Enum):
    SUPPORTS = 2
    NEI = 1
    REFUTES = 0


def make_label(label_str, allow_NEI=True):
    lookup = {
        "SUPPORT": Label.SUPPORTS,
        "NOT_ENOUGH_INFO": Label.NEI,
        "CONTRADICT": Label.REFUTES,
    }

    res = lookup[label_str]
    if (not allow_NEI) and (res is Label.NEI):
        raise ValueError("An NEI was given.")

    return res


####################

# Representations for the corpus and abstracts.


@dataclass(repr=False, frozen=True)
class Document:
    id: str
    title: str
    sentences: Tuple[str]

    def __repr__(self):
        return (
            self.title.upper()
            + "\n"
            + "\n".join(["- " + entry for entry in self.sentences])
        )

    def __lt__(self, other):
        return self.title.__lt__(other.title)

    def dump(self):
        res = {
            "doc_id": self.id,
            "title": self.title,
            "abstract": self.sentences,
            "structured": self.is_structured(),
        }
        return json.dumps(res)


@dataclass(repr=False, frozen=True)
class Corpus:
    """
    A Corpus is just a collection of `Document` objects, with methods to look up
    a single document.
    """

    documents: List[Document]

    def __repr__(self):
        return f"Corpus of {len(self.documents)} documents."

    def __getitem__(self, i):
        "Get document by index in list."
        return self.documents[i]

    def get_document(self, doc_id):
        "Get document by ID."
        res = [x for x in self.documents if x.id == doc_id]
        assert len(res) == 1
        return res[0]

    @classmethod
    def from_jsonl(cls, corpus_file):
        corpus = load_jsonl(corpus_file)
        documents = []
        for entry in corpus:
            doc = Document(entry["doc_id"], entry["title"], entry["abstract"])
            documents.append(doc)

        return cls(documents)


####################

# Gold dataset.


class GoldDataset:
    """
    Class to represent a gold dataset, include corpus and claims.
    """

    def __init__(self, corpus_file, data_file):
        self.corpus = Corpus.from_jsonl(corpus_file)
        self.claims = self._read_claims(data_file)

    def __repr__(self):
        msg = f"{self.corpus.__repr__()} {len(self.claims)} claims."
        return msg

    def __getitem__(self, i):
        return self.claims[i]

    def _read_claims(self, data_file):
        "Read claims from file."
        examples = load_jsonl(data_file)
        res = []
        for this_example in examples:
            entry = copy.deepcopy(this_example)
            entry["release"] = self
            entry["cited_docs"] = [
                self.corpus.get_document(doc) for doc in entry["doc_ids"]
            ]
            assert len(entry["cited_docs"]) == len(entry["doc_ids"])
            del entry["doc_ids"]
            res.append(Claim(**entry))

        res = sorted(res, key=lambda x: x.id)
        return res

    def get_claim(self, example_id):
        "Get a single claim by ID."
        keep = [x for x in self.claims if x.id == example_id]
        assert len(keep) == 1
        return keep[0]


@dataclass
class EvidenceAbstract:
    "A single evidence abstract."
    id: int
    label: Label
    rationales: List[List[int]]


@dataclass(repr=False)
class Claim:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """

    id: int
    claim: str
    evidence: Dict[int, EvidenceAbstract] 
    cited_docs: List[Document]
    release: GoldDataset 
    # abbreviations : List[Abbreviation] = None
    # ner_entities : List[NEREntity] = None

    def __post_init__(self):
        self.evidence = self._format_evidence(self.evidence)

    @staticmethod
    def _format_evidence(evidence_dict):
        # This function is needed because the data schema is designed so that
        # each rationale can have its own support label. But, in the dataset,
        # all rationales for a given claim / abstract pair all have the same
        # label. So, we store the label at the "abstract level" rather than the
        # "rationale level".
        res = {}
        for doc_id, rationales in evidence_dict.items():
            doc_id = int(doc_id)
            labels = [x["label"] for x in rationales]
            if len(set(labels)) > 1:
                msg = (
                    "In this SciFact release, each claim / abstract pair "
                    "should only have one label."
                )
                raise Exception(msg)
            label = make_label(labels[0])
            rationale_sents = [x["sentences"] for x in rationales]
            this_abstract = EvidenceAbstract(doc_id, label, rationale_sents)
            res[doc_id] = this_abstract

        return res

    def __repr__(self):
        msg = f"Example {self.id}: {self.claim}"
        return msg

    def pretty_print(self, evidence_doc_id=None, file=None):
        "Pretty-print the claim, together with all evidence."
        msg = self.__repr__()
        print(msg, file=file)
        # Print the evidence
        print("\nEvidence sets:", file=file)
        for doc_id, evidence in self.evidence.items():
            # If asked for a specific evidence doc, only show that one.
            if evidence_doc_id is not None and doc_id != evidence_doc_id:
                continue
            print("\n" + 20 * "#" + "\n", file=file)
            ev_doc = self.release.corpus.get_document(doc_id)
            print(f"{doc_id}: {evidence.label.name}", file=file)
            for i, sents in enumerate(evidence.rationales):
                print(f"Set {i}:", file=file)
                kept = [sent for i, sent in enumerate(ev_doc.sentences) if i in sents]
                for entry in kept:
                    print(f"\t- {entry}", file=file)

    @classmethod
    def get_claim_by_id(cls, claims: List['Claim'], claim_id: int) -> 'Claim':
        for claim in claims:
            if claim.id == claim_id:
                return claim
        return None
    
    @classmethod
    def get_claim_by_text(cls, claims: List['Claim'], claim_text: str) -> 'Claim':
        for claim in claims:
            if claim.claim == claim_text:
                return claim
        return None
    
    @classmethod
    def get_unique_claims(cls, claims: List['Claim'])-> 'Claim':
        unique_claims = set()
        unique_claim_objects = []
        for claim in claims:
            if claim.claim not in unique_claims:
                unique_claims.add(claim.claim)
                unique_claim_objects.append(claim)
        return unique_claim_objects
    
    @classmethod
    def load_claims_from_file(cls, file_path: str) -> List['Claim']:
        claims = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line, object_pairs_hook=OrderedDict)
                claim = cls(
                    id=data['id'],
                    claim=data['claim'],
                    cited_docs=data['doc_ids'],
                    evidence={},
                    release=None
                )
                claims.append(claim)
        return claims    
####################

# Predicted dataset.


class PredictedDataset:
    """
    Class to handle predictions, with a pointer back to the gold data.
    """

    def __init__(self, gold, prediction_file):
        """
        Takes a GoldDataset, as well as files with rationale and label
        predictions.
        """
        self.gold = gold
        self.predictions = self._read_predictions(prediction_file)

    def __getitem__(self, i):
        return self.predictions[i]

    def __repr__(self):
        msg = f"Predictions for {len(self.predictions)} claims."
        return msg

    def _read_predictions(self, prediction_file):
        res = []

        predictions = load_jsonl(prediction_file)
        for pred in predictions:
            prediction = self._parse_prediction(pred)
            res.append(prediction)

        return res

    def _parse_prediction(self, pred_dict):
        claim_id = pred_dict["id"]
        predicted_evidence = pred_dict["evidence"]

        res = {}

        # Predictions should never be NEI; there should only be predictions for
        # the abstracts that contain evidence.
        for key, this_prediction in predicted_evidence.items():
            label = this_prediction["label"]
            evidence = this_prediction["sentences"]
            pred = PredictedAbstract(
                int(key), make_label(label, allow_NEI=False), evidence
            )
            res[int(key)] = pred

        gold_claim = self.gold.get_claim(claim_id)
        return ClaimPredictions(claim_id, res, gold_claim)


@dataclass
class PredictedAbstract:
    # For predictions, we have a single list of rationale sentences instead of a
    # list of separate rationales (see paper for details).
    abstract_id: int
    label: Label
    rationale: List


@dataclass
class ClaimPredictions:
    claim_id: int
    predictions: Dict[int, PredictedAbstract]
    gold: Claim = None  # For backward compatibility, default this to None.

    def __repr__(self):
        msg = f"Predictions for {self.claim_id}: {self.gold.claim}"
        return msg

    def pretty_print(self, evidence_doc_id=None, file=None):
        if self.gold is None or self.gold.release is None:
            print("No gold data available for this claim.")
            return
        msg = self.__repr__()
        print(msg, file=file)
        # Print the evidence
        print("\nEvidence sets:", file=file)
        for doc_id, prediction in self.predictions.items():
            # If asked for a specific evidence doc, only show that one.
            if evidence_doc_id is not None and doc_id != evidence_doc_id:
                continue
            print("\n" + 20 * "#" + "\n", file=file)
            ev_doc = self.gold.release.corpus.get_document(doc_id)
            print(f"{doc_id}: {prediction.label.name}", file=file)
            # Print the predicted rationale.
            sents = prediction.rationale
            kept = [sent for i, sent in enumerate(ev_doc.sentences) if i in sents]
            for entry in kept:
                print(f"\t- {entry}", file=file)
                
    def pretty_print_simple(self):
        print(f"Claim ID: {self.claim_id}")
        print(f"Gold Claim: {self.gold}\n")
        print("Predictions:")
        for abstract_id, predicted_abstract in self.predictions.items():
            print(f"\nAbstract ID: {abstract_id}")
            print(f"Label: {predicted_abstract.label.name}")
            print("Rationale Sentences:")
            for rationale in predicted_abstract.rationale:
                print(f"- {rationale}")                
                
    @classmethod
    def save_predictions(cls, predictions_list, filename):
        """
        Save a list of ClaimPredictions to a pickle file.

        :param predictions_list: A list of ClaimPredictions objects.
        :param filename: The path to the output file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(predictions_list, f)
                
    @classmethod
    def from_formatted_prediction(cls, prediction_formatted, gold_claim=None):
        """
        Construct a ClaimPredictions object from formatted prediction data.
        For the help of T5ParEvo
        :param prediction_formatted: Formatted prediction data.
        :param gold_claim: Gold Claim object. (Default: None)
        :return: ClaimPredictions object.
        """
        claim_id = prediction_formatted['id']
        predictions = {}
        for abstract_id, evidence in prediction_formatted['evidence'].items():
            label = make_label(evidence['label'])  # Use make_label function here
            rationale = evidence['sentences']
            predicted_abstract = PredictedAbstract(abstract_id, label, rationale)
            predictions[abstract_id] = predicted_abstract
        return cls(claim_id, predictions, gold_claim)        
    
    @classmethod
    def get_count_support_refute(cls, claim_prediction) -> Tuple[int, int, int]:
        """_summary_

        Args:
            claim_prediction (ClaimPredictions): _description_

        Returns:
            Tuple[int, int, int]: _description_
        """
        count_support = 0
        count_refute = 0
        count_nei = 0
        for cur_pred_key in claim_prediction.predictions.keys():
            pred_label = claim_prediction.predictions[cur_pred_key].label
            if pred_label == Label.SUPPORTS:
                count_support += 1
            elif pred_label == Label.REFUTES:
                count_refute += 1
            elif pred_label == Label.NEI:
                count_nei += 1
        return count_support, count_refute, count_nei    
    

# from ..linguistic.ner_abbr import Abbreviation, NEREntity
# # from ..linguistic.entailment import NliLabels
# # from conf import AttackReesult
# @dataclass
# class ParaphrasedClaim:
#     paraphrased_claim: Claim = None
#     original_claim: Claim = None
#     paraphrased_prediction: ClaimPredictions = None
#     original_prediction: ClaimPredictions = None
#     original_claim_ners: List[NEREntity] = None
#     original_claim_abbrs: List[Abbreviation] = None
#     nli_label: 'NliLabels' = None
#     attack_result: AttackResult = None


#     def filter_and_replace_tech_term_paraphrased_claim(claim_paraphrased: str, original_entities: List[Any]) -> bool:
#         """
#         Function to check if a paraphrased sentence preserves all the technical terms or scientific terms that were in the original claim.
        
#         Args:
#         claim_paraphrased (str): The paraphrased claim text.
#         original_entities (List[Union[NEREntity, Abbreviation]]): The list of NER or Abbreviation instances. # Changed for this time being

#         Returns:
#         bool: Returns True if all terms are preserved, False otherwise.
#         """
#         for entity in original_entities:
#             # Using regular expression to check for whole word match in the paraphrased claim
#             if entity.__class__.__name__ == 'NEREntity':
#                 term = entity.ner_text
#             elif entity.__class__.__name__ == 'Abbreviation':
#                 term = entity.abbr
#             else:
#                 raise ValueError(f"Unsupported entity type: {entity.__class__.__name__}")
            
#             # formatted search term for regex
#             term_formatted = r'\b' + re.escape(term) + r'\b'
            
#             if not re.search(term_formatted, claim_paraphrased, re.IGNORECASE):
#                 return False

#         return True

#     def get_difference(self):
#         # calculating the difference in prediction results.
#         raise NotImplementedError()


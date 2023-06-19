from tqdm import tqdm
import spacy
from scispacy.abbreviation import AbbreviationDetector
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
# from T5ParEvo.src.data.data import Claim, Label
from ..data.data import Claim, Label

@dataclass
class Abbreviation:
    claim_id : int
    abrv_text: str
    abr_definition: str
    abr_model: str
    sf_start_char: int
    sf_end_char: int
    lf_start_char: int
    lf_end_char: int
    abr_whole_start: int = -1
    abr_whole_end: int = -1
    
    #get abbreviation by claim id
    @classmethod
    def get_abbr_by_claim_id(cls, claim_id, abbr_list):
        return [abbr for abbr in abbr_list if abbr.claim_id == claim_id]
    
@dataclass
class NEREntity:
    claim_id : int
    ner_text: str
    ner_label: str
    ner_model: str
    start_char: int
    end_char: int    
    
    @classmethod
    def get_ner_by_claim_id(cls, claim_id, ner_list):
        return [ner for ner in ner_list if ner.claim_id == claim_id]


class AbbreviationModel:
    def __init__(self, model_name: str):
        spacy.prefer_gpu()
        self.model = spacy.load(model_name)
        self.model.add_pipe("abbreviation_detector")
        self.abbreviations = set()

    def process_claim(self, claim: Claim) -> List[Abbreviation]:
        doc = self.model(claim.claim)
        abbrs = []
        for abr in doc._.abbreviations:
            self.abbreviations.add(str(abr))
            abbr = Abbreviation(
                claim_id=claim.id,
                abrv_text=str(abr),
                abr_definition=str(abr._.long_form),
                abr_model=self.model.meta['name'],
                sf_start_char=abr.start_char,
                sf_end_char=abr.end_char,
                lf_start_char=abr._.long_form.start_char,
                lf_end_char=abr._.long_form.end_char,
            )
            if (abr.end_char - abr.start_char > 0) & (abr._.long_form.end_char - abr._.long_form.start_char > 0):
                if (abr.start_char - abr._.long_form.end_char < 4) | (abr._.long_form.start_char - abr.end_char < 4):
                    abbr.abr_whole_start = min(abr.start_char, abr._.long_form.start_char)
                    abbr.abr_whole_end = max(abr.end_char, abr._.long_form.end_char)
            abbrs.append(abbr)
        return abbrs
    

    @classmethod
    def merge_overlapping_abbreviations(cls, abbreviations: List[Abbreviation]) -> List[Abbreviation]:
        """_summary_

        Args:
            abbreviations (List[Abbreviation]): _description_

        Returns:
            List[Abbreviation]: _description_
        """
        # Sort abbreviations by start index
        abbreviations.sort(key=lambda x: x.sf_start_char)

        merged_abbreviations = []
        current_abbr = abbreviations[0]

        for abbr in abbreviations[1:]:
            # If this abbreviation overlaps with the current abbreviation, merge them
            if abbr.sf_start_char <= current_abbr.sf_end_char:
                current_abbr = Abbreviation(
                    claim_id=current_abbr.claim_id,
                    abrv_text=current_abbr.abrv_text + abbr.abrv_text[current_abbr.sf_end_char - abbr.sf_start_char:],
                    abr_definition=current_abbr.abr_definition,  # Keep the definition of the first abbreviation
                    abr_model=current_abbr.abr_model,  # Keep the model of the first abbreviation
                    sf_start_char=current_abbr.sf_start_char,
                    sf_end_char=max(current_abbr.sf_end_char, abbr.sf_end_char),
                    lf_start_char=current_abbr.lf_start_char,
                    lf_end_char=max(current_abbr.lf_end_char, abbr.lf_end_char),
                    abr_whole_start=current_abbr.abr_whole_start,
                    abr_whole_end=max(current_abbr.abr_whole_end, abbr.abr_whole_end)
                )
            else:
                # This abbreviation doesn't overlap, so add the current abbreviation to the list and start a new one
                merged_abbreviations.append(current_abbr)
                current_abbr = abbr

        # Don't forget to add the last abbreviation
        merged_abbreviations.append(current_abbr)

        return merged_abbreviations    

class NERModel:
    def __init__(self, model_name: str):
        spacy.prefer_gpu()
        self.model = spacy.load(model_name)
        self.entities = set()

    def process_claim(self, claim: Claim) -> List[NEREntity]:
        """_summary_

        Args:
            claim (Claim): _description_

        Returns:
            List[NEREntity]: _description_
        """
        doc = self.model(claim.claim)
        entities = []
        for ent in doc.ents:
            self.entities.add(str(ent))
            entity = NEREntity(
                claim_id=claim.id,
                ner_text=str(ent),
                ner_label=ent.label_,
                ner_model=self.model.meta['name'],
                start_char=ent.start_char,
                end_char=ent.end_char
            )
            entities.append(entity)
        return entities
    
    @classmethod
    def merge_overlapping_entities(cls, entities: List[NEREntity]) -> List[NEREntity]:
        """
        Merges overlapping entities into a single entity. The merged entity will have the label and model of the first
        """
        # Sort entities by start index
        entities.sort(key=lambda x: x.start_char)

        merged_entities = []
        current_entity = entities[0]

        for entity in entities[1:]:
            # If this entity overlaps with the current entity, merge them
            if entity.start_char <= current_entity.end_char:
                current_entity = NEREntity(
                    claim_id=current_entity.claim_id,
                    ner_text=current_entity.ner_text + entity.ner_text[current_entity.end_char - entity.start_char:],
                    ner_label=current_entity.ner_label,  # Keep the label of the first entity
                    ner_model=current_entity.ner_model,  # Keep the model of the first entity
                    start_char=current_entity.start_char,
                    end_char=max(current_entity.end_char, entity.end_char)
                )
            else:
                # This entity doesn't overlap, so add the current entity to the list and start a new one
                merged_entities.append(current_entity)
                current_entity = entity

        # Don't forget to add the last entity
        merged_entities.append(current_entity)

        return merged_entities    
    
class ClaimProcessor:
    """_summary_
    """
    def __init__(self, ner_models: List[NERModel], abbr_models: List[AbbreviationModel]):
        self.ner_models = ner_models
        self.abbr_models = abbr_models

    def process_claim(self, claim: Claim) -> Tuple[List[NEREntity], List[Abbreviation]]:
        ner_entities = []
        abbreviations = []
        for model in self.ner_models:
            ner_entities.extend(model.process_claim(claim))
        for model in self.abbr_models:
            abbreviations.extend(model.process_claim(claim))
        return ner_entities, abbreviations
        
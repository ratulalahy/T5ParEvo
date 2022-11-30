import abc
import os
from dataclasses import dataclass

from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import MISSING


@dataclass 
class TargetDataset: 
    """
    Abstract dataclass for target dataset to attack or to train/test/validate with. 
    This class can not be instantiated. 
    Inherit this class to show the location of target dataset location.
        
    Args:
        ABC (_type_): Default Abstract class
    """
    base_dir:str =  MISSING

@dataclass
class ScifactTargetDataset(TargetDataset):
    """
    _summary_

    Args:
        LocTargetDataset (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    base_dir:str = f'{os.getcwd()}/target_system/scifact/data'
    loc_target_dataset_corpus: str  = f'{base_dir}/corpus.jsonl'
    loc_target_dataset_train: str  = f'{base_dir}/claims_train.jsonl'
    loc_target_dataset_dev: str  = f'{base_dir}/claims_dev.jsonl'


@dataclass 
class TargetModel(abc.ABC): 
    """
    Parent dataclass for the location of target mode (the model to attack).
    Inherit this class to show the location of target model.
    Args:
        ABC (_type_): Default Abstract class
    """
    base_dir: str  = MISSING


@dataclass
class ScifactTargetModel(TargetModel):
    """
    _summary_

    Args:
        LocTargetDataset (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    base_dir: str =f'{os.getcwd()}/target_system/model'
    loc_label_model: str = f'{base_dir}/label_roberta_large_fever_scifact'
    loc_rationale_model: str = f'{base_dir}/rationale_roberta_large_fever_scifact'
    
    

@dataclass
class ParaphrasingModel:
    model_name: str # keeping it for the sake of experiment tracking!
    tokenizer_name: str
    model_url_or_path: str

@dataclass    
class T5TunedParaphrasingModel(ParaphrasingModel):
    model_name: str = 'finetuned_paws_abstracts'
    tokenizer_name: str = 'Vamsi/T5_Paraphrase_Paws'
    model_url_or_path: str = f'{os.getcwd()}/models/paraphraser/t5_paws_masked_claim_abstract_paws_3_epoch_2/model_3_epochs/'
    
@dataclass
class T5GenParams:
    max_length:int = 512
    do_sample: bool= True
    top_k: int=  50
    top_p: float= 0.99
    repetition_penalty:float = 3.5
    early_stopping:bool = True
    num_return_sequences:int = 5    


    
    
@dataclass
class SciFactT5Config:
    target_dataset: TargetDataset = ScifactTargetDataset()
    target_model: TargetModel = ScifactTargetModel()
    paraphrasing_model: ParaphrasingModel = T5TunedParaphrasingModel()
    t5_generation_param: T5GenParams = T5GenParams()



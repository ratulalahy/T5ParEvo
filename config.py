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
class TargetModel: 
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
class SciFactT5Config:
    target_dataset: TargetDataset = ScifactTargetDataset()
    target_model: TargetModel = ScifactTargetModel()

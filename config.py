from abc import ABC
from dataclasses import dataclass

# @dataclass 
# class LocTargetDataset(ABC): 
#     """
#     Abstract dataclass for target dataset to attack or to train/test/validate with. 
#     This class can not be instantiated. 
#     Inherit this class to show the location of target dataset location.
        
#     Args:
#         ABC (_type_): Default Abstract class
#     """
#     #loc_target_dataset: str
#     def __new__(cls, *args, **kwargs): 
#         if cls == LocTargetDataset or cls.__bases__[0] == LocTargetDataset: 
#             raise TypeError("Cannot instantiate abstract dataclass class.") 
#         return super().__new__(cls)

@dataclass
class ScifactLocTargetDataset:
    """
    _summary_

    Args:
        LocTargetDataset (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    #loc_target_dataset: str
    loc_target_dataset_corpus: str
    loc_target_dataset_train: str
    loc_target_dataset_dev: str


@dataclass 
class LocTargetModel(ABC): 
    """
    Abstract dataclass for the location of target mode (the model to attack).
    this abstract class can not be instantiated directly.  
    Inherit this class to show the location of target model.
    Args:
        ABC (_type_): Default Abstract class
    """
    loc_target_model: str
    def __new__(cls, *args, **kwargs): 
        if cls == LocTargetModel or cls.__bases__[0] == LocTargetModel: 
            raise TypeError("Cannot instantiate abstract dataclass class.") 
        return super().__new__(cls)

@dataclass
class SciFactT5Config:
    loc_target_dataset: ScifactLocTargetDataset

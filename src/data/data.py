from enum import Enum
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from typing import Optional
from abc import ABC, abstractmethod

class Label_enm(Enum):
    REFUTE = 0
    SUPPORT = 1
    NOT_ENOUGH_INFO = 2  
    
class Label: 
    label: Label_enm
    
    @staticmethod    
    def get_enum_rep_label(res_str : str): 
        if res_str == None or res_str == '':
            return Label_enm.NOT_ENOUGH_INFO
        elif res_str == 'SUPPORTS':
            return Label_enm.SUPPORT
        elif res_str == 'REFUTES':
            return Label_enm.REFUTE
        else:
            raise ValueError("Please Provide proper label.")
            

@dataclass
class Claim:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """
#    id: str
    claim_text: str
    
@dataclass
class Rationale:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """
#    id: str
    rationale_text: str    
    
@dataclass
class ClaimRationale:
    id: str
    claim: Claim
    rationales : List[Dict[Rationale, Label]]
#    source: Optional[str] = ""
    

    
@dataclass
class ClaimRationalePredicted:
    claim_original: Claim
    claim_paraphrased: str
    rationale_paraphrased : Dict[Rationale, Label]



from dataclasses import dataclass
from typing import List, Dict, Union

@dataclass
class ParaphraseSettings:
    file_and_dirs: Dict[str, str]
    config_scifact: Dict[str, str]
    paraphrase_model: Dict[str, Union[List[Dict[str, Union[str, bool]]], Dict[str, Union[int, bool, float]]]]
    entailment_model: Dict[str, str]
    labels_multi_nli: Dict[int, str]
    run_settings: Dict[str, Union[float, str]]

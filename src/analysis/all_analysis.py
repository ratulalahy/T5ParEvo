

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Analysis:
    readability_scores: Dict[str, float]
    entailment_score: Optional[float] = None
    grammatical_correctness_score: Optional[float] = None

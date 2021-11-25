from dataclasses import dataclass


@dataclass(frozen=True)
class Hyperparameters:
    discount: float  # gamma in assignment text
    forgetting: float  # lambda in assignment text
    eligibility_range: int
    learning_rate: float

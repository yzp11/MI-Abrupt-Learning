import torch as t
from torch import Tensor
import torch.nn.functional as F

def generate_sample(
    GMMClass: tuple[float[Tensor, 'd_D'], int],
    d_D: int,
    variability: float,
) -> tuple[ float[Tensor, 'd_D'], float[Tensor, 'd_D'] ]:
    generator, label_index = GMMClass

    item = (generator + variability * t.randn(d_D) * (1.0 / d_D)**0.5) / (1+variability**2)**0.5
    label = F.one_hot(t.arange(label_index, label_index+1), num_classes= d_D)

    return item, label


class GMMSequence:
    def __init__(
        self,
        GMMClass_list: list[ tuple[float[Tensor, 'd_D'], int] ],
        d_P: int,
        d_D: int,
        N: int,
        variability: float,
        device: str,
    ):
        self.GMMClass_list = GMMClass_list
        self.d_P = d_P
        self.d_D = d_D
        self.N = N
        self.variability = variability
        self.device = device

    
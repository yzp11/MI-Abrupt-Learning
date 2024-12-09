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
        variability: float,
    ):
        self.GMMClass_list = GMMClass_list
        self.len = 2 * len(GMMClass_list) - 1
        self.d_P = d_P
        self.d_D = d_D
        self.variability = variability

        assert self.d_P >= self.len

    def prompt(self):
        p_start = t.randint(self.d_P - self.len + 1, size= ()).item()
        position_encoding = F.one_hot(t.arange(p_start, p_start + self.len), num_classes= self.d_P)

        prompt = t.zeros(size= (0, self.d_D) )
        for n in range( len(self.GMMClass_list) ):
            item, label = generate_sample(self.GMMClass_list[n], self.d_D, self.variability)
            prompt = t.cat( (prompt, item, label), dim= 0)

        prompt = prompt[:-1, :]

        return t.cat((position_encoding, prompt), dim= -1)

    def target(self):
        _, label = generate_sample(self.GMMClass_list[-1], self.d_D, self.variability)
        return label
    


class GMMDataset:
    def __init__(
        self,
        d_P: int,
        d_D: int,
        N: int,
        variability: float,
        K: int,
        L: int,
        B: int,
        p_B: float,
        alpha: float,
    ):
        self.d_P = d_P
        self.d_D = d_D
        self.N = N
        self.variability = variability
        self.K = K
        self.L = L
        self.B = B
        self.p_B = p_B
        self.alpha = alpha

        self.All_GMMClass = []
        self.All_label = t.randint(L, size=(K,)).tolist()
        for k in range(K):
            pass


    def generate_train(self, batch_size):

        pass


    def generate_IWL(self, batch_size):

        pass


    def generate_novel_ICL(self, batch_size):

        pass


    def generate_swapped_ICL(self, batch_size):

        pass




        

    
import sys
from typing import Any
import numpy as np
from numpy.random import randint
import torch
from torch.nn import Module

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["print_flush", "choice_not_n"]

def print_flush(text: str) -> None:
    print(text)
    sys.stdout.flush()

def choice_not_n(mn: int,
                 mx: int,
                 notn: int) -> int:
    c: int = randint(mn, mx)
    while c == notn:
        c = randint(mn, mx)
    return c

#def sigmoid(x: float) -> float:
#    return 1 / (1 + np.exp(-x))

def torch_models_eq(m1: Module,
                    m2: Module) -> bool:
    for (k1, i1), (k2, i2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if not k1 == k2 or not torch.equal(i1, i2):
            return False
    return True

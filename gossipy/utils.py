import sys
import tarfile
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from numpy.random import randint
import torch
from torch.nn import Module

from . import LOG

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["print_flush", "choice_not_n", "torch_models_eq", "download_and_unzip"]

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


def download_and_unzip(url: str, extract_to: str='.') -> str:
    LOG.info("Downloading %s into %s" %(url, extract_to))
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    return zipfile.namelist()[0]

def download_and_untar(url: str, extract_to: str='.') -> str:
    LOG.info("Downloading %s into %s" %(url, extract_to))
    ftpstream = urlopen(url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=extract_to)
    return thetarfile.getnames()[0]
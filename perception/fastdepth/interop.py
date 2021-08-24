import torch
import numpy as np
from os import listdir
from os.path import isfile, join, dirname

from torch._C import dtype

def isModel(dirname:str, x:str) -> bool:
    suffixTest = (x[-8:] == ".pth.tar")
    fileTest = isfile(join(dirname, x))

    return suffixTest and fileTest

def loadModel() -> torch.nn.Module:
    curdir = dirname(__file__)
    files = [f for f in listdir(curdir) if isModel(curdir, f)]
    assert len(files) == 1

    checkpoint = torch.load(files[0])
    model = checkpoint['model']
    model.eval()
    return model

def pred(model:torch.nn.Module, x:np.ndarray) -> np.ndarray:
    with torch.no_grad():
        lx = torch.Tensor(x).cuda()
        result = model(lx).cpu().detach().numpy()
    
    return result

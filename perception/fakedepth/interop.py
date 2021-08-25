import torch
import numpy as np
from os import listdir
from os.path import isfile, join, dirname
import cv2 as cv

from torch._C import dtype

def loadModel() -> tuple(torch.nn.Module, torch.nn.Module):
    INTEL_MIDAS = "intel-isl/MiDaS"
    MIDAS_MODEL = "MiDaS_small"

    model = torch.hub.load(INTEL_MIDAS, MIDAS_MODEL).cuda().eval()
    transform = torch.hub.load(INTEL_MIDAS, "transforms").small_transform.cuda().eval()

    return model, transform

def pred(model:torch.nn.Module, transform:torch.nn.Module, image:np.ndarray) -> np.ndarray:
    with torch.no_grad():
        embeddedImage = transform(image).cuda()
        rawDepth = model(embeddedImage).cuda()
        depth = torch.nn.functional.interpolate(
            rawDepth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return depth.cpu().numpy()

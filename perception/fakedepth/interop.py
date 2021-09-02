import torch
from torch2trt import torch2trt
import numpy as np

def loadModel(inputSize:torch.Size) -> tuple(torch.nn.Module):
    INTEL_MIDAS = "intel-isl/MiDaS"
    MIDAS_MODEL = "DPT_HYBRID"

    hubmodel = torch.hub.load(INTEL_MIDAS, MIDAS_MODEL).eval().cuda()
    hubtransform = torch.hub.load(INTEL_MIDAS, "transforms").dpt_transform.eval().cuda()

    xtran = torch.ones(inputSize).cuda()
    xmod = torch.ones_like(hubtransform(xtran)).cuda()
    transform = torch2trt(hubtransform, [xtran])
    model = torch2trt(hubmodel, [xmod])

    return model, transform

def pred(model:torch.nn.Module, transform:torch.nn.Module, image:torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        embeddedImage = transform(image).cuda()
        rawDepth = model(embeddedImage).cuda()
        depth = torch.nn.functional.interpolate(
            rawDepth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return depth

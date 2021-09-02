import torch
import numpy as np

def loadModel(inputSize:torch.Size) -> tuple([torch.nn.Module, torch.nn.Module]):
    INTEL_MIDAS = "intel-isl/MiDaS"
    MIDAS_MODEL = "MiDaS"

    model = torch.hub.load(INTEL_MIDAS, MIDAS_MODEL).eval().cuda()
    transform = torch.hub.load(INTEL_MIDAS, "transforms").small_transform

    return model, transform

def pred(model:torch.nn.Module, transform:torch.nn.Module, image:np.array) -> torch.Tensor:
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

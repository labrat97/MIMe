import torch

def loadModel() -> tuple([torch.nn.Module, torch.nn.Module]):
    INTEL_MIDAS = "intel-isl/MiDaS"
    MIDAS_MODEL = "MiDaS_small"

    # Download from torchhub
    model = torch.hub.load(INTEL_MIDAS, MIDAS_MODEL).cuda().eval()
    transform = torch.hub.load(INTEL_MIDAS, "transforms").small_transform

    return model, transform

def pred(model:torch.nn.Module, transform:torch.nn.Module, image, resize:bool=True) -> torch.Tensor:
    # Convert to numpy array
    if isinstance(image, torch.Tensor):
        image = image.cpu().squeeze()
        if len(image.shape) > 3:
            result = [pred(model, transform, image[idx]).unsqueeze(0) for idx in range(image.size(0))]
            return torch.cat(result)
        image = image.numpy()

    # Predict
    with torch.no_grad():
        embeddedImage = transform(image).cuda().detach()
        rawDepth = model(embeddedImage).cuda().detach()
        if not resize: return rawDepth.squeeze()
        depth = torch.nn.functional.interpolate(
            rawDepth.unsqueeze(1),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    
    return depth

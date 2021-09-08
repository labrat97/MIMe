import torch


class PipelineChunk(object):
    def __init__(self, name:str):
        super(PipelineChunk, self).__init__()

        # Make this chunk searchable in a list
        self.name = name

        # Hold the result of the computation for later pipelining
        self.result = None

    def read(self):
        return self.result

    def compute(self, x):
        self.result = x
        return x

def __quickToCuda(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cuda()
    return torch.tensor(x, requires_grad=False).cuda()


import vpiinterop
class DenseFlowPipe(PipelineChunk):
    def __init__(self):
        super(DenseFlowPipe, self).__init__("DenseFlow")

    def compute(self, x, quality:str='high', format:str='bgr'):
        super(DenseFlowPipe, self).compute(x)

        # Force 'x' into GPU/Torch memory
        inter = __quickToCuda(x)
        if len(inter.shape) == 5:
            interCurr = inter[:,0,:,:,:]
            interPrev = inter[:,1,:,:,:]
        if len(inter.shape) == 4:
            interCurr = inter[0,:,:,:].unsqueeze(0)
            interPrev = inter[1,:,:,:].unsqueeze(0)

        self.result = self.__compute(interCurr, interPrev, quality=quality, format=format)
        return self.result

    def compute(self, x, y, quality:str='high', format:str='bgr'):
        super(DenseFlowPipe, self).compute(x)
        
        # Force 'x' and 'y' into GPU/Torch memory
        interCurr = __quickToCuda(x)
        interPrev = __quickToCuda(y)
        if len(interCurr.shape) == 3:
            interCurr.unsqueeze_(0)
        if len(interPrev.shape) == 3:
            interPrev.unsqueeze_(0)

        self.result = self.__compute(interCurr, interPrev, quality=quality, format=format)
        return self.result

    def __compute(self, curr:torch.Tensor, prev:torch.Tensor, quality:str, format:str) -> torch.Tensor:
        return vpiinterop.denseFlow(prev, curr, format=format, quality=quality)

import fastdepth
class FastDepthPipe(PipelineChunk):
    def __init__(self):
        super(FastDepthPipe, self).__init__("FastDepth")
    
    def compute(self, x):
        # TODO: This
        return super(FastDepthPipe, self).compute(x)


import time
import syscamera
class RetinaPipe(PipelineChunk):
    def __init__(self, retina:syscamera.Retina):
        super(RetinaPipe, self).__init__(f'RetinaPipe-{retina.camID}')
        
        # Save retina for later use
        self.retina = retina
    
    def compute(self, x, undistOptimal:bool=True, undistBoth:bool=True):
        super(RetinaPipe, self).compute(x)

        # TODO: I don't know if a tuple like packing can handle appending like this
        self.result = self.retina.read(undist=True, undistOptimal=undistOptimal, \
            undistBoth=undistBoth)
        if x is not None:
            self.result.append(time.time() - x)

        return self.result

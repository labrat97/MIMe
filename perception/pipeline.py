import torch


class PipelineChunk(object):
    def __init__(self, name:str, f_x:function):
        super(PipelineChunk, self).__init__()

        # Make this chunk searchable in a list
        self.name = name

        # Hold the functionality
        self.f_x = f_x

        # Hold the result of the computation for later pipelining
        self.result = None

    def read(self):
        return self.result

    def compute(self, x):
        self.result = self.f_x(x)
        return self.result

def __quickToCuda(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cuda()
    return torch.tensor(x, requires_grad=False).cuda()


import vpiinterop
class DenseFlowPipe(PipelineChunk):
    def __init__(self, f_x:function):
        super(DenseFlowPipe, self).__init__("DenseFlow", f_x)

    def compute(self, x, quality:str='high', format:str='bgr'):
        # Force 'x' into GPU/Torch memory
        inter = __quickToCuda(x)
        if len(inter.shape) == 5:
            interCurr = inter[:,0,:,:,:]
            interPrev = inter[:,1,:,:,:]
        if len(inter.shape) == 4:
            interCurr = inter[0,:,:,:].unsqueeze(0)
            interPrev = inter[1,:,:,:].unsqueeze(0)

        return self.__compute(interCurr, interPrev, quality=quality, format=format)

    def compute(self, x, y, quality:str='high', format:str='bgr'):
        # Force 'x' and 'y' into GPU/Torch memory
        interCurr = __quickToCuda(x)
        interPrev = __quickToCuda(y)
        if len(interCurr.shape) == 3:
            interCurr.unsqueeze_(0)
        if len(interPrev.shape) == 3:
            interPrev.unsqueeze_(0)

        return self.__compute(interCurr, interPrev, quality=quality, format=format)

    def __compute(self, curr:torch.Tensor, prev:torch.Tensor, quality:str, format:str) -> torch.Tensor:
        return vpiinterop.denseFlow(prev, curr, format=format, quality=quality)

import fastdepth
import syscamera


from posixpath import dirname
DIRNAME = dirname(__file__)
MODEL_NAME_BASE = 'fastdepth'
ACCEL_TARGET_MIDSUFFIX = '-dla'
MODEL_NAME_SUFFIX = '.onnx'
COMPILED_NAME_SUFFIX = MODEL_NAME_SUFFIX+'.trt'

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

import torch



class DepthEngine():
    def __init__(self, batchSize:int=1, useDLA:int=None):
        super(DepthEngine, self).__init__()

        # This is going to be a *little* messy for right now
        # ...and by that, I mean that it's a little hard-coded
        self.usingDLA = useDLA is not None
        self.fname = MODEL_NAME_BASE + str(batchSize)
        if self.usingDLA: 
            self.fname += ACCEL_TARGET_MIDSUFFIX + str(useDLA)
        self.fname += COMPILED_NAME_SUFFIX
        # Not even doing input prechecks because of what the potential file open
        #   errors are to turn out to be

        # Open the compiled model for import
        f = open(self.fname, 'rb')

        # Bind the model into this class
        self.LOGGER = TRT_LOGGER
        self.runtime = trt.Runtime(self.LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(torch.cuda.current_device())
        self.batchSize = batchSize

        # Close the template file
        f.close()

    def pred(self, x:torch.Tensor, y:torch.Tensor=None) -> torch.Tensor:
        # Precheck
        assert x.shape == (self.batchSize, 3, 224, 224)
        assert x.is_cuda
        assert x.is_contiguous()
        assert x.dtype == torch.float16
        _x = x.detach()
        
        # The models are compiled in float16 due to a hardware limitation/acceleration
        if y is None:
            _y = torch.zeros((self.batchSize, 224, 224), dtype=torch.float16, requires_grad=False).cuda()
        else:
            assert y.shape == (self.batchSize, 224, 224)
            assert y.is_cuda
            assert y.is_contiguous()
            assert y.dtype == torch.float16
            _y = y.detach()
        
        # 'x' being the input, 'y' being the output
        bindings = [int(_x.data_ptr()), int(_y.data_ptr())]

        # Compute the depth map for the provided frame in torch
        # Shouldn't need any synchronization as the whole thing is a cuda 
        self.context.execute_async_v2(bindings, self.stream.handle)
        
        return y

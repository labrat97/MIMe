import os
from posixpath import dirname
DIRNAME = dirname(__file__)
MODEL_NAME_BASE = 'fastdepth'
MODEL_NAME_SUFFIX = '.onnx'

import pycuda.autoinit
import pycuda.driver as cuda

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def buildEngine(defaultBatch:int = 2, modelPath:str = None):
    # Init trt constructors for engine based inference
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Set up build environment
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30) >> 2 # 256 MB, we aint rich here

    # Find default model if needed
    if modelPath is None:
        modelPath = DIRNAME + os.path.sep + \
            MODEL_NAME_BASE + str(defaultBatch) + MODEL_NAME_SUFFIX

    # Load and parse the onnx model
    with open(modelPath, 'rb') as modelfile:
        if not parser.parse(modelfile.read()):
            print('There was a problem importing the fastdepth ONNX model.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build into a returned engine
    return builder.build_engine(network, config)

def createBindings(engine):
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Using managed allocation enables unified memory according to:
        # https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
        bindmem = cuda.managed_empty(size, dtype)

        bindings.append(int(bindmem))
    
    return bindings

def getTorchStream():
    import torch
    torch.cuda.current_stream(torch.cuda.current_device())

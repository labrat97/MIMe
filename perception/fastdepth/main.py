import os
from posixpath import dirname
DIRNAME = dirname(__file__)
MODEL_PATH = DIRNAME + os.path.sep + 'fastdepth.onnx'

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def buildEngine(modelFile:str = MODEL_PATH):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30) >> 2 # 256 MB

    with open(modelFile, 'rb') as modelfile:
        if not parser.parse(modelfile.read()):
            print('There was a problem importing the fastdepth ONNX model.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    return builder.build_engine(network, config)

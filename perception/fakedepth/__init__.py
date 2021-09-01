import os
os.system("git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && python setup.py install --plugins")


from .interop import *

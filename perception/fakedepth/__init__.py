import os
os.system("wget https://github.com/NVIDIA-AI-IOT/torch2trt/archive/refs/tags/v0.3.0.tar.gz && \
    tar -xzf  v0.3.0.tar.gz && \
    cd torch2trt-0.3.0 && python3 setup.py install")


from .interop import *

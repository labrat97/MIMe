#!/bin/bash

# Enables torch to handle pybind and the NVVPI functionality
export TORCH_USE_RTLD_GLOBAL=YES

# Downloads torch2trt in order to get the intel midas model running at a decent rate
wget https://github.com/NVIDIA-AI-IOT/torch2trt/archive/refs/tags/v0.3.0.tar.gz
tar -xzf v0.3.0.tar.gz
cd torch2trt-0.3.0
python3 setup.py install
cd ..
rm v0.3.0.tar.gz

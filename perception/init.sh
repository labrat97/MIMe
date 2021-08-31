#!/bin/bash
(cd vpiinterop/build && \
    cmake -DTorch_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch/ . .. && \
    make)

export TORCH_USE_RTLD_GLOBAL=YES

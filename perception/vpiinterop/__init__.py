import os

# I had to do this type of JiT because CUDA is a closed source bitch
os.system("(cd vpiinterop/build && \
    cmake -Wno-dev -DTorch_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch/ . .. && \
    make)")

from .build.vpiinterop import *

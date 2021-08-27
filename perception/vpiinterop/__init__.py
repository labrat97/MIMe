from .interop import *

from torch.utils.cpp_extension import load
load("nvvpi", ["layers.cpp"], extra_include_paths='/usr/local/lib/python3.6/dist-packages/torch/include/')

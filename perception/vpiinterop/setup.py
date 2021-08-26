from setuptools import setup, Extension
from torch.utils import cpp_extension

extMeta = Extension('torchvpi', sources=['layers.cpp'], \
    include_dirs=cpp_extension.include_paths().append('/usr/local/lib/python3.6/dist-packages/torch/include/'))

setup(name='torchvpi',
    ext_modules=[extMeta],
    cmd_class={'build_ext':cpp_extension.BuildExtension})

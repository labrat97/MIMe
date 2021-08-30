#from setuptools import setup, Extension
#from torch.utils import cpp_extension

#extMeta = Extension('torchvpi', sources=['layers.cpp'], \
#    include_dirs=cpp_extension.include_paths().append('/usr/local/lib/python3.6/dist-packages/torch/include/'))


from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='nvvpi',
    ext_modules=[cpp_extension.CppExtension('nvvpi', ['denseFlow.cpp'],
    extra_compile_args=['-L/usr/local/cuda-10.2/targets/aarch64-linux/include', '-L/usr/include', '-lcuda',
        '-L/opt/nvidia/vpi1/include', '-lvpi1'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension})

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pieps", version="0.3.0", description="Photometric Stereo from Physical Irradiance Event Streams",
    python_requires=">=3.10,<3.13",
    packages=["pieps", "pieps.graph", "pieps.layers"],
    py_modules=["train", "infer", "evaluate"],
    install_requires=["numpy>=2,<3", "torch==2.4.1", "torch-geometric==2.6.1",
                      "torch-scatter==2.1.2", "torch-cluster==1.6.3", "torch-spline-conv==1.2.2",
                      "torch-sparse==0.6.18", "scipy>=1.10", "Pillow>=9", "PyYAML>=6"],
    package_data={"pieps.graph": ["*.cu", "*.h"]},
    license="MIT AND GPL-3.0-only",
    license_files=["LICENSE", "DAGR_LICENSE", "README.md"],
    ext_modules=[CUDAExtension(name="ev_graph_cuda", sources=["pieps/graph/ev_graph.cu"])],
    cmdclass={"build_ext": BuildExtension},
    entry_points={"console_scripts": ["pieps-train=train:main", "pieps-infer=infer:main", "pieps-evaluate=evaluate:main"]},
)

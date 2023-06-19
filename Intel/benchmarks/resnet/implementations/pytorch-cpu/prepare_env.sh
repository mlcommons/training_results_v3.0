#!/usr/bin/env bash

CONDA_ENV_NAME=rn50-train-mlp30

source /opt/rh/gcc-toolset-11/enable
export CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-11/root/usr/bin/c++

export CONDA_PREFIX="$(dirname $(which conda))/../"
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
export CMAKE_PREFIX_PATH=${CONDA_PREFIX}

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
export WORKDIR=${CUR_DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -rf ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

conda env remove -n ${CONDA_ENV_NAME}
conda create -n ${CONDA_ENV_NAME} python=3.8 --yes
conda activate ${CONDA_ENV_NAME}

echo "Installiing dependencies for RN50"
python -m pip install sklearn onnx dataclasses opencv-python absl-py
python -m pip install matplotlib Pillow pycocotools
conda install -c anaconda git typing_extensions future six requests dataclasses psutil --yes
conda install -c anaconda ninja pyyaml setuptools cmake cffi typing intel-openmp --yes
conda install -c intel mkl=2022.0.1 mkl-include=2022.0.1 cython --yes
conda install -c conda-forge llvm-openmp jemalloc gperftools accimage --yes
pip install "git+https://github.com/mlperf/logging.git@2.0.0"

# Build PyTorch
git clone https://github.com/pytorch/pytorch.git
export Torch_DIR=$PWD/pytorch
cd ${Torch_DIR}
git checkout c19d19f6ffa41f78f0fa377d9bcef176051965db
git apply ${CUR_DIR}/patches/dataloader.patch
git submodule sync
git submodule update --init --recursive
cd third_party/ideep/mkl-dnn/third_party
rm -r oneDNN
git clone -b dev-mlperf-v2.7 https://github.com/oneapi-src/oneDNN.git
cd ${Torch_DIR}
python -m pip install -r requirements.txt
python setup.py install
cd ${WORKDIR}

# Build torchvision
echo "Installiing torch vision"
git clone https://github.com/pytorch/vision
cd vision
git checkout 55799959046fe38e5bd324631e33238a6f8f08a7
python setup.py install
cd ${WORKDIR}

## Build torch-ccl
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl
git checkout tags/v2.0.0+cpu
git submodule sync
git submodule update --init --recursive
# Install Intel Ethernet Fabric Suite: https://www.intel.com/content/www/us/en/download/19816/intel-ethernet-fabric-suite-fs-package.html
cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so.1
cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so
cp /usr/lib64/libfabric/libpsm3-fi.so.1.19.0 third_party/oneCCL/deps/ofi/lib/prov/libpsm3-fi.so
sed -ie 's/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" FALSE)/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" TRUE)/g' third_party/oneCCL/CMakeLists.txt
python setup.py install
cd ${WORKDIR}

git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 3b1eaec6249108f7b059da38b0a2811189a839ee
git apply ${CUR_DIR}/patches/fuse_lars.patch
git submodule sync
git submodule update --init --recursive
python -m pip install -r requirements.txt
python setup.py install
#### Remove if not needed: export IPEX_PATH=${PWD}/build/Release/packages/intel_extension_for_pytorch
cd ${WORKDIR}

python -c 'import torch; print("Torch ver: " + str(torch.__version__))'
python -c 'import intel_extension_for_pytorch as ipex; print("IPEX ver: " + str(ipex.__version__))'
python -c 'import torchvision; print("Torchvision ver: " + str(torchvision.__version__))'
python -c 'import oneccl_bindings_for_pytorch; print("Torch_CCL ver: " + str(oneccl_bindings_for_pytorch.__version__))'

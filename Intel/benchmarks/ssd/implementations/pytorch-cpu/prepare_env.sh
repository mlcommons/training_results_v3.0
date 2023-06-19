#!/bin/bash

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=retinanet-train
export WORKDIR=${CUR_DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -rf ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda create -n ${CONDA_ENV_NAME} python=3.8 --yes
conda activate ${CONDA_ENV_NAME}

echo "Installing dependencies for Retinanet"
conda install -c conda-forge gperftools --yes
python -m pip install scikit-image Pillow pycocotools
python -m pip install sklearn onnx
python -m pip install dataclasses
python -m pip install opencv-python
python -m pip install absl-py matplotlib sympy

pip install "git+https://github.com/mlperf/logging.git"

conda install typing_extensions --yes
conda config --add channels intel
conda install setuptools cmake intel-openmp --yes
conda install -c intel mkl=2023.1.0 --yes
conda install -c intel mkl-include=2023.1.0 --yes
conda install future six requests dataclasses psutil pyyaml --yes

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

echo "=== Installing Pytorch ==="
git clone https://github.com/pytorch/pytorch pytorch
cd pytorch
git checkout 9a18968253e28ba8d8bdf646731087000c7876b7
git submodule sync
git submodule update --init --recursive
cd third_party/ideep/mkl-dnn/
rm -rf third_party/oneDNN
git checkout dbbce0
git submodule update --init --recursive
cd third_party/oneDNN
git checkout v2.7.4
cd ../../../../..
python setup.py install

cd ${WORKDIR}

echo "=== Installing IPEX ==="
git clone https://github.com/intel/intel-extension-for-pytorch ipex
cd ipex
git checkout c66e00ce653af209e2be56e15596e2bfe105e348
git submodule sync
git submodule update --init --recursive

echo -e '\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")' >> cmake/cpu/BuildFlags.cmake
echo -e '\ntarget_link_libraries(${PLUGIN_NAME_CPU} PUBLIC '${WORKDIR}'/pytorch/build/sleef/lib/libsleef.a)' >> csrc/cpu/CMakeLists.txt
cp ${CUR_DIR}/patches/ipex_fl/nn_functional/* intel_extension_for_pytorch/nn/functional/
cp ${CUR_DIR}/patches/ipex_fl/csrc_cpu_aten/FocalLoss.* csrc/cpu/aten/
python setup.py install

cd ${WORKDIR}

echo "=== Installing Torch CCL ==="
git clone https://github.com/intel/torch-ccl torchccl
cd torchccl
git checkout v2.0.0+cpu
git submodule sync
git submodule update --init --recursive
# Install Intel Ethernet Fabric Suite: https://www.intel.com/content/www/us/en/download/19816/intel-ethernet-fabric-suite-fs-package.html
cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so.1
cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so
cp /usr/lib64/libfabric/libpsm3-fi.so.1.19.0 third_party/oneCCL/deps/ofi/lib/prov/libpsm3-fi.so
sed -i 's/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" FALSE)/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" TRUE)/' third_party/oneCCL/CMakeLists.txt
python setup.py install

cd ${WORKDIR}

echo "=== Installing Torchvision ==="
git clone https://github.com/pytorch/vision torchvision
cd torchvision
git checkout 5b07d6c9c6c14cf88fc545415d63021456874744
git submodule sync
git submodule update --init --recursive
python setup.py install

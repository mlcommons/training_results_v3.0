#!/bin/bash

echo -e '\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")' >> cmake/cpu/BuildFlags.cmake
echo -e '\ntarget_link_libraries(${PLUGIN_NAME_CPU} PUBLIC $TORCH_PATH/build/sleef/lib/libsleef.a)' >> csrc/cpu/CMakeLists.txt
cp $RETINANET_PATH/patches/ipex_fl/nn_functional/* intel_extension_for_pytorch/nn/functional/
cp $RETINANET_PATH/patches/ipex_fl/csrc_cpu_aten/FocalLoss.* csrc/cpu/aten/

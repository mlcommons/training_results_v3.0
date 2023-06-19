# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.04-py3
FROM ${FROM_IMAGE_NAME}

# Document build setup
ARG FROM_IMAGE_NAME
ENV CUSTOM_FROM_IMAGE_NAME ${FROM_IMAGE_NAME}
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

# Custom libraries version
WORKDIR /workspace/

## 1. Apex
RUN if [ "${APEX_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/apex && \
      cd apex && \
      echo APEX_REVISION=${APEX_REVISION} && \
      git checkout ${APEX_REVISION} && \
      MAX_JOBS=${APEX_MAX_JOBS} pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./ \
    ; fi

# the following two variables are needed when building TE
ENV NVTE_WITH_USERBUFFERS=1
ENV MPI_HOME=/usr/local/mpi
ARG TE_REVISION=release_v0.8
ENV CUSTOM_TE_REVISION ${TE_REVISION}


## 2. Transformer Engine
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      pip install git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi

ARG NEMO_REVISION=v1.17.0_pt_23.04
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}
ARG NEMO_BASE_VERSION=1.17
ENV CUSTOM_NEMO_BASE_VERSION ${NEMO_BASE_VERSION}

## 3. NeMo
### Base version
RUN if [ "${NEMO_REVISION}" == SKIP ]; then \
      if [ -d /opt/bignlp/NeMo ]; then \
        ln -s /opt/bignlp/NeMo \
      ; else \
        echo "Error: NEMO_REVISION=SKIP but there is no BigNLP NeMo installation in base image." && \
        exit 1 \
      ; fi \
    ; else \
      git clone https://github.com/NVIDIA/NeMo.git && \
      cd NeMo && \
      echo NEMO_REVISION=${NEMO_REVISION} && \
      git checkout ${NEMO_REVISION} && \
      pip uninstall -y nemo-toolkit && \
      pip install -e ".[nlp]" && \
      cd nemo/collections/nlp/data/language_modeling/megatron && \
      make \
    ; fi

## 4. Benchmark dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Benchmark code
WORKDIR /workspace/llm

COPY . .
ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

# Check install
RUN python -c "import nemo.collections.nlp as nemo_nlp"

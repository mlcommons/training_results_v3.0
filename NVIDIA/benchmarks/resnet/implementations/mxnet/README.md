# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [MXNet 23.04-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) (multi-node)

# 2. Directions

## Steps to download and verify data

1. Clone the public DeepLearningExamples repository
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/MxNet/Classification/RN50v1.5
git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```

2. Build the ResNet50 MXNet NGC container
```
docker build . -t nvidia_rn50_mx
```

3. Start an interactive session in the NGC container to run preprocessing
```
nvidia-docker run --rm -it --ipc=host -v <path/to/store/raw/&/processed/data>:/data nvidia_rn50_mx
```

4. Download and unpack the data
* Download **Training images (Task 1 &amp; 2)** and **Validation images (all tasks)** at http://image-net.org/challenges/LSVRC/2012/2012-downloads (require an account)
* Extract the training data:
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
    
* Extract the validation data:
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 
    tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

5. Preprocess the dataset
```
./scripts/prepare_imagenet.sh <path/to/raw/imagenet> <path/to/save/preprocessed/data>
```

## Steps to launch training

### NVIDIA DGX H100

Launch configuration and system-specific hyperparameters for the NVIDIA DGX
H100 submission are in the `../<implementation>/config_DGXH100_<scale>.sh` script.

Steps required to launch training on NVIDIA DGX H100.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_<scale>.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
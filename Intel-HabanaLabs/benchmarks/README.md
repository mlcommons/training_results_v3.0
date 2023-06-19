# Habana MLPerf™ training 3.0 submission
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

- [Habana MLPerf™ training 3.0 submission](#habana-mlperf-training-30-submission)
- [Setup](#setup)
  - [Install firmware, driver, SynapseAI 1.10.97](#install-firmware-driver-synapseai-11097)
  - [Build and deploy HabanaLabs MLPERF training 3.0 container in the cluster for ResNet50, Bert and UNet3D.](#build-and-deploy-habanalabs-mlperf-training-30-container-in-the-cluster-for-resnet50-bert-and-unet3d)
- [Resnet50](#resnet50)
  - [Prepare Imagenet dataset](#prepare-imagenet-dataset)
  - [Run and time TensorFlow Resnet50](#run-and-time-tensorflow-resnet50)
  - [Run and time PyTorch Resnet50](#run-and-time-pytorch-resnet50)
- [Bert TF](#bert-tf)
  - [Prepare packed wiki dataset](#prepare-packed-wiki-dataset)
  - [Run and time](#run-and-time)
- [Bert PT](#bert-pt)
  - [Dataset Preparation](#dataset-preparation)
  - [Training Data Packing](#training-data-packing)
  - [Run and time](#run-and-time-1)
  - [Scaling out the training to 64 Gaudi2](#scaling-out-the-training-to-64-gaudi2)
- [GPT3-175B PT](#gpt3-175b-pt)
  - [Prepare dataset](#prepare-dataset)
  - [Prepare docker container for GPT3](#prepare-docker-container-for-gpt3)
    - [Host file and ssh connection between machines](#host-file-and-ssh-connection-between-machines)
    - [Installing requirements](#installing-requirements)
  - [Prepare checkpoint](#prepare-checkpoint)
  - [Run and time](#run-and-time-2)
    - [Run GPT3 on HLS-Gaudi2-N32-PT system](#run-gpt3-on-hls-gaudi2-n32-pt-system)
    - [Run GPT3 on HLS-Gaudi2-N48-PT system](#run-gpt3-on-hls-gaudi2-n48-pt-system)
- [UNet3D](#unet3d)
  - [Download, preprocess and verify dataset](#download-preprocess-and-verify-dataset)
  - [Run and time](#run-and-time-3)

# Setup
## Install firmware, driver, SynapseAI 1.10.97

Follow the steps in [Setup and Install](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to setup each compute node in the cluster.

## Build and deploy HabanaLabs MLPERF training 3.0 container in the cluster for ResNet50, Bert and UNet3D.

On each compute node, do the following:

1. Create directories for MLPERF, scratch and dataset folders along with Habana subfolder:
```
export MLPERF_DIR=/path/to/mlperf
export SCRATCH_DIR=/path/to/scratch
export DATASETS_DIR=/path/to/datasets
mkdir -p $MLPERF_DIR/Habana
mkdir -p $SCRATCH_DIR
mkdir -p $DATASETS_DIR
```

2. This README is located in `benchmarks` directory corresponding to Habana's submission.
Download this whole `benchmarks` folder along with all subfolders and copy it under `$MLPERF_DIR/Habana`

3. Choose gaudi-docker-mlperf/ver3.0 release container, depending on the framework to be tested.
```
# for TensorFlow:
export MLPERF_DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.0/tensorflow-installer-tf-cpu-2.12.0:1.10.97-59
```
```
# for PyTorch:
export MLPERF_DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.0/pytorch-installer-1.13.1:1.10.97-59
```

Prepare MLPERF training 3.0 container:

1. Create and start container.
```
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf3.0 -td                    \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -e LOG_LEVEL_ALL=6                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $MLPERF_DIR:/root/MLPERF             \
           -v $SCRATCH_DIR:/root/scratch           \
           -v $DATASETS_DIR:/root/datasets/        \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 $MLPERF_DOCKER_IMAGE

docker exec mlperf3.0 bash -c "service ssh start"
docker exec -it mlperf3.0 bash
```

2. In the docker, create `hosts` file that contains a list of hosts in the cluster to `/root/shared`. The example below is for single node.
```
mkdir /root/shared
echo your-machine-ip > /root/shared/hosts
```

3. Install numactl package (required for large scale Gaudi2).
```
apt update
apt install -y numactl
```

# Resnet50
## Prepare Imagenet dataset

 1. Sign up with [image-net.org](http://image-net.org/download-images) and acquire the rights to download original images
 2. Follow the link to the 2012 ILSVRC and download ILSVRC2012_img_val.tar and ILSVRC2012_img_train.tar
 3. Use the script below to unpack the dataset. Set IMAGENET_HOME to a folder where dataset should be placed

```
export IMAGENET_HOME=/path/to/imagenet
mkdir -p $IMAGENET_HOME/val
mkdir -p $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/val
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
rm $IMAGENET_HOME/train/*.tar
cd $IMAGENET_HOME/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Run and time TensorFlow Resnet50
Log into mlperf3.0 TensorFlow container.
Install additional packages required for Resnet50.
```
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
```
Execute the script
```
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-TF
./launch_keras_resnet_hvd.sh --config $(pwd)/batch_256.cfg --cpu-pin cpu --jpeg-data-dir /path/to/imagenet --log_dir /root/scratch
```
for a cluster run based on hostfile.
Use the ```$IMAGENET_HOME``` folder from [prepare imagenet section](#prepare-imagenet-dataset) for ```--jpeg-data-dir```.
Results of the run will be placed on the host, in folder specified by ```--log_dir``` parameter.

## Run and time PyTorch Resnet50
Log into mlperf3.0 PyTorch container. Install additional packages required for Resnet50:
```
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT/PyTorch/requirements.txt
```
Execute the script
```
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT
./launch_resnet.sh --config batch_256.cfg --data-dir /path/to/imagenet --log-dir /root/scratch
```
for a cluster run based on hostfile.
Use the ```$IMAGENET_HOME``` folder from [prepare imagenet section](#prepare-imagenet-dataset) for ```--data-dir```.
Results of the run will be placed on the host, in folder specified by ```--log-dir``` parameter.

# Bert TF

## Prepare packed wiki dataset

Log into mlperf3.0 TensorFlow container.

**Location to download Dataset and Checkpoint:** [Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

**Dataset Preparation:** In order to use dataset one needs to preprocess it similarly as described in [Bert dataset preparation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets).

Each of the 500 dataset files can be converted in the following way:
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pretraining/create_pretraining_data.py \
    --input_file=<path to downloaded and unzipped dataset>/part-00XXX-of-00500 \
    --output_file=<output dir for tfrecord files>/part-00XXX-of-00500 \
    --vocab_file=<path to downloaded vocab.txt> \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10
```


After tfrecord files are ready we pack them using similar code as suggested by [GraphCore for v1.0 submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data)

```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pack_pretraining_data_tfrec.py \
    --input-dir /path-to-tfrecords-dir \
    --output-dir /path-to-tfrecords-packed-dir \
    --max-files 500
```

For additional details please refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).

## Run and time

Log into mlperf3.0 TensorFlow container.

Given a runtime configuration, for instance, 8 Gaudi2 run
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-Gaudi2-TF
```
Edit defaults.cfg with the right location of your packed dataset tf records inside the container

for example, ```INPUT_FILES_DIR_PACKED=/root/datasets/bert_pretraining/packed```
execute the script ```launch_bert_hvd.sh --config defaults.cfg``` for a cluster run based on hostfile
It will place the results of the run at $LOG_DIR on the host.

# Bert PT

## Dataset Preparation

Log into mlperf3.0 PyTorch container and run:
```bash
cd /root/MLPERF/Habana/benchmarks/bert/implementations/PyTorch
pip install -r requirements.txt
export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
bash input_preprocessing/prepare_data.sh -o $PYTORCH_BERT_DATA
```

At this stage, ```$PYTORCH_BERT_DATA/phase1``` checkpoint and  ```$PYTORCH_BERT_DATA/hdf5/eval_varlength``` evaluation data are ready, while ```$PYTORCH_BERT_DATA/hdf5/training_4320/hdf5_4320_shards_uncompressed``` training data requires packing as described in the following section.

## Training Data Packing

After the training data is ready, pack them using a similar code as described in [GraphCore for v1.0 Submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data).

```bash
mkdir $PYTORCH_BERT_DATA/packed
python3 pack_pretraining_data_pytorch.py \
    --input_dir=$PYTORCH_BERT_DATA/hdf5/training-4320/hdf5_4320_shards_uncompressed \
    --output_dir=$PYTORCH_BERT_DATA/packed \
    --max_predictions_per_seq=76
```

For further details, refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027). 

## Run and time

Log into mlperf3.0 PyTorch container.

Given a runtime configuration, for instance, 8 Gaudi2 run
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-Gaudi2-PT
```
Run following code, adjusting parameters to locations of processed datasets and checkpoints.

```
bash launch_bert_pytorch.sh --data-dir <directory-with-datasets-and-checkpoint>
```
It is assumed that directory pointed by ```--data-dir``` has the following structure:

```<directory-with-datasets-and-checkpoint>/phase1/model.ckpt-28252.pt``` - checkpoint from phase1 prepared by prepare_data.sh as described in [Checkpoint and evaluation data](#checkpoint-and-evaluation-data)
```<directory-with-datasets-and-checkpoint>/hdf5/eval_varlength``` - evaluation dataset prepared by prepare_data.sh as described in [Checkpoint and evaluation data](#checkpoint-and-evaluation-data)
```<directory-with-datasets-and-checkpoint>/packed_data_500_pt``` - training dataset generated as described in [Training dataset preparation](#training-data-preparation)

By default results of the training will be placed under ```/tmp/BERT_PRETRAINING```

## Scaling out the training to 64 Gaudi2

Log into mlperf3.0 PyTorch container.

To scale the training to multiple remote machines, use `-H <...>` parameter, whose argument will be passed to the underlying `mpirun` command.
By default `launch_bert_pytorch.sh` script invokes 8 local training processes, each using 8 Gaudi2 accelerators, as if `-H localhost:8` was specified.

To perform 64 Gaudi2 training, prepare identical docker containers on 8 machines, each addressable by unique IP/hostname and each having 8 Gaudi2 accelerators.
As script uses `mpirun` to spawn local and remote processes, provide a passwordless _ssh_ communication:
```
grep -qxF 'StrictHostKeyChecking no' .ssh/config || printf 'Host *\n    StrictHostKeyChecking no' >> .ssh/config
```
It also may be necessary to setup SSH keys and add them to `~/.ssh/authorized_keys`.
Having that, the training may be invoked from any out of 8 machines:
```
bash launch_bert_pytorch.sh <...> --ssh-port <...> -H <IP1>:8,<IP2>:8,<IP3>:8,<IP4>:8,<IP5>:8,<IP6>:8,<IP7>:8,<IP8>:8
```
Parameter `-ssh-port $REMOTE_SSH_PORT` translates to:
```
mpirun <...> --mca plm_rsh_args "-p$REMOTE_SSH_PORT" <...>
```

# GPT3-175B PT
## Prepare dataset

Dataset preparation should be done in the following docker:

```
docker run --ipc=host -it -v <output folder>:/root/output_dataset nvcr.io/nvidia/pytorch:22.11-py3 bash
```
Please specify `<output folder>` to the folder, where dataset should be saved.

MLPerf GPT3 is trained using C4/en/3.0.1 dataset. It can be downloaded from https://huggingface.co/datasets/allenai/c4. Instruction is clear on how to select precisely the files for downloading.

```
apt-get update
apt-get install git-lfs
cd /root/output_dataset
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```

Out of all of the files only 256 will be needed for training and 8 for validation.
One can merge them into 3 .json.gz files using following commands. These were taken from https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md.

```
# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {6..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ../../en/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
  done
done

# merge
mkdir -p en_merge
for shard in {6..7}; do
  cat softlinks/en_${shard}/*gz > en_merge/c4-train.en_${shard}.json.gz
done
cat en/c4-validation.0000* > en_merge/c4-validation.json.gz
```



Prepared files require tokenization with SPM tokenizer. First, download tokenizer model ```vocab_c4_en_301_5Mexp2_spm.model``` and vocabulary file ```vocab_c4_en_301_5Mexp2_spm.vocab``` from https://console.cloud.google.com/storage/browser/mlperf-llm-public2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false. Registration is required to access the files. Tokenization can be performed with the use of the following commands. This conversion might take few hours:

```
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && git checkout f3ad584b94170bc3ea197df29eb9ef9c96061730 && bash ./reinstall.sh && cd ..

mkdir -p preprocessed_c4_spm
for shard in {6..7}; do
python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-train.en_${shard}.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_${shard}_c4_spm \
    --dataset-impl mmap \
    --workers 128
done

python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-validation.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_validation_mc4_spm \
    --dataset-impl mmap \
    --workers 128
```

The resulting files to be used during training are:
* ```c4/preprocessed_c4_spm/c4_en_6_c4_spm_text_document.bin```
* ```c4/preprocessed_c4_spm/c4_en_6_c4_spm_text_document.idx```
* ```c4/preprocessed_c4_spm/c4_en_7_c4_spm_text_document.bin```
* ```c4/preprocessed_c4_spm/c4_en_7_c4_spm_text_document.idx```
* ```c4/preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.bin```
* ```c4/preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.idx```

Apart from the dataset GPT3 implementation requires https://huggingface.co/gpt2/resolve/main/vocab.json and https://huggingface.co/gpt2/resolve/main/merges.txt files:

```
wget "https://huggingface.co/gpt2/resolve/main/vocab.json"
wget "https://huggingface.co/gpt2/resolve/main/merges.txt"
```

Copy `vocab.json` and `merges.txt` to preprocessed dataset dir:
```
cp vocab.json c4/preprocessed_c4_spm/
cp merges.txt c4/preprocessed_c4_spm/
```

## Prepare docker container for GPT3

Make sure to point:
* LOG_DIR to the path you want to save the logs;
* DATASET_DIR to the path where preprocessed data is located;
* CKPT_DIR to the path where universal checkpoint will be stored or is already stored (if steps in point "Prepare checkpoint" are already done);
* MODEL_DIR to the path where MLPERF code is located. The MLPERF code should have a following structure: `MLPERF/Habana/benchmarks`;

For example:
```
export LOG_DIR=/root/logs
export DATASET_DIR=/root/c4/preprocessed_c4_spm/
export CKPT_DIR=/root/universal-checkpoint/
export MODEL_DIR=/root/MLPERF
```

Start docker container with mounting the above paths on all of the machines, that will be used for training. The paths should be shared between the machines:
```
docker run --privileged --security-opt seccomp=unconfined               \
           --name mlperf3.0-gpt -td                                     \
           -v /dev:/dev                                                 \
           --device=/dev:/dev                                           \
           -e LOG_LEVEL_ALL=6                                           \
           -e PYTHONPATH=/usr/lib/habanalabs/:/root:/root/model_garden/ \
           -v /sys/kernel/debug:/sys/kernel/debug                       \
           -v /tmp:/tmp                                                 \
           -v $LOG_DIR:/root/logs                                       \
           -v $DATASET_DIR:/root/dataset/                               \
           -v $CKPT_DIR:/root/universal-checkpoint/                     \
           -v $MODEL_DIR:/root/MLPERF                                   \
           --cap-add=sys_nice --cap-add=SYS_PTRACE                      \
           --user root --workdir=/root --net=host                       \
           --ulimit memlock=-1:-1 vault.habana.ai/gaudi-docker-mlperf/ver3.0/pytorch-installer-1.13.1:1.10.97-48

docker exec -it mlperf3.0-gpt bash
```

### Host file and ssh connection between machines

DeepSpeed requires preparing the file describing all of the machines taking part in the training. Each record in so-called host file contains the address of the machine together with the number of devices that will take part in distributed training. During the training using run_gpt.sh it's location should be specified using --hostsfile argument which by default is /root/hostsfile. Exemplary file specifying two machines including 16 devices can look like this:
```
10.10.100.101 slots=8
10.10.100.102 slots=8
```
DeepSpeed uses ssh to spawn local and remote processes, so in order to allow communication between machines it is needed to provide a passwordless _ssh_ communication and set default port for connection. It needs to be done on all of the machines:
```
mkdir .ssh
printf 'Host *\n    StrictHostKeyChecking no\nPort 3022' >> .ssh/config
```
It also may be necessary to setup SSH keys and add them to `~/.ssh/authorized_keys`.

### Installing requirements

Requirements need to be installed on all of the machines that will take part in training.
Except from installing DeepSpeed what can be done via:
```
pip install /root/MLPERF/Habana/benchmarks/gpt3/deepspeed-fork
```
one needs also to install required packages:
```
pip install -r /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/requirements.txt
```

## Prepare checkpoint

Checkpoint for MLPerf GPT3 in paxml format can be downloaded from [gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000](gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000). The common_bf16.json can be downloaded from: https://github.com/ShriyaPalsamudram/training/tree/LLM-NVIDIA-reference-draft/large_language_model/megatron-lm/scripts. At one point there will be merged dir and universal dir which will be 2 TB disc space each for 96l so the free disc space to finish all the steps is >4TB. On top of that, the machine needs to have at least 32 CPUs and RAM of 755GB to make it work. Before the checkpoint can be used one needs to convert it with the following steps:
* convert paxml checkpoint to megatron distributed using /MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/tools/convert_checkpoint/convert_paxml_optimizer.py

```
python3 /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/tools/convert_checkpoint/convert_paxml_optimizer.py \
        --google_ckpts checkpoint_00004000/ \
        --output_dir megatron_merged_ckpt \
        --num_layers 96 \
        --params_file common_bf16.json \
        --pool 1
```
  
* convert megatron merged checkpoint to DeepSpeed universal

To generate mp-rank-files used in megatron_optim_merged_to_ds_universal_convert.py user has to run GPT-3 which will generate those files for the config used in the run.
It can be obtained by running single step of GPT-3 and save the checkpoint.
It can be done on 8 HLS2 system with data-parallel-size=1:
```
mkdir checkpoint_with_mp_rank_files
bash /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/run_gpt.sh --data-dir /root/dataset/ --hostfile /root/hostsfile --output-dir /root/logs --num-nodes 8 --data-parallel-size 1 --start-from-ckpt false --save-checkpoints-dir checkpoint_with_mp_rank_files --exit-interval 1
```

Run megatron_optim_merged_to_ds_universal_convert.py to create the universal checkpoint:
  
```
python3 /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/tools/convert_checkpoint/megatron_optim_merged_to_ds_universal_convert.py \
    --o /root/universal-checkpoint/ --ds-mp-rank-files-dir checkpoint_with_mp_rank_files --megatron-lm-merged-input-dir megatron_merged_ckpt \ 
    -–tp 8 -–pp 8 --nl 96 --iteration 4000 --global-batch-size 1536 --seq_length 2048 --lr-decay-samples 166809600 --lr-warmup-samples 407040 \
    -–pool 64 --model-parallel-same-config False --update-only-mp-rank-files False
```

## Run and time

Running GPT3 requires multiple machines. For example 32 HLS2 machines: `HLS-Gaudi2-N32-PT system` or 48 HLS2 machines `HLS-Gaudi2-N48-PT system`.

### Run GPT3 on HLS-Gaudi2-N32-PT system
```
bash /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/run_gpt.sh --data-dir /root/dataset/ --universal-ckpt-path /root/universal-checkpoint/ \
--hostfile /root/hostsfile --output-dir /root/logs --num-nodes 32 --data-parallel-size 4 --save-checkpoints false --mllog-output-path /root/logs/result.txt --train-samples 6782976
```

### Run GPT3 on HLS-Gaudi2-N48-PT system
```
bash /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/run_gpt.sh --data-dir /root/dataset/ --universal-ckpt-path /root/universal-checkpoint/ \
--hostfile /root/hostsfile --output-dir /root/logs --num-nodes 48 --data-parallel-size 6 --save-checkpoints false --mllog-output-path /root/logs/result.txt --train-samples 6782976
```

It's important to set `--data-dir` and `--universal-ckpt-path` to the paths set in previous steps. The `--save-checkpoints` is set to `false` as 96l checkpoints take a lot of disc space. In order to save the checkpoint after the run or save it with some frequency, please use `--save-checkpoints true` and manipulate `--save-interval` parameter.
The script will start from universal checkpoint and train up to 416 steps or the time, when validation log perplexity is below 2.69. According to the convergence point of GPT3 on HLS system, it should approximately run for 384 steps in order to reach 2.69 validation log perplexity. To reduce number of steps, you can use `--exit-interval` parameter or reduce train samples by `--train-samples` parameter.


# UNet3D
## Download, preprocess and verify dataset
The instruction for preparing the dataset is based on original MLCommons instruction at: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch#steps-to-download-and-verify-data

1. Download the dataset:
    ```
    cd $MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `$MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch/kits19/data`.
2. Build UNet3D docker container for data preprocessing
    ```
    cd $MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch
    docker build -t unet3d-data-preprocessing .
    ```
3. Preprocess the dataset inside UNet3D docker container.

    Note that you need to mount two directories:
   - directory with original, non-interpolated data e.g. `kits19/data`
   - output directory for preprocessed dataset e.g. `kits/preprocessed_data` and mount it to `/root/data`

    ```
    cd $MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch
    mkdir kits/preprocessed_data
    docker run --ipc=host --rm -it -v $MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch/kits19/data:/root/raw_data -v $MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch/kits19/preprocessed_data/:/root/preprocessed_data  unet3d-data-preprocessing:latest python3 preprocess_dataset.py --data_dir /root/raw_data --results_dir /root/preprocessed_data
    ```
    The command line will preprocess each volume and save it as a numpy array to `/root/MLPERF/Habana/benchmarks/unet3d/implementations/PyTorch/kits19/preprocessed_data/`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity. Also, it will run a checksum on each file comparing it with the source.


## Run and time
Log into mlperf3.0 PyTorch container.
Install additional requirements required for UNet3D and execute `run_and_time.sh` script:
```
cd /root/MLPERF/Habana/benchmarks/unet3d/implementations/PyTorch
pip install -r requirements.txt
DATASET_DIR=/root/MLPERF/Habana/benchmarks/unet3d/implementations/PyTorch/kits19/preprocessed_data/ bash run_and_time.sh --config config_HLS2_1x8x7.sh
```
By default mllog will be saved to `/tmp/result_rank_0.txt`.

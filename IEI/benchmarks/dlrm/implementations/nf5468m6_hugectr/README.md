This file contains the instructions for downloading and preprocessing the dataset,
specifying the location of the input files and building the docker image. Those steps are the same,
irrespective of the hardware platform. However, the actual commands for running the benchmark are different 
on single node and multiple nodes. Please refer to the other README files in this directory for those instructions.

### Dataset downloading and preprocessing 

Steps required to launch DLRMv2 training with HugeCTR on a single NVIDIA DGX H100:

#### Prepare the input dataset.

Input preprocessing steps below are based on the instructions from the official reference implementation repository, see [Running the MLPerf DLRM v2 benchmark](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#running-the-mlperf-dlrm-v2-benchmark). Besides, there is a final step to convert the reference implementation dataset to [raw format](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) in order to make it consumable by HugeCTR training script. For completeness, all the steps are detailed below.

This process can take up to several days and needs 7 TB of fast storage space.

**1.1** Download the dataset from https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/.

**1.2** Clone the reference implementation repository.

```
git clone https://github.com/mlcommons/training.git
cd training/recommendation_v2/torchrec_dlrm
```

**1.3** Build and run the reference docker image.
```
docker build -t dlrmv2_reference .
docker run -it --rm --network=host --ipc=host -v /data:/data dlrmv2_reference
```

**1.4** Run preprocessing steps to get data in NumPy format.

```
./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
    /data/criteo_1tb/raw_input_dataset_dir \
    /data/criteo_1tb/temp_intermediate_files_dir \
    /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```
As a result, files named: `day_*_labels.npy`, `day_*_dense.npy` and `day_0_sparse.npy` will be created (3 per each of 24 days in the original input dataset, 72 files in total). Once completed, the output data can be verified with md5sums provided in [md5sums_preprocessed_criteo_click_logs_dataset.txt](https://github.com/mlcommons/training/blob/master/recommendation_v2/torchrec_dlrm/md5sums_preprocessed_criteo_click_logs_dataset.txt) file.

**1.5** Create a synthetic multi-hot Criteo dataset.

This step produces multi-hot dataset from the original (one-hot) dataset.

```
python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
    --output_path /data/criteo_1tb_sparse_multi_hot \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
```

As a result, `day_*_sparse_multi_hot.npz` files will be created (24 files in total). Once done, the output data can be validated with md5sums provided in [md5sums_MLPerf_v2_synthetic_multi_hot_sparse_dataset.txt](https://github.com/mlcommons/training/blob/master/recommendation_v2/torchrec_dlrm/md5sums_MLPerf_v2_synthetic_multi_hot_sparse_dataset.txt) file.

**1.6** Convert NumPy dataset to raw format.

Because HugeCTR uses, among others, [raw format](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) for input data, we need to convert NumPy files created in the preceding steps to this format. To this end, use [convert_to_raw.py](preprocessing/convert_to_raw.py) script that comes with the container created in section [Build the container and push to a docker registry](#build-the-container-and-push-to-a-docker-registry) below.

```
docker run -it --rm --network=host --ipc=host -v /data:/data <docker/registry>/mlperf-nvidia:recommendation_hugectr
```
In that container, run:
```
python preprocessing/convert_to_raw.py \
   --input_dir_labels_and_dense /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
   --input_dir_sparse_multihot /data/criteo_1tb_sparse_multi_hot \
   --output_dir /data/criteo_1tb_multihot_raw \
   --stages train val
```

As a result, `train_data.bin` and `val_data.bin` will be created. Once done, the output files can be verified with the md5sums provided in [md5sums_raw_dataset.txt](preprocessing/md5sums_raw_dataset.txt) file.

### Specify the preprocessed data paths in the training script.

You may need to manually change the location of the datasets in the [train.py](train.py) file.
The `source` parameter should specify the absolute path to the `train_data.bin` file and the `eval_source`
parameter should point to the `val_data.bin` file from `/data/criteo_1tb_multihot_raw` folder obtained in the previous step.

However, for launching with nvidia-docker, you just need to make sure to set `DATADIR` as the path to the directory containing those two files.

### Build the container and push to a docker registry.

```
cd ../implementations/hugectr
docker build -t <docker/registry>/mlperf-nvidia:recommendation_hugectr .
docker push <docker/registry>/mlperf-nvidia:recommendation_hugectr
```

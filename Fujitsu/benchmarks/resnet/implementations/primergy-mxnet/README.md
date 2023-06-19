## Steps to launch training on a single node

### FUJITSU PRIMERGY server (single node)
Launch configuration and system-specific hyperparameters for the PRIMERGY server is
defined in `config_PG_10gpu.sh`.

Steps required to launch training on PRIMERGY server:

1. Build the container:

```
cd ../mxnet
docker build --pull -t <docker/registry>/mlperf-v30-resnet .
docker push <docker/registry>/mlperf-v30-resnet
```

2. Launch the training:

```
cd ../pytorch
echo <user password for sudo command> > password.txt
bash do_ssd.sh
```

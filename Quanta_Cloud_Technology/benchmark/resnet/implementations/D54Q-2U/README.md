## Steps to launch training

### QuantaGrid D54Q-2U

Launch configuration and system-specific hyperparameters for the QuantaGrid D54Q-2U
submission are in the `../<implementation>/config_D54Q-2U.sh` script.

Steps required to launch training on QuantaGrid D54Q-2U.

1. Build the docker container and push to a docker registry

```
cd ../mxnet
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_D54Q-2U.sh 
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data> ./run_with_docker.sh

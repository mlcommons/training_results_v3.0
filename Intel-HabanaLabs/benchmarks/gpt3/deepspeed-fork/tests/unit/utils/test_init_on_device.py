import torch
import os
import pytest
from unit.simple_model import SimpleModel
from deepspeed import OnDevice
from packaging import version as pkg_version

devices = ['meta', 'cuda:0']
if bool(pytest.use_hpu) == True:
    import habana_frameworks.torch.core as htcore
    devices = ['meta', 'hpu:0']

@pytest.mark.parametrize('device', devices)
def test_on_device(device):
    if device == "meta" and pkg_version.parse(
            torch.__version__) < pkg_version.parse("1.10"):
        pytest.skip("meta tensors only became stable after torch 1.10")
    dtype=torch.half
    if pytest.use_hpu == True:
        if os.getenv("REPLACE_FP16", default = None):
            dtype=torch.float
    with OnDevice(dtype=dtype, device=device):
        model = SimpleModel(4)

    for p in model.parameters():
        assert p.device == torch.device(device)
        assert p.dtype == dtype

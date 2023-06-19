import os
import pytest
import torch
import deepspeed
import transformers.models.bloom.modeling_bloom as modeling_bloom
from deepspeed.model_implementations import DeepSpeedTransformerInference
from transformers import AutoConfig, AutoModelForCausalLM
from unit.common import DistributedTest, DistributedFixture
from deepspeed.module_inject.layers import LinearLayer, Normalize, LinearAllreduce, EmbeddingLayer
from unit.hpu import *


def check_dtype(model, expected_dtype):
    def find_dtype(module):
        for child in module.children():
            if isinstance(child, DeepSpeedTransformerInference):
                return child.attention.attn_qkvw.dtype
            if isinstance(child, (LinearLayer, LinearAllreduce, Normalize, EmbeddingLayer)) and bool(pytest.use_hpu) == True:
                return child.weight.dtype
            else:
                found_dtype = find_dtype(child)
                if found_dtype:
                    return found_dtype

    found_dtype = find_dtype(model)
    assert found_dtype, "Did not find DeepSpeedTransformerInference in model"
    assert (
        found_dtype == expected_dtype
    ), f"Expected transformer dtype {expected_dtype}, but found {found_dtype}"


@pytest.fixture(params=[
    "bigscience/bloom-560m",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-125M",
    "facebook/opt-125m"
])
def model_name(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.int8], ids=["fp16", "int8"])
def dtype(request):
    return request.param


class save_shard(DistributedFixture):
    world_size = 2

    def run(self, model_name, class_tmpdir):
        if bool(pytest.use_hpu) == True:
            # FP16 is not supported on Gaudi1.
            if get_hpu_dev_version() == "Gaudi":
                pytest.skip(f"FP16 tests are not supported by Gaudi1.")

        # Only write a checkpoint if one does not exist
        if not os.path.isdir(os.path.join(class_tmpdir, model_name)):
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            inf_config = {
                "replace_with_kernel_inject": True,
                "dtype": torch.float16,
                "replace_method": "auto",
                "enable_cuda_graph": False,
                "tensor_parallel": {
                    "tp_size": world_size
                },
                "save_mp_checkpoint_path": os.path.join(str(class_tmpdir),
                                                        model_name),
            }

            # Load model and save sharded checkpoint
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         torch_dtype=torch.float16)
            if bool(pytest.use_hpu) == True:
                import deepspeed.module_inject as module_inject
                inf_config["replace_with_kernel_inject"] = False
                inj_policy = {"BertLayer": (module_inject.HFBertLayerPolicy,)}
                if model_name == "bigscience/bloom-560m":
                    inj_policy = {modeling_bloom.BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
                model = deepspeed.init_inference(model, config=inf_config, injection_policy=inj_policy)
            else:
                model = deepspeed.init_inference(model, config=inf_config)


@pytest.mark.seq_inference
class TestCheckpointShard(DistributedTest):
    world_size = 2

    def test(self, model_name, dtype, class_tmpdir, save_shard):

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        inf_config = {
            "replace_with_kernel_inject": True,
            "dtype": dtype,
            "replace_method": "auto",
            "enable_cuda_graph": False,
            "tensor_parallel": {
                "tp_size": world_size
            },
            "checkpoint": os.path.join(class_tmpdir,
                                       model_name,
                                       "ds_inference_config.json"),
        }

        # Load model on meta tensors
        model_config = AutoConfig.from_pretrained(model_name)
        # Note that we use half precision to load initially, even for int8
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            model = AutoModelForCausalLM.from_config(model_config,
                                                     torch_dtype=torch.bfloat16)
        model = model.eval()
        if bool(pytest.use_hpu) == True:
            import deepspeed.module_inject as module_inject
            inj_policy = {"BertLayer": (module_inject.HFBertLayerPolicy,)}
            if model_name == "bigscience/bloom-560m":
                inj_policy = {modeling_bloom.BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
            inf_config["replace_with_kernel_inject"] = False
            model = deepspeed.init_inference(model, injection_policy=inj_policy, config=inf_config)
        else:
            model = deepspeed.init_inference(model, config=inf_config)
        check_dtype(model, dtype)

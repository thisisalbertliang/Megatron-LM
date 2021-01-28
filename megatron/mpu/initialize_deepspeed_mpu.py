"""Model and data parallel groups."""

import torch
from deepspeed.runtime.pipe.topology import PipelineParallelGrid

from .utils import ensure_divisibility


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# ALBERT NOTE: _MPU_WORLD_SIZE and _MPU_RANK is only used for lazy_mpu_init
#              so we don't need them
# # These values enable us to change the mpu sizes on the fly.
# # _MPU_WORLD_SIZE = None
# # _MPU_RANK = None

_DEEPSPEED_MPU: PipelineParallelGrid = None


def is_uninitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DEEPSPEED_MPU is None


def initialize_megatron_model_parallel(deepspeed_mpu: PipelineParallelGrid):
    """
        Initialize model data parallel groups.
    """
    assert deepspeed_mpu is not None, 'DeepSpeed mpu cannot be None'
    global_rank = deepspeed_mpu.get_global_rank()
    if global_rank == 0:
        print('> initializing megatron mpu using DeepSpeed mpu {}'.format(deepspeed_mpu))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(deepspeed_mpu.get_model_parallel_world_size(), world_size)
    ensure_divisibility(world_size, model_parallel_size)

    # Set the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    _DATA_PARALLEL_GROUP = deepspeed_mpu.get_data_parallel_group()

    # Set the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    _MODEL_PARALLEL_GROUP = deepspeed_mpu.get_model_parallel_group()

    global _DEEPSPEED_MPU
    _DEEPSPEED_MPU = deepspeed_mpu


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    raise RuntimeError('Model parallelism is managed by DeepSpeed. '
                       'Setting the model parallel size is prohibited')


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return _DEEPSPEED_MPU.get_model_parallel_world_size()


def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    raise RuntimeError('Model parallelism is managed by DeepSpeed. '
                       'Setting the model parallel rank is prohibited')


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return _DEEPSPEED_MPU.get_model_parallel_rank()


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = _DEEPSPEED_MPU.get_global_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return _DEEPSPEED_MPU.get_data_parallel_world_size()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return _DEEPSPEED_MPU.get_data_parallel_rank()


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.model import GPT2Model
from megatron.training import pretrain, setup_model_and_optimizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses

import deepspeed

from megatron.initialize import initialize_megatron
from megatron.training import build_train_valid_test_data_iterators


class TrainBatchIterator:

    def __init__(self, data_iterator):
        self.data_iterator = data_iterator

    def __iter__(self):
        return self

    def __next__(self):
        """Generate a batch"""
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if self.data_iterator is not None:
            data = next(self.data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return tokens, position_ids, attention_mask, labels, loss_mask


def deepspeed_model_pipeline_engine_provider():
    """Build the model."""
    print_rank_0('building GPT2 model ...')

    args = get_args()

    model = GPT2Model(num_tokentypes=0, parallel_output=True)
    model_layers = model.to_layers()
    model_pipe = deepspeed.pipe.PipelineModule(model_layers, num_stages=1)
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model_pipe
    )

    return model_engine


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
    initialize_megatron(extra_args_provider=None, args_defaults=args_defaults)

    args = get_args()

    model_pipeline_engine = deepspeed_model_pipeline_engine_provider()

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)

    model_pipeline_engine.train()

    train_batch_iter = TrainBatchIterator(train_data_iterator)

    for step in range(args.train_iters):
        loss = model_pipeline_engine.train_batch(data_iter=train_batch_iter)
        print('ALBERT DEBUG:', 'loss', loss)

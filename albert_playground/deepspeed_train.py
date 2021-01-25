import torch
from torch.utils.data import Dataset
import numpy as np

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
from megatron.training import get_optimizer
from megatron.training import setup_model_and_optimizer
from megatron.utils import print_rank_0


class TrainDataIterator:

    def __init__(self, megatron_dataset, batch_size):
        self.megatron_dataset = megatron_dataset
        self.num_batch_done = 0
        self.batch_size = batch_size
        self.global_batch_size = batch_size * mpu.get_data_parallel_world_size()
        self.args = get_args()
        # self.cur_idx = 0
        # print_rank_0('ALBERT DEBUG: ' + 'mpu.get_data_parallel_world_size(): ' + str(mpu.get_data_parallel_world_size()))
        # print_rank_0('ALBERT DEBUG: ' + 'batch_size: ' + str(batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        batch = torch.tensor((), dtype=torch.int64)
        batch = batch.new_zeros((mpu.get_data_parallel_world_size(), self.batch_size, self.args.seq_length + 1), dtype=torch.int64)
        for batch_idx in range(self.global_batch_size):
            dataset_idx = self.num_batch_done * self.global_batch_size + batch_idx
            sample = self.megatron_dataset[dataset_idx]
            # sample = mpu.broadcast_data(['text'], sample, torch.int64)
            tokens = sample['text']
            tokens = torch.tensor(tokens.reshape((1, self.args.seq_length + 1)), dtype=torch.int64)
            batch[batch_idx // self.batch_size, batch_idx % self.batch_size, :] = tokens
        self.num_batch_done += 1
        # print_rank_0('ALBERT DEBUG: ' + 'batch.shape: ' + str(batch.shape))
        # print_rank_0('ALBERT DEBUG: ' + 'batch: ' + str(batch))
        return batch


class DeepSpeedDataset(Dataset):
    def __init__(self, megatron_dataset):
        self.megatron_dataset = megatron_dataset

    def __len__(self):
        return len(self.megatron_dataset)

    def __getitem__(self, idx):
        sample = self.megatron_dataset[idx]
        tokens: numpy.ndarray = sample['text']
        tokens = tokens.reshape((1, 1025))
        print_rank_0('ALBERT DEBUG: ' + 'str(tokens.shape): ' + str(tokens.shape))
        return tokens


def deepspeed_model_pipeline_engine_provider(train_ds):
    """Build the model."""
    print_rank_0('building GPT2 model ...')

    args = get_args()

    model = GPT2Model(num_tokentypes=0, parallel_output=True)
    model = model.to(torch.cuda.current_device())
    model_layers = model.to_layers()
    model_pipe = deepspeed.pipe.PipelineModule(model_layers, num_stages=1)

    optimizer = get_optimizer(model_pipe)

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model_pipe,
        model_parameters=[p for p in model_pipe.parameters() if p.requires_grad],
        optimizer=optimizer,
        training_data=train_ds
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
    args.iteration = 0

    args = get_args()
    print_rank_0('ALBERT DEBUG: ' + '> building train, validation, and test datasets ...')
    # Rank, size, and global batch size.
    data_parallel_size = mpu.get_data_parallel_world_size()
    global_batch_size = args.batch_size * data_parallel_size

    # Number of train/valid/test samples.
    train_iters = args.train_iters
    eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_iters * global_batch_size,
                                  eval_iters * global_batch_size,
                                  test_iters * global_batch_size]
    # print_rank_0('ALBERT DEBUG: ' + ' > datasets target sizes (minimum size):')
    # print_rank_0('ALBERT DEBUG: ' + '    train:      {}'.format(train_val_test_num_samples[0]))
    # print_rank_0('ALBERT DEBUG: ' + '    validation: {}'.format(train_val_test_num_samples[1]))
    # print_rank_0('ALBERT DEBUG: ' + '    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(
        train_val_test_num_samples)

    # print_rank_0('ALBERT DEBUG: ' + 'train_ds.__getitem__(0): ' + str(train_ds.__getitem__(0)))
    # print_rank_0('ALBERT DEBUG: ' + 'type(train_ds.__getitem__(0)): ' + str(type(train_ds.__getitem__(0))))
    # print_rank_0('ALBERT DEBUG: ' + 'len(train_ds.__getitem__(0)): ' + str(len(train_ds.__getitem__(0))))
    # print_rank_0('ALBERT DEBUG: ' + 'type(train_ds.__getitem__(0)["text"]): ' + str(type(train_ds.__getitem__(0)['text'])))
    # print_rank_0('ALBERT DEBUG: ' + 'train_ds.__getitem__(0)["text"].shape: ' + str(train_ds.__getitem__(0)['text'].shape))

    model_pipeline_engine = deepspeed_model_pipeline_engine_provider(None)

    train_data_iter = TrainDataIterator(train_ds, model_pipeline_engine.micro_batch_size)

    # print_rank_0('ALBERT DEBUG: ' + 'torch.cuda.current_device(): ' + str(torch.cuda.current_device()))

    for step in range(args.train_iters):
        loss = model_pipeline_engine.train_batch(data_iter=train_data_iter)
        print('ALBERT DEBUG:', 'loss', loss)

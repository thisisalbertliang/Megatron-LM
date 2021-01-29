import torch

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids


class PipeLoader(object):
    def __init__(self, loader):
        self.loader = loader
        if self.loader is not None:
            self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if self.data_iter is not None:
            data = next(self.data_iter)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, positions = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss
        )

        return (tokens, positions, attention_mask), (labels, loss_mask)

import torch
import torch.nn as nn
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from megatron import get_args
from megatron import mpu
from megatron import print_rank_0
from megatron.mpu import LayerNorm
from megatron.module import MegatronModule
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.language_model import Embedding, init_method_normal, scaled_init_method_normal, parallel_lm_logits
from megatron.model.gpt2_model import gpt2_attention_mask_func

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_pipe_language_model(attention_mask_func, num_tokentypes,
                       init_method=None, scaled_init_method=None):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    language_model = TransformerLanguageModelPipe(
        attention_mask_func=attention_mask_func,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


def tied_embedding_forward(module, inputs):
    outputs = parallel_lm_logits(inputs,
                                 module.word_embeddings.weight,
                                 True)
    return outputs


class TransposePipe(MegatronModule):
    def __init__(self):
        super(TransposePipe, self).__init__()

    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        return (hidden_states, attention_mask)


class FinalLayerNormPipe(MegatronModule):
    def __init__(self):
        super(FinalLayerNormPipe, self).__init__()
        args = get_args()

        self.layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        output = self.layernorm(hidden_states)

        return output
        #return (output, attention_mask)


class EmbeddingPipe(Embedding):
    def forward(self, inputs):
        input_ids, position_ids, attention_mask = inputs

        embedding_output = super().forward(input_ids, position_ids)

        return (embedding_output, attention_mask)


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        output = super().forward(hidden_states=hidden_states,
                                 attention_mask=attention_mask,
                                 layer_past=None,
                                 get_key_value=False)

        return (output, attention_mask)


class CheckpointResetPipe(MegatronModule):
    def __init__(self):
        super(CheckpointResetPipe, self).__init__()

    def forward(self, inputs):
        mpu.reset_checkpointed_activations_memory_buffer()

        return inputs


class ParallelTransformerLayerCheckpointPipe(ParallelTransformerLayer):
    def forward(self, inputs):
        def custom():
            def custom_forward(inputs):
                hidden_states, attention_mask = inputs
                new_states = super.forward(hidden_states, attention_mask)
                return (new_states, attention_mask)
            return custom_forward

        outputs = mpu.checkpoint(custom, inputs)

        return outputs


class TransformerLanguageModelPipe(MegatronModule):
    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0):
        super(TransformerLanguageModelPipe, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method

        # Embeddings
        self.embedding = EmbeddingPipe(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       self.init_method,
                                       self.num_tokentypes)
        self._embedding_key = 'embedding'

        # Transformer
        self.transformer = nn.ModuleList()
        #self.transformer = []
        self.transformer.append(TransposePipe())
        if args.checkpoint_activations:
            self.transformer.append(CheckpointResetPipe())
        for i in range(args.num_layers):
            if args.checkpoint_activations:
                self.transformer.append(
                    ParallelTransformerLayerCheckpointPipe(
                        attention_mask_func,
                        self.init_method,
                        output_layer_init_method,
                        i + 1
                    )
                )
            else:
                self.transformer.append(
                    ParallelTransformerLayerPipe(
                        attention_mask_func,
                        self.init_method,
                        output_layer_init_method,
                        i + 1
                    )
                )
        self.transformer.append(TransposePipe())
        self.transformer.append(FinalLayerNormPipe())
        self._transformer_key = 'transformer'

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None):
        inputs = (input_ids, position_ids, attention_mask, tokentype_ids)
        out = self.embedding(inputs)
        for layer in self.transformer:
            out = layer(out)

        return out

    def to_layers(self):
        layers = [self.embedding]
        for layer in self.transformer:
            layers.append(layer)

        return layers


class GPT2ModelPipe(PipelineModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):

        args = get_args()

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        specs = []
        # Embedding
        embedding_args = [args.hidden_size, args.padded_vocab_size,
                          args.max_position_embeddings, args.hidden_dropout,
                          init_method, num_tokentypes]
        specs.append(TiedLayerSpec("embed", EmbeddingPipe, *embedding_args,
                                   tied_weight_attr="word_embeddings.weight"))
        # Transformer
        specs.append(LayerSpec(TransposePipe))
        if args.checkpoint_activations:
            specs.append(LayerSpec(CheckpointResetPipe))
        for i in range(args.num_layers):
            if args.checkpoint_activations:
                spec = LayerSpec(ParallelTransformerLayerCheckpointPipe,
                                 gpt2_attention_mask_func,
                                 init_method,
                                 scaled_init_method,
                                 i + 1)
            else:
                spec = LayerSpec(ParallelTransformerLayerPipe,
                                 gpt2_attention_mask_func,
                                 init_method,
                                 scaled_init_method,
                                 i + 1)
            specs.append(spec)
        specs.append(LayerSpec(TransposePipe))
        specs.append(LayerSpec(FinalLayerNormPipe))
        specs.append(TiedLayerSpec("embed", EmbeddingPipe, *embedding_args,
                                   forward_fn=tied_embedding_forward,
                                   tied_weight_attr="word_embeddings.weight"))

        def loss_fn(outputs, labels_and_mask):
            labels, loss_mask = labels_and_mask

            if self.fp16_lm_cross_entropy:
                assert outputs.dtype == torch.half
                losses = mpu.vocab_parallel_cross_entropy(outputs, labels)
            else:
                losses = mpu.vocab_parallel_cross_entropy(outputs.float(), labels)

            loss_mask = loss_mask.view(-1)
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            return loss

        if args.num_pp == -1 or args.num_mp == -1 or args.num_dp == -1:
            raise ValueError('To enable DeepSpeed 3D parallelism, '
                             'you must specify each of num-pp, num-mp, and num-dp')
        topology_3d = PipeModelDataParallelTopology(num_pp=args.num_pp,
                                                    num_mp=args.num_mp,
                                                    num_dp=args.num_dp)
        print_rank_0(f'Initializing DeepSpeed 3D parallelism with topology {topology_3d}')

        super().__init__(layers=specs,
                         loss_fn=loss_fn,
                         partition_method="parameters",
                         topology=topology_3d)

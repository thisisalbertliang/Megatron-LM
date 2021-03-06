#! /bin/bash

# Change for multinode config
MP_SIZE=4

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=1

HOSTFILE="/users/sangkeuc/albert/ds-megatron/Megatron-LM-No-PP/albert_scripts/myhostfile"
DATA_PATH=/users/sangkeuc/albert/ds-megatron/Megatron-LM-No-PP/preprocessed_data/my-gpt2_text_document

config_json="/users/sangkeuc/albert/ds-megatron/Megatron-LM-No-PP/albert_scripts/ds_zero2_config.json"
#gpt_options=" \
#       --model-parallel-size ${MP_SIZE} \
#       --num-layers 24 \
#       --hidden-size 1024 \
#       --num-attention-heads 16 \
#       --batch-size 8 \
#       --seq-length 1024 \
#       --max-position-embeddings 1024 \
#       --train-iters 100000 \
#       --resume-dataloader \
#       --train-data webtext \
#       --lazy-loader \
#       --tokenizer-type GPT2BPETokenizer \
#       --split 949,50,1 \
#       --distributed-backend nccl \
#       --lr 0.00015 \
#       --no-load-optim \
#       --lr-decay-style cosine \
#       --weight-decay 1e-2 \
#       --clip-grad 1.0 \
#       --warmup .01 \
#       --checkpoint-activations \
#       --deepspeed-activation-checkpointing \
#       --fp16 \
#"
gpt_options=" \
       --model-parallel-size $MP_SIZE \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json}
#"

run_cmd="deepspeed --hostfile ${HOSTFILE} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} albert_pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
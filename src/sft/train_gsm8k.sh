BITS="2"
GROUP_SIZE=64

RANK=64
SCALE="4e3"
DECAY=0.1

MODEL="meta-llama/Llama-3.2-3B"
DATASET="gsm8k"
INIT="True"
LR="2e-4"

set -x
python train_gsm8k.py \
    --model_name_or_path $MODEL \
    --data_path $DATASET \
    --learning_rate $LR \
    --weight_decay $DECAY \
    --warmup_ratio 0.06 \
    --lr_scheduler_type cosine \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 6 \
    --logging_steps 50 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --output_dir outputs/$DATASET/$MODEL/init-$INIT/$BITS-bits/lr$LR-decay$DECAY-scale$SCALE\
    --resume_from_checkpoint False \
    --bits $BITS \
    --group_size $GROUP_SIZE \
    --lora_r $RANK \
    --scale $SCALE \
    --peft_init $INIT \
    --bf16 True \
    --eval_every_epoch True 
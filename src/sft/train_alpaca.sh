BITS="2"
GROUP_SIZE=64

RANK=64
LR=5e-5
SCALE="4e3"
DECAY=0.01

MODEL="meta-llama/Llama-3.2-3B"
DATASET="alpaca_data_cleaned"
INIT="False"

set -x
python SFT.py \
    --model_name_or_path $MODEL \
    --data_path $DATASET \
    --num_train_epochs 3 \
    --learning_rate $LR \
    --weight_decay $DECAY \
    --warmup_ratio 0.06 \
    --lr_scheduler_type cosine \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --output_dir outputs/$DATASET/$MODEL/init-$INIT/$BITS-bits/lr$LR-decay$DECAY-scale$SCALE \
    --resume_from_checkpoint False \
    --save_total_limit 5 \
    --bits $BITS \
    --group_size $GROUP_SIZE \
    --lora_r $RANK \
    --scale $SCALE \
    --peft_init $INIT \
    --bf16 True \
    --run_name "alpaca rank $RANK lr $LR" \
    --finetune True \
    --eval_csqa True


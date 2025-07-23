
QUANT_METHOD="gptq"
BITS="2"
GROUP_SIZE=64

RANK=64
SCALE="4e3"
DECAY=0.01

MODEL="meta-llama/Llama-3.1-8B"
DATASET="gsm8k" # alpaca_data_cleaned.json or commonsense_170k_sft or gsm8k or wikitext


python test_gsm8k.py \
  --model_name_or_path $MODEL \
  --bits $BITS \
  --group_size $GROUP_SIZE \
  --lora_r $RANK \
  --scale $SCALE \
  --batch_size 64 \
  --adapter_name_or_path outputs/$DATASET/$MODEL/init-$INIT/$BITS-bits/lr$LR-decay$DECAY-scale$SCALE
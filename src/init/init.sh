MODEL_ID=meta-llama/Llama-3.2-3B
BITS=2
GROUPS=64
RANK=64
python initialize.py -m $MODEL_ID -q gptq -b $BITS -g $GROUPS -r $RANK --eval_ppl

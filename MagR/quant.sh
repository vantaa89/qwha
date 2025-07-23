BITS="2"
GROUPS=64

#for BIT in $BITS
#do
    #python mistral.py "mistralai/Mistral-7B-v0.3" --wbits $BIT --groupsize $GROUPS --magr --static-groups --save "gptq_models/Mistral-7B-v0.3-${BIT}bits-g${GROUPS}"
#done

# for BIT in $BITS
# do
#     python llama.py "meta-llama/Llama-3.1-8B" --wbits $BIT --groupsize $GROUPS --magr --static-groups --save "gptq_models/Llama-3.1-8B-${BIT}bits-g${GROUPS}-2e-5-200"
# done

for BIT in $BITS
do
    python llama.py "meta-llama/Llama-3.2-3B" --wbits $BIT --groupsize $GROUPS --magr --static-groups --save "gptq_models/Llama-3.2-3B-${BIT}bits-g${GROUPS}"
done

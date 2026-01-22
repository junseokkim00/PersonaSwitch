# strategy=top_p
# strategy=top_k
# strategy=temperature
# strategy=greedy


# model_name=meta-llama/Llama-3.1-8B-Instruct
# model_name=meta-llama/Llama-3.2-3B-Instruct
# model_name=google/gemma-2-2b-it

# seed=0
# seed=1
# seed=2

python inference_other_decoding_strategy.py \
    --dataset aqua \
    --model_name ${model_name} \
    --method zero_shot \
    --decoding_strategy ${strategy} \
    --random_seed ${seed} \
    --etc seed:${seed}

python inference_other_decoding_strategy.py \
    --dataset gsm8k \
    --model_name ${model_name} \
    --method zero_shot \
    --decoding_strategy ${strategy} \
    --random_seed ${seed} \
    --etc seed:${seed}

python inference_other_decoding_strategy.py \
    --dataset commonsenseqa \
    --model_name ${model_name} \
    --method zero_shot \
    --decoding_strategy ${strategy} \
    --random_seed ${seed} \
    --etc seed:${seed}

python inference_other_decoding_strategy.py \
    --dataset last_letters \
    --model_name ${model_name} \
    --method zero_shot \
    --decoding_strategy ${strategy} \
    --random_seed ${seed} \
    --etc seed:${seed}

python inference_other_decoding_strategy.py \
    --dataset object_tracking \
    --model_name ${model_name} \
    --method zero_shot \
    --decoding_strategy ${strategy} \
    --random_seed ${seed} \
    --etc seed:${seed}
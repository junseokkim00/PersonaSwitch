# model_name=google/gemma-2-2b-it
# model_name=meta-llama/Llama-3.2-3B-Instruct
# model_name=meta-llama/Llama-3.1-8B-Instruct

python inference_base_persona.py \
    --model_name ${model_name} \
    --dataset aqua \
    --limit_dataset_size 0 \
    --method role_play

python inference_base_persona.py \
    --model_name ${model_name} \
    --dataset gsm8k \
    --limit_dataset_size 0 \
    --method role_play

python inference_base_persona.py \
    --model_name ${model_name} \
    --dataset commonsenseqa \
    --limit_dataset_size 0 \
    --method role_play

python inference_base_persona.py \
    --model_name ${model_name} \
    --dataset last_letters \
    --limit_dataset_size 0 \
    --method role_play

python inference_base_persona.py \
    --model_name ${model_name} \
    --dataset object_tracking \
    --limit_dataset_size 0 \
    --method role_play
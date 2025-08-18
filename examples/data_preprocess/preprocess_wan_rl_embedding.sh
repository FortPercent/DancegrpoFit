MODEL_PATH="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-T2V-1.3B"
OUTPUT_DIR="/nvfile-heatstorage/ai_infra/data/wuxn5/wuxuaner/Dancegrpo_verl/data/rl_embeddings"
export PYTHONPATH=/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/wan:$PYTHONPATH

python examples/data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "/nvfile-heatstorage/ai_infra/data/wuxn5/wuxuaner/Dancegrpo_verl/data/rl_embeddings/flattened_wan_results_with_prompt.json" \
    --output_dir $OUTPUT_DIR \

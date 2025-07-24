MODEL_PATH="/nvfile-heatstorage/model_zoo/Wan2___1-T2V-1___3B"
OUTPUT_DIR="data/rl_embeddings"

python examples/data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "data/rl_embeddings/flattened_wan_results_with_prompt.json" \
    --output_dir $OUTPUT_DIR \

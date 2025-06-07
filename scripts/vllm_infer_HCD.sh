
export CUDA_VISIBLE_DEVICES=0,3
python vllm_infer_HCD.py \
    --model_name_or_path "/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_lora_sft" \
    --template "qwen2_vl" \
    --dataset FHM_test_seen_infer_format.json \
    --dataset_dir data_use \
    --cutoff_len 2048 \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --top_p 0.7 \
    --top_k 50 \
    --batch_size 4 \
    --save_name "vllm_infer_HCD_qwen2-vl-2b.json"
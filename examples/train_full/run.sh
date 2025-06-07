CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml

sleep 60 # 等待1分钟
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_full/qwen2vl_full_sft_2.yaml
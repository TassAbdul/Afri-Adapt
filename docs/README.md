

## Get repo ready
0. Python 3.10 or above is recommended
1. git clone https://github.com/TassAbdul/Afri-Adapt.git
2. cd Afri-Adapt
3. pip install -e .
4. pip uninstall transformers
5. pip install git+https://github.com/huggingface/transformers
6. pip install accelerate

## Prepare Train data

1. **Train audio files** should be downloaded to `Afri-Adapt/emnlp_nv_dataset`
2. **Make sure audio files are in true `.wav` format**

### 🎧 Convert `.wav` files to the correct format using `ffmpeg`

```bash
# module load ffmpeg  # (Uncomment if using a cluster environment)

input_dir="Enter correct input dir"
output_dir="emnlp_nv_dataset"

mkdir -p "$output_dir"

for f in "$input_dir"/*.wav; do
    base=$(basename "$f" .wav)
    out="$output_dir/$base.wav"

    if [ ! -f "$out" ]; then
        ffmpeg -v warning -y -i "$f" -ar 16000 -ac 1 "$out"
        if [ $? -eq 0 ]; then
            echo "✅ Converted: $base.wav"
        else
            echo "⚠️ Error converting: $base.wav"
        fi
    else
        echo "⏩ Skipping already converted: $base.wav"
    fi
done
```
# Lora Finetune
1. Make sure you have access to the model via Huggingface
2. Run the code with GPU below: 
```bash
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset /Afri_Adapt/data/cleaned_nv_dataset_swift_transcribe.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_transcribe \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

## Acknowledgment
This project is adapted from [ms-swift](https://github.com/modelscope/ms-swift).  
Original contributors are credited for foundational work in the early implementation.

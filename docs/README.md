

## Get repo ready
1. mm
2. cd ms-swift
3. pip install -e .

## Prepare Train data

1. **Train audio files** should be downloaded to `ms-swift/emnlp_nv_dataset`
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



## Acknowledgment
This project is adapted from [ms-swift](https://github.com/modelscope/ms-swift).  
Original contributors are credited for foundational work in the early implementation.

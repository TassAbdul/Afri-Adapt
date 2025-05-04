from datasets import load_dataset
import soundfile as sf
import os
import json

# Load dataset directly from Hugging Face
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from huggingface_hub import login
login(token = "HF_TOKEN")




#convert to ffmpeg
# module load ffmpeg
# input_dir="naijavoices"
# output_dir="fixed_wav"

# mkdir -p "$output_dir"

# for f in "$input_dir"/*.wav; do
#     base=$(basename "$f" .wav)
#     out="$output_dir/$base.wav"
    
#     if [ ! -f "$out" ]; then
#         ffmpeg -v warning -y -i "$f" -ar 16000 -ac 1 "$out"
#         if [ $? -eq 0 ]; then
#             echo "✅ Converted: $base.wav"
#         else
#             echo "⚠️ Error converting: $base.wav"
#         fi
#     else
#         echo "⏩ Skipping already converted: $base.wav"
#     fi
# done


from datasets import load_dataset
import json
import os

# Load dataset
dataset = load_dataset("intronhealth/afrivox-translate", split="test", trust_remote_code=True)
# dataset = load_dataset("intronhealth/afrivox-transcribe", split="test")

# Output path
output_path = "data/afrivox_inference_translate.jsonl"

# Mode: "transcribe" or "translate"
mode = "translate"  # or "translate"

with open(output_path, "w", encoding="utf-8") as f:
    for idx, sample in enumerate(dataset):
        try:
            
            audio_path = sample["audio"]["path"]  # Absolute path to .wav file
            lang = sample.get("language", "the given language")
            transcription = sample.get("transcription", None)
            translation = sample.get("translation", None)

            # Choose assistant content based on mode
            if mode == "translate":
                system_content = "You are a speech translation model."
                user_content = f"<audio>Translate the following audio from {lang} into English text."
                assistant_content = translation
            else:  # transcribe
                system_content = "You are a speech recognition model."
                user_content = f"<audio>Transcribe the following into {lang} text."
                assistant_content = transcription

            # If assistant content is missing, skip the sample
            if not assistant_content:
                print(f"⚠️ Skipping sample {idx}: missing assistant content.")
                continue

            entry = {
                "audios": [audio_path],
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if idx % 100 == 0:
                print(f"✅ Processed {idx} samples...")

        except Exception as e:
            print(f"❌ Error at sample {idx}: {e}")
            continue

print(f"🎯 Done! Inference-ready JSONL saved to: {output_path}")






# from datasets import load_dataset
# import json
# import os

# # Load dataset
# dataset = load_dataset("intronhealth/afrivox-transcribe", split="test", trust_remote_code=True)

# # Output paths
# output_path = "data/afrivox_transcribe_inference2.jsonl"
# bad_rows_path = "data/afrivox_bad_samples.jsonl"

# good_samples = []
# bad_rows = 0

# for idx, sample in enumerate(dataset):
#     try:
#         # ✅ Use the real extracted audio path
#         audio_path = sample["audio"]["path"]  # Absolute path to .wav file

#         # ✅ Metadata
#         lang = sample.get("language", "the given language")
#         transcription = sample["transcription"]  # (Keep for scoring later)
        
#         entry = {
#             "audios": [audio_path],
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "You are a speech recognition model."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"<audio>Transcribe the following into {lang} text."
#                 }
#             ],
#             "reference": transcription
#         }


#         good_samples.append(entry)

#         if idx % 100 == 0:
#             print(f"✅ Processed {idx} samples...")

#     except Exception as e:
#         print(f"❌ Error at sample {idx}: {e}")
#         bad_rows += 1
#         continue

# # Save only good samples
# with open(output_path, "w", encoding="utf-8") as f_out:
#     for entry in good_samples:
#         f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print(f"✅ Saved {len(good_samples)} good samples to {output_path}")
# print(f"❌ Found {bad_rows} bad rows (skipped)")


# # Load dataset
# dataset = load_dataset("intronhealth/afrivox-transcribe", split="test", trust_remote_code=True)

# # Output path
# output_path = "data/afrivox_transcribe_inference.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for idx, sample in enumerate(dataset):
#         try:
#             # ✅ Use the real extracted audio path
#             audio_path = sample["audio"]["path"]  # Absolute path to .wav file

#             # ✅ Metadata
#             lang = sample.get("language", "the given language")
#             transcription = sample["transcription"]  # (Keep for scoring later)

#             # ✅ Swift-compatible Inference JSONL entry
#             entry = {
#                 "audios": [audio_path],
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "You are a speech recognition model."  # ✅ plain string now
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": f"<audio>Transcribe the following into {lang} text."},
#                             {"type": "audio", "audio": audio_path}
#                         ]
#                     }
#                 ],
#                 "reference": transcription
#             }

#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

#             if idx % 100 == 0:
#                 print(f"✅ Processed {idx} samples...")

#         except Exception as e:
#             print(f"❌ Error at sample {idx}: {e}")
#             continue

# print(f"🎯 Done! Inference-ready JSONL saved to: {output_path}")


# from datasets import load_dataset
# import json
# import os

# # Load dataset
# dataset = load_dataset("intronhealth/afrivox-translate", split="test")

# # Output path
# output_path = "afrivox_translate.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for idx, sample in enumerate(dataset):
#         try:
#             # ✅ Use the real extracted audio path
#             audio_path = sample["audio"]["path"]  # Absolute path to .wav file

#             # ✅ Metadata
#             lang = sample.get("language", "the given language")
#             transcription = sample["transcription"]

#             # ✅ Swift JSONL entry
#             entry = {
#                 "audios": [audio_path],
#                 "messages": [
#                     {"role": "system", "content": "You are a speech translation model."},
#                     {"role": "user", "content": f"<audio>Translate the following audio from {lang} language into English text."},
#                     {"role": "assistant", "content": transcription}
#                 ]
#             }
            
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

#             if idx % 100 == 0:
#                 print(f"✅ Processed {idx} samples...")

#         except Exception as e:
#             print(f"❌ Error at sample {idx}: {e}")
#             continue

# print(f"🎯 Done! Swift-formatted JSONL saved to: {output_path}")

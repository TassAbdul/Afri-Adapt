import pandas as pd
import json
import os

# Original CSV file
input_file = "cleaned_nv_dataset.csv"  # <- Replace with your file

# Generate new filename
base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}_swift_transcribe.jsonl"

############Transcribe###################
# # Load the CSV
# df = pd.read_csv(input_file)
# # Create the JSONL file
# with open(output_file, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         language = row.get("language", "your language")
#         audio_path = row["audio_path"]
        

#         transcription = row["transcription"]

#         entry = {
#             "audios": [audio_path],
#             "messages": [
#                 {"role": "system", "content": "You are a speech recognition model."},
#                 {"role": "user", "content": f"<audio>Transcribe the audio into {language} text."},
#                 # {"role": "user", "content": f"<audio>Transcribe the audio into text."},
#                 {"role": "assistant", "content": transcription}
#             ]
#         }
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print(f"Done! JSONL saved as: {output_file}")




# Original CSV file
input_file = "cleaned_nv_dataset.csv"  # <- Replace with your file

# Generate new filename
base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}_swift_translate.jsonl"

############Transcribe###################
# Load the CSV
df = pd.read_csv(input_file)
# Create the JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        language = row.get("language", "your language")
        audio_path = row["audio_path"]
        

        translation = row["translation"]

        entry = {
            "audios": [audio_path],
            "messages": [
                {"role": "system", "content": "You are a speech translation model."},
                {"role": "user", "content": f"<audio>Translate the following audio from {language} language into English text."},
                {"role": "assistant", "content": translation}
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done! JSONL saved as: {output_file}")



##########Translate


# import pandas as pd
# import json
# import os

# # Original CSV file
# csv_path = "cleaned_nv_dataset.csv"  # <- Replace with your file

# # Generate new filename
# base_name = os.path.splitext(csv_path)[0]
# output_jsonl = f"{base_name}_swift_translate.jsonl"

# df = pd.read_csv(csv_path)

# # Create the JSONL file
# with open(output_jsonl, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         language = row.get("language", "your language")
#         audio_path = row["audio_path"]
       

#         # If needed, you could verify the file exists here
#         entry = {
#             "audios": ["audio_path"],
#             "messages": [
#                 {"role": "system", "content": "You are a speech translation model."},
#                 {"role": "user", "content": "<audio>Translate the following audio into English"},
#                 {"role": "assistant", "content": row["translation"]}
#             ]
#         }
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print(f"✅ Translation-ready JSONL saved to {output_jsonl}")


### This cleans the original csv to make sure the audio files and trancripts are present

# import pandas as pd
# import os

# # Set paths
# csv_path = "emnlp_nv_dataset.csv"
# audio_root = "/gpfs/data/ceickhof/aabdul/ms-swift"  # adjust this
# output_path = "cleaned_nv_dataset.csv"

# # Load CSV
# df = pd.read_csv(csv_path)

# # Filter: keep only rows where the audio file exists
# df["full_audio_path"] = df["audio_path"].apply(lambda x: os.path.join(audio_root, x))
# print(df.full_audio_path.iloc[5])
# print(df.full_audio_path.iloc[50])
# df = df[df["full_audio_path"].apply(os.path.exists)]


# # Drop the helper column and save cleaned CSV
# df.drop(columns=["full_audio_path"]).to_csv(output_path, index=False)

# print(f"✅ Cleaned dataset saved: {output_path}")
# print(f"✅ Rows kept: {len(df)}")




########## This is a command line code#########
#first i module load ffmpeg
# #!/bin/bash

# input_dir="/gpfs/data/ceickhof/aabdul/ms-swift/emnlp_nv_dataset"
# output_dir="/gpfs/data/ceickhof/aabdul/ms-swift/emnlp_nv_dataset_wav"

# mkdir -p "$output_dir"

# for f in "$input_dir"/*.wav; do
#     base=$(basename "$f" .wav)
#     ffmpeg -y -i "$f" -ar 16000 -ac 1 "$output_dir/$base.wav"
#     echo "✅ Converted: $base.wav"
# done

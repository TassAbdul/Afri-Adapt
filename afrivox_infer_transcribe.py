import json
from tqdm import tqdm
from evaluate import load
from swift.llm import get_model_tokenizer, get_template, InferRequest, RequestConfig, PtEngine
from modelscope import snapshot_download
from swift.tuners import Swift

# Set how many samples you want to process
MAX_SAMPLES = 50  # <--- Set this to any number you want

# Load model and tokenizer via Swift
model_dir = snapshot_download('Qwen/Qwen2.5-Omni-7B')
adapter_dir = '/gpfs/data/ceickhof/aabdul/ms-swift/output_transcribe/v0-20250424-195749/checkpoint-5960'

model, tokenizer = get_model_tokenizer(model_dir, device_map='auto')
model = Swift.from_pretrained(model, adapter_dir)
template = get_template(model.model_meta.template, tokenizer)
engine = PtEngine.from_model_template(model, template)

# Load Hugging Face WER metric
wer_metric = load("wer")

# Load your JSONL dataset
input_jsonl = "/gpfs/data/ceickhof/aabdul/ms-swift/data/afrivox_inference_transcribe.jsonl"

output_jsonl = "/gpfs/data/ceickhof/aabdul/ms-swift/data/afrivox_transcribe_output.jsonl"

predictions = []
references = []
results = []
request_config = RequestConfig(max_tokens=512, temperature=0)
with open(input_jsonl, "r", encoding="utf-8") as f:
    progress_bar = tqdm(f, desc="Running inference", dynamic_ncols=True)
    
    for idx, line in enumerate(progress_bar):

        # if idx >= MAX_SAMPLES:
        #     break
    
            
    
        sample = json.loads(line)
        progress_bar.set_postfix({"Sample": idx + 1}, refresh=False)
    
        messages = sample["messages"]
    
        # 💥 Fix: Remove assistant ground-truth before inference
        if messages[-1]["role"] == "assistant":
            reference = messages[-1]["content"]
            messages = messages[:-1]
        else:
            reference = None
    
        audio_paths = sample.get("audios", None)
    
        infer_req = InferRequest(
            messages=messages,
            audios=audio_paths
        )
    
        response_list = engine.infer([infer_req], request_config=request_config)
        generated_text = response_list[0].choices[0].message.content
    
        predictions.append(generated_text)
        references.append(reference)
    
        result_entry = {
            "input": messages,
            "reference": reference,
            "prediction": generated_text
        }
        results.append(result_entry)


# Save all inference outputs
with open(output_jsonl, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Compute WER
wer_score = wer_metric.compute(predictions=predictions, references=references)
print(f"🎯 Word Error Rate (WER): {wer_score:.4f}")

print(f"✅ Inference complete! Processed {len(results)} samples. Results saved to {output_jsonl}")


# import json
# from tqdm import tqdm
# from evaluate import load
# from swift.llm import get_model_tokenizer, get_template, InferRequest, RequestConfig, PtEngine
# from modelscope import snapshot_download
# from swift.tuners import Swift
# import torch

# # Set how many samples you want to process
# MAX_SAMPLES = 10

# # Load model and tokenizer via Swift
# model_dir = snapshot_download('Qwen/Qwen2.5-Omni-7B')
# adapter_dir = '/gpfs/data/ceickhof/aabdul/ms-swift/output_transcribe/v0-20250424-195749/checkpoint-5960'

# model, tokenizer = get_model_tokenizer(model_dir, device_map='auto')
# model = Swift.from_pretrained(model, adapter_dir)
# template = get_template(model.model_meta.template, tokenizer)
# engine = PtEngine.from_model_template(model, template)

# # Load Hugging Face WER metric
# wer_metric = load("wer")

# # Load your JSONL dataset
# input_jsonl = "/gpfs/data/ceickhof/aabdul/ms-swift/data/afrivox_transcribe.jsonl"
# output_jsonl = "/gpfs/data/ceickhof/aabdul/ms-swift/data/afrivox_transcribe_outputX.jsonl"

# predictions = []
# references = []
# results = []


# request_config = RequestConfig(max_tokens=512, temperature=0)
# with open(input_jsonl, "r", encoding="utf-8") as f:
#     progress_bar = tqdm(f, desc="Running inference", dynamic_ncols=True)
 
#     for idx, line in enumerate(progress_bar):

#         if idx >= MAX_SAMPLES:
#             break

#         sample = json.loads(line)
#         progress_bar.set_postfix({"Sample": idx + 1}, refresh=False)

#         messages = sample["messages"]
#         reference = sample["reference"]  # ✅ Get true transcription separately
        
#         # ✅ Print audio file path(s)
#         for msg in messages:
#             if msg["role"] == "user":
#                 for content in msg["content"]:
#                     if isinstance(content, dict) and content.get("type") == "audio":
#                         print(f"🔊 Audio file detected for Sample {idx + 1}: {content['audio']}")


#         infer_req = InferRequest(
#             messages=messages
#             # 🚫 No audios=audio_paths anymore
#         )

        

#         with torch.inference_mode():
#             response_list = engine.infer([infer_req], request_config=request_config)

#         generated_text = response_list[0].choices[0].message.content

#         predictions.append(generated_text)
#         references.append(reference)

#         result_entry = {
#             "input": messages,
#             "reference": reference,
#             "prediction": generated_text
#         }
#         results.append(result_entry)

# # Save all inference outputs
# with open(output_jsonl, "w", encoding="utf-8") as f:
#     for r in results:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")

# # Compute WER
# wer_score = wer_metric.compute(predictions=predictions, references=references)
# print(f"🎯 Word Error Rate (WER): {wer_score:.4f}")

# print(f"✅ Inference complete! Processed {len(results)} samples. Results saved to {output_jsonl}")

# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# import torch

# # 1. Load model and processor
# # model_id = "google/gemma-3n-e2b-it"
# # processor = AutoProcessor.from_pretrained(model_id)
# # model = AutoModelForImageTextToText.from_pretrained(model_id)

# # model.save_pretrained("./gemma3n_model")
# # processor.save_pretrained("./gemma3n_processor")




# # # 2. Load an image (for testing)
# image = Image.open("Blurr.jpg").convert("RGB")  # replace with your own image file

# # 3. Create prompt


# from transformers import AutoProcessor, AutoModelForImageTextToText

# processor = AutoProcessor.from_pretrained("./gemma3n_processor")
# model = AutoModelForImageTextToText.from_pretrained("./gemma3n_model")


# # Add <|image|> to tokenizer special tokens if not present
# if "<|image|>" not in processor.tokenizer.special_tokens_map.get("additional_special_tokens", []):
#     processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})
#     model.resize_token_embeddings(len(processor.tokenizer))  # Resize model embeddings

# prompt = "<|image|> Describe this image in one line."

# # 4. Preprocess input
# inputs = processor(images=image, text=prompt, return_tensors="pt")

# # 5. Generate output
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100)

# # 6. Decode result
# print(processor.batch_decode(outputs, skip_special_tokens=True)[0])



from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# Step 1: Load processor and model
processor = AutoProcessor.from_pretrained("./gemma3n_processor")
model = AutoModelForImageTextToText.from_pretrained("./gemma3n_model")

# Step 2: Add special token <|image|> if not present
special_token = "<|image|>"
tokenizer = processor.tokenizer

if special_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Special token added and model resized.")
    # Save updated processor and model
    tokenizer.save_pretrained("./gemma3n_processor")
    model.save_pretrained("./gemma3n_model")

# Step 3: Load image
image = Image.open("Blurr.jpg").convert("RGB")

# Step 4: Prompt
prompt = "<|image|> Describe this image in one line."

# Step 5: Preprocess
inputs = processor(images=image, text=prompt, return_tensors="pt")

# Step 6: Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Step 7: Decode result
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])

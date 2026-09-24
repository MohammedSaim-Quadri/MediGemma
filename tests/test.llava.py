import torch
from PIL import Image

from transformers import BitsAndBytesConfig

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates

model_path = "./LLaVA-Medical-Director"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model...")

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=False,
    load_8bit=False,
    quantization_config=quant,
    device_map="cuda",
)

print("Model loaded")

image = Image.open("phase3_training/datasets/azh_wound/dataset/Test/N/76.jpg").convert("RGB")

image_tensor = process_images(
    [image],
    image_processor,
    model.config,
)

if isinstance(image_tensor, list):
    image_tensor = [
        img.to(model.device, dtype=torch.float16)
        for img in image_tensor
    ]
else:
    image_tensor = image_tensor.to(
        model.device,
        dtype=torch.float16,
    )

prompt = "Describe the wound."

if model.config.mm_use_im_start_end:
    qs = (
        DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_TOKEN
        + DEFAULT_IM_END_TOKEN
        + "\n"
        + prompt
    )
else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

conv = conv_templates["vicuna_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)

prompt_str = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt_str,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(model.device)

print("Generating...")

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=128,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
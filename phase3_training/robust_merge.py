# phase3_training_robust_merge.py
import torch
import os
import argparse
from peft import PeftModel
from transformers import AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def merge_lora(base_model_path, adapter_path, save_path):
    print(f"1. Loading Base Model: {base_model_path}")
    # Load the base LLaVA model directly (bypassing the buggy builder.py)
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    print(f"2. Loading LoRA Adapter: {adapter_path}")
    # Apply your fine-tuned sticky notes to the base model
    model = PeftModel.from_pretrained(model, adapter_path)

    print("3. Merging Weights (This may take a moment)...")
    # Force the mathematical addition of weights
    model = model.merge_and_unload()

    print(f"4. Saving Final Model to: {save_path}")
    model.save_pretrained(save_path)
    
    # Don't forget the tokenizer!
    print("5. Saving Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    tokenizer.save_pretrained(save_path)
    
    print("✅ DONE! Your standalone model is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to your LoRA checkpoint")
    parser.add_argument("--base", type=str, default="liuhaotian/llava-v1.5-7b", help="Base model name")
    parser.add_argument("--save", type=str, required=True, help="Output folder")
    args = parser.parse_args()

    merge_lora(args.base, args.adapter, args.save)
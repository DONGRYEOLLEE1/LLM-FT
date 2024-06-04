import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parsing_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type = str, default = "..")
    parser.add_argument("--peft_model_path", type = str, default = "..")
    parser.add_argument("--output_dir", type = str, default = "..")
    parser.add_argument("--device", type = str, default = "cuda:1")
    args = parser.parse_args()
    
    return args


def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype = torch.bfloat16, device_map = args.device)
    print(f"*** Base Model Load in {args.device} ***")
    
    config = PeftConfig.from_pretrained(args.peft_model_path)
    model = PeftModel(model, config)
    print(f"*** Peft Model Load Completion ***")
    
    merged_model = model.merge_and_unload()
    
    print(f"*** Peft Model Converting... ***")
    tokenizer.save_pretrained(args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization = True)
    print(f"*** Save Completion ***")

if __name__ == "__main__":
    
    args = parsing_args()
    
    main(args)
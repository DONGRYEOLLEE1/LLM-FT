import os
import sys
import json
sys.path.append(os.path.abspath(".."))
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union
from datasets import load_dataset, concatenate_datasets, DatasetDict

from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOConfig, ORPOTrainer
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig
from get_dataset import DatasetLoader
from util import mkdir_p

_SYSTEM_PROMPT = """친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."""

_REMAIN_COLS = ["chosen", "rejected", "prompt"]

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default = None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_r: Optional[int] = field(default=64, metadata = {"help" : "lora R"})
    lora_alpha: Optional[int] = field(default = 16, metadata = {"help" : "lora alpha"})
    lora_dropout: Optional[float] = field(default = 0.05, metadata = {"help" : "lora dropout"})
    lora_target_modules: Union[List[str], str, None] = field(
        default="all-linear",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    orpo_beta: Optional[float] = field(
        default = 0.1,
        metadata = {"help": "ORPO Config BETA"}
    )
    peft_model_id: Optional[str] = field(
        default = None,
        metadata = {"help": "PEFT MODEL ID"}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    test_size: Optional[float] = field(
        default = None,
        metadata = {"help": "Train Test data slice ratio"}
    )
    max_length: Optional[int] = field(default = 2048)
    max_prompt_length: Optional[int] = field(default = 1024)


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # bnb config
    if model_args.use_4bit_quantization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)
        
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
    
    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        quantization_config = bnb_config,
        attn_implementation = "flash_attention_2" if model_args.use_flash_attn else "eager",
        torch_dtype = torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    peft_config = LoraConfig(
        r = model_args.lora_r,
        target_modules = model_args.lora_target_modules,
        lora_alpha = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout
    )
    model = get_peft_model(model, peft_config)
    
    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}


    def process(example):
        
        example['prompt'] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": example['system'] if example['system'] != "" else _SYSTEM_PROMPT},
                {"role": "user", "content": example['question']}
            ], tokenize = False
        )
        example['chosen'] = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "conetent": example['chosen']}
            ], tokenize = False
        )
        example['rejected'] = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "conetent": example['rejected']}
            ], tokenize = False
        )
        
        example['prompt'] = example['prompt'][0]
        
        return example
    
    def making_dataset():
        
        DATA_CONFIG = {"mncai/orca_dpo_pairs_ko": 1.0, "Ja-ck/Orca-DPO-Pairs-KO": 1.0}
        
        mixer = DatasetLoader(data_config = DATA_CONFIG, columns = ['system', 'question', 'chosen', 'rejected'])
        data = mixer.get_datasets(is_test = False, test_size = data_args.test_size, shuffle = True)
        data = data.map(
            process, 
            num_proc = os.cpu_count(),
            remove_columns = [
                col for col in data.column_names['train'] if col not in _REMAIN_COLS
            ])
        
        return data, mixer
    

    orpo_args = ORPOConfig(
        output_dir = training_args.output_dir,
        beta = model_args.orpo_beta,
        lr_scheduler_type = training_args.lr_scheduler_type,
        max_length = data_args.max_length,
        max_prompt_length = data_args.max_prompt_length,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        per_device_eval_batch_size = training_args.per_device_eval_batch_size,
        gradient_accumulation_steps = training_args.gradient_accumulation_steps,
        evaluation_strategy = training_args.evaluation_strategy,
        logging_steps = training_args.logging_steps,
        save_steps = training_args.save_steps,
        report_to = "wandb",
        run_name = training_args.run_name,
        bf16 = training_args.bf16,
        learning_rate = training_args.learning_rate,
        optim = "paged_adamw_8bit"
    )
    
    data, mixer = making_dataset()
    
    print("--------------------------- ORPO Arguments ---------------------------")
    print(orpo_args)
    print("--------------------------- --------------------------- ---------------------------")
    
    trainer = ORPOTrainer(
        model = model,
        tokenizer = tokenizer,
        args = orpo_args,
        train_dataset = data['train']
    )
    
    # save to data-collection
    mixer.data2json(training_args.output_dir + "/data-collection.json")
    
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
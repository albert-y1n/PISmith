import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .Model import Model


def _get_model_device(model):
    """获取模型所在的设备（支持 device_map="auto" 的情况）"""
    try:
        # 对于 device_map="auto" 的模型，获取第一个参数的设备
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_current_device_map():
    """获取当前进程应该使用的 device_map。
    
    在分布式训练中，device_map="auto" 会把模型分片到所有可见 GPU 上，
    导致 LoRA 层跨 GPU 计算时出现 device mismatch 错误。
    这里将模型限制在当前进程的 GPU 上（4-bit Mistral-7B ~4GB，单卡足够）。
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        # 优先使用 LOCAL_RANK（分布式训练中每个进程对应的 GPU）
        # 但要确保其在当前可见设备范围内，避免 LOCAL_RANK 泄露导致越界。
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            try:
                local_rank_int = int(local_rank)
                if 0 <= local_rank_int < device_count:
                    return {"": local_rank_int}
                print(
                    f"⚠️ LOCAL_RANK={local_rank_int} out of visible CUDA range "
                    f"[0, {max(device_count - 1, 0)}], fallback to device 0"
                )
                return {"": 0}
            except ValueError:
                print(f"⚠️ Invalid LOCAL_RANK={local_rank}, fallback to current CUDA device")
        # 否则使用当前 CUDA 设备；若不可用则回退到 0
        try:
            current_device = torch.cuda.current_device()
            if 0 <= current_device < device_count:
                return {"": current_device}
        except Exception:
            pass
        return {"": 0}
    return "auto"


class QLoraModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"]) # 128
        self.device = config["params"]["device"]

        self.base_model_id = config["model_info"]['name'] 
        self.ft_path = config["params"]['ft_path']

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        if "eval_only" not in config or not config["eval_only"]:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,  
                quantization_config=self.bnb_config, 
                device_map=_get_current_device_map(),
                trust_remote_code=True,
            )

            if 'phi2' in self.provider or 'phi-2' in self.base_model_id:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_id, 
                    add_bos_token=True, 
                    trust_remote_code=True,
                    use_fast=False
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_id, 
                    add_bos_token=True, 
                    trust_remote_code=True
                )
            if self.ft_path == '' or self.ft_path == 'base':
                self.ft_model = self.base_model
            else:
                try:
                    self.ft_model = PeftModel.from_pretrained(self.base_model, self.ft_path)#"mistral-7-1000-naive-original-finetune/checkpoint-5000")
                except ValueError:
                    raise ValueError(f"Bad ft path: {self.ft_path}")
            
            # 缓存模型设备
            self._model_device = _get_model_device(self.ft_model)
        else:
            self.ft_model = self.base_model = self.tokenizer = None
            self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| FT Path: {self.ft_path}\n{'-'*len(f'| Model name: {self.name}')}")

    def formatting_func(self, example):
        if isinstance(example, dict):
            input_split = example['input'].split('\nText: ')
        elif isinstance(example, str):
            input_split = example.split('\nText: ')
        else:
            raise ValueError(f'{type(example)} is not supported for querying Mistral')
        assert (len(input_split) == 2)
        text = f"### Instruction: {input_split[0]}\n### Text: {input_split[1]}"
        return text

    def query(self, msg):
        if self.ft_path == '' and 'DGDSGNH' not in msg:
            print('self.ft_model is None. Forward the query to the backend LLM')
            return self.backend_query(msg)
        
        processed_eval_prompt = self.formatting_func(msg)
        
        processed_eval_prompt = f'{processed_eval_prompt}\n### Response: '

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._model_device)

        self.ft_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.ft_model.generate(
                    **input_ids, 
                    max_new_tokens=10, 
                    repetition_penalty=1.2
                )[0], 
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output

    def query_localization(self, msg):
        if self.ft_path == '' and 'DGDSGNH' not in msg:
            print('self.ft_model is None. Forward the query to the backend LLM')
            return self.backend_query(msg)
        
        processed_eval_prompt = self.formatting_func(msg)
        
        processed_eval_prompt = f'{processed_eval_prompt}\n'

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._model_device)

        self.ft_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.ft_model.generate(
                    **input_ids, 
                    max_new_tokens=10, 
                    repetition_penalty=1.2,
                    do_sample=False,
                    temperature=0,
                    pad_token_id=self.tokenizer.eos_token_id
                )[0], 
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output
    
    def backend_query(self, msg):
        if '\nText: ' in msg or (isinstance(msg, dict) and '\nText: ' in msg['input']):
            if isinstance(msg, dict):
                input_split = msg['input'].split('\nText: ')
            elif isinstance(msg, str):
                input_split = msg.split('\nText: ')
            else:
                raise ValueError(f'{type(msg)} is not supported for querying Mistral')
            assert (len(input_split) == 2)

            processed_eval_prompt = f"{input_split[0]}\nText: {input_split[1]}.{self.tokenizer.eos_token}"
        
        else:
            processed_eval_prompt = f"{msg} {self.tokenizer.eos_token}"

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._model_device)

        self.base_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.base_model.generate(
                    **input_ids, 
                    max_new_tokens=self.max_output_tokens, 
                    repetition_penalty=1.2
                )[0], 
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output
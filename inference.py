from PIL import Image
import torch
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

from src.gemma_model import KVCache, PaliGemmaModel, PaliGemmaConfig
from src.paligemma_processor import PaligemmaProcessor


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaModel, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
    model = PaliGemmaModel(config).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(tensors, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.tie_weights()

    return (model, tokenizer)

def move_inputs_to_device(model_inputs: dict, device:str):
    model_inputs = {k:v.to(device) for k,v in model_inputs.items()}
    return model_inputs

def get_model_inputs(
            processor: PaligemmaProcessor, prompt:str, image_file_path: str, device: str
        ):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
  model : PaliGemmaModel,
  processor: PaligemmaProcessor,
  device : str,
  prompt : str,
  image_file_path : str,
  max_tokens_to_generate: int,
  temperature: float,
  top_p: float,
  do_sample: bool  
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    min_length = 20
    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    for step in range(max_tokens_to_generate):
        
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = output["kv_cache"]
        next_token_logits = output["logits"][:, -1, :]
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits/temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1,1)
        
        if next_token.item() == stop_token and step < min_length:
            print(f"Warning: Generated problematic token {next_token.item()}. Trying next best token.")
            # Set this token's probability to 0 and sample again
            next_token_logits[0, next_token.item()] = float('-inf')
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        next_token = next_token.squeeze(0)
        token_decoded = processor.tokenizer.decode([next_token.item()], skip_special_tokens=True)
        print(f"Step {step}: Generated token ID: {next_token.item()}, Decoded: '{token_decoded}'")
        generated_tokens.append(next_token)
        
        if next_token.item() == stop_token and step >= min_length:
            break
        
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1,1), device=input_ids.device)], dim=-1
        )
    
    generated_tokens  = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("output : ",prompt+decoded)
    

def _sample_top_p(probs: torch.Tensor, p:float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token =torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main(
    model_path: str = None,
    prompt:str = None,
    image_file_path:str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p : float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
    print("device : ", device)
    
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    print("loading model : ")
    
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaligemmaProcessor(tokenizer, num_image_tokens, image_size)
    
    with torch.no_grad():
        test_inference(model, processor, device, prompt, image_file_path, max_tokens_to_generate, temperature, top_p, do_sample) 
        
        
if __name__ == "__main__":
    MODEL_PATH="paligemma-weights/paligemma-3b-pt-224"
    PROMPT="Which instrument is used in this picture? explain the image. "
    IMAGE_FILE_PATH="test/test2.jpg"
    MAX_TOKENS_TO_GENERATE=100
    TEMPERATURE=0.8
    TOP_P=0.7
    DO_SAMPLE="False"
    ONLY_CPU="False"
    
    main(
        model_path = MODEL_PATH,
        prompt = PROMPT,
        image_file_path = IMAGE_FILE_PATH,
        max_tokens_to_generate = MAX_TOKENS_TO_GENERATE,
        temperature = TEMPERATURE,
        top_p = TOP_P,
        do_sample = DO_SAMPLE,
        only_cpu = ONLY_CPU,
    )
    
            
        
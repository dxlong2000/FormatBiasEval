from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch

MODEL_SAVE_PATH = '/some/local/path'

access_token = 'YOUR ACCESS TOKEN'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token = access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map= "auto" , torch_dtype=torch.bfloat16, token = access_token)

tokenizer.save_pretrained(MODEL_SAVE_PATH)
model.save_pretrained(MODEL_SAVE_PATH)


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4) #Change the number of GPUs

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
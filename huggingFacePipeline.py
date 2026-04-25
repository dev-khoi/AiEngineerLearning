from transformers import pipeline
from accelerate import Accelerator

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Use Accelerator to automatically detect an available accelerator for inference.

# Accelerator is a utility from the Hugging Face Hugging Face Accelerate
# that figures out and manages where your code should run.

# Core idea: 
# It automatically detects and handles:

# CPU
# GPU (CUDA)
# multiple GPUs
# sometimes TPUs (in supported setups)

# So you don’t manually write:

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = Accelerator().device

pipeline = pipeline("text-generation", model=MODEL_NAME, device=device)


result = pipeline("The secret to baking a good cake is ", max_new_tokens=50)

text = result[0]["generated_text"]

print(text)


#output: The secret to baking a good cake is 10% chemistry and 90% artistry.
#That’s according to the world-renowned chef, author, and food expert, Tom Colicchio. In his latest book, “How I Built This,” he describes how he uses science
print("\n", result) 

#output: {'generated_text': 'The secret to baking a good cake is 10% chemistry and 90% artistry.\nThat’s according to the world-renowned chef, author, and food expert, Tom Colicchio. In his latest book, “How I Built This,” he describes how he uses science'}]

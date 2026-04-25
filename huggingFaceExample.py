from transformers import AutoModelForCausalLM, AutoTokenizer

# lower level
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "The secret to baking a good cake is "

# tensor is a multi-dimensional array (1-d tensor, 2-d tensor, etc.), similar to numpy arrays but
# with additional capabilities for GPU acceleration and automatic differentiation,
#  which are essential for training and inference in deep learning models.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)  # raw text to tensors | returns A tokenizer class instance
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
# 1. AutoModel
# Automatically selects the correct model architecture based on MODEL_NAME
# You don’t need to manually specify (e.g., GPT, LLaMA, etc.)


model_inputs = tokenizer(PROMPT, return_tensors="pt").to(
    model.device
)  # Output includes: input_ids (token IDs), attention_mask
generated_ids = model.generate(
    **model_inputs, max_new_tokens=30
)  # It returns a tensor of token IDs (numbers), representing the full generated sequence.
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(result)

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "The secret to baking a good cake is "


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model_inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=30)
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(result)

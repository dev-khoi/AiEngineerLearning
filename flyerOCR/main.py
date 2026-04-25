# from transformers import pipeline

# # Use text-generation pipeline with a small chat model
# model = pipeline(
#     task="text-generation",
#     model="Qwen/Qwen2.5-1.5B-Instruct",  # small, free, no login required
# )

# # Use chat-style messages format
# messages = [{"role": "user", "content": "What is the capital of France?"}]

# response = model(messages, max_new_tokens=100)
# print(response[0]["generated_text"][-1]["content"])
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

# ── Single text-generation model replaces summarization + QA ──────────────────
gen_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=0,  # remove this line if you have no GPU
    max_new_tokens=512,
)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# ── Prompt templates ──────────────────────────────────────────────────────────
summary_template = PromptTemplate.from_template(
    "<|im_start|>user\nSummarize the following text in a {length} way:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
)

qa_template = PromptTemplate.from_template(
    "<|im_start|>user\nUsing only the context below, answer the question.\n\nContext: {context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
)

# ── Chains ────────────────────────────────────────────────────────────────────
summarization_chain = summary_template | llm | StrOutputParser()
qa_chain = qa_template | llm | StrOutputParser()

# ── Run ───────────────────────────────────────────────────────────────────────
text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    answer = qa_chain.invoke({"context": summary, "question": question})

    print("\n🔹 **Answer:**")
    print(answer)

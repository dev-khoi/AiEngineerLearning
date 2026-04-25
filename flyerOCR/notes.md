# just some notes from: https://huggingface.co/docs/transformers/pipeline_tutorial

Transformers has two pipeline classes, a generic Pipeline and many individual task-specific pipelines like TextGenerationPipeline.

1. Use pipeline() when the model is standard

Use Hugging Face Transformers pipeline() when:

The model follows a standard task format It is registered under a known task like: text-generation text-classification image-to-text Inputs/outputs are simple and standardized

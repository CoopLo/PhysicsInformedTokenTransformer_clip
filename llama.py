### Load model directly
##from transformers import AutoTokenizer, AutoModelForCausalLM
##
token = "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"
##
##tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", token=token)
##model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", token=token)
##
sentence = "We are simulating the 2D Heat equation."
##
###print(model.encode(sentence))
##
##from transformers import LlamaModel, LlamaConfig
##
### Initializing a LLaMA llama-7b style configuration
##configuration = LlamaConfig()
##
### Initializing a model from the llama-7b style configuration
##model = LlamaModel(configuration)
##
### Accessing the model configuration
##configuration = model.config

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('meta-llama/Llama-2-13b-hf', token=token)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token=token)
tokenizer.pad_token = tokenizer.eos_token
#embed_model = HuggingFaceEmbedding(model=model, model_name='meta-llama/Llama-2-13b-hf', tokenizer=tokenizer,
#                                   tokenizer_name='meta-llama/Llama-2-13b-hf')

print(type(tokenizer.encode(sentence)))
print(type(model))
print(model(tokenizer.encode(sentence)))

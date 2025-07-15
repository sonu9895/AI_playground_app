#######################################
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

#################################



### AI chat model
chat_model_1 = "Qwen/Qwen3-4B"
chat_model_2 = "openai-community/gpt2"
chat_model_3 = "meta-llama/Llama-3.1-8B-Instruct"


### Speech to Text model
speech_to_text_model = "openai/whisper-large-v3-turbo"


#Text to Speech model
speech_to_text_model = "suno/bark-small"
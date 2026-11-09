import os

# Tests never touch the network. Anything that needs a Hugging Face file (e.g. the Qwen
# tokenizer) must already be in the local cache, and is skipped otherwise.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

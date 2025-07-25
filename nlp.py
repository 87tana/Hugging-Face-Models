# -*- coding: utf-8 -*-
"""NLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l2csG5wwr4t7puJln-LIh_64u016YKQH
"""

# 1) Mount to Drive
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# 2) Upgrade Transformers & Hub
!pip install --quiet --upgrade transformers huggingface_hub

# 3)  Force offline mode so we never hit HF Hub again
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 4) Download the BlenderBot files into /content/models
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="facebook/blenderbot-400M-distill",
    cache_dir="/content/models",
    library_name="transformers"
)
print(f"Model directory: {model_path}")
print("Contents:", os.listdir(model_path))

# 5) Load the tokenizer & model from that local snapshot
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    local_files_only=True
)

from transformers import pipeline

chatbot = pipeline(
    "text2text-generation",
    model= model,
    tokenizer =tokenizer
)

user_message = """
What are some fun activities I can do in the winter?
"""

# Generate a response (you can tweak max_length or add num_return_sequences if you like)
output = chatbot(
    user_message,
    max_length=60,          # adjust based on how long you want the answer
    num_return_sequences=1  # how many alternate replies you want
)

# The pipeline returns a list of dicts; each dict has a "generated_text" field
reply = output[0]["generated_text"]

print("Bot reply:", reply)

# NEW, text2text-generation style
output = chatbot(
    "What else do you recommend?",
    max_length=60,            # tweak as needed
    num_return_sequences=1
)
reply = output[0]["generated_text"]
print(reply)

"""# LLM dont keep the memory of your previous messages.
# when we use transformer object we can append,
"""

# 1) start with your existing history in user_message
user_message = """User: What are some fun activities I can do in the winter?
Bot: I like to go skiing and snowboarding"""

# 2) your new turn
new_message = "What else do you recommend?"

# 3) append that turn to your history, ending with "Bot:" so the model knows to reply
user_message += f"\nUser: {new_message}\nBot:"

# 4) call the text2text pipeline on the full history
output = chatbot(
    user_message,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

# 5) grab the bot’s reply, and tack it back onto your history
reply = output[0]["generated_text"].strip()
user_message += f" {reply}"

print("Bot:", reply)
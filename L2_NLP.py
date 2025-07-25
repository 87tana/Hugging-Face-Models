#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: Natural Language Processing (NLP)

# - In the classroom, the libraries are already installed for you.
# - If you would like to run this code on your own machine, you can install the following:
# ```
#     !pip install transformers
# ```

# ### Build the `chatbot` pipeline using 🤗 Transformers Library

# - Here is some code that suppresses warning messages.

# In[3]:


from transformers.utils import logging
logging.set_verbosity_error()


# In[4]:


from transformers import pipeline


# - Define the conversation pipeline

# In[5]:


chatbot = pipeline(task="conversational",
                   model="./models/facebook/blenderbot-400M-distill")


# Info about ['blenderbot-400M-distill'](https://huggingface.co/facebook/blenderbot-400M-distill)

# In[6]:


user_message = """
What are some fun activities I can do in the winter?
"""


# In[7]:


from transformers import Conversation


# In[8]:


conversation = Conversation(user_message)


# In[9]:


print(conversation)


# In[10]:


conversation = chatbot(conversation)


# In[11]:


print(conversation)


# - You can continue the conversation with the chatbot with:
# ```
# print(chatbot(Conversation("What else do you recommend?")))
# ```
# - However, the chatbot may provide an unrelated response because it does not have memory of any prior conversations.
# 
# - To include prior conversations in the LLM's context, you can add a 'message' to include the previous chat history.

# In[12]:


conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })


# In[13]:


print(conversation)


# In[14]:


conversation = chatbot(conversation)

print(conversation)


# - [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
# - [LMSYS Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

# ### Try it yourself! 
# - Try chatting with the model!

# In[ ]:





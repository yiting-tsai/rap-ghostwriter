import streamlit as st
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Rap Ghostwriter',
    layout='wide')

#---------------------------------#
# Model loading
model = GPT2LMHeadModel.from_pretrained('model/out').to('cpu') # because its loaded on xla by default
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#---------------------------------#
st.write("""
# The Rap Ghostwriter App
The model is built with GPT-2 trained on top 20 popular song lyrics of each top 50 rap/hip-hop artists listed on BillBoard for last 10 years. Trained on TPUs via Pytorch/XLA for less than 30 mins.

In the following section, please input a word, a phrase or a paragraph as you wish, 
and also how long would you like the text to be?  
""")
st.write(":exclamation: Some ***profane words*** and ***racial slurs*** might be present in generated text.")

default_value_start_prompt="I'm too turned up"

start_prompt=st.text_area("Start prompt (a word, phrase, paragraph)", default_value_start_prompt)
max_len=int(round(st.text_input("label goes here", 250)))

inputs=tokenizer.encode(start_prompt, add_special_tokens=False, return_tensors="pt")
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=max_len, do_sample=True, top_p=0.95, top_k=60)
generated = tokenizer.decode(outputs[0])[prompt_length:]

if st.button('Write me some texts, Ghost!'):
    st.header('Text generated:')
    st.write(generated)
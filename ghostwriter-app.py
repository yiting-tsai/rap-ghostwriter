import streamlit as st
import urllib.request
from pathlib import Path
from transformers import TFGPT2LMHeadModel #, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
#import tensorflow as tf
#import gpt_2_simple as gpt2

#---------------------------------#
# Page layout
st.set_page_config(page_title='Rap Ghostwriter')

#---------------------------------#
# Model loading function to cache
## HuggingFace gpt-2
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    #url='https://github.com/yiting-tsai/rap-ghostwriter-app/releases/download/v2.0/pytorch_model.bin'
    #filename=url.split('/')[-1]
    #file_name, headers=urllib.request.urlretrieve(url, filename)
    drive_file_id='1XofyhhJLo4E2LE7PBfO5X6Zo__IuuBRc'

    with st.spinner("Downloading model .. this may be awhile! \n Don't quit!"):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        gdd.download_file_from_google_drive(file_id=drive_file_id,
                                    dest_path='./model/pytorch_model.bin',
                                    overwrite=True)
   
    config=GPT2Config.from_json_file('./model/config.json')      # local_files_only=True
    model=TFGPT2LMHeadModel.from_pretrained('./model/pytorch_model.bin', from_pt=True, config=config)#.to('cpu')
    return model, tokenizer
#---------------------------------#

st.write("""
# The Rap Ghostwriter App
The model is built with GPT-2 trained on top 30 popular song lyrics of each rap/hip-hop artist listed on [annual Top 50](https://www.billboard.com/charts/year-end/2019/top-r-and-b-hip-hop-artists) of BillBoard from last 10 years. 
Trained on TPUs via Pytorch/XLA for 20 epochs.
    
:exclamation: Explicit contents: some ***profane words*** and ***racial slurs*** might be present in generated text.

In the following section, please input a word, a phrase or a paragraph as you wish, 
and also how long would you like the text to be?
""")

default_value_start_prompt="""I've been pop, whippin', wrist is on another rhythm
I was not kiddin', don't know why they playin' with him
"""

start_prompt=st.text_area("Start prompt (a word, phrase, paragraph)", default_value_start_prompt,height=200)
max_len=st.text_input("Length for texts to be generated", 250)
max_len_int=int(max_len)

#---------------------------------#
# Sidebar
st.sidebar.header('Advanced: specify decoding params')
st.sidebar.write('This is for more advanced users, who have the notion of decoding methods of text generation, please consult this [blog on HuggingFace.](https://huggingface.co/blog/how-to-generate) Otherwise, leave default values as they are. :point_down:')
top_P=st.sidebar.slider('Top-p sampling', 0.0, 1.0, 0.92)
top_K=st.sidebar.slider('Top-k sampling', 1, max_len_int, 60)
temperaTure=st.sidebar.slider('Temperature (higher=crazier the text)', 0.0, 1.0, 1.0)
#---------------------------------#
#@st.cache(allow_output_mutation=True, show_spinner=False)
#def generate(inputs, max_len_int,top_P, top_K):
#    outputs=model.generate(input_ids=inputs, max_length=max_len_int, do_sample=True, top_p=top_P, top_k=top_K)
#    generated=tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
#    return generated
#---------------------------------#  

st.write(":bow: model takes some times to load, currently working on app performance improvement :construction_worker:")
model, tokenizer=load_model()
inputs=tokenizer.encode(start_prompt, return_tensors="tf") #add_special_tokens=False,

if st.button('Write me some texts, Ghost!'):
    st.write(":ghost: ghost might need a couple of minutes to write (hey, it's not easy for them to grab physical objects!) and once you reclick that button, previous generated texts would be gone :dash:")
    outputs=model.generate(input_ids=inputs, max_length=max_len_int, do_sample=True, top_p=top_P, top_k=top_K, temperature=temperaTure)
    generated1=tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    st.text_area('Text generated:',generated1,height=800)
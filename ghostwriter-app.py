import streamlit as st
#from transformers import GPT2LMHeadModel,TFGPT2LMHeadModel 
#from transformers import GPT2Tokenizer, GPT2Config
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import gpt_2_simple as gpt2

#---------------------------------#
# Page layout
st.set_page_config(page_title='Rap Ghostwriter')

#---------------------------------#
# Model loading function to cache
#@st.cache(show_spinner=False)
#def generate():
#    sess = gpt2.start_tf_sess()
#    gpt2.load_gpt2(sess, run_name='run1')
#    generated=gpt2.generate(sess,length=max_len_int, temperature=0.9, top_k=88, top_p=0.9, prefix=start_prompt, return_as_list=True)[0]
#    return generated
## HuggingFace gpt-2
    #config=GPT2Config.from_json_file('./model/out/config.json')      # local_files_only=True
    #model=TFGPT2LMHeadModel.from_pretrained('./model/out/pytorch_model.bin', from_pt=True, config=config, local_files_only=True).to('cpu')
    #model=GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path='./model/out/').to('cpu') # because its loaded on xla by default
    #tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    #return model, tokenizer
#---------------------------------#

st.write("""
# The Rap Ghostwriter App
The model is built with GPT-2 trained on top 20 popular song lyrics of each rap/hip-hop artist listed on [annual Top 50](https://www.billboard.com/charts/year-end/2019/top-r-and-b-hip-hop-artists) of BillBoard from last 10 years. 
Trained on TPUs via Pytorch/XLA for less than 30 mins.
    
:exclamation: Explicit contents: some ***profane words*** and ***racial slurs*** might be present in generated text.

In the following section, please input a word, a phrase or a paragraph as you wish, 
and also how long would you like the text to be?
""")

default_value_start_prompt="""I'm tired of being the one
'Cause I see the sunrise when it comes
In your face
A new woman, that's how I feel
It's all I see"""

start_prompt=st.text_area("Start prompt (a word, phrase, paragraph)", default_value_start_prompt,height=200)
max_len=st.text_input("Length for texts to be generated", 250)
max_len_int=int(max_len)

# inference
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')
generated=gpt2.generate(sess,length=max_len_int, temperature=0.9, top_k=88, top_p=0.9, prefix=start_prompt, return_as_list=True)[0]

## HuggingFace gpt-2
#model, tokenizer=load_model()
#inputs=tokenizer.encode(start_prompt, add_special_tokens=False, return_tensors="pt")
#prompt_length=len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
#outputs=model.generate(inputs, max_length=max_len_int, do_sample=True, top_p=0.95, top_k=60)
#generated=tokenizer.decode(outputs[0])

st.write(":ghost: ghost might need a couple of minutes to write (hey, it's not easy for them to grab physical objects!) and once you reclick that button beneath, previous generated texts would be gone :dash:")
#if st.button('Write me some texts, Ghost!'):
st.text_area('Text generated:',generated,height=800)
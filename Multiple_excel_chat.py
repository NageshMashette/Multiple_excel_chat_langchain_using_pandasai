import os
import openai
from langchain.chat_models import AzureChatOpenAI
API_KEY=""
RESOURCE_ENDPOINT= ""
OPENAI_DEPLOYMENT_NAME = ""
OPENAI_MODEL_NAME= ""
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = RESOURCE_ENDPOINT
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-03-15-preview"

import streamlit as st
from pandasai.llm.openai import OpenAI
from langchain.chat_models import AzureChatOpenAI
import os
import pandas as pd
from pandasai import SmartDataframe
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def chat_with_csv(df,prompt):
    llm = AzureChatOpenAI(
    deployment_name="gpt-35",
    model_name="gpt-35-turbo-16k",
    )
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(prompt)
    return result

st.set_page_config(layout='wide')
st.title("Excel Chat")
st.markdown('<style>h1{color: Black; text-align: left;}</style>', unsafe_allow_html=True)

# Upload multiple CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

if input_csvs:
    # Select a CSV file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index], encoding='latin1')
    st.dataframe(data,use_container_width=True)

    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    #Perform analysis
    if input_text:
        if st.button("Submit"):
            st.info("Your Query: "+ input_text)
            result = chat_with_csv(data,input_text)
            fig_number = plt.get_fignums()
            if fig_number:
                st.pyplot(plt.gcf())
            else:
                st.success(result)
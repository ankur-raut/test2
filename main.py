import streamlit as st
from PyPDF2 import PdfReader


def append_to_file(file_path, llm, api, text):
    try:
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(str(llm) + '^' + api + '^' + text + '^')
        print("Parameters appended to the file successfully.")
    except IOError:
        print("An error occurred while appending parameters to the file.")


def read_and_print_file(file):
    if uploaded_file.type == 'application/pdf':
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)

        text = ""
        for page in range(num_pages):
            text += pdf_reader.pages[page].extract_text()
        return text
    elif uploaded_file.type == 'text/plain':
        return file.read().decode('utf-8')
    else:
        return 'Unsupported file format. Only PDF and text files are accepted.'


# Page title
st.set_page_config(page_title='Chatbot - Private Docs')
st.title('Chatbot - Private Docs')

with st.form('story_form', clear_on_submit=False):
    uploaded_file = st.file_uploader("Choose a Text/PDF file", type=['pdf', 'txt'], accept_multiple_files=False)

    with st.sidebar:
        llm_name = st.radio('Select LLM type: ', ('OpenAI', 'Cohere'))
        api_key = st.text_input('LLM API Key', type='password')
        submitted = st.form_submit_button('Submit')

if (submitted and api_key.startswith('4a')) or (submitted and api_key.startswith('sk-') and uploaded_file != ''):
    with st.spinner('Calculating...'):
        append_to_file(r'pages/file.txt', llm_name, api_key, read_and_print_file(uploaded_file))
        st.markdown('<a href="/QA_Page" target="_self">Go to Chat -></a>', unsafe_allow_html=True)
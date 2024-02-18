import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from googleapiclient.discovery import build
import spacy
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


# Set page configuration
st.set_page_config(page_title="Sumquiry", page_icon=":robot:")

# Define OpenAI API key
openai_api_key = "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"  # Replace with your OpenAI API key

# Load English language model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Google Custom Search API key and search engine ID
API_KEY = 'AIzaSyA-S9UyBaXHXoWolmJlGrAdh0-qN-9_UTw'
SEARCH_ENGINE_ID = '246a2a6c987584fd8'

# Function to extract keywords from the summary
def extract_keywords(summary):
    # Process the summary text with spaCy
    doc = nlp(summary)
    
    # Extract nouns and proper nouns (entities) as keywords
    keywords = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}]
    
    return keywords

# Function to search for relevant links of articles
def search_related_links(keywords):
    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=" ".join(keywords), cx=SEARCH_ENGINE_ID).execute()
    return res['items']

# Main content
st.header("Sumquiry")

# Upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

# Process user choice after uploading file
if pdf is not None:
    user_choice = st.text_input("Enter your choice (summary, title, generate questions):")

    if user_choice:
        if user_choice.lower() == "summary":
            st.write("You chose to generate a summary.")
            # Process the PDF to generate summary and relevant links of articles
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            
            # Create docs
            docs = [Document(page_content=t) for t in chunks[:3]]
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

            # Show summarized doc
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summarized_docs = chain.run(docs)
            st.write("Summary")
            st.write(summarized_docs)

            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            knowledge_base = FAISS


            # Extract keywords from the summary
            keywords = extract_keywords(summarized_docs)

            # Search for relevant links of articles using the keywords
            links = search_related_links(keywords)

            # Display the links to the relevant articles
            if links:
                st.write("Related Links:")
                for link in links:
                    st.write(f"[{link['title']}]({link['link']})")
            else:
                st.write("No related links found.")

        elif user_choice.lower() == "title":
            st.write("You chose to display the title.")
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            
            # Create docs
            docs = [Document(page_content=t) for t in chunks[:3]]
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

            # Show summarized doc
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summarized_docs = chain.run(docs)

            titles = [page.split("\n")[0] for page in chunks[:1]]
            st.write("Title:")
            for title in titles:
                st.write(title)
            generated_text = "\n".join(titles)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            knowledge_base = FAISS


            # Extract keywords from the summary
            keywords = extract_keywords(summarized_docs)

            # Search for relevant links of articles using the keywords
            links = search_related_links(keywords)

            # Display the links to the relevant articles
            if links:
                st.write("Related Links:")
                for link in links:
                    st.write(f"[{link['title']}]({link['link']})")
            else:
                st.write("No related links found.")
            # Process the PDF to display the title (you can implement this logic)

        elif user_choice.lower() == "generate questions":
            st.write("You chose to generate questions.")
            questions = ["What is the main idea of the text?", "Who is the author?", "What are the key points discussed?", "Why is this topic important?"]
            st.write("Questions:")
            for question in questions:
                st.write(question)
            generated_text = "\n".join(questions)

            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            
            # Create docs
            docs = [Document(page_content=t) for t in chunks[:3]]
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

            # Show summarized doc
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summarized_docs = chain.run(docs)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            knowledge_base = FAISS


            # Extract keywords from the summary
            keywords = extract_keywords(summarized_docs)

            # Search for relevant links of articles using the keywords
            links = search_related_links(keywords)

            # Display the links to the relevant articles
            if links:
                st.write("Related Links:")
                for link in links:
                    st.write(f"[{link['title']}]({link['link']})")
            else:
                st.write("No related links found.")
            # Process the PDF to display the title (you can implement this logic)

            # Process the PDF to generate questions (you can implement this logic)

        else:
            st.write("Invalid choice. Please enter 'summary', 'title', or 'generate questions'.")

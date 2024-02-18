from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub 
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
import fitz
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from googleapiclient.discovery import build
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import spacy
from streamlit_option_menu import option_menu

nltk.download('punkt')

openai_api_key = "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"  # Replace with your OpenAI API key

# Load English language model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Google Custom Search API key and search engine ID
API_KEY = 'AIzaSyA-S9UyBaXHXoWolmJlGrAdh0-qN-9_UTw'
SEARCH_ENGINE_ID = '246a2a6c987584fd8'

st.set_page_config(page_title='GLOBAL NEWS SUMMARIZER 📰 Portal', page_icon='./Meta/newspaper.ico')

def fetch_news_search_topic(topic):
    # Replace spaces with %20 to form a valid URL
    topic = topic.replace(" ", "%20")
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open('./Meta/no_image.jpg')
        st.image(image, use_column_width=True)

def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        st.write('({}) {}'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
            continue  # Skip to the next article if an error occurs
        
        if news_data.top_image:  # Check if an image is available
            fetch_news_poster(news_data.top_image)
            with st.expander(news.title.text):
                st.markdown(
                    '''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                    unsafe_allow_html=True)
                st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
            st.success("Published Date: " + news.pubDate.text)
        
        if c >= news_quantity:
            break


def get_article_info(url, info_type):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        if info_type == "Summary":
            return article.summary
        elif info_type == "Title":
            return article.title
        elif info_type == "Publish Date":
            return article.publish_date
        else:
            return None
    except Exception as e:
        st.error(f"Error occurred while fetching information: {e}")
        return None


def get_related_news(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    keywords = article.keywords
    related_news = []
    for keyword in keywords:
        news_list = fetch_news_search_topic(keyword)
        if news_list:
            related_news.extend(news_list)
    return related_news

def RAG_chain(document, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(documents=document)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key='AIzaSyAOnfRr1T6Q9hKljdbXDi-P9qA2d2S7VNs')
    )
    retriever = vectorstore.as_retriever() 
    prompt = hub.pull('rlm/rag-prompt')
    llm = GoogleGenerativeAI(
        model='models/text-bison-001',
        google_api_key='AIzaSyAOnfRr1T6Q9hKljdbXDi-P9qA2d2S7VNs'
    )
    rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

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


def run():
    st.markdown("<h1 style='text-align: center;'>INSIGHT DIGEST 📰</h1>", unsafe_allow_html=True)
    image = Image.open('./Meta/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")

    with st.sidebar:
        selected=option_menu('CHOOSE THE OPTION : ',
                         ['Topic','URL','Pdf'],
                         icons=['activity','person'],
                         default_index=0)
        


    

    if selected == "URL":
        user_input = st.text_input("Enter the URL🔗")
        info_type = st.selectbox("Select Information Type:", ["Summary", "Title", "Publish Date"])

        if st.button("Search URL") and user_input != '':
            if info_type:
                info = get_article_info(user_input, info_type)
                if info:
                    st.subheader(f"✅ {info_type.capitalize()}")
                    st.write(info)
                else:
                    st.error("Failed to fetch information.")
                
                related_news = get_related_news(user_input)
                if related_news:
                    st.subheader("✅ Related News Articles")
                    display_news(related_news, 5)  # Display 5 related news articles
                else:
                    st.warning("No related news articles found.")
    
    elif selected == "Topic":
        user_input = st.text_input("Enter the Topic🔍")
        if st.button("Search Topic") and user_input != '':
            news_list = fetch_news_search_topic(topic=user_input)
            if news_list:
                st.subheader(f"✅ Here are some News related to {user_input.capitalize()}")
                display_news(news_list, 5)  # Display 5 news articles
            else:
                st.error("No News found for {}".format(user_input))
    
    elif selected=='Pdf':
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        # Process user choice after uploading file
        if pdf is not None:
            user_choice = st.text_input("Enter your choice (summary, title, generate questions):")

            if user_choice:
                if user_choice.lower() == "summary":
                    st.write("You choose to generate a summary.")
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

run()

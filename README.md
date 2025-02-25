# InsightDigest - Global Article Summarization and Link Analysis chatbot

The problem at hand revolves around addressing the challenge of information overload in the modern world. With an abundance of articles and updates available, it becomes increasingly difficult for individuals to identify and comprehend the most significant ones. Thus, the objective is to develop an interactive platform that utilizes advanced language processing techniques to filter articles based on user queries and provide concise, relevant summaries. Additionally, the platform should analyze relationships between articles to offer further insights to users.
Here we present InsightDigest to address this problem.
InsightDigest is an interactive platform designed to filter articles, generate summaries, and analyze relationships between articles based on user queries. It leverages advanced language processing techniques and a variety of libraries to provide concise and relevant information to users.

## Video-



https://github.com/lucky0612/InsightDigest/assets/145666325/526f5fa6-69e9-48ff-b84a-510ec0ac4553



## Features

- Filter articles based on user queries or abstracts.
- Generate concise summaries of filtered articles.
- Analyze relationships between articles to provide further insights.
- Interactive UI for user input and customization of summarization length.
- Offer multilingual translation capabilities to overcome language barriers and access information in different languages.

## Future Goals 
- Make the platform accessible to users withdiverse needs and abilities.
- Analyze the sentiment of the article to provide users with an understanding of the author's tone and potential biases.
- Integrate fact-checking mechanisms to ensure the accuracy and credibility of the information presented in the articles.
- Enhance the translation capabilities to provide summaries and insights in a broader range of languages.

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/your-username/ArticleSummarizer.git
```
2. Navigate to the project directory:
```bash
   cd InsightDigest
```
3. Run the application:
```bash
   streamlit run app.py
```

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import heapq
from collections import Counter, defaultdict
import string
import itertools
import textwrap
from gensim.summarization import summarize as gensim_summarize
from gensim.summarization import keywords as gensim_keywords
import networkx as nx

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing, text extraction, and embedding generation."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.document_sections = {}  # Will store sections of documents
        self.doc_embeddings = {}  # Will store document embeddings
        self.section_embeddings = {}  # Will store section embeddings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            max_df=0.85,         # Ignore terms that appear in >85% of docs
            min_df=2,            # Ignore terms that appear in <2 docs
        )
        self.count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from the knowledge base.
        
        Returns:
            Dictionary mapping document names to their content
        """
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.documents[file_name] = f.read()
                        
                elif file_extension == '.docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.documents[file_name] = content
                    
                elif file_extension == '.pdf':
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ''
                        for page_num in range(len(pdf_reader.pages)):
                            content += pdf_reader.pages[page_num].extract_text()
                        self.documents[file_name] = content
                
                else:
                    print(f"Unsupported file type: {file_extension} for file {file_name}")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                
        print(f"Loaded {len(self.documents)} documents from knowledge base")
        
        # Create document sections for more granular retrieval
        self._create_document_sections()
        
        return self.documents
    
    def _create_document_sections(self):
        """Create sections from the documents for more granular retrieval."""
        for doc_name, content in self.documents.items():
            # Split documents into paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            
            # Group paragraphs into coherent sections (at most 5 paragraphs per section)
            sections = []
            current_section = []
            section_length = 0
            
            for paragraph in paragraphs:
                # If paragraph is very short, combine with the previous one
                if len(paragraph.split()) < 10 and current_section:
                    current_section[-1] += " " + paragraph
                else:
                    current_section.append(paragraph)
                    section_length += 1
                
                # Start a new section if current one is getting too long
                if section_length >= 3:
                    sections.append('\n'.join(current_section))
                    current_section = []
                    section_length = 0
            
            # Add the last section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                
            # Store the sections for this document
            self.document_sections[doc_name] = sections
            
        print(f"Created {sum(len(sections) for sections in self.document_sections.values())} sections from {len(self.documents)} documents")
    
    def create_document_embeddings(self):
        """Create TF-IDF embeddings for all documents and their sections."""
        if not self.documents:
            self.load_all_documents()
            
        # Prepare document content for vectorization
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        # Fit and transform to get document embeddings
        tfidf_matrix = self.vectorizer.fit_transform(docs_content)
        
        # Store document embeddings with their names
        for i, doc_name in enumerate(doc_names):
            self.doc_embeddings[doc_name] = tfidf_matrix[i]
        
        # Create section embeddings
        section_texts = []
        section_ids = []
        
        for doc_name, sections in self.document_sections.items():
            for i, section in enumerate(sections):
                section_texts.append(section)
                section_ids.append((doc_name, i))
        
        if section_texts:
            # Transform the section texts using the already-fitted vectorizer
            section_matrix = self.vectorizer.transform(section_texts)
            
            # Store section embeddings
            for idx, (doc_name, section_idx) in enumerate(section_ids):
                self.section_embeddings[(doc_name, section_idx)] = section_matrix[idx]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(text)
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, similarity_score, content)
        """
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_sections(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        """
        Search for document sections relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, section_idx, similarity_score, content)
        """
        if not self.section_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all sections
        for (doc_name, section_idx), section_vector in self.section_embeddings.items():
            similarity = cosine_similarity(query_vector, section_vector)[0][0]
            content = self.document_sections[doc_name][section_idx]
            results.append((doc_name, section_idx, similarity, content))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def exact_keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """
        Search for exact keyword matches in documents.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of tuples (doc_name, relevant_context)
        """
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            # Check if keyword exists in the document
            if keyword_lower in content.lower():
                # Extract context around the keyword
                relevant_context = self.extract_context(content, keyword_lower)
                if relevant_context:
                    results.append((doc_name, relevant_context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 1000) -> str:
        """
        Extract context around a keyword in text.
        
        Args:
            text: Full text to extract from
            keyword: Keyword to find
            context_size: Number of characters to extract around the keyword
            
        Returns:
            Text with context around keyword
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
        
        # Extract contexts for all occurrences
        all_contexts = []
        for match_pos in matches[:3]:  # Limit to first 3 occurrences
            # Extract context around occurrence
            start_pos = max(0, match_pos - context_size // 2)
            end_pos = min(len(text), match_pos + len(keyword) + context_size // 2)
            
            # Find sentence boundaries if possible
            if start_pos > 0:
                # Try to start at the beginning of a sentence
                sentence_start = text.rfind('.', 0, start_pos)
                if sentence_start != -1:
                    start_pos = sentence_start + 1
            
            if end_pos < len(text):
                # Try to end at the end of a sentence
                sentence_end = text.find('.', end_pos)
                if sentence_end != -1 and sentence_end - end_pos < 200:  # Don't extend too far
                    end_pos = sentence_end + 1
            
            context = text[start_pos:end_pos].strip()
            all_contexts.append(context)
        
        # Combine all contexts
        return "\n\n".join(all_contexts)
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key phrases from text using gensim.
        
        Args:
            text: Text to analyze
            top_n: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        try:
            # Try using gensim's keyword extraction
            if len(text.split()) > 20:  # Only use for texts with sufficient length
                keywords = gensim_keywords(text, words=top_n, split=True)
                return keywords
        except Exception as e:
            print(f"Error using gensim keyword extraction: {str(e)}")
            
        # Fallback to spaCy-based extraction
        doc = nlp(text)
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            clean_chunk = ' '.join(token.lemma_ for token in chunk if not token.is_stop and not token.is_punct)
            if clean_chunk and len(clean_chunk.split()) <= 4:  # Limit to reasonable length phrases
                noun_phrases.append(clean_chunk)
        
        # Count phrase frequencies
        phrase_counts = Counter(noun_phrases)
        
        # Return most common phrases
        return [phrase for phrase, _ in phrase_counts.most_common(top_n)]


class Summarizer:
    """Class for text summarization using multiple techniques."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.nlp = nlp
        self.stopwords = STOPWORDS
    
    def textrank_summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Summarize text using TextRank algorithm.
        
        Args:
            text: Text to summarize
            ratio: Proportion of sentences to keep
            
        Returns:
            Summarized text
        """
        try:
            # Try gensim's implementation first
            if len(text.split()) > 100:  # Need sufficient text
                summary = gensim_summarize(text, ratio=ratio)
                if summary:
                    return summary
        except Exception:
            pass
            
        # Fallback to manual TextRank implementation
        return self._textrank_summarize_manual(text, ratio)
    
    def _textrank_summarize_manual(self, text: str, ratio: float = 0.3) -> str:
        """
        Manual implementation of TextRank summarization.
        
        Args:
            text: Text to summarize
            ratio: Proportion of sentences to keep
            
        Returns:
            Summarized text
        """
        # Extract sentences and remove short ones
        sentences = nltk.sent_tokenize(text)
        filtered_sentences = [s for s in sentences if len(s.split()) > 5]
        
        if not filtered_sentences:
            return text  # Can't summarize
            
        # Create similarity matrix
        similarity_matrix = np.zeros((len(filtered_sentences), len(filtered_sentences)))
        
        for i in range(len(filtered_sentences)):
            for j in range(len(filtered_sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(filtered_sentences[i], filtered_sentences[j])
        
        # Create graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Select top sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(filtered_sentences)), reverse=True)
        
        # Calculate how many sentences to keep
        select_count = max(1, int(len(filtered_sentences) * ratio))
        
        # Get top sentences
        top_sentences = [s for _, s in ranked_sentences[:select_count]]
        
        # Reorder sentences to maintain document flow
        ordered_top_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_top_sentences.append(sentence)
                
        # Return summarized text
        return " ".join(ordered_top_sentences)
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate similarity between two sentences.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Similarity score
        """
        # Process sentences to get clean tokens
        words1 = [word.lower() for word in sent1.split() if word.lower() not in self.stopwords]
        words2 = [word.lower() for word in sent2.split() if word.lower() not in self.stopwords]
        
        # Create word sets
        words1_set = set(words1)
        words2_set = set(words2)
        
        # Find common words
        common_words = words1_set.intersection(words2_set)
        
        # Calculate Jaccard similarity
        if not words1_set or not words2_set:
            return 0.0
            
        return len(common_words) / (len(words1_set) + len(words2_set) - len(common_words))
    
    def topic_based_summarize(self, sections: List[str], query: str, max_length: int = 300) -> str:
        """
        Create a topic-based summary focused on the query.
        
        Args:
            sections: List of text sections to summarize
            query: User query to focus the summary on
            max_length: Maximum word count for the summary
            
        Returns:
            Topic-focused summary
        """
        # Process query to get key terms
        query_terms = [term.lower() for term in query.split() if term.lower() not in self.stopwords]
        
        # Process each section
        all_sentences = []
        for section in sections:
            sentences = nltk.sent_tokenize(section)
            all_sentences.extend(sentences)
        
        # Score sentences based on query relevance
        scored_sentences = []
        for sentence in all_sentences:
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            # Count query terms in sentence
            sentence_lower = sentence.lower()
            term_count = sum(1 for term in query_terms if term in sentence_lower)
            
            # Score includes term frequency and sentence position
            score = term_count / max(1, len(query_terms))
            scored_sentences.append((score, sentence))
        
        # Sort by score
        scored_sentences.sort(reverse=True)
        
        # Take top sentences
        top_sentences = [s for _, s in scored_sentences[:10]]
        
        # Also include some TextRank sentences to ensure overall coverage
        textrank_summary = self.textrank_summarize("\n".join(sections), ratio=0.2)
        textrank_sentences = nltk.sent_tokenize(textrank_summary)
        
        # Com

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import heapq
from collections import Counter, defaultdict
import string
import itertools
import textwrap

# For Sumy-based summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing, text extraction, and embedding generation."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.document_sections = {}  # Will store sections of documents
        self.doc_embeddings = {}  # Will store document embeddings
        self.section_embeddings = {}  # Will store section embeddings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            max_df=0.85,         # Ignore terms that appear in >85% of docs
            min_df=2,            # Ignore terms that appear in <2 docs
        )
        self.count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from the knowledge base.
        
        Returns:
            Dictionary mapping document names to their content
        """
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.documents[file_name] = f.read()
                        
                elif file_extension == '.docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.documents[file_name] = content
                    
                elif file_extension == '.pdf':
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ''
                        for page_num in range(len(pdf_reader.pages)):
                            content += pdf_reader.pages[page_num].extract_text()
                        self.documents[file_name] = content
                
                else:
                    print(f"Unsupported file type: {file_extension} for file {file_name}")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                
        print(f"Loaded {len(self.documents)} documents from knowledge base")
        
        # Create document sections for more granular retrieval
        self._create_document_sections()
        
        return self.documents
    
    def _create_document_sections(self):
        """Create sections from the documents for more granular retrieval."""
        for doc_name, content in self.documents.items():
            # Split documents into paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            
            # Group paragraphs into coherent sections (at most 5 paragraphs per section)
            sections = []
            current_section = []
            section_length = 0
            
            for paragraph in paragraphs:
                # If paragraph is very short, combine with the previous one
                if len(paragraph.split()) < 10 and current_section:
                    current_section[-1] += " " + paragraph
                else:
                    current_section.append(paragraph)
                    section_length += 1
                
                # Start a new section if current one is getting too long
                if section_length >= 3:
                    sections.append('\n'.join(current_section))
                    current_section = []
                    section_length = 0
            
            # Add the last section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                
            # Store the sections for this document
            self.document_sections[doc_name] = sections
            
        print(f"Created {sum(len(sections) for sections in self.document_sections.values())} sections from {len(self.documents)} documents")
    
    def create_document_embeddings(self):
        """Create TF-IDF embeddings for all documents and their sections."""
        if not self.documents:
            self.load_all_documents()
            
        # Prepare document content for vectorization
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        # Fit and transform to get document embeddings
        tfidf_matrix = self.vectorizer.fit_transform(docs_content)
        
        # Store document embeddings with their names
        for i, doc_name in enumerate(doc_names):
            self.doc_embeddings[doc_name] = tfidf_matrix[i]
        
        # Create section embeddings
        section_texts = []
        section_ids = []
        
        for doc_name, sections in self.document_sections.items():
            for i, section in enumerate(sections):
                section_texts.append(section)
                section_ids.append((doc_name, i))
        
        if section_texts:
            # Transform the section texts using the already-fitted vectorizer
            section_matrix = self.vectorizer.transform(section_texts)
            
            # Store section embeddings
            for idx, (doc_name, section_idx) in enumerate(section_ids):
                self.section_embeddings[(doc_name, section_idx)] = section_matrix[idx]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(text)
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, similarity_score, content)
        """
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_sections(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        """
        Search for document sections relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, section_idx, similarity_score, content)
        """
        if not self.section_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all sections
        for (doc_name, section_idx), section_vector in self.section_embeddings.items():
            similarity = cosine_similarity(query_vector, section_vector)[0][0]
            content = self.document_sections[doc_name][section_idx]
            results.append((doc_name, section_idx, similarity, content))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def exact_keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """
        Search for exact keyword matches in documents.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of tuples (doc_name, relevant_context)
        """
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            # Check if keyword exists in the document
            if keyword_lower in content.lower():
                # Extract context around the keyword
                relevant_context = self.extract_context(content, keyword_lower)
                if relevant_context:
                    results.append((doc_name, relevant_context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 1000) -> str:
        """
        Extract context around a keyword in text.
        
        Args:
            text: Full text to extract from
            keyword: Keyword to find
            context_size: Number of characters to extract around the keyword
            
        Returns:
            Text with context around keyword
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
        
        # Extract contexts for all occurrences
        all_contexts = []
        for match_pos in matches[:3]:  # Limit to first 3 occurrences
            # Extract context around occurrence
            start_pos = max(0, match_pos - context_size // 2)
            end_pos = min(len(text), match_pos + len(keyword) + context_size // 2)
            
            # Find sentence boundaries if possible
            if start_pos > 0:
                # Try to start at the beginning of a sentence
                sentence_start = text.rfind('.', 0, start_pos)
                if sentence_start != -1:
                    start_pos = sentence_start + 1
            
            if end_pos < len(text):
                # Try to end at the end of a sentence
                sentence_end = text.find('.', end_pos)
                if sentence_end != -1 and sentence_end - end_pos < 200:  # Don't extend too far
                    end_pos = sentence_end + 1
            
            context = text[start_pos:end_pos].strip()
            all_contexts.append(context)
        
        # Combine all contexts
        return "\n\n".join(all_contexts)
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key phrases from text using spaCy.
        
        Args:
            text: Text to analyze
            top_n: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        # Process with spaCy
        doc = nlp(text)
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            clean_chunk = ' '.join(token.lemma_ for token in chunk if not token.is_stop and not token.is_punct)
            if clean_chunk and len(clean_chunk.split()) <= 4:  # Limit to reasonable length phrases
                noun_phrases.append(clean_chunk)
        
        # Count phrase frequencies
        phrase_counts = Counter(noun_phrases)
        
        # Extract individual keywords as well
        keywords = [token.lemma_ for token in doc 
                   if token.is_alpha and not token.is_stop and token.pos_ in ('NOUN', 'PROPN')]
        keyword_counts = Counter(keywords)
        
        # Combine noun phrases and keywords, prioritizing phrases
        combined_counts = Counter()
        for phrase, count in phrase_counts.items():
            combined_counts[phrase] = count * 3  # Weight phrases higher
            
        for keyword, count in keyword_counts.items():
            if keyword not in combined_counts:
                combined_counts[keyword] = count
        
        # Return most common terms
        return [phrase for phrase, _ in combined_counts.most_common(top_n)]


class Summarizer:
    """Class for text summarization using Sumy library."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.language = "english"
        self.stemmer = Stemmer(self.language)
        self.stop_words = get_stop_words(self.language)
        
        # Initialize summarizers
        self.lexrank_summarizer = LexRankSummarizer(self.stemmer)
        self.lexrank_summarizer.stop_words = self.stop_words
        
        self.lsa_summarizer = LsaSummarizer(self.stemmer)
        self.lsa_summarizer.stop_words = self.stop_words
        
        self.luhn_summarizer = LuhnSummarizer(self.stemmer)
        self.luhn_summarizer.stop_words = self.stop_words
    
    def summarize_text(self, text: str, sentences_count: int = 10, method: str = 'lexrank') -> str:
        """
        Summarize text using the specified method.
        
        Args:
            text: Text to summarize
            sentences_count: Number of sentences to include in summary
            method: Summarization method ('lexrank', 'lsa', or 'luhn')
            
        Returns:
            Summarized text
        """
        if not text or len(text.strip()) < 100:
            return text
            
        # Create parser
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Select summarizer
        if method == 'lsa':
            summarizer = self.lsa_summarizer
        elif method == 'luhn':
            summarizer = self.luhn_summarizer
        else:  # default to lexrank
            summarizer = self.lexrank_summarizer
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentences_count)
        
        # Join sentences into a coherent text
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        
        return summary
    
    def multi_method_summarize(self, text: str, sentences_count: int = 10) -> str:
        """
        Combine multiple summarization methods for better results.
        
        Args:
            text: Text to summarize
            sentences_count: Total number of sentences to include
            
        Returns:
            Combined summary
        """
        # Make sure we have enough text to summarize
        if not text or len(text.strip()) < 100:
            return text
            
        # Create parser
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Get sentences from each method
        lexrank_sentences = set(str(s) for s in self.lexrank_summarizer(parser.document, sentences_count // 2))
        lsa_sentences = set(str(s) for s in self.lsa_summarizer(parser.document, sentences_count // 2))
        luhn_sentences = set(str(s) for s in self.luhn_summarizer(parser.document, sentences_count // 3))
        
        # Combine sentences, prioritizing those that appear in multiple methods
        all_sentences = []
        
        # First add sentences that appear in all methods
        for sentence in lexrank_sentences & lsa_sentences & luhn_sentences:
            all_sentences.append(sentence)
        
        # Then add sentences that appear in two methods
        for sentence in (lexrank_sentences & lsa_sentences) - set(all_sentences):
            all_sentences.append(sentence)
        
        for sentence in (lexrank_sentences & luhn_sentences) - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in (lsa_sentences & luhn_sentences) - set(all_sentences):
            all_sentences.append(sentence)
        
        # Then add remaining unique sentences from each method
        for sentence in lexrank_sentences - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in lsa_sentences - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in luhn_sentences - set(all_sentences):
            all_sentences.append(sentence)
        
        # Limit to desired number of sentences
        all_sentences = all_sentences[:sentences_count]
        
        # Original sentences from the document to preserve order
        original_sentences = nltk.sent_tokenize(text)
        
        # Reorder sentences to maintain flow
        ordered_sentences = [s for s in original_sentences if s in all_sentences]
        
        # Return summary
        return " ".join(ordered_sentences)
    
    def query_focused_summarize(self, text: str, query: str, sentences_count: int = 10) -> str:
        """
        Create a summary focused on the query terms.
        
        Args:
            text: Text to summarize
            query: Query to focus on
            sentences_count: Number of sentences to include
            
        Returns:
            Query-focused summary
        """
        # Process query to get key terms
        query_doc = nlp(query.lower())
        query_terms = [token.lemma_ for token in query_doc 
                      if not token.is_stop and token.is_alpha]
        
        # If n

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import heapq
from collections import Counter, defaultdict
import string
import itertools
import textwrap

# For Sumy-based summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing, text extraction, and embedding generation."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.document_sections = {}  # Will store sections of documents
        self.doc_embeddings = {}  # Will store document embeddings
        self.section_embeddings = {}  # Will store section embeddings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            max_df=0.85,         # Ignore terms that appear in >85% of docs
            min_df=2,            # Ignore terms that appear in <2 docs
        )
        self.count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from the knowledge base.
        
        Returns:
            Dictionary mapping document names to their content
        """
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.documents[file_name] = f.read()
                        
                elif file_extension == '.docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.documents[file_name] = content
                    
                elif file_extension == '.pdf':
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ''
                        for page_num in range(len(pdf_reader.pages)):
                            content += pdf_reader.pages[page_num].extract_text()
                        self.documents[file_name] = content
                
                else:
                    print(f"Unsupported file type: {file_extension} for file {file_name}")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                
        print(f"Loaded {len(self.documents)} documents from knowledge base")
        
        # Create document sections for more granular retrieval
        self._create_document_sections()
        
        return self.documents
    
    def _create_document_sections(self):
        """Create sections from the documents for more granular retrieval."""
        for doc_name, content in self.documents.items():
            # Split documents into paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            
            # Group paragraphs into coherent sections (at most 5 paragraphs per section)
            sections = []
            current_section = []
            section_length = 0
            
            for paragraph in paragraphs:
                # If paragraph is very short, combine with the previous one
                if len(paragraph.split()) < 10 and current_section:
                    current_section[-1] += " " + paragraph
                else:
                    current_section.append(paragraph)
                    section_length += 1
                
                # Start a new section if current one is getting too long
                if section_length >= 3:
                    sections.append('\n'.join(current_section))
                    current_section = []
                    section_length = 0
            
            # Add the last section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                
            # Store the sections for this document
            self.document_sections[doc_name] = sections
            
        print(f"Created {sum(len(sections) for sections in self.document_sections.values())} sections from {len(self.documents)} documents")
    
    def create_document_embeddings(self):
        """Create TF-IDF embeddings for all documents and their sections."""
        if not self.documents:
            self.load_all_documents()
            
        # Prepare document content for vectorization
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        # Fit and transform to get document embeddings
        tfidf_matrix = self.vectorizer.fit_transform(docs_content)
        
        # Store document embeddings with their names
        for i, doc_name in enumerate(doc_names):
            self.doc_embeddings[doc_name] = tfidf_matrix[i]
        
        # Create section embeddings
        section_texts = []
        section_ids = []
        
        for doc_name, sections in self.document_sections.items():
            for i, section in enumerate(sections):
                section_texts.append(section)
                section_ids.append((doc_name, i))
        
        if section_texts:
            # Transform the section texts using the already-fitted vectorizer
            section_matrix = self.vectorizer.transform(section_texts)
            
            # Store section embeddings
            for idx, (doc_name, section_idx) in enumerate(section_ids):
                self.section_embeddings[(doc_name, section_idx)] = section_matrix[idx]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(text)
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, similarity_score, content)
        """
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_sections(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        """
        Search for document sections relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, section_idx, similarity_score, content)
        """
        if not self.section_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all sections
        for (doc_name, section_idx), section_vector in self.section_embeddings.items():
            similarity = cosine_similarity(query_vector, section_vector)[0][0]
            content = self.document_sections[doc_name][section_idx]
            results.append((doc_name, section_idx, similarity, content))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def exact_keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """
        Search for exact keyword matches in documents.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of tuples (doc_name, relevant_context)
        """
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            # Check if keyword exists in the document
            if keyword_lower in content.lower():
                # Extract context around the keyword
                relevant_context = self.extract_context(content, keyword_lower)
                if relevant_context:
                    results.append((doc_name, relevant_context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 1000) -> str:
        """
        Extract context around a keyword in text.
        
        Args:
            text: Full text to extract from
            keyword: Keyword to find
            context_size: Number of characters to extract around the keyword
            
        Returns:
            Text with context around keyword
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
        
        # Extract contexts for all occurrences
        all_contexts = []
        for match_pos in matches[:3]:  # Limit to first 3 occurrences
            # Extract context around occurrence
            start_pos = max(0, match_pos - context_size // 2)
            end_pos = min(len(text), match_pos + len(keyword) + context_size // 2)
            
            # Find sentence boundaries if possible
            if start_pos > 0:
                # Try to start at the beginning of a sentence
                sentence_start = text.rfind('.', 0, start_pos)
                if sentence_start != -1:
                    start_pos = sentence_start + 1
            
            if end_pos < len(text):
                # Try to end at the end of a sentence
                sentence_end = text.find('.', end_pos)
                if sentence_end != -1 and sentence_end - end_pos < 200:  # Don't extend too far
                    end_pos = sentence_end + 1
            
            context = text[start_pos:end_pos].strip()
            all_contexts.append(context)
        
        # Combine all contexts
        return "\n\n".join(all_contexts)
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key phrases from text using spaCy.
        
        Args:
            text: Text to analyze
            top_n: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        # Process with spaCy
        doc = nlp(text)
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            clean_chunk = ' '.join(token.lemma_ for token in chunk if not token.is_stop and not token.is_punct)
            if clean_chunk and len(clean_chunk.split()) <= 4:  # Limit to reasonable length phrases
                noun_phrases.append(clean_chunk)
        
        # Count phrase frequencies
        phrase_counts = Counter(noun_phrases)
        
        # Extract individual keywords as well
        keywords = [token.lemma_ for token in doc 
                   if token.is_alpha and not token.is_stop and token.pos_ in ('NOUN', 'PROPN')]
        keyword_counts = Counter(keywords)
        
        # Combine noun phrases and keywords, prioritizing phrases
        combined_counts = Counter()
        for phrase, count in phrase_counts.items():
            combined_counts[phrase] = count * 3  # Weight phrases higher
            
        for keyword, count in keyword_counts.items():
            if keyword not in combined_counts:
                combined_counts[keyword] = count
        
        # Return most common terms
        return [phrase for phrase, _ in combined_counts.most_common(top_n)]


class Summarizer:
    """Class for text summarization using Sumy library."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.language = "english"
        self.stemmer = Stemmer(self.language)
        self.stop_words = get_stop_words(self.language)
        
        # Initialize summarizers
        self.lexrank_summarizer = LexRankSummarizer(self.stemmer)
        self.lexrank_summarizer.stop_words = self.stop_words
        
        self.lsa_summarizer = LsaSummarizer(self.stemmer)
        self.lsa_summarizer.stop_words = self.stop_words
        
        self.luhn_summarizer = LuhnSummarizer(self.stemmer)
        self.luhn_summarizer.stop_words = self.stop_words
    
    def summarize_text(self, text: str, sentences_count: int = 10, method: str = 'lexrank') -> str:
        """
        Summarize text using the specified method.
        
        Args:
            text: Text to summarize
            sentences_count: Number of sentences to include in summary
            method: Summarization method ('lexrank', 'lsa', or 'luhn')
            
        Returns:
            Summarized text
        """
        if not text or len(text.strip()) < 100:
            return text
            
        # Create parser
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Select summarizer
        if method == 'lsa':
            summarizer = self.lsa_summarizer
        elif method == 'luhn':
            summarizer = self.luhn_summarizer
        else:  # default to lexrank
            summarizer = self.lexrank_summarizer
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentences_count)
        
        # Join sentences into a coherent text
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        
        return summary
    
    def multi_method_summarize(self, text: str, sentences_count: int = 10) -> str:
        """
        Combine multiple summarization methods for better results.
        
        Args:
            text: Text to summarize
            sentences_count: Total number of sentences to include
            
        Returns:
            Combined summary
        """
        # Make sure we have enough text to summarize
        if not text or len(text.strip()) < 100:
            return text
            
        # Create parser
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Get sentences from each method
        lexrank_sentences = set(str(s) for s in self.lexrank_summarizer(parser.document, sentences_count // 2))
        lsa_sentences = set(str(s) for s in self.lsa_summarizer(parser.document, sentences_count // 2))
        luhn_sentences = set(str(s) for s in self.luhn_summarizer(parser.document, sentences_count // 3))
        
        # Combine sentences, prioritizing those that appear in multiple methods
        all_sentences = []
        
        # First add sentences that appear in all methods
        for sentence in lexrank_sentences & lsa_sentences & luhn_sentences:
            all_sentences.append(sentence)
        
        # Then add sentences that appear in two methods
        for sentence in (lexrank_sentences & lsa_sentences) - set(all_sentences):
            all_sentences.append(sentence)
        
        for sentence in (lexrank_sentences & luhn_sentences) - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in (lsa_sentences & luhn_sentences) - set(all_sentences):
            all_sentences.append(sentence)
        
        # Then add remaining unique sentences from each method
        for sentence in lexrank_sentences - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in lsa_sentences - set(all_sentences):
            all_sentences.append(sentence)
            
        for sentence in luhn_sentences - set(all_sentences):
            all_sentences.append(sentence)
        
        # Limit to desired number of sentences
        all_sentences = all_sentences[:sentences_count]
        
        # Original sentences from the document to preserve order
        original_sentences = nltk.sent_tokenize(text)
        
        # Reorder sentences to maintain flow
        ordered_sentences = [s for s in original_sentences if s in all_sentences]
        
        # Return summary
        return " ".join(ordered_sentences)
    
    def query_focused_summarize(self, text: str, query: str, sentences_count: int = 10) -> str:
        """
        Create a summary focused on the query terms.
        
        Args:
            text: Text to summarize
            query: Query to focus on
            sentences_count: Number of sentences to include
            
        Returns:
            Query-focused summary
        """
        # Process query to get key terms
        query_doc = nlp(query.lower())
        query_terms = [token.lemma_ for token in query_doc 
                      if not token.is_stop and token.is_alpha]
        
        # If no meaningful query terms, fall back to general summarization
        if not query_terms:
            return self.multi_method_summarize(text, sentences_count)
            
        # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Score sentences based on query term presence
        scored_sentences = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Process sentence
            sent_doc = nlp(sentence.lower())
            sent_terms = [token.lemma_ for token in sent_doc if token.is_alpha]
            
            # Count query term matches
            matches = sum(1 for term in query_terms if term in sent_terms)
            
            # Calculate score (query match ratio + position boost for early sentences)
            score = matches / max(1, len(query_terms))
            
            # Position boost for the first few sentences
            position = sentences.index(sentence)
            if position < 3:
                score += (3 - position) * 0.1
                
            scored_sentences.append((score, sentence))
        
        # Sort by score
        scored_sentences.sort(reverse=True)
        
        # Select top sentences
        query_focused_sentences = [s for _, s in scored_sentences[:sentences_count]]
        
        # Get general summary sentences too
        general_summary = self.summarize_text(text, sentences_count // 2)
        general_sentences = nltk.sent_tokenize(general_summary)
        
        # Combine query-focused and general sentences
        combined_sentences = list(query_focused_sentences)
        for sentence in general_sentences:
            if sentence not in combined_sentences:
                combined_sentences.append(sentence)
        
        # Reorder to maintain document flow
        ordered_sentences = []
        for sentence in sentences:
            if sentence in combined_sentences:
                ordered_sentences.append(sentence)
                if len(ordered_sentences) >= sentences_count:
                    break
        
        # Join into coherent text
        return " ".join(ordered_sentences)
    
    def structure_summary(self, summary: str, key_phrases: List[str] = None) -> str:
        """
        Structure the summary into a more readable format with sections.
        
        Args:
            summary: Raw summary text
            key_phrases: Key phrases to highlight
            
        Returns:
            Structured summary
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(summary)
        
        if not sentences:
            return summary
            
        # Identify the introduction (first 1-2 sentences)
        intro_count = min(2, len(sentences) // 4)
        intro = " ".join(sentences[:intro_count])
        
        # Identify the conclusion (last 1-2 sentences)
        conclusion_count = min(2, len(sentences) // 4)
        conclusion = " ".join(sentences[-conclusion_count:])
        
        # Main content (everything in between)
        main_content = sentences[intro_count:-conclusion_count] if conclusion_count > 0 else sentences[intro_count:]
        
        # Group main content into paragraphs (roughly 3-4 sentences per paragraph)
        paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(main_content):
            current_paragraph.append(sentence)
            
            # Start a new paragraph every few sentences
            if (i + 1) % 3 == 0 and i < len(main_content) - 1:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        
        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        # Format the structured summary
        structured_summary = f"{intro}\n\n"
        structured_summary += "\n\n".join(paragraphs)
        if conclusion:
            structured_summary += f"\n\n{conclusion}"
            
        return structured_summary


class RAGSystem:
    """Implements the Retrieval-Augmented Generation system with Sumy-based summarization."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.summarizer = Summarizer()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with response information
        """
        print(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Check if query is a simple keyword or a complete question
        is_question = self._is_question(query)
        print(f"Query identified as {'question' if is_question else 'keyword search'}")
        
        # Different retrieval strategies based on query type
        if is_question:
            # For questions, search both full documents and sections
            doc_results = self.document_processor.search_documents(query, top_k=2)
            section_results = self.document_processor.search_sections(query, top_k=5)
            
            # Combine document and section results
            retrieved_docs = [(doc_name, content) for doc_name, _, content in doc_results]
            for doc_name, _, _, content in section_results:
                # Avoid duplicating entire documents
                if not any(doc_name == d_name for d_name, _ in retrieved_docs):
                    retrieved_docs.append((doc_name, content))
        else:
            # For keywords, use exact matching first
            retrieved_docs = self.document_processor.exact_keyword_search(query)
            
            # If no exact matches, fall back to semantic search with sections
            if not retrieved_docs:
                print("No exact matches found, falling back to semantic search")
                section_results = self.document_processor.search_sections(query, top_k=5)
                retrieved_docs = [(doc_name, content) for doc_name, _, _, content in section_results]
        
        print(f"Found {len(retrieved_docs)} relevant document segments")
        
        if not retrieved_docs:
            # No relevant documents found
            return {
                "success": False,
                "error": "No relevant information found for your query.",
                "query": query,
                "retrieved_docs": []
            }
        
        # Get key phrases from the query to focus the summary
        query_key_phrases = []
        for word in query.split():
            if word.lower() not in STOPWORDS and len(word) > 3:
                query_key_phrases.append(word.lower())
        
        # Extract key phrases from retrieved documents
        all_content = "\n\n".join([content for _, content in retrieved_docs])
        doc_key_phrases = self.document_processor.extract_key_phrases(all_content, top_n=5)
        
        # Combine query and document key phrases
        key_phrases = list(set(query_key_phrases + doc_key_phrases))
        
        # Generate a proper summary
        summary = self._generate_true_summary(query, retrieved_docs, key_phrases)
        
        return {
            "success": True,
            "query": query,
            "response": summary,
            "retrieved_docs": [doc_name for doc_name, _ in retrieved_docs]
        }
    
    def _is_question(self, text: str) -> bool:
        """
        Determine if text is a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be a question
        """
        # Check for question marks
        if '?' in text:
            return True
            
        # Check for question words
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'tell', 'explain', 'describe']
        first_word = text.lower().split()[0] if text else ""
        
        return first_word in question_starters
    
    def _generate_true_summary(self, query: str, retrieved_docs: List[Tuple[str, str]], key_phrases: List[str]) -> str:
        """
        Generate a true summary that synthesizes information from retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            key_phrases: Key phrases to focus on
            
        Returns:
            Generated summary
        """
        # Combine all content
        all_content = "\n\n".join([content for _, content in retrieved_docs])
        
        # Create a query-focused summary
        query_summary = self.summarizer.query_focused_summarize(all_content, query, sentences_count=15)
        
        # Structure the summary into a readable format
        structured_summary = self.summarizer.structure_summary(query_summary, key_phrases)
        
        # Format the response with proper structure
        formatted_summary = f"# Summary: {query}\n\n"
        formatted_summary += structured_summary
        
        # Add key points section
        if key_phrases:
            formatted_summary += "\n\n## Key Points\n"
            for phrase in key_phrases[:5]:
                # Capitalize the first letter of each phrase
                formatted_phrase = phrase[0].upper() + phrase[1:] if phrase else ""
                formatted_summary += f"- {formatted_phrase}\n"
        
        # Add sources section
        formatted_summary += "\n\n## Sources\n"
        for doc_name, _ in retrieved_docs:
            formatted_summary += f"- {doc_name}\n"
        
        return formatted_summary
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """
        Generate a professionally formatted PDF report of the response.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: List of document names used
            
        Returns:
            Path to the generated PDF
        """
        # Create a temporary directory for the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"response_{timestamp}.pdf")
            
            # Create a more professional PDF with ReportLab Platypus
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='Justify',
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY
            ))
            
            # Create elements for the PDF
            elements = []
            
            # Add title
            title_style = styles['Heading1']
            title_style.alignment = TA_LEFT
            elements.append(Paragraph(f"Summary: {query}", title_style))
            elements.append(Spacer(1, 0.25 * inch))
            
            # Process markdown in response
            paragraphs = response.split("\n\n")
            for paragraph in paragraphs:
                if paragraph.startswith("# "):
                    # Main heading
                    heading_text = paragraph[2:].strip()
                    elements.append(Paragraph(heading_text, styles['Heading1']))
                    elements.append(Spacer(1, 0.15 * inch))
                elif paragraph.startswith("## "):
                    # Subheading
                    subheading_text = paragraph[3:].strip()
                    elements.append(Paragraph(subheading_text, styles['Heading2']))
                    elements.append(Spacer(1, 0.1 * inch))
                elif paragraph.startswith("- "):
                    # List item
                    list_text = paragraph[2:].strip()
                    elements.append(Paragraph("• " + list_text, styles['Normal']))
                    elements.append(Spacer(1, 0.05 * inch))
                else:
                    # Regular paragraph
                    elements.append(Paragraph(paragraph, styles['Justify']))
                    elements.append(Spacer(1, 0.1 * inch))
            
            # Add footer with timestamp
            footer_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph(footer_text, styles['Normal']))
            
            # Build the PDF
            doc.build(elements)
            
            # In Vertex AI Workbench notebook environment
            final_path = f"/home/jupyter/response_{timestamp}.pdf"
            os.system(f"cp {output_path} {final_path}")
            
            return final_path
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width in characters
            
        Returns:
            List of wrapped lines
        """
        return textwrap.wrap(text, width=width)


# Main function to run the RAG system directly
def main():
    """Main function to run the RAG system."""
    
    # Configuration parameters - using your specified values
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing RAG system with:")
    print(f"- Knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = RAGSystem(KNOWLEDGE_BASE_PATH)
    
    # Example usage
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
            
        print("\nProcessing your query...\n")
        
        # Process the query
        result = rag_system.process_query(query)
        
        if result["success"]:
            print("=" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(result["response"])
            print("=" * 80)
            print("\nSources:", ", ".join(result["retrieved_docs"]))
            
            # Generate PDF report
            try:
                pdf_path = rag_system.generate_pdf_report(
                    query, 
                    result["response"], 
                    result["retrieved_docs"]
                )
                print(f"\nPDF report generated at: {pdf_path}")
            except Exception as e:
                print(f"\nError generating PDF report: {str(e)}")
        else:
            print(f"Error: {result['error']}")


# For use in a Jupyter notebook
def create_rag_system():
    """Create and return a RAG system instance for use in a notebook."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    return RAGSystem(KNOWLEDGE_BASE_PATH)


def process_query_and_generate_pdf(rag_system, query):
    """Process a query and generate a PDF report."""
    result = rag_system.process_query(query)
    
    if result["success"]:
        print("=" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(result["response"])
        print("=" * 80)
        print("\nSources:", ", ".join(result["retrieved_docs"]))
        
        # Generate PDF report
        try:
            pdf_path = rag_system.generate_pdf_report(
                query, 
                result["response"], 
                result["retrieved_docs"]
            )
            print(f"\nPDF report generated at: {pdf_path}")
            return result["response"], pdf_path
        except Exception as e:
            print(f"\nError generating PDF report: {str(e)}")
            return result["response"], None
    else:
        print(f"Error: {result['error']}")
        return None, None


# First run installer if needed
def install_required_packages():
    """Install required packages if they're not already installed."""
    try:
        import sumy
        print("Sumy is already installed.")
    except ImportError:
        print("Installing sumy...")
        os.system("pip install sumy")
        print("Sumy installed successfully.")


if __name__ == "__main__":
    # First install required packages
    install_required_packages()
    
    # Then run the main system
    main()
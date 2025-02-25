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


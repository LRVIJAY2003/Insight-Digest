ni# InsightDigest - Global Article Summarization and Link Analysis chatbot

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
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    # Application settings
    APP_NAME = "RAG Confluence & Remedy"
    APP_VERSION = "2.0.0"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24).hex())
    PORT = int(os.getenv("PORT", "8000"))
    
    # Cache settings
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "3600"))  # 1 hour
    CACHE_DIR = os.getenv("CACHE_DIR", "cache")
    
    # Confluence configuration
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
    CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")
    CONFLUENCE_MAX_RESULTS = int(os.getenv("CONFLUENCE_MAX_RESULTS", "20"))
    CONFLUENCE_INCLUDE_ATTACHMENTS = os.getenv("CONFLUENCE_INCLUDE_ATTACHMENTS", "True").lower() == "true"
    CONFLUENCE_INCLUDE_COMMENTS = os.getenv("CONFLUENCE_INCLUDE_COMMENTS", "True").lower() == "true"
    
    # Remedy configuration
    REMEDY_SERVER = os.getenv("REMEDY_SERVER")
    REMEDY_USERNAME = os.getenv("REMEDY_USERNAME")
    REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD")
    REMEDY_AUTH_TYPE = os.getenv("REMEDY_AUTH_TYPE", "AR-JWT")
    REMEDY_MAX_RESULTS = int(os.getenv("REMEDY_MAX_RESULTS", "20"))
    REMEDY_INCLUDE_ATTACHMENTS = os.getenv("REMEDY_INCLUDE_ATTACHMENTS", "True").lower() == "true"
    REMEDY_INCLUDE_HISTORY = os.getenv("REMEDY_INCLUDE_HISTORY", "True").lower() == "true"
    
    # Content parsing settings
    ENABLE_OCR = os.getenv("ENABLE_OCR", "True").lower() == "true"
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
    SUPPORTED_DOCUMENT_FORMATS = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt"]
    TEMP_DIR = os.getenv("TEMP_DIR", "temp")
    
    # Summarization settings
    MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "500"))
    MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH", "100"))
    SUMMARY_METHOD = os.getenv("SUMMARY_METHOD", "extractive")  # 'extractive' or 'abstractive'
    
    # Search settings
    SEARCH_HYBRID_WEIGHT = float(os.getenv("SEARCH_HYBRID_WEIGHT", "0.7"))  # Weight for semantic search vs keyword
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))  # Minimum relevance score
    
    # UI settings
    DEFAULT_THEME = os.getenv("DEFAULT_THEME", "light")  # 'light' or 'dark'
    RESULTS_PER_PAGE = int(os.getenv("RESULTS_PER_PAGE", "5"))
    
    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate that required configuration settings are present.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate Confluence settings
        if not cls.CONFLUENCE_URL:
            errors.append("CONFLUENCE_URL is required")
        if not cls.CONFLUENCE_USERNAME:
            errors.append("CONFLUENCE_USERNAME is required")
        if not cls.CONFLUENCE_API_TOKEN:
            errors.append("CONFLUENCE_API_TOKEN is required")
        if not cls.CONFLUENCE_SPACE_KEY:
            errors.append("CONFLUENCE_SPACE_KEY is required")
        
        # Validate Remedy settings
        if not cls.REMEDY_SERVER:
            errors.append("REMEDY_SERVER is required")
        if not cls.REMEDY_USERNAME:
            errors.append("REMEDY_USERNAME is required")
        if not cls.REMEDY_PASSWORD:
            errors.append("REMEDY_PASSWORD is required")
        
        # Create necessary directories
        Path(cls.CACHE_DIR).mkdir(exist_ok=True)
        Path(cls.TEMP_DIR).mkdir(exist_ok=True)
        
        return errors
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """
        Get all configuration settings as a dictionary.
        
        Returns:
            Dictionary of settings
        """
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not callable(getattr(cls, k))}
    
    @classmethod
    def to_json(cls) -> str:
        """
        Get all configuration settings as a JSON string.
        
        Returns:
            JSON string of settings
        """
        config_dict = {k: v for k, v in cls.__dict__.items() 
                      if not k.startswith('__') and not callable(getattr(cls, k))}
        
        # Remove sensitive information
        sensitive_keys = ['CONFLUENCE_API_TOKEN', 'REMEDY_PASSWORD', 'SECRET_KEY']
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "********"
        
        return json.dumps(config_dict, sort_keys=True, indent=4)


# Validate configuration on import
config_errors = Config.validate()
if config_errors:
    for error in config_errors:
        print(f"Configuration Error: {error}")
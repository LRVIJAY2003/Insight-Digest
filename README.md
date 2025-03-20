import logging
import time
import asyncio
import base64
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile

from atlassian import Confluence
import requests
from requests.exceptions import RequestException

from config.config import Config
from services.content_parser import ContentParser

# Set up logging
logger = logging.getLogger(__name__)

class ConfluenceService:
    """
    Enhanced service for interacting with Confluence API and retrieving content.
    """
    
    def __init__(self):
        """Initialize the Confluence service with API connection."""
        self.url = Config.CONFLUENCE_URL
        self.username = Config.CONFLUENCE_USERNAME
        self.api_token = Config.CONFLUENCE_API_TOKEN
        self.space_key = Config.CONFLUENCE_SPACE_KEY
        
        self.confluence = Confluence(
            url=self.url,
            username=self.username,
            password=self.api_token,
            cloud=True  # Set to True for Confluence Cloud, False for Server
        )
        
        # Initialize content parser
        self.content_parser = ContentParser()
        
        # Simple cache implementation
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_expiry = Config.CACHE_EXPIRY
        self.enable_cache = Config.ENABLE_CACHE
    
    async def search_content(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for content in Confluence.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing page info (id, title, url)
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"search_{query}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved search results from cache for query: {query}")
                return self._cache[cache_key]
        
        logger.info(f"Searching Confluence for: {query}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            # Perform CQL search
            cql = f'space = "{self.space_key}" AND text ~ "{query}"'
            search_results = await loop.run_in_executor(
                None, 
                lambda: self.confluence.cql(cql, limit=limit)
            )
            
            results = []
            if 'results' in search_results:
                for result in search_results['results']:
                    content = result.get('content', {})
                    content_type = content.get('type', '')
                    
                    # Get title and excerpt
                    title = content.get('title', '')
                    excerpt = result.get('excerpt', '')
                    
                    # Clean up excerpt
                    excerpt = self._clean_excerpt(excerpt)
                    
                    # Get URL
                    url = f"{self.url}{content.get('_links', {}).get('webui', '')}"
                    
                    # Get last updated info
                    last_updated = None
                    if 'lastModified' in result:
                        last_updated = result['lastModified'].get('when', '')
                    
                    # Get content ID
                    content_id = content.get('id', '')
                    
                    # Get space information
                    space = {}
                    if 'space' in content:
                        space = {
                            'key': content['space'].get('key', ''),
                            'name': content['space'].get('name', '')
                        }
                    
                    results.append({
                        'id': content_id,
                        'title': title,
                        'type': content_type,
                        'excerpt': excerpt,
                        'url': url,
                        'last_updated': last_updated,
                        'space': space
                    })
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = results
                self._cache_timestamps[cache_key] = time.time()
                
            return results
        except Exception as e:
            logger.error(f"Error searching Confluence: {str(e)}")
            return []
    
    async def get_page_content(self, page_id: str, include_attachments: bool = None) -> Dict[str, Any]:
        """
        Get the content of a specific page with enhanced parsing.
        
        Args:
            page_id: The Confluence page ID
            include_attachments: Whether to include attachments (defaults to Config setting)
            
        Returns:
            Dictionary with page content and metadata
        """
        if include_attachments is None:
            include_attachments = Config.CONFLUENCE_INCLUDE_ATTACHMENTS
            
        cache_key = f"page_{page_id}_{include_attachments}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved page content from cache for page ID: {page_id}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving Confluence page content for ID: {page_id}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            # Get page content with body storage and additional info
            expand_params = "body.storage,history,space,version,descendants.attachment"
            if Config.CONFLUENCE_INCLUDE_COMMENTS:
                expand_params += ",children.comment"
                
            page = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_page_by_id(
                    page_id, 
                    expand=expand_params
                )
            )
            
            # Extract basic metadata
            title = page.get('title', '')
            space_key = page.get('space', {}).get('key', '')
            space_name = page.get('space', {}).get('name', '')
            version = page.get('version', {}).get('number', 0)
            
            # Extract creator and last updater
            creator = page.get('history', {}).get('createdBy', {}).get('displayName', '')
            last_updater = page.get('history', {}).get('lastUpdated', {}).get('by', {}).get('displayName', '')
            
            # Extract dates
            created_date = page.get('history', {}).get('createdDate', '')
            last_updated = page.get('history', {}).get('lastUpdated', {}).get('when', '')
            
            # Get URL
            url = f"{self.url}/pages/viewpage.action?pageId={page_id}"
            
            # Get HTML content
            body = page.get('body', {}).get('storage', {}).get('value', '')
            
            # Parse HTML content
            parsed_content = self.content_parser.parse_html_content(body)
            
            # Collect attachments if requested
            attachments = []
            parsed_attachments = []
            
            if include_attachments and 'descendants' in page and 'attachments' in page['descendants']:
                attachment_list = page['descendants']['attachments'].get('results', [])
                
                for attachment in attachment_list:
                    attachment_id = attachment.get('id', '')
                    attachment_title = attachment.get('title', '')
                    attachment_metadata = {
                        'id': attachment_id,
                        'title': attachment_title,
                        'filename': attachment.get('title', ''),
                        'mediaType': attachment.get('metadata', {}).get('mediaType', ''),
                        'size': attachment.get('extensions', {}).get('fileSize', 0),
                        'created': attachment.get('extensions', {}).get('fileCreatedDate', ''),
                        'comment': attachment.get('metadata', {}).get('comment', '')
                    }
                    
                    attachments.append(attachment_metadata)
                    
                    # Download and parse attachment content if it's a supported type
                    if self._is_supported_attachment(attachment_metadata['mediaType'], attachment_metadata['filename']):
                        try:
                            # Download attachment
                            attachment_data = await loop.run_in_executor(
                                None,
                                lambda: self.confluence.download_attachment(
                                    attachment_id,
                                    path=None,  # Don't save to disk
                                    return_content=True
                                )
                            )
                            
                            # Parse attachment content
                            parsed_attachment = self.content_parser.parse_attachment(
                                attachment_data,
                                attachment_metadata['filename'],
                                attachment_metadata['mediaType']
                            )
                            
                            # Add metadata to parsed content
                            parsed_attachment['metadata'].update(attachment_metadata)
                            parsed_attachments.append(parsed_attachment)
                            
                        except Exception as att_error:
                            logger.error(f"Error processing attachment {attachment_title}: {str(att_error)}")
                            parsed_attachments.append({
                                "text": f"[Error processing attachment: {attachment_title}]",
                                "metadata": {
                                    **attachment_metadata,
                                    "error": str(att_error)
                                }
                            })
            
            # Get comments if enabled
            comments = []
            if Config.CONFLUENCE_INCLUDE_COMMENTS and 'children' in page and 'comment' in page['children']:
                comment_list = page['children']['comment'].get('results', [])
                
                for comment in comment_list:
                    comment_body = comment.get('body', {}).get('storage', {}).get('value', '')
                    comment_author = comment.get('author', {}).get('displayName', '')
                    comment_created = comment.get('created', '')
                    
                    # Parse comment HTML
                    parsed_comment = self.content_parser.parse_html_content(comment_body)
                    
                    comments.append({
                        'author': comment_author,
                        'created': comment_created,
                        'content': parsed_comment,
                        'id': comment.get('id', '')
                    })
            
            # Combine all information
            result = {
                'id': page_id,
                'title': title,
                'space_key': space_key,
                'space_name': space_name,
                'url': url,
                'version': version,
                'creator': creator,
                'last_updater': last_updater,
                'created_date': created_date,
                'last_updated': last_updated,
                'content': parsed_content,
                'attachments': attachments,
                'parsed_attachments': parsed_attachments,
                'comments': comments
            }
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving page content: {str(e)}")
            return {
                'id': page_id,
                'title': '',
                'error': str(e),
                'content': self.content_parser._create_empty_content_dict(error=str(e))
            }
    
    async def get_pages_in_space(self, limit: int = None) -> List[Dict[str, str]]:
        """
        Get all pages in the configured space.
        
        Args:
            limit: Maximum number of pages to retrieve
            
        Returns:
            List of dictionaries with page information
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"space_pages_{self.space_key}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved pages list from cache for space: {self.space_key}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving pages from Confluence space: {self.space_key}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            pages = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_all_pages_from_space(
                    space=self.space_key,
                    start=0,
                    limit=limit
                )
            )
            
            result = []
            for page in pages:
                # Extract basic metadata
                page_id = page.get('id', '')
                title = page.get('title', '')
                
                # Get URL
                url = f"{self.url}{page.get('_links', {}).get('webui', '')}"
                
                # Get page type
                page_type = page.get('type', '')
                
                # Get space info
                space = {}
                if 'space' in page:
                    space = {
                        'key': page['space'].get('key', ''),
                        'name': page['space'].get('name', '')
                    }
                
                result.append({
                    'id': page_id,
                    'title': title,
                    'url': url,
                    'type': page_type,
                    'space': space
                })
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving pages from space: {str(e)}")
            return []
    
    import logging
import time
import asyncio
import base64
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile

from atlassian import Confluence
import requests
from requests.exceptions import RequestException

from config.config import Config
from services.content_parser import ContentParser

# Set up logging
logger = logging.getLogger(__name__)

class ConfluenceService:
    """
    Enhanced service for interacting with Confluence API and retrieving content.
    """
    
    def __init__(self):
        """Initialize the Confluence service with API connection."""
        self.url = Config.CONFLUENCE_URL
        self.username = Config.CONFLUENCE_USERNAME
        self.api_token = Config.CONFLUENCE_API_TOKEN
        self.space_key = Config.CONFLUENCE_SPACE_KEY
        
        self.confluence = Confluence(
            url=self.url,
            username=self.username,
            password=self.api_token,
            cloud=True  # Set to True for Confluence Cloud, False for Server
        )
        
        # Initialize content parser
        self.content_parser = ContentParser()
        
        # Simple cache implementation
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_expiry = Config.CACHE_EXPIRY
        self.enable_cache = Config.ENABLE_CACHE
    
    async def search_content(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for content in Confluence.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing page info (id, title, url)
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"search_{query}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved search results from cache for query: {query}")
                return self._cache[cache_key]
        
        logger.info(f"Searching Confluence for: {query}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            # Perform CQL search
            cql = f'space = "{self.space_key}" AND text ~ "{query}"'
            search_results = await loop.run_in_executor(
                None, 
                lambda: self.confluence.cql(cql, limit=limit)
            )
            
            results = []
            if 'results' in search_results:
                for result in search_results['results']:
                    content = result.get('content', {})
                    content_type = content.get('type', '')
                    
                    # Get title and excerpt
                    title = content.get('title', '')
                    excerpt = result.get('excerpt', '')
                    
                    # Clean up excerpt
                    excerpt = self._clean_excerpt(excerpt)
                    
                    # Get URL
                    url = f"{self.url}{content.get('_links', {}).get('webui', '')}"
                    
                    # Get last updated info
                    last_updated = None
                    if 'lastModified' in result:
                        last_updated = result['lastModified'].get('when', '')
                    
                    # Get content ID
                    content_id = content.get('id', '')
                    
                    # Get space information
                    space = {}
                    if 'space' in content:
                        space = {
                            'key': content['space'].get('key', ''),
                            'name': content['space'].get('name', '')
                        }
                    
                    results.append({
                        'id': content_id,
                        'title': title,
                        'type': content_type,
                        'excerpt': excerpt,
                        'url': url,
                        'last_updated': last_updated,
                        'space': space
                    })
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = results
                self._cache_timestamps[cache_key] = time.time()
                
            return results
        except Exception as e:
            logger.error(f"Error searching Confluence: {str(e)}")
            return []
    
    async def get_page_content(self, page_id: str, include_attachments: bool = None) -> Dict[str, Any]:
        """
        Get the content of a specific page with enhanced parsing.
        
        Args:
            page_id: The Confluence page ID
            include_attachments: Whether to include attachments (defaults to Config setting)
            
        Returns:
            Dictionary with page content and metadata
        """
        if include_attachments is None:
            include_attachments = Config.CONFLUENCE_INCLUDE_ATTACHMENTS
            
        cache_key = f"page_{page_id}_{include_attachments}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved page content from cache for page ID: {page_id}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving Confluence page content for ID: {page_id}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            # Get page content with body storage and additional info
            expand_params = "body.storage,history,space,version,descendants.attachment"
            if Config.CONFLUENCE_INCLUDE_COMMENTS:
                expand_params += ",children.comment"
                
            page = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_page_by_id(
                    page_id, 
                    expand=expand_params
                )
            )
            
            # Extract basic metadata
            title = page.get('title', '')
            space_key = page.get('space', {}).get('key', '')
            space_name = page.get('space', {}).get('name', '')
            version = page.get('version', {}).get('number', 0)
            
            # Extract creator and last updater
            creator = page.get('history', {}).get('createdBy', {}).get('displayName', '')
            last_updater = page.get('history', {}).get('lastUpdated', {}).get('by', {}).get('displayName', '')
            
            # Extract dates
            created_date = page.get('history', {}).get('createdDate', '')
            last_updated = page.get('history', {}).get('lastUpdated', {}).get('when', '')
            
            # Get URL
            url = f"{self.url}/pages/viewpage.action?pageId={page_id}"
            
            # Get HTML content
            body = page.get('body', {}).get('storage', {}).get('value', '')
            
            # Parse HTML content
            parsed_content = self.content_parser.parse_html_content(body)
            
            # Collect attachments if requested
            attachments = []
            parsed_attachments = []
            
            if include_attachments and 'descendants' in page and 'attachments' in page['descendants']:
                attachment_list = page['descendants']['attachments'].get('results', [])
                
                for attachment in attachment_list:
                    attachment_id = attachment.get('id', '')
                    attachment_title = attachment.get('title', '')
                    attachment_metadata = {
                        'id': attachment_id,
                        'title': attachment_title,
                        'filename': attachment.get('title', ''),
                        'mediaType': attachment.get('metadata', {}).get('mediaType', ''),
                        'size': attachment.get('extensions', {}).get('fileSize', 0),
                        'created': attachment.get('extensions', {}).get('fileCreatedDate', ''),
                        'comment': attachment.get('metadata', {}).get('comment', '')
                    }
                    
                    attachments.append(attachment_metadata)
                    
                    # Download and parse attachment content if it's a supported type
                    if self._is_supported_attachment(attachment_metadata['mediaType'], attachment_metadata['filename']):
                        try:
                            # Download attachment
                            attachment_data = await loop.run_in_executor(
                                None,
                                lambda: self.confluence.download_attachment(
                                    attachment_id,
                                    path=None,  # Don't save to disk
                                    return_content=True
                                )
                            )
                            
                            # Parse attachment content
                            parsed_attachment = self.content_parser.parse_attachment(
                                attachment_data,
                                attachment_metadata['filename'],
                                attachment_metadata['mediaType']
                            )
                            
                            # Add metadata to parsed content
                            parsed_attachment['metadata'].update(attachment_metadata)
                            parsed_attachments.append(parsed_attachment)
                            
                        except Exception as att_error:
                            logger.error(f"Error processing attachment {attachment_title}: {str(att_error)}")
                            parsed_attachments.append({
                                "text": f"[Error processing attachment: {attachment_title}]",
                                "metadata": {
                                    **attachment_metadata,
                                    "error": str(att_error)
                                }
                            })
            
            # Get comments if enabled
            comments = []
            if Config.CONFLUENCE_INCLUDE_COMMENTS and 'children' in page and 'comment' in page['children']:
                comment_list = page['children']['comment'].get('results', [])
                
                for comment in comment_list:
                    comment_body = comment.get('body', {}).get('storage', {}).get('value', '')
                    comment_author = comment.get('author', {}).get('displayName', '')
                    comment_created = comment.get('created', '')
                    
                    # Parse comment HTML
                    parsed_comment = self.content_parser.parse_html_content(comment_body)
                    
                    comments.append({
                        'author': comment_author,
                        'created': comment_created,
                        'content': parsed_comment,
                        'id': comment.get('id', '')
                    })
            
            # Combine all information
            result = {
                'id': page_id,
                'title': title,
                'space_key': space_key,
                'space_name': space_name,
                'url': url,
                'version': version,
                'creator': creator,
                'last_updater': last_updater,
                'created_date': created_date,
                'last_updated': last_updated,
                'content': parsed_content,
                'attachments': attachments,
                'parsed_attachments': parsed_attachments,
                'comments': comments
            }
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving page content: {str(e)}")
            return {
                'id': page_id,
                'title': '',
                'error': str(e),
                'content': self.content_parser._create_empty_content_dict(error=str(e))
            }
    
    async def get_pages_in_space(self, limit: int = None) -> List[Dict[str, str]]:
        """
        Get all pages in the configured space.
        
        Args:
            limit: Maximum number of pages to retrieve
            
        Returns:
            List of dictionaries with page information
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"space_pages_{self.space_key}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved pages list from cache for space: {self.space_key}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving pages from Confluence space: {self.space_key}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            pages = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_all_pages_from_space(
                    space=self.space_key,
                    start=0,
                    limit=limit
                )
            )
            
            result = []
            for page in pages:
                # Extract basic metadata
                page_id = page.get('id', '')
                title = page.get('title', '')
                
                # Get URL
                url = f"{self.url}{page.get('_links', {}).get('webui', '')}"
                
                # Get page type
                page_type = page.get('type', '')
                
                # Get space info
                space = {}
                if 'space' in page:
                    space = {
                        'key': page['space'].get('key', ''),
                        'name': page['space'].get('name', '')
                    }
                
                result.append({
                    'id': page_id,
                    'title': title,
                    'url': url,
                    'type': page_type,
                    'space': space
                })
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving pages from space: {str(e)}")
            return []
    
    async def get_space_content_tree(self) -> Dict[str, Any]:
        """
        Get the content tree for the configured space.
        
        Returns:
            Dictionary with space content tree
        """
        cache_key = f"space_tree_{self.space_key}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved content tree from cache for space: {self.space_key}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving content tree for Confluence space: {self.space_key}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            # Get space info
            space = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_space(self.space_key, expand='description.view')
            )
            
            # Get space home page
            home_id = space.get('homepage', {}).get('id', '')
            
            # Build tree structure starting from home page
            tree = {
                'space_key': self.space_key,
                'space_name': space.get('name', ''),
                'description': space.get('description', {}).get('view', {}).get('value', ''),
                'home_page_id': home_id,
                'pages': []
            }
            
            # Get all pages for this space
            pages = await self.get_pages_in_space()
            
            # Create lookup for parent-child relationships
            page_lookup = {page['id']: page for page in pages}
            children = {}
            
            # Get child pages for each page
            for page in pages:
                page_id = page['id']
                
                try:
                    # Get children for this page
                    child_pages = await loop.run_in_executor(
                        None, 
                        lambda: self.confluence.get_page_child_by_type(page_id, type='page')
                    )
                    
                    # Store child IDs
                    children[page_id] = [child.get('id', '') for child in child_pages]
                except Exception as child_error:
                    logger.error(f"Error getting children for page {page_id}: {str(child_error)}")
                    children[page_id] = []
            
            # Build hierarchical tree
            def build_tree_node(page_id):
                if page_id not in page_lookup:
                    return None
                    
                page = page_lookup[page_id]
                node = {
                    'id': page['id'],
                    'title': page['title'],
                    'url': page['url'],
                    'children': []
                }
                
                # Add children recursively
                if page_id in children:
                    for child_id in children[page_id]:
                        child_node = build_tree_node(child_id)
                        if child_node:
                            node['children'].append(child_node)
                
                return node
            
            # Start building from home page if available
            if home_id and home_id in page_lookup:
                tree['pages'].append(build_tree_node(home_id))
                
                # Add orphaned top-level pages (those without parent or with parent outside space)
                for page_id, page in page_lookup.items():
                    # Skip home page and its descendants
                    if page_id == home_id or any(page_id in children.get(ancestor_id, []) 
                                               for ancestor_id in page_lookup):
                        continue
                    
                    # Add as top-level page
                    tree['pages'].append(build_tree_node(page_id))
            else:
                # No home page, add all pages at top level
                for page_id in page_lookup:
                    tree['pages'].append(build_tree_node(page_id))
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = tree
                self._cache_timestamps[cache_key] = time.time()
                
            return tree
        except Exception as e:
            logger.error(f"Error retrieving content tree: {str(e)}")
            return {
                'space_key': self.space_key,
                'error': str(e),
                'pages': []
            }
    
    async def get_labels(self, content_id: str) -> List[str]:
        """
        Get labels for a specific content item.
        
        Args:
            content_id: The Confluence content ID
            
        Returns:
            List of label names
        """
        cache_key = f"labels_{content_id}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved labels from cache for content ID: {content_id}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving labels for Confluence content ID: {content_id}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            labels = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_page_labels(content_id)
            )
            
            result = []
            if 'results' in labels:
                result = [label.get('name', '') for label in labels['results']]
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving labels: {str(e)}")
            return []
    
    async def search_by_label(self, label: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for content with a specific label.
        
        Args:
            label: Label to search for
            limit: Maximum number of results to return
            
        Returns:
            List of content items
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"label_search_{label}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved label search from cache for label: {label}")
                return self._cache[cache_key]
        
        logger.info(f"Searching Confluence for content with label: {label}")
        try:
            # Construct CQL query for label search
            cql = f'space = "{self.space_key}" AND label = "{label}"'
            
            # Use the search_content method with the CQL query
            results = await self.search_content(cql, limit)
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = results
                self._cache_timestamps[cache_key] = time.time()
                
            return results
        except Exception as e:
            logger.error(f"Error searching by label: {str(e)}")
            return []
    
    async def get_recently_updated_pages(self, days: int = 30, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get recently updated pages in the space.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            List of recently updated pages
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        cache_key = f"recent_pages_{self.space_key}_{days}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved recent pages from cache for space: {self.space_key}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving recently updated pages from Confluence space: {self.space_key}")
        try:
            # Construct CQL query for recent updates
            cql = f'space = "{self.space_key}" AND lastmodified >= "-{days}d"'
            
            # Use the search_content method with the CQL query
            results = await self.search_content(cql, limit)
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = results
                self._cache_timestamps[cache_key] = time.time()
                
            return results
        except Exception as e:
            logger.error(f"Error retrieving recently updated pages: {str(e)}")
            return []
    
    async def get_page_history(self, page_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the version history of a page.
        
        Args:
            page_id: The Confluence page ID
            limit: Maximum number of versions to return
            
        Returns:
            List of page versions
        """
        cache_key = f"history_{page_id}_{limit}"
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved page history from cache for page ID: {page_id}")
                return self._cache[cache_key]
        
        logger.info(f"Retrieving history for Confluence page ID: {page_id}")
        try:
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            history = await loop.run_in_executor(
                None, 
                lambda: self.confluence.get_content_history(page_id, limit=limit)
            )
            
            result = []
            
            # Process each version
            for version in history.get('latest', []):
                version_info = {
                    'number': version.get('number', 0),
                    'when': version.get('when', ''),
                    'message': version.get('message', ''),
                    'by': {
                        'username': version.get('by', {}).get('username', ''),
                        'display_name': version.get('by', {}).get('displayName', '')
                    }
                }
                result.append(version_info)
                
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving page history: {str(e)}")
            return []
    
    async def advanced_search(self, 
                            query: str = None,
                            title: str = None, 
                            label: str = None,
                            creator: str = None,
                            space_key: str = None,
                            content_type: str = "page",
                            days: int = None,
                            limit: int = None) -> List[Dict[str, Any]]:
        """
        Perform an advanced search with multiple criteria.
        
        Args:
            query: Text to search for
            title: Title to search for
            label: Label to search for
            creator: Creator username to search for
            space_key: Space key to search in (defaults to configured space)
            content_type: Type of content to search for (page, blogpost, etc.)
            days: Number of days to look back for updates
            limit: Maximum number of results to return
            
        Returns:
            List of matching content items
        """
        if limit is None:
            limit = Config.CONFLUENCE_MAX_RESULTS
            
        if space_key is None:
            space_key = self.space_key
            
        # Generate cache key based on parameters
        cache_params = f"{query}_{title}_{label}_{creator}_{space_key}_{content_type}_{days}_{limit}"
        cache_key = f"adv_search_{hash(cache_params)}"
        
        if self.enable_cache and cache_key in self._cache:
            # Check if cache is still valid
            if time.time() - self._cache_timestamps[cache_key] < self.cache_expiry:
                logger.info(f"Retrieved advanced search results from cache: {cache_params}")
                return self._cache[cache_key]
        
        logger.info(f"Performing advanced Confluence search with criteria: {cache_params}")
        try:
            # Build CQL query
            cql_parts = []
            
            # Add space restriction
            cql_parts.append(f'space = "{space_key}"')
            
            # Add content type if specified
            if content_type:
                cql_parts.append(f'type = "{content_type}"')
                
            # Add text query if specified
            if query:
                cql_parts.append(f'text ~ "{query}"')
                
            # Add title search if specified
            if title:
                cql_parts.append(f'title ~ "{title}"')
                
            # Add label search if specified
            if label:
                cql_parts.append(f'label = "{label}"')
                
            # Add creator search if specified
            if creator:
                cql_parts.append(f'creator = "{creator}"')
                
            # Add date restriction if specified
            if days:
                cql_parts.append(f'lastmodified >= "-{days}d"')
                
            # Combine all parts with AND
            cql = " AND ".join(cql_parts)
            
            # Wrap synchronous API call in a thread
            loop = asyncio.get_event_loop()
            
            search_results = await loop.run_in_executor(
                None, 
                lambda: self.confluence.cql(cql, limit=limit)
            )
            
            results = []
            if 'results' in search_results:
                for result in search_results['results']:
                    content = result.get('content', {})
                    
                    # Skip if no content
                    if not content:
                        continue
                        
                    content_type = content.get('type', '')
                    
                    # Get title and excerpt
                    title = content.get('title', '')
                    excerpt = result.get('excerpt', '')
                    
                    # Clean up excerpt
                    excerpt = self._clean_excerpt(excerpt)
                    
                    # Get URL
                    url = f"{self.url}{content.get('_links', {}).get('webui', '')}"
                    
                    # Get last updated info
                    last_updated = None
                    if 'lastModified' in result:
                        last_updated = result['lastModified'].get('when', '')
                    
                    # Get content ID
                    content_id = content.get('id', '')
                    
                    # Get space information
                    space = {}
                    if 'space' in content:
                        space = {
                            'key': content['space'].get('key', ''),
                            'name': content['space'].get('name', '')
                        }
                    
                    results.append({
                        'id': content_id,
                        'title': title,
                        'type': content_type,
                        'excerpt': excerpt,
                        'url': url,
                        'last_updated': last_updated,
                        'space': space
                    })
            
            # Update cache
            if self.enable_cache:
                self._cache[cache_key] = results
                self._cache_timestamps[cache_key] = time.time()
                
            return results
        except Exception as e:
            logger.error(f"Error performing advanced search: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = {}
        self._cache_timestamps = {}
        logger.info("Confluence service cache cleared")
    
    def _clean_excerpt(self, excerpt: str) -> str:
        """
        Clean up excerpt text from search results.
        
        Args:
            excerpt: Raw excerpt text with HTML
            
        Returns:
            Cleaned excerpt text
        """
        if not excerpt:
            return ""
            
        # Remove HTML tags
        excerpt = re.sub(r'<[^>]+>', '', excerpt)
        
        # Replace highlight markers
        excerpt = excerpt.replace("@@@hl@@@", "").replace("@@@endhl@@@", "")
        
        # Clean up whitespace
        excerpt = re.sub(r'\s+', ' ', excerpt).strip()
        
        return excerpt
    
    def _is_supported_attachment(self, media_type: str, filename: str) -> bool:
        """
        Check if an attachment type is supported for parsing.
        
        Args:
            media_type: MIME type of the attachment
            filename: Filename of the attachment
            
        Returns:
            True if the attachment type is supported, False otherwise
        """
        # Check MIME type
        if media_type:
            if (media_type.startswith('image/') or
                media_type in ['application/pdf', 'application/msword', 
                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                              'application/vnd.ms-excel',
                              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              'text/plain', 'text/csv', 'text/markdown']):
                return True
        
        # Fall back to extension-based detection
        ext = Path(filename).suffix.lower()
        
        if (ext in Config.SUPPORTED_IMAGE_FORMATS or
            ext[1:] in Config.SUPPORTED_IMAGE_FORMATS or
            ext in Config.SUPPORTED_DOCUMENT_FORMATS or
            ext[1:] in Config.SUPPORTED_DOCUMENT_


def _is_supported_attachment(self, media_type: str, filename: str) -> bool:
        """
        Check if an attachment type is supported for parsing.
        
        Args:
            media_type: MIME type of the attachment
            filename: Filename of the attachment
            
        Returns:
            True if the attachment type is supported, False otherwise
        """
        # Check MIME type
        if media_type:
            if (media_type.startswith('image/') or
                media_type in ['application/pdf', 'application/msword', 
                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                              'application/vnd.ms-excel',
                              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              'text/plain', 'text/csv', 'text/markdown']):
                return True
        
        # Fall back to extension-based detection
        ext = Path(filename).suffix.lower()
        
        if (ext in Config.SUPPORTED_IMAGE_FORMATS or
            ext[1:] in Config.SUPPORTED_IMAGE_FORMATS or
            ext in Config.SUPPORTED_DOCUMENT_FORMATS or
            ext[1:] in Config.SUPPORTED_DOCUMENT_FORMATS):
            return True
            
        return False
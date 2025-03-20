import os
import re
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import base64
import json
import mimetypes
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import pandas as pd
import docx
from pdfminer.high_level import extract_text
import openpyxl
from langdetect import detect

from config.config import Config

# Set up logging
logger = logging.getLogger(__name__)

class ContentParser:
    """
    Advanced content parser for various types of content from Confluence and Remedy.
    Handles HTML, plain text, images, tables, PDFs, Office documents, etc.
    """
    
    def __init__(self):
        """Initialize the content parser."""
        self.temp_dir = Path(Config.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure OCR if enabled
        self.ocr_enabled = Config.ENABLE_OCR
        if self.ocr_enabled:
            # Set OCR language
            pytesseract.pytesseract.tesseract_cmd = os.environ.get(
                'TESSERACT_CMD', 
                'tesseract'  # Default command
            )
    
    def parse_html_content(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content from Confluence.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Dictionary with extracted content including text, tables, images, etc.
        """
        if not html_content:
            return self._create_empty_content_dict()
            
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()
            
            # Extract main text content
            main_text = self._extract_text_from_soup(soup)
            
            # Extract tables
            tables = self._extract_tables_from_soup(soup)
            
            # Extract images with captions
            images = self._extract_images_from_soup(soup)
            
            # Extract links
            links = self._extract_links_from_soup(soup)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks_from_soup(soup)
            
            # Extract headings and structure
            structure = self._extract_structure_from_soup(soup)
            
            # Extract any embedded content (macros, etc.)
            embedded = self._extract_embedded_content_from_soup(soup)
            
            # Detect language
            language = self._detect_language(main_text)
            
            return {
                "text": main_text,
                "tables": tables,
                "images": images,
                "links": links,
                "code_blocks": code_blocks,
                "structure": structure,
                "embedded": embedded,
                "language": language,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "html",
                    "content_length": len(html_content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            return self._create_empty_content_dict(error=str(e))
    
    def parse_image(self, image_data: Union[bytes, str], image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse image content using OCR if enabled.
        
        Args:
            image_data: Raw image data as bytes or base64 string
            image_path: Optional path to the image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.ocr_enabled:
            logger.info("OCR is disabled, skipping image parsing")
            return {
                "text": "",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "image",
                    "ocr_enabled": False
                }
            }
            
        try:
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir, suffix='.png') as temp_file:
                temp_path = temp_file.name
                
                # Convert base64 string to bytes if needed
                if isinstance(image_data, str) and image_data.startswith(('data:image', 'base64')):
                    # Extract base64 data
                    if 'base64,' in image_data:
                        image_data = image_data.split('base64,')[1]
                    
                    # Decode base64
                    image_data = base64.b64decode(image_data)
                
                # Write image data to temp file
                temp_file.write(image_data)
            
            # Process image with OCR
            try:
                # Open the image with PIL
                img = Image.open(temp_path)
                
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(img, lang=Config.OCR_LANGUAGE)
                
                # Get image metadata
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                # Try to extract any EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        for tag, value in exif.items():
                            if isinstance(value, (str, int, float, bool)):
                                exif_data[str(tag)] = value
                
                # Clean up
                os.unlink(temp_path)
                
                return {
                    "text": ocr_text.strip(),
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "image",
                        "ocr_enabled": True,
                        "format": format_name,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "exif": exif_data,
                        "original_path": image_path
                    }
                }
                
            except Exception as img_error:
                logger.error(f"Error processing image with OCR: {str(img_error)}")
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return {
                    "text": "",
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "image",
                        "ocr_enabled": True,
                        "ocr_error": str(img_error),
                        "original_path": image_path
                    }
                }
                
        except Exception as e:
            logger.error(f"Error parsing image content: {str(e)}")
            return {
                "text": "",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "image",
                    "error": str(e),
                    "original_path": image_path
                }
            }
    
    def parse_pdf(self, pdf_data: bytes, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse PDF content.
        
        Args:
            pdf_data: Raw PDF data
            pdf_path: Optional path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Create a temporary file for the PDF
            with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir, suffix='.pdf') as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_data)
            
            # Extract text from PDF
            extracted_text = extract_text(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            # Detect language
            language = self._detect_language(extracted_text)
            
            return {
                "text": extracted_text,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "pdf",
                    "content_length": len(pdf_data),
                    "language": language,
                    "original_path": pdf_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF content: {str(e)}")
            return {
                "text": "",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "pdf",
                    "error": str(e),
                    "original_path": pdf_path
                }
            }
    
    def parse_word_document(self, docx_data: bytes, docx_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse Word document content.
        
        Args:
            docx_data: Raw DOCX data
            docx_path: Optional path to the DOCX file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Create a temporary file for the DOCX
            with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir, suffix='.docx') as temp_file:
                temp_path = temp_file.name
                temp_file.write(docx_data)
            
            # Extract text from DOCX
            doc = docx.Document(temp_path)
            full_text = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                full_text.append(para.text)
            
            # Extract tables
            tables = []
            for i, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                
                tables.append({
                    "id": i,
                    "data": table_data
                })
            
            # Clean up
            os.unlink(temp_path)
            
            # Join paragraphs with newlines
            extracted_text = "\n".join(full_text)
            
            # Detect language
            language = self._detect_language(extracted_text)
            
            return {
                "text": extracted_text,
                "tables": tables,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "docx",
                    "content_length": len(docx_data),
                    "language": language,
                    "original_path": docx_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing DOCX content: {str(e)}")
            return {
                "text": "",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "docx",
                    "error": str(e),
                    "original_path": docx_path
                }
            }
    
    def parse_excel_document(self, xlsx_data: bytes, xlsx_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse Excel document content.
        
        Args:
            xlsx_data: Raw XLSX data
            xlsx_path: Optional path to the XLSX file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Create a temporary file for the XLSX
            with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir, suffix='.xlsx') as temp_file:
                temp_path = temp_file.name
                temp_file.write(xlsx_data)
            
            # Extract data from XLSX
            workbook = openpyxl.load_workbook(temp_path, data_only=True)
            sheet_data = []
            
            for sheet in workbook.sheetnames:
                current_sheet = workbook[sheet]
                
                # Convert sheet to list of lists
                rows = []
                for row in current_sheet.rows:
                    rows.append([cell.value for cell in row])
                
                sheet_data.append({
                    "name": sheet,
                    "data": rows
                })
            
            # Create a text representation
            text_content = []
            for sheet in sheet_data:
                text_content.append(f"Sheet: {sheet['name']}")
                for row in sheet['data']:
                    text_content.append("\t".join([str(cell) if cell is not None else "" for cell in row]))
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "text": "\n".join(text_content),
                "sheets": sheet_data,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "xlsx",
                    "sheet_count": len(sheet_data),
                    "original_path": xlsx_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing XLSX content: {str(e)}")
            return {
                "text": "",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "xlsx",
                    "error": str(e),
                    "original_path": xlsx_path
                }
            }
    
    def parse_attachment(self, 
                        attachment_data: bytes, 
                        filename: str, 
                        content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse attachment content based on file type.
        
        Args:
            attachment_data: Raw attachment data
            filename: Filename of the attachment
            content_type: Optional MIME type
            
        Returns:
            Dictionary with extracted content
        """
        if not attachment_data:
            return self._create_empty_content_dict()
            
        try:
            # Determine file type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
            
            # Handle different file types
            if content_type:
                if content_type.startswith('image/'):
                    return self.parse_image(attachment_data, filename)
                elif content_type == 'application/pdf':
                    return self.parse_pdf(attachment_data, filename)
                elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                    'application/msword']:
                    return self.parse_word_document(attachment_data, filename)
                elif content_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                     'application/vnd.ms-excel']:
                    return self.parse_excel_document(attachment_data, filename)
            
            # Fall back to extension-based detection
            ext = Path(filename).suffix.lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return self.parse_image(attachment_data, filename)
            elif ext == '.pdf':
                return self.parse_pdf(attachment_data, filename)
            elif ext in ['.doc', '.docx']:
                return self.parse_word_document(attachment_data, filename)
            elif ext in ['.xls', '.xlsx']:
                return self.parse_excel_document(attachment_data, filename)
            elif ext in ['.txt', '.csv', '.md', '.log']:
                # Handle plain text
                try:
                    text_content = attachment_data.decode('utf-8')
                except UnicodeDecodeError:
                    # Try another encoding
                    try:
                        text_content = attachment_data.decode('latin-1')
                    except:
                        text_content = f"[Binary file: {filename}]"
                
                return {
                    "text": text_content,
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "text",
                        "original_path": filename
                    }
                }
            else:
                # Unhandled file type
                return {
                    "text": f"[Unsupported file type: {filename}]",
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "unknown",
                        "original_path": filename
                    }
                }
                
        except Exception as e:
            logger.error(f"Error parsing attachment {filename}: {str(e)}")
            return {
                "text": f"[Error parsing {filename}]",
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "error",
                    "error": str(e),
                    "original_path": filename
                }
            }
    
    def parse_table(self, table_data: List[List[Any]]) -> Dict[str, Any]:
        """
        Parse and format table data.
        
        Args:
            table_data: Table data as list of lists
            
        Returns:
            Dictionary with formatted table information
        """
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(table_data)
            
            # Use first row as header if it looks like a header
            if len(df) > 1:
                # Check if first row is different from other rows
                first_row_types = [type(x) for x in df.iloc[0] if x is not None]
                other_rows_types = [type(x) for row in df.iloc[1:].values for x in row if x is not None]
                
                if len(first_row_types) > 0 and all(t == str for t in first_row_types):
                    # First row is all strings, likely a header
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
            
            # Generate text representation
            text_representation = df.to_string(index=False)
            
            # Generate statistics for numeric columns
            stats = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[str(col)] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median())
                    }
            
            return {
                "text": text_representation,
                "data": table_data,
                "columns": list(df.columns),
                "row_count": len(df),
                "column_count": len(df.columns),
                "statistics": stats,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "table"
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing table data: {str(e)}")
            return {
                "text": str(table_data),
                "data": table_data,
                "metadata": {
                    "parsed_at": datetime.now().isoformat(),
                    "content_type": "table",
                    "error": str(e)
                }
            }
    
    def _extract_text_from_soup(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Cleaned text content
        """
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up special characters
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}-]', '', text)
        
        return text.strip()
    
    def _extract_tables_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract tables from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of table dictionaries
        """
        tables = []
        
        for i, table_elem in enumerate(soup.find_all('table')):
            try:
                # Extract rows
                rows = []
                for row in table_elem.find_all('tr'):
                    # Extract cells (th or td)
                    cells = []
                    for cell in row.find_all(['th', 'td']):
                        # Get cell content and clean it
                        cell_text = cell.get_text(strip=True)
                        cells.append(cell_text)
                    
                    if cells:  # Skip empty rows
                        rows.append(cells)
                
                if rows:  # Skip empty tables
                    # Parse table data
                    table_info = self.parse_table(rows)
                    table_info['id'] = i
                    
                    # Add caption if available
                    caption = table_elem.find('caption')
                    if caption:
                        table_info['caption'] = caption.get_text(strip=True)
                    
                    tables.append(table_info)
                    
            except Exception as e:
                logger.error(f"Error extracting table {i}: {str(e)}")
                tables.append({
                    "id": i,
                    "error": str(e),
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "table"
                    }
                })
        
        return tables
    
    def _extract_images_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract images from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of image dictionaries
        """
        images = []
        
        for i, img in enumerate(soup.find_all('img')):
            try:
                # Get image attributes
                src = img.get('src', '')
                alt = img.get('alt', '')
                title = img.get('title', '')
                
                # Skip empty sources or known icons
                if not src or '/icons/' in src or '/emoji/' in src:
                    continue
                
                # Look for caption (sometimes in figcaption or nearby p element)
                caption = ""
                if img.parent and img.parent.name == 'figure':
                    figcaption = img.parent.find('figcaption')
                    if figcaption:
                        caption = figcaption.get_text(strip=True)
                
                # Create image info
                image_info = {
                    "id": i,
                    "src": src,
                    "alt": alt,
                    "title": title,
                    "caption": caption,
                    "ocr_text": "",  # Will be filled if OCR is performed
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "image"
                    }
                }
                
                # Try to perform OCR if enabled and source is reachable
                if self.ocr_enabled and src.startswith(('http://', 'https://')):
                    try:
                        # Download image (with timeout)
                        response = requests.get(src, timeout=5)
                        if response.status_code == 200:
                            # Process with OCR
                            ocr_result = self.parse_image(response.content)
                            image_info["ocr_text"] = ocr_result.get("text", "")
                    except Exception as ocr_error:
                        logger.warning(f"Error performing OCR on image {i}: {str(ocr_error)}")
                
                images.append(image_info)
                
            except Exception as e:
                logger.error(f"Error extracting image {i}: {str(e)}")
                images.append({
                    "id": i,
                    "error": str(e),
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "image"
                    }
                })
        
        return images
    
    def _extract_links_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract links from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of link dictionaries
        """
        links = []
        
        for a in soup.find_all('a', href=True):
            try:
                href = a.get('href', '')
                text = a.get_text(strip=True)
                
                # Skip empty links or anchors
                if not href or href.startswith('#'):
                    continue
                
                links.append({
                    "href": href,
                    "text": text
                })
                
            except Exception as e:
                logger.error(f"Error extracting link: {str(e)}")
        
        return links
    
    def _extract_code_blocks_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract code blocks from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of code block dictionaries
        """
        code_blocks = []
        
        # Look for pre and code elements
        for i, elem in enumerate(soup.find_all(['pre', 'code'])):
            try:
                # Get code content
                code = elem.get_text(strip=True)
                
                # Skip empty code blocks
                if not code:
                    continue
                
                # Get language if available
                language = ""
                if 'class' in elem.attrs:
                    for cls in elem['class']:
                        if cls.startswith(('language-', 'brush:')):
                            language = cls.replace('language-', '').replace('brush:', '')
                            break
                
                code_blocks.append({
                    "id": i,
                    "code": code,
                    "language": language
                })
                
            except Exception as e:
                logger.error(f"Error extracting code block {i}: {str(e)}")
        
        return code_blocks
    
    def _extract_structure_from_soup(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract document structure from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with document structure
        """
        structure = {
            "title": "",
            "headings": []
        }
        
        # Extract title
        title_elem = soup.find('title')
        if title_elem:
            structure["title"] = title_elem.get_text(strip=True)
        
        # Extract headings
        for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            try:
                level = int(heading.name[1])
                text = heading.get_text(strip=True)
                
                structure["headings"].append({
                    "id": i,
                    "level": level,
                    "text": text
                })
                
            except Exception as e:
                logger.error(f"Error extracting heading {i}: {str(e)}")
        
        return structure
    
    def _extract_embedded_content_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract embedded content from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of embedded content dictionaries
        """
        embedded = []
        
        # Look for iframes, embeds, objects, etc.
        for i, elem in enumerate(soup.find_all(['iframe', 'embed', 'object'])):
            try:
                # Get element attributes
                tag_name = elem.name
                attrs = {k: v for k, v in elem.attrs.items()}
                
                embedded.append({
                    "id": i,
                    "type": tag_name,
                    "attributes": attrs,
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "embedded"
                    }
                })
                
            except Exception as e:
                logger.error(f"Error extracting embedded content {i}: {str(e)}")
        
        # Look for Confluence macros
        for i, elem in enumerate(soup.find_all(class_=lambda c: c and 'confluence-macro' in c)):
            try:
                # Get macro data
                macro_name = ""
                macro_params = {}
                
                # Extract data from 'data-macro-name' attribute
                if 'data-macro-name' in elem.attrs:
                    macro_name = elem.attrs['data-macro-name']
                
                # Extract parameters
                if 'data-macro-parameters' in elem.attrs:
                    try:
                        macro_params = json.loads(elem.attrs['data-macro-parameters'])
                    except:
                        macro_params = {"raw": elem.attrs['data-macro-parameters']}
                
                # Extract content
                macro_content = elem.get_text(strip=True)
                
                embedded.append({
                    "id": i,
                    "type": "confluence-macro",
                    "name": macro_name,
                    "parameters": macro_params,
                    "content": macro_content,
                    "metadata": {
                        "parsed_at": datetime.now().isoformat(),
                        "content_type": "macro"
                    }
                })
                
            except Exception as e:
                logger.error(f"Error extracting Confluence macro {i}: {str(e)}")
        
        return embedded
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text content.
        
        Args:
            text: Text content
            
        Returns:
            Language code
        """
        # Skip empty or very short texts
        if not text or len(text) < 50:
            return "unknown"
            
        try:
            return detect(text[:1000])  # Use first 1000 chars for efficiency
        except:
            return "unknown"
    
    def _create_empty_content_dict(self, error: str = None) -> Dict[str, Any]:
        """
        Create an empty content dictionary.
        
        Args:
            error: Optional error message
            
        Returns:
            Empty content dictionary
        """
        result = {
            "text": "",
            "tables": [],
            "images": [],
            "links": [],
            "code_blocks": [],
            "structure": {
                "title": "",
                "headings": []
            },
            "embedded": [],
            "language": "unknown",
            "metadata": {
                "parsed_at": datetime.now().isoformat(),
                "content_type": "empty"
            }
        }
        
        if error:
            result["metadata"]["error"] = error
            
        return result
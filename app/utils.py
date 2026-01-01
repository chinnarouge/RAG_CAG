import os
import io
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document


def read_file(file, filename=None):
    if filename is None:
        filename = getattr(file, 'name', getattr(file, 'filename', ''))
    
    ext = os.path.splitext(filename)[1].lower()
    
    if hasattr(file, 'read'):
        content = file.read()
        if hasattr(file, 'seek'):
            file.seek(0)
    else:
        content = file
    
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    if ext == '.pdf':
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    
    elif ext == '.docx':
        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs)
    
    elif ext == '.txt':
        return content.decode('utf-8', errors='ignore')
    
    elif ext == '.csv':
        df = pd.read_csv(io.BytesIO(content))
        return df.to_string()
    
    else:
        return content.decode('utf-8', errors='ignore')

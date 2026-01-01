import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def chunk_text(text, max_len=2000, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_len + sentence_len > max_len and current_chunk:
            chunks.append(" ".join(current_chunk))
            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_len = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        
        current_chunk.append(sentence)
        current_len += sentence_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

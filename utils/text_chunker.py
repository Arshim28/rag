from typing import List

def chunk_text(text: str, max_chunk_size: int = 8000, overlap: int = 200) -> List[str]:
    """
    Chunk a text into smaller pieces for embedding.
    
    Args:
        text: The text to chunk.
        max_chunk_size: The maximum size of each chunk.
        overlap: The amount of overlap between chunks.
        
    Returns:
        A list of chunks.
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        if end < len(text):
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1 and last_newline > start + max_chunk_size // 2:
                end = last_newline + 1
            else:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + max_chunk_size // 2:
                    end = last_space + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else start + 1
    
    return chunks
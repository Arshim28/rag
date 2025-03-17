import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from rag_pipeline import RAGPipeline

def main():
    """Main entry point for the RAG pipeline."""
    parser = argparse.ArgumentParser(description='Hybrid RAG Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration file')
    parser.add_argument('--pdf', type=str, help='Path to the PDF file to process')
    parser.add_argument('--query', type=str, help='Question to ask')
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline(args.config)
    
    if args.pdf:
        pdf_path = args.pdf
        metadata = {
            "source": Path(pdf_path).name,
            "processed_at": datetime.now().isoformat(),
        }
        
        ids = pipeline.process_document(pdf_path, metadata)
        print(f"Processed {pdf_path} and added {len(ids)} documents to the vector store.")
    
    if args.query:
        answer = pipeline.query(args.query)
        print(f"Query: {args.query}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
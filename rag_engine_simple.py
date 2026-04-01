from typing import List, Dict, Tuple
from pathlib import Path
import re

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config import config
from logger import logger


class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata or {}


class SimpleRAGEngine:
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.documents = []
        self.chunks = []
        self.demo_mode = False
        
        self.logger.info("Simple RAG Engine initialized (in-memory)")
    
    def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Simple RAG Engine...")
            
            if not OLLAMA_AVAILABLE:
                self.logger.warning("Ollama not installed → Demo mode ON")
                self.demo_mode = True
                return True
            
            try:
                ollama.list()
                self.logger.info("Ollama connection successful")
                return True
            except Exception:
                self.logger.warning("Ollama not running → Demo mode ON")
                self.demo_mode = True
                return True
            
        except Exception as e:
            self.logger.error("Initialization failed", exception=e)
            self.demo_mode = True
            return True
    
    def ingest_documents(self, file_paths: List[str]) -> Dict:
        stats = {"successful": 0, "failed": 0, "total_chunks": 0}
        
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self._split_text(content)
                
                for chunk in chunks:
                    self.chunks.append({
                        'content': chunk,
                        'source': Path(path).name,
                        'path': path
                    })
                
                stats["successful"] += 1
                stats["total_chunks"] += len(chunks)
                
            except Exception as e:
                stats["failed"] += 1
        
        return stats
    
    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored_chunks[:top_k]]
    
    def query(self, question: str, language: str = "en") -> Tuple[str, List[Document]]:
        try:
            # 🚀 DEMO MODE (NO OLLAMA)
            if self.demo_mode:
                relevant_chunks = self._simple_search(question, top_k=1)
                
                if relevant_chunks:
                    return relevant_chunks[0]['content'], []
                
                return "Demo mode: Try asking about shipping, refund, payment, or support.", []
            
            # 🔥 NORMAL MODE (WITH OLLAMA)
            relevant_chunks = self._simple_search(question, top_k=3)
            
            if not relevant_chunks:
                return self._get_direct_answer(question), []
            
            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
            
            prompt = f"""You are a helpful customer support agent.

Context:
{context}

Question: {question}

Answer:"""
            
            response = ollama.generate(
                model=self.config.llm.model_name,
                prompt=prompt
            )
            
            return response['response'].strip(), []
            
        except Exception:
            return "Something went wrong. Try again.", []
    
    def _get_direct_answer(self, question: str) -> str:
        try:
            response = ollama.generate(
                model=self.config.llm.model_name,
                prompt=question
            )
            return response['response'].strip()
        except:
            return "Unable to process request."
    
    def get_statistics(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(set(c['source'] for c in self.chunks)),
        }
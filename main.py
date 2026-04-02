import asyncio
import sys
from pathlib import Path

from config import config
from logger import logger
from rag_engine_simple import SimpleRAGEngine  
from audio_handler import AudioHandler
from websocket_server import SupportAgentServer


class CustomerSupportAgent:
    def __init__(self):
        self.logger = logger
        self.rag = None
        self.audio = None
        self.server = None
        
        print("=" * 60)
        print("Customer Support Agent - Starting")
        print(f"Python Version: {sys.version}")
        print("=" * 60)
    
    def _create_sample_kb(self):
        kb_file = config.paths.knowledge_base_dir / "sample_faq.txt"
        kb_file.write_text("""
Customer Support FAQ

Q: What are your business hours?
A: We're available 24/7 for customer support through this chat system. 
Our live agents are available Monday-Friday 9 AM - 6 PM EST.
        """)
        self.logger.info("Created sample knowledge base")
    
    async def initialize(self):
        try:
            kb_files = list(config.paths.knowledge_base_dir.glob("*.txt"))
            if not kb_files:
                self.logger.info("No knowledge base found, creating sample...")
                self._create_sample_kb()
                kb_files = list(config.paths.knowledge_base_dir.glob("*.txt"))
            
            self.logger.info("Initializing RAG Engine...")
            self.rag = SimpleRAGEngine()
            
            self.rag.initialize()
            
            if kb_files:
                self.logger.info(f"Loading {len(kb_files)} knowledge base files...")
                stats = self.rag.ingest_documents([str(f) for f in kb_files])
                self.logger.info(f"Loaded {stats['successful']} files")
            
            self.audio = AudioHandler()
            self.audio.initialize()
            
            self.server = SupportAgentServer(self.rag, self.audio)
            
            return True
            
        except Exception as e:
            self.logger.error("Initialization failed", exception=e)
            return False
    
    async def run(self):
        if not await self.initialize():
            sys.exit(1)
        
        try:
            await self.server.start()
        except Exception as e:
            self.logger.error("Server error", exception=e)
            sys.exit(1)


def main():
    try:
        app = CustomerSupportAgent()
        asyncio.run(app.run())
    except Exception as e:
        logger.error("Application failed", exception=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
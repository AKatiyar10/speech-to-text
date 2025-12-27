from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import tempfile
from typing import Optional
import whisper
import logging
import time
import torch
import ollama

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech-to-Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GrammarCorrector:
    """Gemma grammar correction"""
    def __init__(self, model_name="gemma2:2b"):
        self.model_name = model_name
        logger.info(f"Loading Grammar Corrector: {model_name}")
        
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            # Quick test
            self.client.generate(model=model_name, prompt="test", options={'num_predict': 1})
            logger.info("✓ Ollama connected!")
            self.enabled = True
        except Exception as e:
            logger.error(f"Ollama failed: {e}")
            self.enabled = False
    
    async def correct_text(self, text: str) -> str:
        """Correct grammar"""
        if not self.enabled or len(text.strip()) < 3:
            return text
        
        try:
            start = time.time()
            
            prompt = f"""Fix grammar and punctuation. Output only corrected text.

{text}

Corrected:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.0,
                    'num_predict': 80,
                    'stop': ['\n']
                }
            )
            
            corrected = response['response'].strip().strip('"\'').replace('Corrected:', '').strip()
            
            if corrected != text:
                logger.info(f"Corrected ({time.time()-start:.1f}s): '{text}' → '{corrected}'")
            
            return corrected if corrected else text
            
        except Exception as e:
            logger.error(f"Grammar error: {e}")
            return text

class WhisperProcessor:
    """
    Simple, reliable processor
    Key: Process each complete chunk immediately
    """
    def __init__(self, model_size="base", enable_grammar=True):
        logger.info("Loading Whisper...")
        self.model = whisper.load_model(model_size)
        logger.info("✓ Whisper loaded!")
        
        self.grammar = GrammarCorrector() if enable_grammar else None
        self.grammar_enabled = enable_grammar and (self.grammar.enabled if self.grammar else False)
        
        # Simpler approach: accumulate 3-4 chunks before processing
        self.audio_chunks = []
        self.chunks_threshold = 3  # Process every 3 chunks (3 seconds if 1s chunks)
        self.last_process_time = time.time()
        
        self.stats = {"total": 0, "processed": 0, "skipped": 0}
        
        logger.info(f"Grammar: {'✓ Enabled' if self.grammar_enabled else '✗ Disabled'}")
        logger.info(f"Processing every {self.chunks_threshold} chunks (~{self.chunks_threshold}s)")
    
    async def add_chunk(self, audio_data: bytes) -> Optional[dict]:
        """
        Add chunk and process when we have enough
        """
        self.audio_chunks.append(audio_data)
        self.stats["total"] += 1
        
        # Process when we have enough chunks
        if len(self.audio_chunks) >= self.chunks_threshold:
            logger.info(f"Processing {len(self.audio_chunks)} chunks ({sum(len(c) for c in self.audio_chunks)/1024:.1f}KB)")
            return await self._process_chunks()
        
        return None
    
    async def _process_chunks(self) -> Optional[dict]:
        """Process accumulated chunks"""
        if not self.audio_chunks:
            return None
        
        start_time = time.time()
        
        try:
            # Combine all chunks
            combined_audio = b''.join(self.audio_chunks)
            total_size = len(combined_audio)
            
            logger.info("="*60)
            logger.info(f"Processing {len(self.audio_chunks)} chunks = {total_size/1024:.1f}KB")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                temp_path = f.name
                f.write(combined_audio)
            
            # Verify file
            file_size = os.path.getsize(temp_path)
            logger.info(f"File created: {file_size/1024:.1f}KB")
            
            if file_size < 5000:  # Less than 5KB
                logger.warning("File too small, skipping")
                self.audio_chunks.clear()
                os.remove(temp_path)
                return {"text": "", "skipped": True, "reason": "too_small"}
            
            # Transcribe
            whisper_start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, temp_path)
            whisper_time = time.time() - whisper_start
            
            logger.info(f"Whisper: {whisper_time:.2f}s")
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not transcription or len(transcription.strip()) < 2:
                logger.info("Empty transcription")
                self.audio_chunks.clear()
                self.stats["skipped"] += 1
                return {"text": "", "skipped": True, "reason": "empty"}
            
            logger.info(f"Raw: '{transcription}'")
            
            # Grammar correction
            if self.grammar_enabled:
                grammar_start = time.time()
                corrected = await self.grammar.correct_text(transcription)
                grammar_time = time.time() - grammar_start
                logger.info(f"Grammar: {grammar_time:.2f}s")
                transcription = corrected
            
            # Stats
            self.stats["processed"] += 1
            total_time = time.time() - start_time
            
            # Clear chunks
            self.audio_chunks.clear()
            self.last_process_time = time.time()
            
            logger.info(f"✓ DONE: {total_time:.2f}s | '{transcription}'")
            logger.info("="*60)
            
            return {
                "text": transcription,
                "skipped": False,
                "processing_time": total_time,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.audio_chunks.clear()
            return None
    
    def _transcribe(self, audio_path: str) -> str:
        """Transcribe with Whisper"""
        try:
            result = self.model.transcribe(
                audio_path,
                language="en",
                fp16=False,
                temperature=0.0,
                verbose=False,
                beam_size=5,
                best_of=5
            )
            
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    async def flush(self) -> Optional[dict]:
        """Process remaining chunks"""
        if len(self.audio_chunks) > 0:
            logger.info(f"Flushing {len(self.audio_chunks)} remaining chunks")
            return await self._process_chunks()
        return None

class ConnectionManager:
    """Manages connections"""
    def __init__(self):
        self.connections = {}
        self.processors = {}
        self.count = 0
        
        logger.info("="*60)
        logger.info(" INITIALIZING SYSTEM")
        logger.info("="*60)
        
        # Shared model
        self.shared_model = whisper.load_model("base")
        self.shared_grammar = GrammarCorrector()
        
        logger.info("="*60)
        logger.info(" READY")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = WhisperProcessor("base", True)
        processor.model = self.shared_model
        processor.grammar = self.shared_grammar
        
        self.processors[client_id] = processor
        self.count += 1
        logger.info(f"✓ Connected: {client_id} | Total: {self.count}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.connections.get(client_id)
            
            final = await processor.flush()
            if final and final.get('text') and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": final['text'],
                        "is_final": True
                    })
                except:
                    pass
            
            logger.info(f"Stats: {processor.stats}")
        
        if client_id in self.connections:
            del self.connections[client_id]
            self.count -= 1
        if client_id in self.processors:
            del self.processors[client_id]
        
        logger.info(f"✗ Disconnected: {client_id}")
    
    async def process_and_send(self, client_id: str, audio_data: bytes):
        processor = self.processors.get(client_id)
        websocket = self.connections.get(client_id)
        
        if not processor or not websocket:
            return
        
        try:
            result = await processor.add_chunk(audio_data)
            
            if result and result.get('text'):
                await websocket.send_json({
                    "type": "transcription",
                    "text": result['text'],
                    "is_final": True,
                    "metadata": {
                        "time": result.get('processing_time', 0)
                    }
                })
                logger.info(f"✓ Sent: '{result['text']}'")
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"service": "Speech-to-Text", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "connections": manager.count}

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Ready - Speak for 3-4 seconds",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" SPEECH-TO-TEXT SERVER")
    print(" Whisper + Gemma Grammar")
    print(" Process every 3 seconds")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

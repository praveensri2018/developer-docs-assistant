from io import BytesIO
from pathlib import Path
from threading import RLock
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from core import Assistant

app = FastAPI(title='Developer Documentation Assistant')
assistant = Assistant()
lock = RLock()
ROOT = Path(__file__).parent

def seed():
    return [(p.name, 1, p.read_text()) for p in sorted((ROOT / 'samples').glob('*.md'))]
assistant.ingest(seed())

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ok', 'chunks': len(assistant.chunks)}

@app.post('/ask')
def ask(body: Question):
    with lock:
        return assistant.ask(body.question)

@app.post('/reset')
def reset():
    with lock:
        return {'chunks': assistant.ingest(seed())}

@app.post('/upload')
async def upload(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(400, 'Maximum 10 files.')
    docs, total = [], 0
    for file in files:
        content = await file.read(5 * 1024 * 1024 + 1)
        total += len(content)
        if len(content) > 5 * 1024 * 1024 or total > 10 * 1024 * 1024:
            raise HTTPException(413, 'Maximum 5 MB per file, 10 MB total.')
        name = Path(file.filename or 'document').name
        try:
            if name.lower().endswith('.pdf'):
                reader = PdfReader(BytesIO(content))
                if len(reader.pages) > 100:
                    raise ValueError('Maximum 100 PDF pages.')
                docs.extend((name, i+1, page.extract_text() or '') for i, page in enumerate(reader.pages))
            elif name.lower().endswith(('.txt', '.md')):
                docs.append((name, 1, content.decode('utf-8')))
            else:
                raise ValueError('Use PDF, TXT or MD files.')
        except Exception:
            raise HTTPException(400, 'Unable to read document; use text-based PDF, UTF-8 TXT or MD.')
    if sum(len(d[2]) for d in docs) > 500000:
        raise HTTPException(413, 'Extracted text exceeds 500,000 characters.')
    try:
        with lock:
            return {'chunks': assistant.ingest(docs)}
    except ValueError:
        raise HTTPException(400, 'No indexable text found.')

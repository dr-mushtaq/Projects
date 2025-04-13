from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import stanza
import os

app = FastAPI()
templates = Jinja2Templates(directory="mini")

# Only Indian languages officially supported by Stanza
supported_indian_langs = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil", 
    "te": "Telugu",
    "mr": "Marathi",
    "sa": "Sanskrit"
}

loaded_pipelines = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str
    lang: str

def download_model(lang):
    """Download model with error handling"""
    try:
        print(f"Downloading {lang} model...")
        stanza.download(lang)
        return True
    except Exception as e:
        print(f"Failed to download {lang}: {str(e)}")
        return False

@app.post("/tag")
async def pos_tag(input: InputText):
    lang = input.lang
    if lang not in supported_indian_langs:
        return {"result": f"Language not supported. Available: {', '.join(supported_indian_langs.values())}"}
    
    try:
        if lang not in loaded_pipelines:
            if not download_model(lang):
                return {"result": f"Failed to download model for {supported_indian_langs[lang]}"}
            
            # Special config for Bengali
            config = {"tokenize_pretokenized": True} if lang == "bn" else {}
            loaded_pipelines[lang] = stanza.Pipeline(lang=lang, **config)

        doc = loaded_pipelines[lang](input.text)
        return {"result": " ".join(f"{word.text}/{word.upos}" for sent in doc.sentences for word in sent.words)}
    
    except Exception as e:
        return {"result": f"Error processing text: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": supported_indian_langs
    })


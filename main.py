from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import base64
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer(req: Request):
    image_bytes = base64.b64decode(req.image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    response = model.generate_content([
        req.question,
        image
    ])

    return {"answer": response.text.strip()}

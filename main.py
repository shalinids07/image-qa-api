from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import base64

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            req.question,
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        ]
    )

    return {"answer": response.text.strip()}

from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fake News Classification API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def classify_text(input: TextInput):
    result = predict(input.text)
    return result

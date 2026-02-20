from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# Use relative path to your local model folder
model_path = os.path.join(os.path.dirname(__file__), "distilbert_fake_news_model")

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    return {"prediction": int(pred_class), "confidence": float(confidence)}

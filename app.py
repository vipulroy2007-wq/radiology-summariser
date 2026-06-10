from flask import Flask, request, jsonify
from transformers import BartTokenizer, AutoModelForSeq2SeqLM
import torch
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model")

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print("Loading tokenizer...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model.eval()
        print("Model loaded")

def summarise(text, max_length=60, min_length=5):
    load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/summarise", methods=["POST"])
def summarise_endpoint():
    data = request.get_json()
    if not data or "findings" not in data:
        return jsonify({"error": "Missing findings field"}), 400
    findings = data["findings"].strip()
    if not findings:
        return jsonify({"error": "findings field is empty"}), 400
    return jsonify({"impression": summarise(findings)}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

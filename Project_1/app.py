from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

print("Loading model...")
summariser = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    tokenizer="sshleifer/distilbart-cnn-6-6"
)
print("Model loaded and ready")

def summarise(text):
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])
    
    input_length = len(text.split())
    max_len = min(60, max(10, input_length - 1))
    min_len = min(5, max_len - 1)
    
    result = summariser(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=2  
    )
    return result[0]["summary_text"]

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

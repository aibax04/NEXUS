from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("leo")

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_question = data.get("query", "")
    print("Received:", user_question)
    
    # Step 1: Embed the question
    query_embedding = embedding_model.encode(user_question).tolist()
    
    # Step 2: Query Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    matches = results.get("matches", [])
    
    if not matches:
        return jsonify({"answer": "❌ Sorry, I couldn't find any relevant information."})

    # Step 3: Prepare context from retrieved documents
    context = "\n\n".join([
        match["metadata"].get("text", "") for match in matches
    ])[:1500]  # Optional trim to avoid token overflow
    
    # Step 4: Construct prompt for the LLM
    prompt = f"""
    You are a helpful assistant. Review the data retrieved and frame a good answer.

    Context:
    {context}

    Question:
    {user_question}
    """
    
    # Step 5: Generate answer using FLAN-T5
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Bot response:", answer)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load model and Pinecone
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("leo")

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_question = data.get("query", "")
    print("Received:", user_question)
    
    # Encode the query
    query_embedding = embedding_model.encode(user_question).tolist()
    
    # Query Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    if not results.get("matches"):
        answer = "❌ Sorry, I couldn't find any relevant information."
    else:
        answer = results["matches"][0]["metadata"].get("text", "⚠️ No data found.")
    
    print("Bot response:", answer)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)


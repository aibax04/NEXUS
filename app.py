from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory user store
users = {}

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def get_id(self):
        return self.id

@login_manager.user_loader
def load_user(user_id):
    user = users.get(user_id)
    if user:
        return User(user_id, user['username'], user['password_hash'])
    return None

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in [u['username'] for u in users.values()]:
            flash('Username already exists')
            return redirect(url_for('register'))
        user_id = str(len(users) + 1)
        users[user_id] = {
            'username': username,
            'password_hash': generate_password_hash(password)
        }
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        for user_id, user in users.items():
            if user['username'] == username and check_password_hash(user['password_hash'], password):
                login_user(User(user_id, username, user['password_hash']))
                return redirect(url_for('chat'))
        flash('Invalid credentials')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/account')
@login_required
def account():
    return render_template('account.html', username=current_user.username)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone (old syntax, as you requested)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("leo")

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route("/")
@login_required
def chat():
    return render_template("chat.html", username=current_user.username)

@app.route("/query", methods=["POST"])
@login_required
def query():
    try:
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

        # Step 3: Return the most relevant retrieved text
        answer = matches[0]["metadata"].get("text", "⚠️ No data found.")
        print("Bot response:", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in /query:", str(e))
        return jsonify({"answer": f"⚠️ Internal error: {str(e)}"}), 500

@app.route('/courses')
@login_required
def courses():
    return render_template('courses.html', username=current_user.username)

CORS(app, origins=["http://localhost:3000"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


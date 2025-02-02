from flask import Flask, render_template, request, session
from langchain_ollama import ChatOllama
import markdown

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session storage

# Function to generate response using Ollama
def generate_response(user_input):
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434/")
    response = model.invoke(user_input)
    return markdown.markdown(response.content)  # Converts response to HTML

@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input:
            bot_response = generate_response(user_input)
            session["chat_history"].append({"user": user_input, "ollama": bot_response})
            session.modified = True  # Save session updates

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/clear", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    session.modified = True
    return ("", 204)  # Empty response, triggers page reload in frontend

if __name__ == "__main__":
    app.run(debug=True)

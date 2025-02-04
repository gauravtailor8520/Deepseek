# RAG Application using Ollama & Deepseek r1: 1.5B

This repository contains a Streamlit-based RAG (Retrieval-Augmented Generation) application utilizing the Ollama and Deepseek R1 1.5B models. Additionally, a simple chatbot is implemented using only the Deepseek model.

## Features
- **RAG-based AI Assistant**: Uses Ollama and Deepseek R1 1.5B models for intelligent responses.
- **Simple Chatbot**: A lightweight chatbot powered by Deepseek R1 1.5B.
- **Streamlit UI**: Interactive and user-friendly interface.

## Installation
Clone the repository and install dependencies:

```sh
git clone https://github.com/your-username/rag-app.git
cd rag-app
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:

```sh
streamlit run App.py
```

## File Structure
```
rag-app/
│── App.py            # Main Streamlit app for RAG
│── Rag.py
│── chatbot.py
│── Build RAG Locally with DeepSeek.ipynb     # Ipynb File For RAG Pipeline
│── requirements.txt
│── Simple.py  # Simple Deepseek Chatbot Directly. 
│── assets/           # Images and videos
│   ├── app_screenshot.png
│   ├── chatbot_demo.mp4
│── templates/           # Html file for the Simple Deepseek ChatBot.
│   ├── index.html

```

## Screenshots
### RAG Application
![RAG Application](assets/app_screenshot.png)

### Chatbot Demo
![Chatbot Demo](assets/chatbot_demo.mp4)

## License
This project is licensed under the MIT License.

## Author
[Your Name](https://github.com/your-username)

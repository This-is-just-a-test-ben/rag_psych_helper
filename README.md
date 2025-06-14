# RAG-based AI Assistant: Psychology Helper

This project is part of the Agentic AI Developer Certification (AAIDC) Module 1. It implements a Retrieval-Augmented Generation (RAG) system using ChromaDB, SentenceTransformers, and OpenAI’s GPT model to answer questions based on a local corpus of psychology-related documents, specifically Organizational Psychology, Methodology, and Survey Design.

## Project Overview
- Loads and chunks Wikipedia psychology articles
- Embeds text using `sentence-transformers/all-MiniLM-L6-v2`
- Stores chunks in a persistent ChromaDB vector store
- Accepts natural language questions
- Retrieves relevant documents and feeds them into OpenAI’s GPT
- Returns a contextual, grounded response

## How to Run

### 1. Clone the repo and navigate into the directory:

```bash
git clone <https://github.com/XYZZZZ/rag-psych-helper>
cd <rag-psych-helper>

#Set up a virtual environment
python -m venv rag_env
rag_env\Scripts\activate  # On Windows

#install dependencies
pip install -r requirements.txt

#add your API key in a .env file
OPENAI_API_KEY=your_openai_api_key_here

#Run the script
python rag_project_psych.py

#Folder Structure
> data/
> > wikipedia_psych_articles/
> .env_example
> .gitignore
> LICENSE
> rag_project_psych.py
> README.md
> requirements.txt

#Security Notes
The .env file is ignored via .gitignore

.env_example is included to guide API setup

#Dependencies
langchain
chromadb
sentence-transformers
openai
python-dotenv

#Data License
Wikipedia - Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
If you wish to reuse or share this project, ensure your usage complies with the original license terms of the content used.

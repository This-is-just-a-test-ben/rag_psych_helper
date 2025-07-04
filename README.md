# RAG-based AI Assistant: Psychology Helper

This project is part of the Agentic AI Developer Certification (AAIDC) Module 1. It implements a Retrieval-Augmented Generation (RAG) system using ChromaDB, SentenceTransformers, and OpenAI’s GPT model to answer questions based on a local corpus of psychology-related documents, specifically Organizational Psychology, Methodology, and Survey Design.

## Target Audience

This assistant is ideal for:
- Students and early-career researchers in psychology
- Practitioners in behavioral science or workforce development
- Anyone interested in building AI tools for grounded question answering

## Project Overview

This project implements a domain-specific Retrieval-Augmented Generation (RAG) system focused on psychology-related content. It leverages a combination of open-source tools and APIs to ground LLM responses in local documents.

The system workflow:

1. Load & Chunk: After scrapping the relevant pages for text files are saved in a local folder. Local .txt files are recursively loaded and chunked into an appropriate size with LangChain's RecursiveCharacterTextSplitter.
2. Embed & Store: Chunks are embedded using SentenceTransformers (all-MiniLM-L6-v2) and stored in local ChromaDB instance, which allows for consistent search.
3. Query & Retrieve: A user asks a question, the prompt is embedded and the most relevant chunks are retrieved.
4. Generate Answer: A system prompt is created, and OpenAI’s GPT model answers using the retrieved context (personal ChatGPT API required). The model generates a natural language answer using only the retrieved context.
5. Continued User Interaction: Users continue to interact with the assistant through the terminal interface, submitting questions. The assistant returns a grounded, contextual response using the embedded content. Questions may include, but are not limited to:
- “What is employee engagement?”
- “How do researchers define survey consistency?

## Future Directions

Several enhancements are planned or possible:

- UX: Develop a basic web interface or chatbot frontend for non-technical users.
- Additional source material - Allow input from PDFs, URLs, or APIs in addition to `.txt` files. In addition, draw from a wider knowledge base than Wikipedia.
- Agentic pipeline - Introduce workflow agents (e.g., summarizers, taggers) to assist researchers beyond Q&A.
- Broader domains - Adapt the RAG assistant to other disciplines such as law, finance, or healthcare.
- Creativity - Adjust temperature and test to determine how creative the RAG can be before results stray.

This project lays the foundation for intelligent assistants that can operate within real-world constraints, grounded in trustworthy content and guided by user intent.

## Conclusion

This project demonstrates how a Retrieval-Augmented Generation (RAG) system can be used to ground large language models in a domain-specific dataset. By embedding curated psychology content and connecting it with OpenAIs GPT, the assistant delivers contextually accurate responses. The modular and open-source design makes it easy to adapt for other academic or professional domains.

This assistant satisfies the goals of agentic AI by combining external tools (ChromaDB), autonomous reasoning (query formulation and response synthesis), and memory-based decision-making (semantic chunk retrieval). The project also adheres to open-source best practices.

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

#add your API key in a .env file (must be created by the user)
OPENAI_API_KEY=your_openai_api_key_here

#Run the script
python rag_project_psych.py

#You’ll be prompted to type a question:
## What is employee engagement?
## How do researchers measure consistency in survey design?


#Folder Structure
> data/
> > wikipedia_psych_articles/
> research_db/ # Generated by Chroma when running script
> .env_example
> .env # (NOT INCLUDED IN REPO)
> .gitignore
> LICENSE
> rag_project_psych.py
> README.md
> requirements.txt

#Security Notes
The .env file is ignored via .gitignore
The .env file is not included as this is my personal API Key, create an .env file with the text 
OPENAI_API_KEY="your-api-key-here" 
and save as a .env file to fully utilize the project. 

.env_example is included to guide API setup

#Dependencies
langchain
chromadb
sentence-transformers
openai
python-dotenv

#Data License
Wikipedia - Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
Code: Licensed under CC BY-NC-SA 4.0
If you wish to reuse or share this project, ensure your usage complies with the original license terms of the content used.

# RAG_demo
This is the repository for the demo of RAG (Retrieval-Augmented Generation).
## Installation
Set up the virtual environment
```bash
python -m venv rag_env
```
Activate the created virtual environment
```bash
# Windows
rag_env/Scripts/activate
```
or
```bash
# macOS, Linux
source rag_env/bin/activate
```
Install dependencies using pip
```bash
pip install -r requirements.txt
```
Change the name of the file ```.env.demo``` to ```.env``` and enter values to the corresponding fields.

Set up the Postgres database to store embeddings
```bash
docker compose up -d
```
## Run demo
To start the demo:
```bash
python app.py
```
When the URL ```http://127.0.0.1:7860``` appears, click on it to run the demo.
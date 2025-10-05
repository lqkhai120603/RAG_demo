import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import psycopg2
import gradio as gr
import atexit

# Load environment variables from .env
load_dotenv()

# Constants for chunking
CHUNK_SIZE = 512 
CHUNK_OVERLAP = 50

# Set up LLM
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Set up embedding model
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Database connection
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port="5432"
)

# Close connection when the program ends
atexit.register(conn.close)

# Prompt template to feed to LLM
TEMPLATE = """
Your task is to answer all the questions that users ask based only on the context information provided below.
Please answer the question at length and in detail, with full meaning.
In the answer there is no sentence such as: based on the context provided.
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: 
"""

# Chunking received text
def recursive_character_splitter(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks based on common separators."""
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = [text]
    
    # Simple recursive splitting
    for separator in separators:
        new_chunks = []
        for chunk in chunks:
            # If a chunk is larger than the desired size, split it further
            if len(chunk) > chunk_size:
                parts = chunk.split(separator)
                temp_chunk = ""
                for part in parts:
                    if len(temp_chunk) + len(part) + len(separator) <= chunk_size:
                        temp_chunk += part + separator
                    else:
                        if temp_chunk:
                            new_chunks.append(temp_chunk.strip())
                        temp_chunk = part + separator
                if temp_chunk:
                    new_chunks.append(temp_chunk.strip())
            else:
                new_chunks.append(chunk.strip())
        
        chunks = new_chunks
        
        # Apply overlap correction (simplified: only takes effect if the chunking reduced size)
        if all(len(c) <= chunk_size for c in chunks):
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0 and chunk_overlap > 0:
                    # Append a small overlapping tail from the previous chunk
                    overlap_text = chunks[i-1][-chunk_overlap:]
                    final_chunks.append(overlap_text + chunk)
                else:
                    final_chunks.append(chunk)
            
            # Remove duplicates and ensure chunks are reasonable size
            return [c for c in final_chunks if len(c) > 10]

    return chunks

# Answer a question using RAG
def answer(query, top_k=3, threshold=0.4):
    cur = conn.cursor()
    emb = embed_model.encode([query], convert_to_tensor=False)[0].tolist()
    cur.execute("""
        SELECT content, embedding <=> %s::vector AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s;
    """, (emb, top_k))
    results = cur.fetchall()
    cur.close()

    if results:
        retrieved_texts = [r[0] for r in results if r[1] < threshold]
        if retrieved_texts:
            context = "\n---\n".join(retrieved_texts)
            prompt = TEMPLATE.format(context=context, query=query)
            response = llm.generate_content(prompt)
            return response.text
        else:
            return "No relevant information was found."
    else:
        return "No relevant information was found."

# Add document to database
def add_document(document):
    cur = conn.cursor()
    try:
        if not document.strip():
            return "Document cannot be empty."
        chunks = recursive_character_splitter(document, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return "Document was too short or improperly formatted to create chunks."
        total_chunks_added = 0
        for chunk in chunks:
            # Encode the chunk
            emb = embed_model.encode(
                [chunk],
                convert_to_tensor=False,
                show_progress_bar=False
            )[0].tolist()
            # Save to DB
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
                (chunk, emb)
            )
            total_chunks_added += 1
        conn.commit()
        return f"Document added to the database as {total_chunks_added} chunks!"
    except Exception as e:
        conn.rollback()
        return f"Error: {e}"
    finally:
        cur.close()

# --- Function: Show latest registered data ---
def show_latest_documents(limit=5):
    cur = conn.cursor()
    cur.execute(f"SELECT content FROM documents ORDER BY id DESC LIMIT {limit}")
    rows = cur.fetchall()
    cur.close()
    if rows:
        formatted_docs = "\n".join([f"- {row[0][:150]}..." for row in rows])
        return formatted_docs
    else:
        return "No documents found in the database"

initial_doc_display = show_latest_documents()

# Clear the document input after submission
def clear_doc_input():
    return ""

# Chatbot interface
with gr.Blocks(title="RAG Chatbot Demo") as demo:
    def user(message, chat_history: list):
        chat_history.append({"role": "user", "content": message})
        return "", chat_history

    def bot(chat_history: list):
        if not chat_history:
            return []
        user_input = chat_history[-1]['content']
        bot_message = answer(user_input)
        chat_history.append({"role": "assistant", "content": ""})
        for word in bot_message.split():
            chat_history[-1]['content'] += word + " "
            time.sleep(0.05)
            yield chat_history
        return chat_history

    # --- UI Layout ---
    gr.Markdown("# RAG Chatbot Demo")    
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot(type="messages", height=300)
        msg = gr.Textbox(label="Query")
        clear_btn = gr.Button("Clear")
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    with gr.Tab("Documents"):        
        gr.Markdown("## Add New Document")
        status_message = gr.Textbox(
            label="Status",
            value=f"System ready. Documents available: {initial_doc_display.count('-')}",
            lines=2,
            interactive=False
        )        
        with gr.Row():
            document_input = gr.Textbox(
                label="New document text",
                lines=3,
                placeholder="Enter a new fact or document chunk here"
            )
            add_btn = gr.Button("Add", variant="primary")
            
        gr.Markdown("## Latest Registered Documents")        
        document_display_output = gr.Markdown(
            value=initial_doc_display,
            elem_id="doc_display"
        )

        add_btn.click(
            fn=add_document,
            inputs=[document_input],
            outputs=[status_message]
        ).then(
            fn=clear_doc_input,
            inputs=[],
            outputs=[document_input]
        ).then(
            fn=show_latest_documents,
            inputs=[],
            outputs=[document_display_output]
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
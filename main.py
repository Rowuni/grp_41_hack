import streamlit as st
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np

# === Azure OpenAI Config ===
client = openai.AzureOpenAI(
    api_key="KEY", # à trouver dans Clés et point de terminaison sur le portail Azure
    api_version="2023-05-15",
    azure_endpoint="ENDPOINT" # pareil que la clé
)

EMBEDDING_DEPLOYMENT = "embedding-ada"
GPT_DEPLOYMENT = "gpt-4"

# === Session State Init ===
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None

# === PDF Processing ===
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text(text, max_chars=1000):
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def embed_chunks(chunks):
    return np.array([get_embedding(c) for c in chunks], dtype="float32")

def search_similar_chunks(query, chunks, chunk_embeddings, k=4):
    query_embedding = get_embedding(query)
    index = faiss.IndexFlatL2(len(query_embedding))
    index.add(chunk_embeddings)
    _, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]

def ask_gpt(context, question):
    prompt = f"""Tu es une IA spécialisée en analyse documentaire. En te basant uniquement sur le contenu suivant :

{context}

Réponds à la question suivante :
{question}

Réponds clairement en français sans inventer.
"""
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

# === Streamlit UI ===
st.title("📄 Posez vos questions à votre PDF (RAG + Azure OpenAI)")

uploaded_file = st.file_uploader("📎 Téléchargez un fichier PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📚 Traitement du document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(raw_text)
        embeddings = embed_chunks(chunks)
        st.session_state['chunks'] = chunks
        st.session_state['embeddings'] = embeddings
    st.success(f"✅ {len(chunks)} morceaux extraits et encodés.")

prompt = st.chat_input("💬 Posez votre question ici...")

if prompt:
    st.session_state.past.append(prompt)
    top_chunks = search_similar_chunks(prompt, st.session_state['chunks'], st.session_state['embeddings'])
    context = "\n---\n".join(top_chunks)
    answer = ask_gpt(context, prompt)

    st.markdown(f"**👤 Vous :** {prompt}")
    st.markdown(f"**🤖 IA :** {answer}")

if st.button("🔁 Réinitialiser la session"):
    st.session_state.past.clear()
    st.session_state.chunks.clear()
    st.session_state.embeddings = None

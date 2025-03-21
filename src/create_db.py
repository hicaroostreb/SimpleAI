import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import textwrap

# Configurar API do Gemini
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Carregar modelo de embeddings
model = SentenceTransformer("all-mpnet-base-v2")


def create_embeddings(texts):
    """
    Gera embeddings para uma lista de textos usando processamento em lote.
    """
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True).tolist()
    return embeddings


def create_vector_database(data, persist_directory="vector_db"):
    """
    Cria e armazena embeddings no ChromaDB de forma incremental.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("my_collection")

    # Gerar embeddings em lotes
    embeddings = create_embeddings(data)
    ids = [str(i) for i in range(len(data))]

    # Adicionar ao banco vetorial
    collection.add(embeddings=embeddings, ids=ids, documents=data)

    return collection


def chunk_text(text, max_chunk_size=1000):
    """
    Divide um texto longo em chunks de tamanho máximo, preferindo quebras naturais.
    """
    paragraphs = text.split("\n\n")  # Quebrar por parágrafos
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    """
    Processa o arquivo Markdown e cria o banco vetorial no ChromaDB.
    """
    filepath = "data input/D&D 5.5 - Livro do Jogador 2024.md"

    # Carregar todo o texto e dividir em chunks
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    data = chunk_text(text, max_chunk_size=1000)

    # Criar o banco vetorial
    create_vector_database(data)

    print(f"Vector database criado com {len(data)} chunks!")


if __name__ == "__main__":
    main()

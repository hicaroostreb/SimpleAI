import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

# Set up the Gemini API
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")


def create_embeddings(text):
    """
    Creates embeddings for the given text using the Sentence Transformer model.
    """
    embeddings = model.encode(text).tolist()
    return embeddings


def search_vector_database(collection, query, top_k=5):
    """
    Searches the Chroma DB vector database for the most relevant chunks of data.
    """
    query_embedding = create_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


system_prompt = """Você é um assistente de IA útil que responde a perguntas com base no contexto fornecido."""


def get_gemini_response(query, context):
    """
    Passes the query and the relevant chunks of data to the Gemini model and returns the response.
    """
    prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer: {system_prompt}"
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    response = gemini_model.generate_content(prompt)
    return response.text


def main():
    """
    Main function to load the vector database, take user input, search for relevant chunks of data, and get the response from the Gemini model.
    """
    # Load the vector database
    persist_directory = "vector_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection("my_collection")

    while True:
        # Take user input
        query = input("Digite sua pergunta (ou digite 'exit' para sair): ")
        if query.lower() == "exit":
            break

        # Search the vector database for the most relevant chunks of data
        results = search_vector_database(collection, query)

        # Get the relevant chunks of data
        context = results["documents"][0]

        # Get the response from the Gemini model
        response = get_gemini_response(query, context)

        # Print the response
        print(response)


if __name__ == "__main__":
    main()

import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Load PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    mypages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError(f"No readable text found in {pdf_path}")

    return full_text


# Create text chunks
def create_chunks(full_text, chunksize=500, overlap=100):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# Build vectors
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(embed_vectors, dtype="float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# Load multiple PDFs
def load_base(pdf_path="college_docs"):
    pdf_folder = Path(__file__).resolve().parent / pdf_path

    if not pdf_folder.exists():
        raise ValueError("college_docs folder not found")

    pdf_files = [
        file for file in pdf_folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".pdf"
    ]

    if not pdf_files:
        raise ValueError("No PDF files found in college_docs folder")

    chunks = []

    for pdf in sorted(pdf_files):
        full_text = load_pdf(pdf)
        chunks.extend(create_chunks(full_text))

    if not chunks:
        raise ValueError("No text chunks created from PDFs")

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# Retrieve relevant content
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        return ""

    if index.ntotal == 0:
        return ""

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(query_embed, dtype="float32")
    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# Generate answer using Groq
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):
    if not api_key:
        raise ValueError("Groq API key is missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    if not background.strip():
        return "Information Missing"

    prompt = f"""
    Answer the question using only the information given below.

    If the answer is not available in the documents,
    reply exactly: Information Missing.

    Do not make assumptions.

    Documents:
    {background}

    Question:
    {question}
    """

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

def build_faiss_index(embeddings: np.ndarray, num_labels: int | None = None):
    embeddings = embeddings.astype("float32")
    embeddings = np.ascontiguousarray(embeddings)
    if faiss is None:
        return None, False
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        if num_labels is not None:
            print(f"FAISS enabled with {num_labels} embeddings.")
        return index, True
    except Exception as e:
        print(f"Failed to initialize FAISS. Falling back to NumPy search. Error: {e}")
        return None, False

def search_embedding(
    faiss_index,
    use_faiss: bool,
    db_embs_norm: np.ndarray,
    emb_norm: np.ndarray,
):
    if use_faiss and faiss_index is not None:
        query = emb_norm.reshape(1, -1).astype("float32")
        distances, indices = faiss_index.search(query, 1)
        best_idx = int(indices[0][0])
        best_sim = float(distances[0][0])
    else:
        sims = db_embs_norm @ emb_norm
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
    return best_idx, best_sim
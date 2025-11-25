import os
import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def format_label(raw_label: str) -> str:
    base = os.path.splitext(str(raw_label))[0]
    return base

def collect_identities(images_dir: str):
    identities = []
    if not os.path.isdir(images_dir):
        return identities
    for name in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, name)
        if not os.path.isdir(path):
            continue
        image_files = []
        for fn in os.listdir(path):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(path, fn))
        if not image_files:
            continue
        identities.append((name, image_files))
    return identities

def compute_embeddings(app: FaceAnalysis, identities):
    all_embeddings = []
    all_labels = []
    for label, image_paths in identities:
        identity_embs = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if len(faces) == 0:
                continue
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            emb = face.embedding
            if emb is None:
                continue
            identity_embs.append(emb)
        if not identity_embs:
            continue
        mean_emb = np.mean(np.stack(identity_embs, axis=0), axis=0)
        all_embeddings.append(mean_emb)
        formatted_label = format_label(label)
        all_labels.append(formatted_label)
    if not all_embeddings:
        return None, None
    embeddings_array = np.stack(all_embeddings, axis=0).astype("float32")
    labels_array = np.array(all_labels)
    return embeddings_array, labels_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="data/embeddings")
    parser.add_argument("--model_name", type=str, default="buffalo_l")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    identities = collect_identities(args.images_dir)
    if not identities:
        print(f"No identity folders found in {args.images_dir}")
        return

    app = FaceAnalysis(name=args.model_name)
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings, labels = compute_embeddings(app, identities)
    if embeddings is None:
        print("No embeddings were computed. Check your images.")
        return

    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    labels_path = os.path.join(args.output_dir, "labels.npy")
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)

    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of identities: {len(labels)}")

if __name__ == "__main__":
    main()
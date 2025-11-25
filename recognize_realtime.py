import os
import argparse
import re
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_database(embeddings_dir: str):
    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    labels_path = os.path.join(embeddings_dir, "labels.npy")
    if not os.path.exists(emb_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("embeddings.npy or labels.npy not found in " + embeddings_dir)
    embeddings = np.load(emb_path).astype("float32")
    labels = np.load(labels_path)
    return embeddings, labels

def normalize_embeddings(embeddings: np.ndarray):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings / norms

def format_label(raw_label: str) -> str:
    base = os.path.splitext(str(raw_label))[0]
    name_part = base.replace("_", " ")
    name_part = re.sub(r"(?<!\s)(?<!^)([A-Z])", r" \1", name_part)
    return name_part.strip()

def recognize_realtime(
    embeddings_dir: str,
    model_name: str = "buffalo_l",
    source: int | str = 0,
    threshold: float = 0.4,
):
    db_embs, db_labels = load_database(embeddings_dir)
    db_embs_norm = normalize_embeddings(db_embs)

    face_app = FaceAnalysis(name=model_name)
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = face_app.get(frame)
        if len(faces) == 0:
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        h, w, _ = frame.shape

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            emb = face.embedding
            if emb is None:
                continue

            emb = emb.astype("float32")
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sims = db_embs_norm @ emb_norm
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            raw_label = str(db_labels[best_idx])
            pretty_label = format_label(raw_label)

            if best_sim >= threshold:
                text = f"{pretty_label} {best_sim:.2f}"
                color = (0, 255, 0)
            else:
                text = f"Unknown {best_sim:.2f}"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings")
    parser.add_argument("--model_name", type=str, default="buffalo_l")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video file path")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    # Chuyển source về int nếu là số (webcam), ngược lại giữ string (file video)
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    recognize_realtime(
        embeddings_dir=args.embeddings_dir,
        model_name=args.model_name,
        source=source,
        threshold=args.threshold,
    )

if __name__ == "__main__":
    main()
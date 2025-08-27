import os
from typing import List, Tuple

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except ImportError:
    faiss = None  # We handle the case gracefully

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "/tmp")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"xls", "xlsx", "xlsm", "xlsb"}

# ---------------------------
# Demo corpus + embeddings
# ---------------------------
DEMO_CORPUS = [
    "Customer asked about refund policy and timeline.",
    "Inquiry regarding shipping delays to California region.",
    "User reported app crash when uploading large files.",
    "Request for enterprise pricing and volume discounts.",
    "Feedback: checkout flow confusing on mobile Safari."
]
DEMO_IDS = np.arange(len(DEMO_CORPUS))

EMBED_DIM = 384  # set any fixed dim you like for the hashing trick


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def hashing_trick_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Deterministic, dependency-light 'embedding' using a hashing trick.
    Replace this with your real embedding model.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        # stable bucket using Python's built-in hash, then mod by dim
        bucket = hash(tok) % dim
        vec[bucket] += 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build_demo_faiss(corpus: List[str]) -> Tuple[object, np.ndarray]:
    """
    Build an in-memory FAISS index for the demo corpus.
    """
    if faiss is None:
        return None, None

    xb = np.vstack([hashing_trick_embed(t) for t in corpus]).astype("float32")
    index = faiss.IndexFlatIP(xb.shape[1])  # inner product works with normalized vectors
    index.add(xb)
    return index, xb


FAISS_INDEX, _ = build_demo_faiss(DEMO_CORPUS)


def faiss_search(query_text: str, top_n: int) -> List[Tuple[int, float, str]]:
    """
    Returns list of (corpus_id, score, corpus_text) for the top_n most similar.
    If FAISS isn't installed, we fall back to a pure-numpy search.
    """

    if not query_text:
        return [{"error": "Query text is required."}]

    q = hashing_trick_embed(query_text).reshape(1, -1).astype("float32")

    if FAISS_INDEX is not None:
        scores, ids = FAISS_INDEX.search(q, top_n)
        hits = []
        for i, s in zip(ids[0], scores[0]):
            if i == -1:
                continue
            hits.append((int(i), float(s), DEMO_CORPUS[int(i)]))
        return hits

    # Fallback: numpy cosine similarity
    xb = np.vstack([hashing_trick_embed(t) for t in DEMO_CORPUS])
    sims = (q @ xb.T).ravel()
    top_idx = np.argsort(-sims)[:top_n]
    return [(int(i), float(sims[i]), DEMO_CORPUS[int(i)]) for i in top_idx]


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        input_text=session.get("input_text"),
        last_results=session.pop("last_results", None),
        last_top_n=session.pop("last_top_n", None),
    )


@app.route("/submit_input", methods=["POST"])
def submit_input():
    """
    Accept either:
      - file upload (Excel with a single row and a column named 'customer note')
      - free-form text from textarea 'free_query'
    Save into session['input_text'] and show it.
    """

    free_query = (request.form.get("free_query") or "").strip()
    file = request.files.get("excel_file")

    input_text = None

    # 1) If a file is present, try to read it (Excel or CSV)
    if file and file.filename:
        fname = secure_filename(file.filename)
        ext = fname.rsplit(".", 1)[-1].lower()
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(path)
        try:
            if ext in {"xls", "xlsx", "xlsm", "xlsb"}:
                df = pd.read_excel(path)
            elif ext == "csv":
                df = pd.read_csv(path)
            else:
                flash("Unsupported file type. Please upload Excel or CSV.", "error")
                return redirect(url_for("index"))
            # accept case-insensitive column name
            df.columns = [str(c).strip() for c in df.columns]
            lower_map = {c.lower(): c for c in df.columns}
            if "customer note" not in lower_map:
                flash("File must have a column named 'customer note' (case-insensitive).", "error")
                return redirect(url_for("index"))
            if len(df) != 1:
                flash("File input must contain exactly one row.", "error")
                return redirect(url_for("index"))
            col_name = lower_map["customer note"]
            val = df.iloc[0][col_name]
            if pd.isna(val):
                flash("'customer note' value is empty.", "error")
                return redirect(url_for("index"))
            input_text = str(val).strip()
        except Exception as e:
            flash(f"Failed to read file: {e}", "error")
            return redirect(url_for("index"))
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    # 2) Otherwise try free-form text
    if input_text is None:
        if not free_query:
            flash("Please upload an Excel/CSV file OR enter a free-form query.", "error")
            return redirect(url_for("index"))
        input_text = free_query

    # Save into session and show it
    session["input_text"] = input_text
    flash("Input text captured.", "success")
    return redirect(url_for("index"))


@app.route("/download_results", methods=["GET"])
def download_results():
    # Download the last search results as an Excel file
    last_results = session.get("last_results")
    if not last_results:
        flash("No results to download.", "error")
        return redirect(url_for("index"))
    df = pd.DataFrame(last_results)
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Similar Complaints")
    output.seek(0)
    from flask import send_file
    return send_file(
        output,
        download_name="similar_complaints.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/search", methods=["POST"])
def search():
    if "input_text" not in session or not session["input_text"]:
        flash("No input text found. Submit a file or query first.", "error")
        return redirect(url_for("index"))

    raw_top_n = (request.form.get("top_n") or "").strip()
    try:
        top_n = int(raw_top_n)
        if top_n <= 0:
            raise ValueError
    except Exception:
        flash("top_n must be a positive integer.", "error")
        return redirect(url_for("index"))

    query_text = session["input_text"]

    # === your search function ===
    indices, distances = my_search_function(query_text, top_n)
    # indices -> list[int], distances -> list[float]

    # === complaint library (for demo, pretend it's a pandas DataFrame) ===
    # Example structure: complaints_df = pd.DataFrame({"id":[0,1], "complaint":["...", "..."]})
    results_df = complaints_df.iloc[indices].copy()
    results_df["score"] = distances

    # Convert to dicts for Jinja
    session["last_results"] = results_df.to_dict(orient="records")
    session["last_top_n"] = top_n

    return redirect(url_for("index"))


if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)

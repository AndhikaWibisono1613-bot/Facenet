# verify_pair.py
from utils_facenet import embed_from_path, cosine_similarity

# Ganti dengan path gambar Anda
img1 = "data/train/Andhika/a1.jpg"
img2 = "data/val/Andhika/a1.jpg"

emb1 = embed_from_path(img1)
emb2 = embed_from_path(img2)

if emb1 is None or emb2 is None:
    print("Wajah tidak terdeteksi pada salah satu gambar.")
else:
    sim = cosine_similarity(emb1, emb2)
    print("Cosine similarity:", sim)
    # Threshold umum (awal): 0.8-0.9 (semakin tinggi = semakin ketat)
    threshold = 0.85
    print("Match?", "YA" if sim >= threshold else "TIDAK")
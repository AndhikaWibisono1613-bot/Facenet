# eval_folder.py
import os
import glob
import numpy as np
import joblib
from collections import defaultdict
from utils_facenet import embed_from_path

# Load model
try:
    clf = joblib.load("facenet_svm.joblib")
    print("✅ Model berhasil diload")
    print(f"   Classes available: {clf.classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def predict_emb(emb):
    proba = clf.predict_proba([emb])[0]
    idx = int(np.argmax(proba))
    
    # PERBAIKAN: Gunakan classes_ bukan classes
    return clf.classes_[idx], float(proba[idx])

root = "data/val"
y_true, y_pred = [], []
per_cls = defaultdict(lambda: {"ok": 0, "total": 0})

print("Memulai evaluasi pada folder:", root)

# Cek apakah folder val exists
if not os.path.exists(root):
    print(f"❌ Folder {root} tidak ditemukan!")
    print("Struktur folder yang diperlukan:")
    print("data/")
    print("├── train/")
    print("│   ├── Andhika/")
    print("│   └── Zalda/")
    print("└── val/")
    print("    ├── Andhika/")
    print("    └── Zalda/")
    exit(1)

for cls in sorted(os.listdir(root)):
    pdir = os.path.join(root, cls)
    if not os.path.isdir(pdir):
        continue
    
    print(f"Memproses kelas: {cls}")
    
    for p in glob.glob(os.path.join(pdir, "*")):
        if p.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"  Memproses: {os.path.basename(p)}")
            emb = embed_from_path(p)
            if emb is None:
                print(f"  ⚠️  Wajah tidak terdeteksi: {p}")
                continue
            
            pred, conf = predict_emb(emb)
            y_true.append(cls)
            y_pred.append(pred)
            per_cls[cls]["total"] += 1
            per_cls[cls]["ok"] += int(pred == cls)
            
            status = "✓" if pred == cls else "✗"
            print(f"  {status} {os.path.basename(p)}: True={cls}, Pred={pred}, Conf={conf:.3f}")

if y_true:
    acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    print(f"\n{'='*50}")
    print(f"Hasil Evaluasi:")
    print(f"Accuracy overall: {acc:.4f} ({sum([t==p for t,p in zip(y_true,y_pred)])}/{len(y_true)})")
    
    print(f"\nDetail per kelas:")
    for c, st in per_cls.items():
        if st["total"] > 0:
            acc_cls = st["ok"] / st["total"]
            print(f"  {c}: {st['ok']}/{st['total']} = {acc_cls:.3f}")
else:
    print("❌ Tidak ada gambar yang berhasil diproses!")
    print("   Periksa:")
    print("   - Struktur folder data/val/")
    print("   - Format gambar (.jpg, .png, .jpeg)")
    print("   - Deteksi wajah pada gambar")
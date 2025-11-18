# train_classifier.py - GUNAKAN KNN UNTUK DATA KECIL
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# TAMBAH IMPORT KNN
from sklearn.neighbors import KNeighborsClassifier

# Load data
X = np.load("X_train.npy")
y = np.load("y_train.npy", allow_pickle=True)

print(f"📊 Data shape: {X.shape}")
print(f"📊 Jumlah sampel: {len(X)}")
print(f"📊 Kelas: {np.unique(y)}")
print(f"📊 Distribusi kelas: {np.unique(y, return_counts=True)}")

# GUNAKAN KNN UNTUK DATA KECIL
if len(X) <= 3:
    print("🔧 Menggunakan KNN untuk data kecil...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1, metric="euclidean"))
    ])
else:
    # Gunakan SVM untuk data lebih besar
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"))
    ])

# Training sederhana untuk data kecil
if len(X) < 5:
    print("📝 Training dengan semua data...")
    clf.fit(X, y)
    
    # Test dengan data training
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"✅ Training Accuracy: {accuracy:.4f}")
    
    print("\n📈 Classification Report:")
    print(classification_report(y, y_pred))
else:
    # Gunakan cross-validation untuk data lebih besar
    scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring="accuracy")
    print(f"✅ CV acc mean: {scores.mean():.4f} ± {scores.std():.4f}")
    clf.fit(X, y)

# Simpan model
joblib.dump(clf, "facenet_svm.joblib")
print("💾 Model disimpan ke facenet_svm.joblib")

# Test prediksi dengan data training untuk verifikasi
print("\n🔍 Verifikasi prediksi pada data training:")
for i, (emb, true_label) in enumerate(zip(X, y)):
    pred = clf.predict([emb])[0]
    proba = clf.predict_proba([emb])[0]
    conf = np.max(proba)
    status = "✓" if true_label == pred else "✗"
    print(f"   Sample {i+1}: {true_label} -> {pred} {status} (conf={conf:.4f})")
# predict_one.py - TAMBAHKAN DEBUG SECTION
import joblib
from utils_facenet import embed_from_path, cosine_similarity  # IMPORT cosine_similarity
import numpy as np
import os

def load_model(model_path="facenet_svm.joblib"):
    """Load model dengan error handling"""
    try:
        clf = joblib.load(model_path)
        print(f"✅ Model loaded: {type(clf).__name__}")
        print(f"   Classes: {clf.classes_}")
        return clf
    except FileNotFoundError:
        print(f"❌ Model file {model_path} tidak ditemukan!")
        print("   Jalankan train_classifier.py terlebih dahulu")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image(clf, path, unknown_threshold=0.55):
    """Predict satu gambar dengan model yang sudah diload"""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND", 0.0
    
    emb = embed_from_path(path)
    if emb is None:
        return "NO_FACE", 0.0
    
    try:
        proba = clf.predict_proba([emb])[0]
        idx = int(np.argmax(proba))
        label = clf.classes_[idx]
        conf = float(proba[idx])
        
        if conf < unknown_threshold:
            return "UNKNOWN", conf
        
        return label, conf
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "PREDICTION_ERROR", 0.0

if __name__ == "__main__":
    # Load model
    clf = load_model()
    if clf is None:
        exit(1)
    
    # Test image
    test_img = "data/val/Andhika/a1.jpg"
    
    if not os.path.exists(test_img):
        print(f"❌ File {test_img} tidak ditemukan!")
        print("📁 File yang tersedia:")
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"   {os.path.join(root, file)}")
        exit(1)
    
    # ========== TAMBAH DEBUG DI SINI ==========
    print("\n🔍 DEBUG: Analisis Similarity dengan Training Data")
    
    # Load data training
    try:
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy", allow_pickle=True)
        print(f"📊 Data training: {X_train.shape}")
        
        # Test embedding gambar yang akan diprediksi
        emb_test = embed_from_path(test_img)
        if emb_test is not None:
            print(f"📊 Embedding test shape: {emb_test.shape}")
            
            # Hitung similarity dengan semua data training
            similarities = []
            for i, train_emb in enumerate(X_train):
                sim = cosine_similarity(emb_test, train_emb)
                similarities.append(sim)
                print(f"   Similarity dengan {y_train[i]}: {sim:.3f}")
            
            # Cari yang paling mirip
            most_similar_idx = np.argmax(similarities)
            most_similar_label = y_train[most_similar_idx]
            max_similarity = similarities[most_similar_idx]
            
            print(f"🎯 Paling mirip dengan: {most_similar_label} (similarity: {max_similarity:.3f})")
            
            # Analisis probabilitas model
            print(f"\n🔍 DEBUG: Probabilitas Model")
            proba = clf.predict_proba([emb_test])[0]
            for i, class_name in enumerate(clf.classes_):
                print(f"   {class_name}: {proba[i]:.3f}")
                
        else:
            print("❌ Tidak bisa mengekstrak embedding dari gambar test")
    except Exception as e:
        print(f"❌ Error dalam debug: {e}")
    # ========== AKHIR DEBUG ==========
    
    # Predict seperti biasa
    label, conf = predict_image(clf, test_img)
    print(f"\n🎯 Hasil prediksi:")
    print(f"   Gambar: {test_img}")
    print(f"   Prediksi: {label}")
    print(f"   Confidence: {conf:.3f}")
    
    # Interpretasi
    if label == "NO_FACE":
        print("   ⚠️  Wajah tidak terdeteksi")
    elif label == "UNKNOWN":
        print("   ❓ Wajah dikenali tapi confidence rendah")
    elif label == "FILE_NOT_FOUND":
        print("   ❌ File tidak ditemukan")
    else:
        print(f"   ✅ Diprediksi sebagai: {label}")
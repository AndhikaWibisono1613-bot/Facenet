Analisis Praktikum

utils_facenet.py
Modul inti ini jadi tulang punggung sistem, di mana MTCNN bekerja seperti mata digital yang jeli menemukan dan meluruskan setiap wajah dengan presisi, lalu model InceptionResnetV1 mengubahnya menjadi kode rahasia 512 angka yang unik untuk setiap orang. Fungsi cosine_similarity-nya bertindak seperti pengukur kedekatan hubungan antar wajah, sementara berbagai penanganan error memastikan sistem tetap jalan meski dapat gambar bermasalah.

build_embeddings.py
Script ini mirip petugas arsip yang rajin, menjelajahi setiap folder training dan mengubah foto wajah menjadi data numerik yang rapi. Prosesnya bisa dilacak real-time berkat progress bar, dan ada mekanisme khusus yang mencatat gambar-gambar yang gagal diproses. Hasil akhirnya dua file numpy yang berisi semua pola wajah dalam bentuk angka siap latih.

train_classifier.py
Di sini terjadi proses pelatihan sebenarnya, di mana data wajah yang sudah dinormalisasi diajarkan ke model SVM untuk mengenali pola-pola khas setiap orang. Sistemnya pintar menyesuaikan metode validasi berdasarkan jumlah data, dan model akhirnya disimpan lengkap dengan tes verifikasi untuk memastikan kinerjanya bagus.

predict_one.py
Script ini jadi wajah sistem bagi pengguna, memungkinkan testing gambar tunggal dengan proses dari deteksi sampai prediksi. Ada fitur keamanan yang menandai wajah tak dikenal jika confidence-nya rendah, plus berbagai penanganan masalah dari file hilang sampai wajah tak terdeteksi.

eval_folder.py
Sebagai pengawas kualitas, file ini menguji model secara menyeluruh pada data validasi, menghitung akurasi per kelas dan keseluruhan. Setiap prediksi dicatat detailnya untuk analisis lebih lanjut, memberikan gambaran komprehensif tentang kekuatan dan kelemahan model.

verify_pair.py
Khusus untuk verifikasi satu-satu, script ini membandingkan dua gambar langsung dengan mengukur kemiripan numeriknya. Threshold yang bisa diatur memungkinkan penyesuaian tingkat keamanan, cocok untuk aplikasi yang butuh konfirmasi identitas cepat.

train_knn.py
Alternatif klasifikasi ini menggunakan pendekatan tetangga terdekat, yang lebih sederhana namun efektif untuk data sedikit. Modelnya mencari kemiripan pola langsung dalam data training, tanpa membangun model kompleks seperti SVM.

verify_cli.py
Interface command-line ini memungkinkan operasi verifikasi via terminal, cocok untuk integrasi dengan sistem lain atau automasi tugas. Desainnya sederhana tapi powerful, menerima input gambar dan threshold langsung dari parameter command.

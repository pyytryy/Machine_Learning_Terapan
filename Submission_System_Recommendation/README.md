# Laporan Proyek Machine Learning - Putri Pita Mutia

## Project Overview
**Temukan Lagu Mirip Favoritmu: Rekomendasi Musik Berdasarkan Fitur Akustik dan Genre**

Musik telah menjadi bagian penting dari kehidupan manusia sehari-hari. Di era digital saat ini, jutaan lagu tersedia secara online melalui berbagai platform seperti Spotify, Apple Music, dan YouTube Music. Namun, jumlah lagu yang sangat besar ini justru menimbulkan tantangan baru: pengguna sering kali kesulitan menemukan lagu yang sesuai dengan preferensi mereka. Oleh karena itu, diperlukan sistem yang mampu membantu pengguna dalam menemukan musik yang relevan dan sesuai selera mereka secara otomatis. 
Salah satu solusi dari permasalahan tersebut adalah penggunaan sistem rekomendasi musik. Sistem ini dirancang untuk menyarankan lagu-lagu kepada pengguna berdasarkan pola preferensi, karakteristik lagu, dan interaksi sebelumnya. Terdapat berbagai pendekatan dalam membangun sistem rekomendasi, seperti collaborative filtering dan content-based filtering. Namun, dalam banyak kasus, terutama ketika data interaksi pengguna terbatas, pendekatan content-based filtering menjadi pilihan yang efektif. Pendekatan content-based filtering bekerja dengan menganalisis fitur-fitur dari lagu itu sendiri, seperti genre, danceability, energy, acousticness, instrumentalness, dan sebagainya. Dengan memanfaatkan data tersebut, sistem dapat merekomendasikan lagu-lagu yang memiliki kemiripan dengan lagu yang disukai pengguna, tanpa memerlukan data eksplisit tentang penilaian atau interaksi dari pengguna lain.
Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi musik berbasis content-based filtering yang mampu memberikan rekomendasi lagu serupa berdasarkan karakteristik akustik dan popularitas. Dengan demikian, pengguna dapat menemukan lagu-lagu baru yang relevan dengan preferensi mereka secara lebih mudah dan menyenangkan.

Sistem rekomendasi merupakan komponen inti dalam berbagai platform digital modern, terutama dalam industri hiburan seperti musik. Dengan ratusan juta lagu yang tersedia secara daring, kemampuan untuk secara otomatis menyarankan musik yang relevan dapat meningkatkan pengalaman pengguna secara signifikan, menjaga keterlibatan (engagement), dan bahkan berdampak pada peningkatan retensi pengguna.
Menurut Statista (2024), jumlah pengguna aktif Spotify telah mencapai lebih dari 600 juta pengguna di seluruh dunia, dan sistem rekomendasi mereka (seperti "Discover Weekly" dan "Daily Mix") berperan besar dalam membuat pengguna tetap aktif menggunakan layanan mereka. Bahkan disebutkan bahwa lebih dari 30% waktu mendengarkan di Spotify berasal dari hasil rekomendasi otomatis.
Namun, tidak semua sistem rekomendasi membutuhkan data pengguna yang kompleks. Dalam konteks tertentu, seperti pengembangan aplikasi musik lokal, proyek tugas akhir, atau eksperimen AI, pendekatan content-based filtering sangat relevan karena:
- Tidak memerlukan data rating pengguna.
- Menggunakan fitur objektif dari musik itu sendiri (acousticness, danceability, energy, dll)

## Business Understanding


### Problem Statements
- Bagaimana cara memahami struktur dan karakteristik dataset musik untuk membangun sistem rekomendasi?
- Bagaimana cara membangun sistem rekomendasi musik menggunakan pendekatan content-based filtering yang hanya bergantung pada fitur audio lagu?
- Bagaimana sistem dapat menyarankan lagu-lagu yang relevan berdasarkan satu lagu favorit pengguna, tanpa data rating eksplisit?
- Bagaimana cara menilai apakah sistem rekomendasi yang dibangun mampu memberikan hasil yang relevan dan memuaskan?

### Goals
- Melakukan eksplorasi awal (EDA) dan visualisasi terhadap dataset musik untuk memahami distribusi dan korelasi fitur lagu (seperti danceability, energy, acousticness, dll).
- Membangun sistem rekomendasi musik berbasis content-based filtering dengan menggunakan kemiripan fitur audio antar lagu.
- Mengembangkan fungsi yang mampu memberikan daftar rekomendasi lagu berdasarkan lagu favorit pengguna.
- Mengevaluasi kualitas hasil rekomendasi secara kualitatif (relevansi lagu) dan kuantitatif (jika memungkinkan, menggunakan metrik seperti cosine similarity score atau uji coba user).

## Data Understanding
Dataset yang digunakan pada proyek ini berasal dari dataset pada Kaggle dengan nama [Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) milik Zaheen Hamidani. Dataset ini berisi kumpulan informasi mengenai lagu-lagu populer yang dapat digunakan untuk membangun sistem rekomendasi musik. 
Masing-masing entri dalam dataset mewakili satu lagu dengan beberapa fitur numerik dan kategorikal yang mendeskripsikan karakteristik lagu tersebut.

**Berikut adalah penjelasan masing-masing fitur dalam dataset:**
**Berikut adalah penjelasan masing-masing fitur dalam dataset:**
- `genre`: Kategori genre lagu, berguna untuk segmentasi preferensi musik pengguna (tipe data kategori).
- `artist_name`: Nama artis/band yang membawakan lagu tersebut (tipe data kategori).
- `track_name`: Judul lagu (tipe data kategori).
- `track_id`: ID unik dari lagu, sebagai pengenal dalam sistem (tipe data kategori).
- `popularity`: Skor popularitas lagu berdasarkan jumlah pemutaran atau eksposur di platform digital (tipe data numerik).
- `acousticness`: Nilai lebih tinggi maka lagu lebih akustik (tipe data numerik).
- `danceability`: Menggambarkan seberapa cocok lagu untuk menari, berdasarkan tempo, irama, dan kestabilan beat (tipe data numerik).
- `duration_ms`: 	Durasi lagu dalam milidetik (tipe data numerik).
- `energy`: Lagu dengan nilai tinggi biasanya cepat dan keras (tipe data numerik).
- `instrumentalness`: Nilai mendekati 1 menunjukkan lagu sangat instrumental (tipe data numerik).
- `key`: Menunjukkan tangga nada dasar lagu (tipe data kategori).
- `liveness`: Mengindikasikan kemungkinan lagu direkam live (tipe data numerik).
- `loudness`: Tingkat kekerasan lagu (dalam dB). Bisa berpengaruh ke energi atau mood (tipe data numerik).
- `mode`: Mayor/minor (biasanya 1 = mayor, 0 = minor) (tipe data kategori).
- `speechiness`: Mengukur seberapa banyak elemen vokal/spoken. Mebedakan lagu instrumental vs lagu rap (tipe data numerik).
- `tempo`: Kecepatan lagu dalam BPM. Berpengaruh pada gaya musik dan mood (tipe data numerik).
- `time_signature`: Menunjukkan struktur ketukan lagu (umumnya 4/4) (tipe data kategori).
- `valence`: Mengukur kebahagiaan/mood lagu (0 = sedih, 1 = bahagia) (tipe data numerik).
- 
Variabel-variabel tersebut kemudian dianalisis untuk membangun model sistem rekomendasi yang dapat menyarankan lagu-lagu lain berdasarkan kemiripan karakteristik audio dari lagu yang disukai pengguna. Dalam hal ini, sistem menggunakan pendekatan content-based filtering, yang mengandalkan fitur seperti tempo, energi, danceability, valence, dan lain-lain untuk menghitung kemiripan antar lagu.

_Namun sebelum itu, untuk menambah wawasan awal tentang data dan membantu pengambilan keputusan analisis selanjutnya akan dilakukan exploratory data analysis._

✨ Saran Penggunaan dalam Model:
✅ Gunakan untuk menghitung similarity:
acousticness, danceability, energy, instrumentalness, speechiness, valence, tempo, liveness, loudness

⚠️ Opsional / Boleh Digunakan jika Relevan:
popularity, duration_ms, mode, key (kalau dikodekan), time_signature

❌ Abaikan dari input fitur numerik (tapi tetap disimpan sebagai info):
track_id, track_name, artist_name

---

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

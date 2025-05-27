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
Dataset yang digunakan pada proyek ini berasal dari dataset pada Kaggle dengan nama [Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db).

Dataset ini berisi kumpulan informasi mengenai lagu-lagu populer yang dapat digunakan untuk membangun sistem rekomendasi musik. 
Masing-masing entri dalam dataset mewakili satu lagu dengan beberapa fitur numerik dan kategorikal yang mendeskripsikan karakteristik lagu tersebut.

**Berikut adalah penjelasan masing-masing fitur dalam dataset:**
- genre: Kategori genre lagu, berguna untuk segmentasi preferensi musik pengguna (tipe data kategori).
- artist_name: Nama artis/band yang membawakan lagu tersebut (tipe data kategori).
- track_name: Judul lagu (tipe data kategori).
- track_id: ID unik dari lagu, sebagai pengenal dalam sistem (tipe data kategori).
- popularity: Angka Harapan Hidup (AHH) dalam tahun.
- avg_schooltime: Rata-rata lama sekolah (dalam tahun), mencerminkan capaian pendidikan.
- exp_percap: Pengeluaran per kapita, yaitu rata-rata pengeluaran individu di wilayah tersebut (dalam ribuan rupiah).

Variabel-variabel tersebut kemudian dianalisis untuk membangun model clustering yang dapat memprediksi tingkat pencapaian pendidikan (dalam hal ini diwakili oleh rata-rata lama sekolah) berdasarkan kondisi ekonomi suatu wilayah. Namun sebelum itu, untuk menambah wawasan awal tentang data dan membantu pengambilan keputusan analisis selanjutnya akan dilakukan exploratory data analysis.

---
**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

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

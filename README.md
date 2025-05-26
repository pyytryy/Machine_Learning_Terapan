# Laporan Proyek Machine Learning - Putri Pita Mutia

## Project #thewinnertakesitall: Ketimpangan Ekonomi di Balik Mimpi Kuliah

## Latar Belakang (Domain Proyek)

Belakangan ini, media sosial diramaikan dengan tren tagar **#thewinnertakesiitall** yang diunggah oleh pelajar di Indonesia. Dalam unggahan-unggahan tersebut, mereka menyampaikan harapan besar untuk dapat melanjutkan pendidikan ke jenjang perguruan tinggi. Namun, mereka juga mengungkapkan realitas pahit: keterbatasan ekonomi menjadi penghalang utama untuk mewujudkan impian tersebut.

Fenomena ini mencerminkan kondisi nyata di Indonesia, di mana kesenjangan ekonomi masih menjadi tantangan besar dalam pemerataan akses pendidikan. Pendidikan sendiri merupakan fondasi utama dalam pembangunan sumber daya manusia dan peningkatan kualitas hidup masyarakat.

Data dari Badan Pusat Statistik (BPS) menunjukkan bahwa **rata-rata lama sekolah di Indonesia masih berada pada angka 9,08 atau setara kelas 9 SMP/Sederajat tahun 2022**, yang berarti belum mencapai jenjang SMA secara penuh. Status ekonomi rumah tangga merupakan salah satu faktor yang memiliki pengaruh terhadap tinggi rendahnya tingkat pendidikan. Selain itu, provinsi dengan **tingkat kemiskinan dan pengeluaran per kapita yang rendah** cenderung memiliki capaian pendidikan yang lebih rendah pula (BPS: Statistik Pendidikan, 2022).

Menurut UNESCO (2020), **setiap tambahan satu tahun pendidikan dapat meningkatkan pendapatan individu sebesar 10%**, yang menegaskan adanya hubungan erat antara pendidikan dan kesejahteraan ekonomi. Maka, ketimpangan dalam akses pendidikan berpotensi memperparah siklus kemiskinan antar generasi.

Untuk mengatasi masalah ini, dilakukan pendekatan analitik berbasis data menggunakan model klasifikasi machine learning. Dengan memanfaatkan data ekonomi regional, model ini diharapkan dapat memprediksi tingkat pencapaian pendidikan di berbagai wilayah. Hasil dari model ini tidak hanya memberikan pemetaan potensi pendidikan, tetapi juga menjadi dasar pengambilan keputusan dalam merancang intervensi kebijakan pendidikan yang lebih terarah dan berbasis data.

Untuk menjawab isu tersebut, digunakan dataset yang berasal dari Kaggle dan bersumber dari data terbuka milik Badan Pusat Statistik (BPS). Dataset ini berisi data dari berbagai provinsi di Indonesia dengan beberapa variabel ekonomi dan sosial, di antaranya:

* Persentase penduduk miskin,
* Produk Domestik Regional Bruto (PDRB),
* Angka Harapan Hidup (AHH),
* Rata-rata lama sekolah,
* Pengeluaran per kapita.

Variabel-variabel tersebut kemudian dianalisis untuk membangun model klasifikasi yang dapat memprediksi tingkat pencapaian pendidikan (dalam hal ini diwakili oleh rata-rata lama sekolah) berdasarkan kondisi ekonomi suatu wilayah. Dengan pendekatan ini, diharapkan model dapat memberikan gambaran potensi pendidikan di suatu daerah, serta menjadi bahan pertimbangan dalam merancang kebijakan pemerataan pendidikan.

## Referensi

1. Badan Pusat Statistik. (2022). *Statistik Pendidikan 2022*.
   [https://www.bps.go.id/id/publication/2022/11/25/xyz/statistik-pendidikan-2022.html](https://www.bps.go.id/id/publication/2022/11/25/xyz/statistik-pendidikan-2022.html)

2. UNESCO. (2020). *Education Transforms Lives*.
   [https://en.unesco.org/themes/education](https://en.unesco.org/themes/education)

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

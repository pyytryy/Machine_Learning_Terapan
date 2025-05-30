# Laporan Proyek Machine Learning - Putri Pita Mutia

## Temukan Lagu Mirip Favoritmu: Rekomendasi Musik Berdasarkan Fitur Akustik dan Genre

**Latar Belakang (Domain Proyek)**
Musik merupakan bagian penting dalam kehidupan manusia sehari-hari. Di era digital, jutaan lagu kini tersedia secara daring melalui berbagai platform seperti Spotify, Apple Music, dan YouTube Music. Namun, ketersediaan lagu dalam jumlah sangat besar ini justru menimbulkan tantangan baru: pengguna sering kali kesulitan menemukan musik yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sebuah sistem yang mampu membantu pengguna menemukan lagu yang relevan secara otomatis.

Salah satu solusi atas permasalahan tersebut adalah penerapan sistem rekomendasi musik. Sistem ini dirancang untuk menyarankan lagu kepada pengguna berdasarkan pola preferensi, karakteristik lagu, maupun interaksi sebelumnya. Terdapat berbagai pendekatan dalam membangun sistem rekomendasi, seperti collaborative filtering dan content-based filtering. Pada kasus yakni ketika data interaksi pengguna terbatas, pendekatan **content-based filtering** menjadi pilihan yang lebih efektif karena hanya mengandalkan fitur-fitur dari lagu tanpa memerlukan data eksplisit dari pengguna lain.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi musik berbasis content-based filtering yang dapat memberikan saran lagu serupa berdasarkan karakteristik akustik dan tingkat popularitas. Dengan pendekatan ini, pengguna dapat menemukan lagu baru yang sesuai dengan selera mereka dengan lebih mudah dan menyenangkan.

Berdasarkan artikel[^1] jumlah pengguna aktif Spotify telah mencapai lebih dari 600 juta pengguna secara global. Fitur rekomendasi mereka, seperti "Discover Weekly" dan "Daily Mix", memainkan peran penting dalam mempertahankan keterlibatan pengguna. Bahkan disebutkan bahwa lebih dari 30% waktu mendengarkan musik di Spotify berasal dari rekomendasi otomatis. Meskipun sistem rekomendasi Spotify sangat kompleks dan berbasis data pengguna besar, pendekatan content-based filtering tetap relevan, terutama dalam konteks tertentu seperti pengembangan aplikasi musik lokal, tugas akhir, atau eksperimen AI. Keunggulan pendekatan ini antara lain:
- Tidak membutuhkan data rating atau preferensi pengguna secara eksplisit.
- Mengandalkan fitur objektif dari lagu, seperti acousticness, danceability, dan energy.
  
Penelitian oleh[^2] menunjukkan bahwa sistem rekomendasi berbasis konten dapat meningkatkan interaksi antara pengguna dan artis dengan memungkinkan penyesuaian preferensi berdasarkan fitur audio spesifik. Studi lain juga menyebutkan bahwa pendekatan ini mampu memberikan rekomendasi yang lebih akurat dan memuaskan, khususnya saat data pengguna lain tidak tersedia.
Dengan mempertimbangkan keunggulan dan ketersediaan data fitur audio dari Spotify, proyek ini diarahkan untuk membangun sistem rekomendasi musik berbasis content-based filtering yang mampu memberikan rekomendasi lagu secara personal dan relevan bagi pengguna.

## Business Understanding

### Problem Statements
- Bagaimana membangun sistem rekomendasi lagu yang dipersonalisasi berdasarkan karakteristik konten lagu, tanpa bergantung pada data pengguna lain?
- Bagaimana cara merekomendasikan lagu lain yang relevan secara musikal dengan lagu yang sudah disukai pengguna, namun belum pernah mereka dengarkan sebelumnya?

### Goals
- Menghasilkan rekomendasi lagu yang dipersonalisasi berdasarkan karakteristik konten dari lagu yang dipilih pengguna (misalnya genre, danceability, energy, valence, dll) dengan pendekatan content-based filtering.
- Menyediakan daftar lagu-lagu baru dan belum pernah didengarkan oleh pengguna, namun memiliki kemiripan secara musikal, sehingga memungkinkan pengguna untuk mengeksplorasi lagu-lagu baru yang sesuai dengan preferensinya.

### Solution Statements
Solusi yang ditawarkan terdiri dari beberapa langkah inti:
- Ekstraksi dan normalisasi fitur audio untuk memastikan seluruh nilai berada pada skala yang setara.
- Kombinasi fitur tekstual dan numerik untuk membentuk representasi lagu yang lebih kaya dan informatif.
- Penghitungan kemiripan antar lagu menggunakan metode cosine similarity dan MAE (Mean Absolute Error) untuk mengukur seberapa dekat karakteristik antar lagu.
- Fungsi rekomendasi yang mampu memberikan daftar lagu yang paling mirip berdasarkan input lagu pilihan pengguna.
- Evaluasi terhadap model yang telah dibuat

## Data Understanding
Dataset yang digunakan pada proyek ini berasal dari dataset pada Kaggle dengan nama [Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) milik Zaheen Hamidani. Dataset ini berisi kumpulan informasi mengenai lagu-lagu populer yang dapat digunakan untuk membangun sistem rekomendasi musik. 
Masing-masing entri dalam dataset mewakili satu lagu dengan beberapa fitur numerik dan kategorikal yang mendeskripsikan karakteristik lagu tersebut.

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
  
Variabel-variabel tersebut kemudian dianalisis untuk membangun model sistem rekomendasi yang dapat menyarankan lagu-lagu lain berdasarkan kemiripan karakteristik audio dari lagu yang disukai pengguna. Dalam hal ini, sistem menggunakan pendekatan content-based filtering, yang mengandalkan fitur seperti tempo, energi, danceability, valence, dan lain-lain untuk menghitung kemiripan antar lagu. Namun sebelum itu, untuk menambah wawasan awal tentang data dan membantu pengambilan keputusan analisis selanjutnya akan dilakukan exploratory data analysis.

### Exploratory Data Analysis

Exploratory Data Analysis (EDA) atau Analisis Data Eksploratif adalah tahap awal atau pondasi utama dalam proses analisis data yang bertujuan untuk memahami struktur, pola, dan karakteristik data sebelum dilakukan pemodelan atau analisis lebih lanjut.

---

1. Informasi Dataset

  ```python
  df.info()
  ```
  Diperoleh output sebagai berikut:

  ```python
      <class 'pandas.core.frame.DataFrame'>
      RangeIndex: 232725 entries, 0 to 232724
      Data columns (total 18 columns):
       #   Column            Non-Null Count   Dtype  
      ---  ------            --------------   -----  
       0   genre             232725 non-null  object 
       1   artist_name       232725 non-null  object 
       2   track_name        232724 non-null  object 
       3   track_id          232725 non-null  object 
       4   popularity        232725 non-null  int64  
       5   acousticness      232725 non-null  float64
       6   danceability      232725 non-null  float64
       7   duration_ms       232725 non-null  int64  
       8   energy            232725 non-null  float64
       9   instrumentalness  232725 non-null  float64
       10  key               232725 non-null  object 
       11  liveness          232725 non-null  float64
       12  loudness          232725 non-null  float64
       13  mode              232725 non-null  object 
       14  speechiness       232725 non-null  float64
       15  tempo             232725 non-null  float64
       16  time_signature    232725 non-null  object 
       17  valence           232725 non-null  float64
      dtypes: float64(9), int64(2), object(7)
      memory usage: 32.0+ MB
  ```

  **Insight:**
  
  Dataset `df` memiliki jumlah baris sebanyak 232.725 baris dengan 18 fitur

  Selanjutnya akan diperiksa data duplikat dengan code:
  
  ```python
    df.duplicated().sum()
  ```
  
  Diperoleh output sebagai berikut:
  ```python
    np.int64(0)
  ```
  **Insight:**
   
  Diperoleh bahwa dataset `df` merupakan dataset yang tidak memiliki data dengan missing value.

  Selanjutnya akan diperiksa data missing value dengan code:
  ```python
    df.isnull().sum()
  ```

  Diperoleh output sebagai berikut:
  ```python
        		0
    genre	0
    artist_name	0
    track_name	1
    track_id	0
    popularity	0
    acousticness	0
    danceability	0
    duration_ms	0
    energy	0
    instrumentalness	0
    key	0
    liveness	0
    loudness	0
    mode	0
    speechiness	0
    tempo	0
    time_signature	0
    valence	0
    
```
**Insight:**
  
  Diperoleh bahwa terdapat sebanyak 1 missing value pada fitur track_name, sehingga perlu dilakukan penanganan .

  Selanjutnya akan diperiksa distribusi data dengan code:
  ```python
      df.describe()
  ```

  Diperoleh output sebagai berikut:
  ```python
      popularity	acousticness	danceability	duration_ms	energy	instrumentalness	liveness	loudness	speechiness	tempo	valence
      count	232725.000000	232725.000000	232725.000000	2.327250e+05	232725.000000	232725.000000	232725.000000	232725.000000	232725.000000	232725.000000	232725.000000
      mean	41.127502	0.368560	0.554364	2.351223e+05	0.570958	0.148301	0.215009	-9.569885	0.120765	117.666585	0.454917
      std	18.189948	0.354768	0.185608	1.189359e+05	0.263456	0.302768	0.198273	5.998204	0.185518	30.898907	0.260065
      min	0.000000	0.000000	0.056900	1.538700e+04	0.000020	0.000000	0.009670	-52.457000	0.022200	30.379000	0.000000
      25%	29.000000	0.037600	0.435000	1.828570e+05	0.385000	0.000000	0.097400	-11.771000	0.036700	92.959000	0.237000
      50%	43.000000	0.232000	0.571000	2.204270e+05	0.605000	0.000044	0.128000	-7.762000	0.050100	115.778000	0.444000
      75%	55.000000	0.722000	0.692000	2.657680e+05	0.787000	0.035800	0.264000	-5.501000	0.105000	139.054000	0.660000
      max	100.000000	0.996000	0.989000	5.552917e+06	0.999000	0.999000	1.000000	3.744000	0.967000	242.903000	1.000000
```

**Insight:**

1. Dataset ini terdiri dari 232725 data
2. Dataset ini terdiri dari 18 fitur
3. Terdapat 1 missing value
4. Tidak terdapat data yang duplikat
5. Rangkuman Statistik Deskriptif

  Selanjutnya akan dilakukan CountAndPlot

  Untuk menghitung jumlah dan persentase kemunculan tiap kategori dalam sebuah fitur (kolom) DataFrame, menampilkannya dalam bentuk tabel, lalu memvisualisasikannya sebagai grafik batang. Dalam proyek ini akan difokuskan pada fitur `genre`

  ```python
    def CountAndPlot(df, feature):
      count = df[feature].value_counts()
      percent = 100*df[feature].value_counts(normalize=True)
      samples = pd.DataFrame({'Sample Count':count, 'Percentage':percent.round(1)})
      print(samples)
      count.plot(kind='bar', title=feature)
  ```

  ```python
  CountAndPlot(df, 'genre')
  ```
 
 Diperoleh output sebagai berikut:
   ```
                     Sample Count  Percentage
    genre                                     
    Comedy                    9681         4.2
    Soundtrack                9646         4.1
    Indie                     9543         4.1
    Jazz                      9441         4.1
    Pop                       9386         4.0
    Electronic                9377         4.0
    Children’s Music          9353         4.0
    Folk                      9299         4.0
    Hip-Hop                   9295         4.0
    Rock                      9272         4.0
    Alternative               9263         4.0
    Classical                 9256         4.0
    Rap                       9232         4.0
    World                     9096         3.9
    Soul                      9089         3.9
    Blues                     9023         3.9
    R&B                       8992         3.9
    Anime                     8936         3.8
    Reggaeton                 8927         3.8
    Ska                       8874         3.8
    Reggae                    8771         3.8
    Dance                     8701         3.7
    Country                   8664         3.7
    Opera                     8280         3.6
    Movie                     7806         3.4
    Children's Music          5403         2.3
    A Capella                  119         0.1
  ```

  ![image](https://github.com/user-attachments/assets/7f17b80b-6c00-4f78-b5c4-690b4a69a168)

  **Insight:**
  Terdapat genre musik Children's Music yang muncul dua kali dengan jumlah berbeda, perlu dilakukan analisis lebih lanjut

## Data Preprocessing

Pada tahap ini, dilakukan pembersihan data menangani missing values, mengatasi kedua nama genre yang mirip, dan memilih fitur yang relevan. Proses ini bertujuan untuk memastikan bahwa data yang digunakan bersih dan siap untuk dianalisis lebih lanjut.

- Mengatasi Missing Value

  Pada tahapan ini akan dilakukan penanganan missing value dengan menghapus baris dengan missing value pada fitur `track_name`
  
- Menangani 2 genre yang sama

   Diperoleh bahwa kedua genre memiliki beberapa entri dengan judul lagu yang sama namun dinyanyikan oleh artis yang berbeda. Hal ini kemungkinan disebabkan oleh perbedaan versi lagu, seperti versi asli dan cover. Oleh karena itu, akan dilakukan generalisasi atau standarisasi nama genre untuk menjaga konsistensi data. Namun, data tidak akan dihapus karena meskipun judul lagunya sama, karena kontennya berbeda, sehingga tetap relevan untuk dianalisis secara terpisah.

- Mengambil Subset Dataset Secara Acak

  Karena ukuran dataset yang sangat besar membutuhkan waktu komputasi yang lama dan ruang penyimpanan yang besar, maka diputuskan untuk mengambil subset data **sebanyak >= 10.000 baris**. Pemilihan subset dilakukan secara acak, kecuali untuk data dengan genre yang jarang muncul, seperti `"A Capella"`, yang akan dipertahankan seluruhnya agar tidak hilang dari analisis. Sementara itu, data dari genre lain akan diambil secara acak hingga total data mencapai batas 10.000 baris.
  
- Mengambil fitur yang relevan

  Dataset memiliki total 18 fitur, namun tidak semuanya digunakan dalam proses pembuatan sistem rekomendasi. Pada proyek ini, sistem rekomendasi dikembangkan tidak hanya berdasarkan kemiripan 'Penyanyi' dan 'Judul Lagu', tetapi juga **mempertimbangkan atribut-atribut audio dari setiap lagu**. Fitur-fitur yang dipilih mencakup:
  
  - `track_id`, `artist_name`, `track_name`, `genre`,
  - serta fitur audio seperti `acousticness`, `danceability`, `energy`, `instrumentalness`, `loudness`, `speechiness`, `tempo`, dan `valence`.
    Pemilihan fitur ini bertujuan untuk menghasilkan rekomendasi lagu yang lebih relevan secara musikalitas dan karakteristik audio, bukan sekadar berdasarkan kemiripan teks atau artis.

**Kesimpulan Akhir dari Proses Data Preprocessing**

Dataset telah dibersihkan dari missing value, nama genre yang mirip, dan telah mensortir fitur yang relevan digunakan dan telah disimpan dalam variabel `data`.


## Data Preparation

Pada tahap selanjutnya dilakukan normalisasi nilai fitur audio dengan MinmAxScaler, membuat kombinasi nama, TF-IDF vektorizer dan penggabungan

- Normalisasi Fitur Audio

  Dalam proyek ini digunakan metode **MinMaxScaler** untuk mengubah semua nilai fitur ke dalam rentang [0, 1], sehingga tidak ada fitur yang mendominasi perhitungan kemiripan.
  
- Membuat Kombinasi Nama

  Hal ini dilakukan agar TF-IDF dapat menghitung kemiripan berdasarkan dua aspek yakni `artist_name` dan `genre` dan menyimpannya dalam `'combined_text'`

Saat mencari sebuah lagu, agar sistem akan merekomendasikan lagu-lagu lain yang mirip baik dari segi penyanyi maupun karakteristik audio seperti `acousticness, danceability, energy, valence`, dan `tempo`, maka dilakukan penggabungan dari TF-IDF dan `scaled_audio`

- Dilakukan TF-IDF pada combined_text

  ```python
  tf = TfidfVectorizer()
  tfidf_matrix = tf.fit_transform(data['combined_text'])
  ```
  
- Dilakukan Penggabungan

  ```python
  hybrid_matrix = hstack([tfidf_matrix, scaled_audio])
  ```

  Disimpan pada variabel `hybrid_matrix`


## **Model Development Content Based Filtering**

Dalam pengembangan sistem rekomendasi ini, model yang digunakan adalah **Content-Based Filtering**. Pemilihan pendekatan ini didasarkan pada karakteristik dataset yang tidak memuat informasi interaksi pengguna seperti user ID, rating, atau riwayat pemutaran lagu. Oleh karena itu, metode Collaborative Filtering tidak dapat diterapkan secara efektif. Dalam konteks proyek ini, fitur-fitur seperti nama artis, judul lagu, genre, serta karakteristik audio seperti acousticness, danceability, energy, valence, dan tempo digunakan sebagai dasar kemiripan. Selain lebih sederhana untuk diterapkan, pendekatan ini juga memberikan fleksibilitas dalam menyesuaikan bobot atau kombinasi fitur sesuai kebutuhan. 

Content-Based Filtering sangat cocok untuk sistem rekomendasi berbasis konten yang tidak melibatkan perilaku pengguna secara langsung, serta mempermudah interpretasi hasil karena rekomendasi diberikan berdasarkan kemiripan konten antar lagu. Namun metode ini memiliki kelemahan, adapauun kelemahan utamanya adalah masalah _overspecialization_, di mana sistem hanya merekomendasikan item yang sangat mirip dengan yang sudah dikenal atau disukai sebelumnya, sehingga mengurangi keberagaman rekomendasi. Selain itu, metode ini tidak dapat menangkap preferensi kolektif dari pengguna lain karena tidak melibatkan data komunitas atau perilaku pengguna secara luas. Kekurangan lainnya adalah ketergantungan tinggi terhadap kualitas dan kelengkapan fitur konten.

- Cosine Similarity

  Gunakan teknik cosine similarity pada variabel `hybrid_matrix` untuk menemukan lagu-lagu yang mirip secara musikal dengan lagu yang dicari

  ```python
  cosine_sim = cosine_similarity(hybrid_matrix)
  cosine_sim
  ```
- Membuat Dictionary Berisi 10 Lagu Paling Mirip

  Menyusun top 10 rekomendasi lagu yang paling mirip untuk setiap lagu berdasarkan skor cosine similarity, dengan memasukkan lagu yang diinginkan juga

  ```python
  top_k_sim = {}
  for i in range(cosine_sim.shape[0]):
      top_indices = np.argsort(cosine_sim[i])[::-1][:10]  
      key = data['track_name'].iloc[i] + '-' + data['artist_name'].iloc[i] 
      similar_items = data.iloc[top_indices][['track_name', 'artist_name']].apply(
          lambda x: f"{x['track_name']} - {x['artist_name']}", axis=1).tolist()
      top_k_sim[key] = similar_items
  ```
- Mendapatkan Rekomendasi

  Memberikan rekomendasi lagu yang mirip dengan lagu input, berdasarkan kemiripan fitur (hasil dari cosine_similarity), dengan referensi pada `track_name` dan `artist_name`.

   ```python
  def music_recommendations(track_name_input, artist_input, similarity_data=top_k_sim, k=5):
    """
    Rekomendasikan lagu mirip berdasarkan track dan artist
    """
    key = f"{track_name_input}-{artist_input}"

    if key not in similarity_data:
        print("Lagu tidak ditemukan di data.")
        return pd.DataFrame()

    recommended_keys = similarity_data[key][:k]

    recommended_data = []
    for item in recommended_keys:
        parts = item.rsplit(" - ", 1)
        if len(parts) == 2:
            recommended_data.append(parts)
        else:
            print(f"Warning: Skipping item '{item}' as it could not be parsed into track and artist.")
            continue

    if not recommended_data:
        print("No valid recommendations found after parsing.")
        return pd.DataFrame()

    recommended_df = pd.DataFrame(recommended_data, columns=["track_name", "artist_name"])

    return pd.merge(recommended_df, data, on=["track_name", "artist_name"], how="left")[[
         'track_id','artist_name','track_name','genre','acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]]
  ```

  - Melakukan testing pada model

    ```python
    music_recommendations("How to Be Married", "Jackie Kashian")
    ```

     Diperoleh output rekomendasi lagu yang similiar dengan How to Be Married:

    ```python
    
      track_id	artist_name	track_name	genre	acousticness	danceability	energy	instrumentalness	loudness	speechiness	tempo	valence
    0	0a4lqrikZRONK8GFqRgRoi	Jackie Kashian	How to Be Married	Comedy	0.676706	0.704187	0.757695	0.0	0.802148	0.961747	0.203107	0.732053
    1	3doVn4FNryNBnKjGFx2bMC	Jackie Kashian	Bite Suit	Comedy	0.668674	0.708537	0.700623	0.0	0.825096	0.961747	0.389149	0.702730
    2	5ahc9pNqLSwDAF5PJZMo9S	T.J. Miller	Episode 6 "Smokes, Jokes And Froze-Tokes"	Comedy	0.838353	0.510604	0.743677	0.0	0.827039	0.979811	0.167757	0.588473
    3	6sauVvIY5BNHLjxoyDoSju	Tommy Ryman	Toots and Farts	Comedy	0.783132	0.683524	0.857821	0.0	0.821125	0.948996	0.259631	0.564206
    4	0u6cXeKT83UPslAot9FwQA	Sinbad	Life Is Short	Comedy	0.734939	0.592170	0.674590	0.0	0.816172	0.927744	0.384774	0.625885

    ```
## Evaluation

- Visual Similarity Inspection

  Membandingkan nilai fitur audio (seperti acousticness, danceability, dll) antara lagu input dan lagu hasil rekomendasi. Visualisasi ini memudahkan untuk melihat seberapa mirip pola fitur kedua lagu dalam bentuk garis.

**Cara Kerja Visual Similarity Inspection**

- Ekstraksi Fitur Audio:

  Ambil nilai dari fitur-fitur audio
  
- Plotting Nilai Fitur:
   - Visualisasi berupa plot garis (line plot), sumbu-X merepresentasikan nama fitur audio, dan sumbu-Y adalah nilai normalisasi dari masing-masing fitur. Lagu input digambarkan dengan satu garis, dan setiap lagu rekomendasi digambarkan dengan garis lain

- Perbandingan Visual:

  Semakin mirip bentuk kurva/grafiknya, semakin mirip juga karakteristik musikal dari lagu-lagu tersebut.

  Bentuk output dari hasil rekomendasi terhadap `How to Be Married - Jackie Kashian`

  ![image](https://github.com/user-attachments/assets/8b046a7a-9e0c-41ee-aaae-2aa7bd2faefb)

  **Salah satu interpretasi dari grafik diatas:**
  
  Rekomendasi "Life Is Short" cukup mirip dengan lagu How to Be Married karena memiliki kesamaan pada aspek `instrumentalness`, `loudness`, dan `speechiness`, meskipun sedikit berbeda dalam `tempo` dan `valence`.


---
- Menggunakan MAE
  
  **Mean Absolute Error (MAE)** adalah metrik evaluasi yang digunakan untuk mengukur seberapa mirip dua buah entitas berdasarkan fitur numerik mereka. Dalam konteks **sistem rekomendasi musik**, MAE digunakan untuk menghitung rata-rata selisih absolut antara nilai fitur lagu input dan lagu yang direkomendasikan. Fitur-fitur ini bisa berupa:

**Rumus MAE:**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |
$$

**Keterangan:**
- yᵢ: nilai fitur lagu asli
- ŷᵢ: nilai fitur lagu hasil rekomendasi
- n: jumlah fitur audio yang dibandingkan

**Dengan cara kerja:**

- Untuk setiap lagu rekomendasi, dibandingkan nilai fitur audionya dengan lagu input.

- Seluruh selisih absolut dihitung untuk fitur seperti acousticness, danceability, energy, valence, dan tempo.

- MAE yang lebih kecil menunjukkan bahwa fitur-fitur lagu tersebut memiliki kriteria yang sangat mirip dengan lagu input.

---

**Interpretasi Nilai MAE**

| Nilai MAE        | Interpretasi                             |
|------------------|------------------------------------------|
| 0.00 - 0.05      | Sangat mirip                             |
| 0.05 - 0.10      | Cukup mirip                              |
| > 0.10           | Mulai berbeda signifikan                 |

Semakin kecil nilai MAE, semakin **mirip secara musikal** antara lagu input dan rekomendasinya.

---

Bentuk output dari hasil rekomendasi terhadap `How to Be Married - Jackie Kashian`

| Lagu Rekomendasi yang dihasilkan                   | MAE    | Interpretasi          |
|----------------------------------------------------|--------|-----------------------|
| Bite Suit - Jackie Kashian                         | 0.0385 | Sangat mirip          |
| Life Is Short - Sinbad                             | 0.0737 | Cukup mirip           |
| Toots and Farts - Tommy Ryman                      | 0.0604 | Cukup mirip           |

**Insight:**

- Diperoleh nilai MAE dari kemiripan lagu 'How to Be Married' dengan 'How to Be Married' adalah 0, yang artinya sama persis karena memang membandingkan dengan lagu yang sama
- Diperoleh nilai MAE dari kemiripan lagu 'How to Be Married' dengan 'Bite Suit' adalah 0.0385, menunjukkan bahwa rekomendasi lagu 'Bite Suit' **sangat mirip** dengan lagu "How to Be Married".
- Diperoleh nilai MAE dari kemiripan lagu 'How to Be Married' dengan 'Episode 6 "Smokes, Jokes And Froze-Tokes"' - 'T.J. Miller' adalah 0.0739, menunjukkan bahwa rekomendasi lagu 'Episode 6 "Smokes, Jokes And Froze-Tokes"' - 'T.J. Miller' **cukup mirip** dengan lagu "How to Be Married".
- Diperoleh nilai MAE dari kemiripan lagu 'How to Be Married' dengan 'Toots and Farts' adalah 0.0604, menunjukkan bahwa rekomendasi lagu 'Toots and Farts' **cukup mirip** dengan lagu "How to Be Married".
- Diperoleh nilai MAE dari kemiripan lagu 'How to Be Married' dengan 'Life Is Short' - 'Sinbad' adalah 0.0737, menunjukkan bahwa rekomendasi lagu 'Life Is Short' - 'Sinbad' **cukup mirip** dengan lagu "How to Be Married".

Diantara kedua bentuk evaluasi, metode evaluasi dengan MAE menunjukkan hasil yang dapat dipercaya dibandingkan hanya dengan melihat visualisasinya.

## Kesimpulan
Sistem rekomendasi lagu berbasis konten (Content-Based Filtering) yang telah dibangun memungkinkan pemberian rekomendasi musik secara personal tanpa memerlukan data pengguna lain. Dengan memanfaatkan fitur-fitur audio seperti acousticness, danceability, energy, valence, tempo, dan atribut tambahan seperti genre dan nama artis, sistem ini mengukur kemiripan antar lagu menggunakan teknik seperti cosine similarity dan Mean Absolute Error (MAE). Pendekatan ini efektif dalam menjawab dua tantangan utama:

- Membangun sistem rekomendasi tanpa data pengguna lain dicapai dengan menganalisis karakteristik konten lagu itu sendiri, sehingga sistem tetap dapat bekerja meskipun pengguna belum memiliki histori interaksi.
- 
- Memberikan rekomendasi musik yang relevan secara musikal, bahkan untuk lagu yang belum pernah didengarkan pengguna, dilakukan dengan membandingkan lagu yang disukai dengan seluruh koleksi berdasarkan fitur-fitur numerik dan metadata, sehingga hasilnya tetap relevan secara audio maupun genre.

Dengan metode ini, pengguna bisa mendapatkan saran lagu yang memiliki karakteristik musik serupa dengan yang mereka sukai, sekaligus memperluas eksplorasi musik ke artis atau lagu yang sebelumnya belum dikenal.

## **Referensi**

[^1]: Kompas.com, "Jumlah pengguna Spotify tumbuh pada 2023, tembus 600 juta," Kompas Tekno, Feb. 12, 2024. [Online]. Available: https://tekno.kompas.com/read/2024/02/12/11030007/jumlah-pengguna-spotify-tumbuh-pada-2023-tembus-600-juta.
[^2]: S. Bangera, V. Nagaonkar, A. Tiwari, S. Ansari, and K. Talekar, "Spotify recommendation system," Int. Res. J. Mod. Eng. Technol. Sci., vol. 6, no. 2, Feb. 2024. [Online]. Available: https://www.researchgate.net/publication/381853790_SPOTIFY_RECOMMENDATION_SYSTEM.




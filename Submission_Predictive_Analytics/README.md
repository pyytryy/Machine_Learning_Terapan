# Laporan Proyek Machine Learning - Putri Pita Mutia

## Project #thewinnertakesitall: Ketimpangan Ekonomi di Balik Mimpi Kuliah

**Latar Belakang (Domain Proyek)**

Belakangan ini, media sosial diramaikan dengan tren tagar _#thewinnertakesiitall_ yang diunggah oleh pelajar Indonesia. Dalam unggahan-unggahan tersebut, mereka menyampaikan harapan untuk dapat melanjutkan pendidikan ke jenjang perguruan tinggi. Namun, mereka juga mengungkapkan realitas pahit: keterbatasan ekonomi menjadi penghalang utama untuk mewujudkan impian tersebut.

Fenomena ini mencerminkan kondisi nyata yang terjadi di Indonesia, yakni kesenjangan ekonomi masih menjadi tantangan besar dalam pemerataan akses pendidikan. Pendidikan sendiri merupakan fondasi utama dalam pembangunan sumber daya manusia dan peningkatan kualitas hidup masyarakat. Data dari Badan Pusat Statistik (BPS) menunjukkan bahwa rata-rata lama sekolah di Indonesia hanya mencapai 9,08 tahun atau setara kelas 9 
SMP/sederajat (2022), yang berarti belum mencapai jenjang SMA secara penuh. Status ekonomi rumah tangga menja merupakan salah satu faktor yang memiliki pengaruh terhadap tinggi rendahnya tingkat pendidikan. Selain itu, provinsi dengan tingkat kemiskinan dan pengeluaran per kapita yang rendah cenderung memiliki capaian pendidikan yang lebih rendah pula. Menurut[^1], setiap tambahan satu tahun pendidikan dapat meningkatkan pendapatan individu sebesar 10%, yang menegaskan adanya hubungan erat antara pendidikan dan kesejahteraan ekonomi. Maka, ketimpangan dalam akses pendidikan berpotensi memperparah siklus kemiskinan antar generasi [^2].

Untuk mengatasi masalah ini, dilakukan pendekatan analitik berbasis data menggunakan model klasifikasi machine learning. Dengan memanfaatkan data ekonomi regional, model ini diharapkan dapat memprediksi tingkat pencapaian pendidikan di berbagai wilayah. Hasil dari model ini tidak hanya memberikan pemetaan potensi pendidikan, tetapi juga menjadi dasar pengambilan keputusan dalam merancang intervensi kebijakan pendidikan yang lebih terarah dan berbasis data.


# Business Understanding
Pada tahap ini akan dilakukan identifikasi konteks permasalahan yang dihadapi dan tujuan bisnis, yang berfungsi untuk memastikan bahwa solusi yang dikembangkan nantinya benar-benar selaras dengan kebutuhan dan harapan pihak terkait.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana hubungan keterkaitan antara kondisi ekonomi suatu daerah dengan tingkat pencapaian pendidikan masyarakatnya?
- Bagaimana kelompok - kelompok wilayah yang memiliki kemiripan kondisi ekonomi dapat diidentifikasi melalui clustering?
- Bagaimana sebuah model klasifikasi dapat memprediksi tingkat pencapaian pendidikan berdasarkan indikator-indikator ekonomi daerah?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Menganalisis hubungan keterkaitan antara faktor-faktor ekonomi suatu daerah dengan tingkat pencapaian pendidikan masyarakatnya.
- Mengidentifikasi kelompok-kelompok wilayah dengan karakteristik ekonomi yang mirip menggunakan metode clustering, sebagai langkah segmentasi untuk memahami kondisi wilayah secara lebih terstruktur.
- Membangun model klasifikasi yang mampu memprediksi tingkat pencapaian pendidikan berdasarkan indikator ekonomi daerah, guna memberikan gambaran prediktif yang dapat digunakan dalam perencanaan kebijakan pendidikan.
  
### Solution statements
- Melakukan eksplorasi data dan segmentasi terhadap fitur ekonomi seperti GDP regional, pengeluaran per kapita, persentase penduduk miskin, harapan hidup, dan rata-rata lama sekolah untuk menemukan pola-pola yang menggambarkan kondisi sosial ekonomi tiap wilayah. Visualisasi seperti heatmap untuk melihat tingkat korelasi antar fitur, barplot untuk memudahkan melihat wilayah dengan rata-rata pendidikan tertinggi dan terendah, histogram untuk melihat distribusi data normal atau tidak, boxplot untuk melihat berapa banyak outlier pada dataset serta scatter plot digunakan untuk mengidentifaksi korelasi antara 2 fitur.
- Melakukan preprocessing dataset hingga sesuai yang diharapkan agar proses clustering untuk mengelompokkan wilayah berdasarkan kesamaan indikator ekonomi, sehingga dapat diperoleh segmentasi wilayah dengan karakteristik ekonomi yang optimal. Hal ini membantu mengenali kelompok daerah yang memiliki potensi atau tantangan serupa dalam pengembangan pendidikan.
- Membuat model klasifikasi berbasis machine learning untuk memprediksi kategori pencapaian pendidikan suatu kota/kabupaten berdasarkan kondisi ekonominya. Model ini dapat digunakan sebagai alat bantu untuk menyusun kebijakan berbasis data, terutama dalam menargetkan intervensi pendidikan di wilayah dengan ekonomi yang kurang menguntungkan.

## Data Understanding
Dataset yang digunakan pada proyek ini berasal dari dataset pada Kaggle dengan nama [Socio-Economic of Indonesia in 2021](https://www.kaggle.com/datasets/dannytheodore/socio-economic-of-indonesia-in-2021/data) dan bersumber dari data terbuka milik Badan Pusat Statistik (BPS). 

**Dataset ini berisi data dari berbagai provinsi di Indonesia dengan beberapa variabel ekonomi dan sosial, di antaranya:**
- `province`: Nama provinsi di Indonesia (tipe data kategori).
- `cities_reg`: Nama kabupaten/kota yang termasuk dalam provinsi tersebut (tipe data kategori).
- `poorpeople_percentage`: Persentase penduduk miskin di wilayah tersebut (dalam persentase).
- `reg_gdp`: Produk Domestik Regional Bruto (PDRB) dalam satuan miliar rupiah
- `life_exp`: Angka Harapan Hidup (AHH) dalam tahun.
- `avg_schooltime`: Rata-rata lama sekolah (dalam tahun), mencerminkan capaian pendidikan.
- `exp_percap`: Pengeluaran per kapita, yaitu rata-rata pengeluaran individu di wilayah tersebut (dalam ribuan rupiah).

Variabel-variabel tersebut kemudian dianalisis untuk membangun model clustering yang dapat memprediksi tingkat pencapaian pendidikan (dalam hal ini diwakili oleh rata-rata lama sekolah) berdasarkan kondisi ekonomi suatu wilayah. Namun sebelum itu, untuk menambah wawasan awal tentang data dan membantu pengambilan keputusan analisis selanjutnya akan dilakukan exploratory data analysis.

---
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
    RangeIndex: 514 entries, 0 to 513
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   province               514 non-null    object 
     1   cities_reg             514 non-null    object 
     2   poorpeople_percentage  514 non-null    float64
     3   reg_gdp                514 non-null    float64
     4   life_exp               514 non-null    float64
     5   avg_schooltime         514 non-null    float64
     6   exp_percap             514 non-null    int64  
    dtypes: float64(4), int64(1), object(2)
    memory usage: 28.2+ KB
  ```

  **Insight:**
  
  Diperoleh bahwa dataset `df` merupakan dataset yang berisikan 514 baris dengan 7 fitur (5 fitur numerik dan 2 fitur object)

  
  Selanjutnya akan diperiksa data duplikat dengan code:
  
  ```python
      df.duplicated().sum()
  ```
  
  Diperoleh output sebagai berikut:
  
  ```python
    np.int64(0)
  ```

  **Insight:**
  
  Diperoleh bahwa dataset `df` merupakan dataset yang tidak memiliki data yang duplikat.
  
  Selanjutnya akan diperiksa data missing value dengan code:
 
  ```python
      df.isnull().sum()
  ```

  Diperoleh output sebagai berikut:
 
  ```python
                            0
      province	            0
      cities_reg	          0
      poorpeople_percentage	0
      reg_gdp	              0
      life_exp	            0
      avg_schooltime	      0
      exp_percap	          0
      
      dtype: int64
  ```
  **Insight:**
    
   Diperoleh bahwa dataset `df` merupakan dataset yang tidak memiliki missing value.

  Selanjutnya akan diperiksa distribusi data dengan code:
  
  ```python
      df.describe()
  ```

  Diperoleh output sebagai berikut:
  
  ```python
          poorpeople_percentage	reg_gdp	life_exp	avg_schooltime	exp_percap
          count	514.000000	514.000000	514.000000	514.000000	514.000000
          mean	12.273152	34.798333	69.619076	8.436615	10324.787938
          std	7.458703	84.155498	3.455911	1.630842	2717.144186
          min	2.380000	1.042000	55.370000	1.420000	3976.000000
          25%	7.150000	5.587500	67.336250	7.510000	8574.000000
          50%	10.455000	13.068500	69.922500	8.305000	10196.500000
          75%	14.887500	28.849500	72.018750	9.337500	11719.000000
          max	41.660000	819.000000	77.855000	12.830000	23888.000000
  ```
    
  **Insight:**
  
   - `poorpeople_percentage` (Persentase Penduduk Miskin)
      - **Rata-rata**: 12.27%
      - **Minimum**: 2.38%, **Maksimum**: 41.66%
      - **Sebaran (std)**: 7.46%
      - *Sebagian besar wilayah memiliki penduduk miskin <15%, namun ada wilayah ekstrem dengan >40%.*
   - `reg_gdp` (PDRB, dalam miliar rupiah)
      - **Rata-rata**: 34.80 miliar
      - **Minimum**: 1.04 miliar, **Maksimum**: 819 miliar
      - **Sebaran (std)**: 84.16
      - Tingkat ketimpangan sangat tinggi. 75% wilayah memiliki PDRB <28.85 miliar.
    - `life_exp` (Angka Harapan Hidup)
      - **Rata-rata**: 69.62 tahun
      - **Minimum**: 55.37 tahun, **Maksimum**: 77.86 tahun
      - **Sebaran (std)**: 3.46
      - Sebagian besar wilayah memiliki AHH yang relatif homogen antara 67–72 tahun.
    - `avg_schooltime` (Rata-rata Lama Sekolah)
      - **Rata-rata**: 8.44 tahun (setara SMP kelas 2)
      - **Minimum**: 1.42 tahun, **Maksimum**: 12.83 tahun
      - Terdapat ketimpangan, namun mayoritas wilayah berada di kisaran 7.5 – 9.3 tahun.
   - `exp_percap` (Pengeluaran Per Kapita / Tahun)
      - **Rata-rata**: Rp10.324.788 (~Rp860.000/bulan)
      - **Minimum**: Rp3.976.000, **Maksimum**: Rp23.888.000
      - **Sebaran (std)**: Rp2.717.000
      - Terdapat perbedaan signifikan daya beli antar daerah.

**Kesimpulan Awal**:

  1. Terdapat **ketimpangan antar daerah** dalam hal ekonomi dan pendidikan.
  2. Fitur-fitur seperti `reg_gdp`, `exp_percap`, dan `poorpeople_percentage` menunjukkan **distribusi yang sangat lebar** dan memerlukan teknik preprocessing khusus (log transform, scaling).

---

2. Visualisasikan Dataset
   
  - Heatmap untuk melihat korelasi antar fitur
    
     ```python
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Heatmap Korelasi')
        plt.show()
     ```
     Diperoleh output sebagai berikut:

     ![image](https://github.com/user-attachments/assets/75c60d05-2d8f-4305-85b8-f32ed61c55c4)

     **Insight:**
    
       - Rata-rata lama sekolah berkorelasi **positif** dengan pengeluaran per kapita (0.67) dan angka harapan hidup (0.41). Semakin tinggi rata-rata lama sekolah di suatu wilayah, maka akan semakin tinggi pengeluaran per kapita dan angka harapan hidup. Ini menunjukkan bahwa pendidikan berpotensi menjadi indikator umum kualitas hidup dan daya beli masyarakat.
       - Persentase penduduk miskin berkorelasi **negatif** dengan rata-rata lama sekolah (-0.54), angka harapan hidup (-0.54), dan pengeluaran per kapita (-0.64). Di wilayah dengan pendidikan lebih tinggi, angka kemiskinan cenderung lebih rendah, dan kualitas hidup lebih baik.
       - Pengeluaran per kapita berkorelasi **positif** dengan angka harapan hidup (0.57) dan dengan PDRB (Produk Domestik Regional Bruto) dalam miliar rupiah (0.33). Pengeluaran per kapita berpotensi menjadi indikator umum kesejahteraan ekonomi, berkaitan erat dengan PDRB dan harapan hidup.
---

  - Barplot untuk memudahkan melihat wilayah dengan rata-rata pendidikan tertinggi dan terendah
    
   ```python
        avg_school_by_prov = df.groupby('province')['avg_schooltime'].mean().sort_values()
        lowest = avg_school_by_prov.head(5)
        highest = avg_school_by_prov.tail(5)
        
        combined = pd.concat([lowest, highest])
        
        colors = ['gray'] * len(combined)
        colors[0] = 'darkred'    
        colors[-1] = 'darkgreen' 
        
        plt.figure(figsize=(10,6))
        sns.barplot(x=combined.values, y=combined.index, palette=colors)
        
        plt.title('Provinsi dengan Rata-rata Lama Sekolah Tertinggi dan Terendah')
        plt.xlabel('Rata-rata Lama Sekolah (Tahun)')
        plt.ylabel('Provinsi')
        plt.tight_layout()
        plt.show()
   ```
   Diperoleh output sebagai berikut:

   ![image](https://github.com/user-attachments/assets/577cfe9f-82f6-4593-8b7a-425ac51f6525)

   **Insight:**
   
   - Diperoleh 5 provinsi dengan Rata-rata Lama Sekolah tertinggi yakni DKI Jakarta, DI Yogyakarta, Aceh, Kalimantan Timur dan Maluku.
   - Diperoleh 5 provinsi dengan Rata-rata Lama Sekolah terendah yakni Papua, Kalimantan Barat, Nusa Tenggara Timur, Papua Barat, dan Gorontalo.
 
---
  - Histogram untuk melihat distribusi data normal atau tidak

    ```python
    for feature in num_features:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()
    ```
    
   Diperoleh output sebagai berikut:

   ![image](https://github.com/user-attachments/assets/066f681f-b1a4-4084-aab4-deb9ee8f1291)

   **Insight:**
   - poorpeople_percentage (Persentase Penduduk Miskin): distribusi: **Right-skewed**
   - reg_gdp (PDRB dalam miliar rupiah): distribusi: Sangat **right-skewed** (ada outlier ekstrem)
   - life_exp (Angka Harapan Hidup): distribusi: **Hampir normal** (sedikit skew kiri)
   - avg_schooltime (Rata-rata Lama Sekolah): distribusi: **Hampir normal** (sedikit skew kanan dan outlier)
   - exp_percap (Pengeluaran Per Kapita): distribusi: **Right-skewed** (banyak nilai ekstrem tinggi)

---
  - Barplot untuk melihat data yang outlier
    ```python
     for feature in num_features:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()
     ```
    **Insight**
    Pada beberapa fitur terdapat banyak outlier sehingga perlu penanganan

---
    
  - Scatter Plot digunakan untuk mengidentifaksi korelasi antara 2 fitur.

     ```python
     plt.figure(figsize=(10,6))
      sns.regplot(data=df, x='avg_schooltime', y='poorpeople_percentage', scatter_kws={'s': 30}, line_kws={'color': 'red'})
      plt.title('Hubungan Rata-rata Lama Sekolah dengan Persentase Penduduk Miskin')
      plt.xlabel('Rata-rata Lama Sekolah (Tahun)')
      plt.ylabel('Persentase Penduduk Miskin (%)')
      plt.grid(True)
      plt.show()
     ```
     
      Diperoleh output sebagai berikut:
     ![image](https://github.com/user-attachments/assets/09f4bac4-ca03-4481-b32d-eca73c825edc)

     **Insight**
     - Wilayah dengan pendidikan rendah (rata-rata lama sekolah < 6 tahun) cenderung memiliki tingkat kemiskinan yang sangat tinggi (bisa >30%).
     - Wilayah dengan lama sekolah > 9 tahun hampir seluruhnya memiliki kemiskinan di bawah 15%, bahkan sebagian besar kurang dari 10%.
       Dari scatter plot diperoleh bahwa semakin tinggi persentase penduduk miskin di suatu daerah, maka semakin rendah rata-rata lama sekolah. Hal ini menunjukkan bahwa kemiskinan menjadi faktor penghambat utama terhadap keberlangsungan pendidikan. Oleh karena itu, peningkatan akses dan dukungan pendidikan di daerah-daerah miskin menjadi sangat penting untuk memutus rantai kemiskinan dalam jangka panjang
   
## Data Preparation

Pada bagian ini akan dibagi menjadi 2 bagian, yakni data preparation untuk clustering & data preparation untuk klasifikasi. 

### Data Preparation Untuk Clustering

Berdasarkan hasil EDA diketahui bahwa dataset yang digunakan tidak terdapat missing value, duplikat data sehingga tidak perlu ditangani, namun terdapat data yang outlier sehingga yang ditangani hanya dari sisi outlier, karena clustering dengan metode K-Means sangat sensitif dengan outlier sehingga perlu ditangani lebih lanjut. 

- Penanganan Outlier
  Pada tahapan ini akan dilakukan penanganan outlier dengan menggunakan RobustScaler dikarenakan dataset terlihat skewed. RobustScaler bekerja dengan median dan IQR (interquartile range) daripada mean dan standar deviasi sehingga cocok untuk data yang memiliki nilai ekstrim (outlier)
- Dataset dengan Fitur Terpilih
  - Dataset awal yang disalin = `df_scaled`, dilakukan salinan dari dataset awal agar mencegah terjadinya pencampuran dalam     dataset
  - Dataset baru dengan fitur terpilih = `df_fitur`, yang berisikan `reg_gdp`, `exp_percap`, `poorpeople_percentage`, `life_exp` dan `avg_schooltime`. Fitur `provinci` dan `cities_reg` tidak digunakan karena hanya menandakan tempat dan tidak berpengeruh begitu besar pada bagian clustering. Tujuan dari model ini adalah memahami hubungan antara kondisi ekonomi dan pencapaian pendidikan berdasarkan indikator numerik. Informasi geografis tetap disimpan secara terpisah untuk keperluan interpretasi hasil clustering dan klasifikasi.

### Data Preparation Untuk Klasifikasi

Pada bagian ini akan dilakukan pendekatan clustering menggunakan algoritma KMeans. Tujuannya adalah untuk mengelompokkan data ke dalam beberapa klaster berdasarkan kemiripan fitur, tanpa menggunakan label atau target tertentu seperti pada klasifikasi. Sebagai langkah lanjutan, hasil clustering tersebut digunakan sebagai label baru untuk membangun model klasifikasi. Model klasifikasi ini bertujuan untuk memprediksi klaster dari data baru yang masuk di masa mendatang. Dengan demikian, sistem tidak hanya mampu mengelompokkan data historis, tetapi juga dapat melakukan prediksi secara otomatis terhadap data yang belum pernah dilihat sebelumnya.

Untuk mengelompokkan wilayah berdasarkan kondisi ekonomi yang serupa, digunakan metode KMeans Clustering. Proses ini diawali dengan evaluasi model menggunakan metode Elbow, Silhouette Score, dan PCA untuk memahami struktur data. Setelah nilai K ditentukan, hasil clustering divisualisasikan dan dianalisis secara mendalam melalui barplot dan proyeksi PCA. Selanjutnya, hasil klaster digunakan sebagai label baru dalam tahap klasifikasi, dengan pemisahan fitur dan target yang disesuaikan untuk membangun model prediktif terhadap data baru.

a. Model Kmeans
  - Dilakukan metode Elbow tanpa cluster diawal untuk melihat grafik metode elbow agar dapat menentukan pemisahan kelompok data.
  - Mengitung Silhouette Score sebagai ukuran kualitas di setiap cluster.
 
     ![image](https://github.com/user-attachments/assets/20e13a07-e465-49a6-a727-942d436c09ff)
    ```pyhton
    Silhouette Score untuk 2 cluster: 0.8700
    Silhouette Score untuk 3 cluster: 0.6358
    Silhouette Score untuk 4 cluster: 0.3011
    Silhouette Score untuk 5 cluster: 0.2963
    Silhouette Score untuk 6 cluster: 0.2968
    Silhouette Score untuk 7 cluster: 0.2981
    Silhouette Score untuk 8 cluster: 0.2600
    Silhouette Score untuk 9 cluster: 0.2532
    Silhouette Score untuk 10 cluster: 0.2554
    ```

  **Insight:**
  
  Dapat dilihat pada nilai silhouette score dari tiap cluster, nilai berada di bawah 0.7 untuk nilai k dari 3 hingga 10. Sebelumnya, dataset telah dilakukan pemilihan fitur yang relevan. Namun, hasil clustering masih belum optimal. Oleh karena itu, langkah selanjutnya adalah menerapkan PCA, karena PCA dapat mereduksi dimensi data dengan tetap mempertahankan informasi penting. Dengan cara ini, pola atau struktur dalam data menjadi lebih jelas terlihat, sehingga dapat membantu meningkatkan kualitas pemisahan klaster dan menaikkan nilai silhouette score.

 - PCA

  Setelah dilakukan PCA pada dataset, dan dilakukan kembali proses evaluasi dengan metode Elbow dan silhouette score
   
  ![image](https://github.com/user-attachments/assets/00013492-8d5e-4804-8781-03f152399480)
 
   ```pyhton
  Silhouette Score untuk 2 cluster: 0.8899
  Silhouette Score untuk 3 cluster: 0.7027
  Silhouette Score untuk 4 cluster: 0.3944
  Silhouette Score untuk 5 cluster: 0.3929
  Silhouette Score untuk 6 cluster: 0.3904
  Silhouette Score untuk 7 cluster: 0.4109
  Silhouette Score untuk 8 cluster: 0.4274
  Silhouette Score untuk 9 cluster: 0.4116
  Silhouette Score untuk 10 cluster: 0.4132
  ```
**Insight**

Dapat dilihat pada nilai silhouette score dari tiap cluster setelah dilakukan proses PCA, nilai yang diperoleh mengalami kenaikan yakni pada nilai `K=3` memiliki nilai silhoutte score menyentung 0.70. Maka nilai `K=3` akan dipilih sebagai nilai k optimal untuk clustering.

- Penetapan jumlah optimal clustering yakni`K=3`
  
- Visualisasi hasil clustering

  Setelah model clustering dilatih dan jumlah cluster optimal ditentukan, langkah selanjutnya adalah menampilkan hasil clustering melalui visualisasi. Visualisasi dilakukan dengan dua metode yakni barplot dan PCA
   
   - Barplot
   
   Barplot untuk menampilkan perbandingan jumlah data pada setiap cluster

   ![image](https://github.com/user-attachments/assets/92440447-0c9b-44e5-8ae4-44d561b1510e)
     ```pyhton
    Distribusi Data dalam Setiap Klaster:
    Cluster
    0    476
    2     30
    1      8
    Name: count, dtype: int64
    ```
    **Insight**
    
  Diperoleh bahwa setelah dilakukan clustering terlihat bahwa cluster 0 memiliki data yang lebih banyak dibandingkan cluster 1 & 2, pada cluster 1 memiliki data yang lebih sedikit dibandingkan cluster 2

  - Ruang PCA (2D PCA Projection)
 
    Ruang PCA (2D PCA Projection), karena jumlah fitur yang digunakan lebih dari dua, sehingga diperlukan reduksi dimensi agar data dapat divisualisasikan secara dua dimensi dengan tetap mempertahankan informasi utama dari data.
    
   ![image](https://github.com/user-attachments/assets/5c7d26aa-aec1-4876-8302-f99bc476d6eb)

    **Insight**

   Visualisasi PCA menunjukkan bahwa data terbagi ke dalam tiga klaster yang cukup terpisah. Klaster 1 (warna jingga) terlihat memiliki karakteristik ekonomi yang sangat berbeda dibanding dua klaster lainnya, ditunjukkan oleh jarak centroid yang jauh di sisi kanan. Klaster 0 dan 2 memiliki distribusi data yang lebih rapat dan saling berdekatan, namun tetap dapat dibedakan oleh posisi centroid masing-masing. Ini mengindikasikan adanya segmentasi wilayah dengan kondisi ekonomi yang mirip namun tetap berbeda secara signifikan.
  
- Analisis dan Interpretasi Hasil Cluster
  **Cluster 1 – Daerah Tertinggal**
  
  GDP regional sangat rendah (17.29) → kemungkinan daerah miskin secara ekonomi. Pengeluaran per kapita rendah, angka kemiskinan tertinggi (12.54%), usia harapan hidup rendah, dan rata-rata lama sekolah juga rendah. Cluster ini merepresentasikan wilayah tertinggal, dengan kombinasi kesejahteraan rendah dan kualitas hidup buruk. Perlu fokus kebijakan pembangunan ekonomi dasar dan akses pendidikan.
  
  **Cluster 2 – Wilayah Maju**
  
  GDP regional sangat tinggi (611.26), pengeluaran per kapita tertinggi, dan angka pendidikan paling tinggi. Usia harapan hidup juga paling tinggi (73.63 tahun). Walaupun angka kemiskinan tidak paling rendah, tapi tetap relatif kecil (11.17%).Cluster ini merepresentasikan wilayah maju dan berkembang pesat, dengan tingkat pendidikan dan kesehatan tinggi. Fokus pembangunan bisa diarahkan pada pengembangan inovasi, ekonomi kreatif, dan investasi jangka panjang.
  
  **Cluster 3 – Wilayah Berkembang**
  
  GDP regional dan pengeluaran per kapita menengah, memiliki tingkat kemiskinan terendah (8.26%) meskipun bukan yang paling kaya. Usia harapan hidup dan rata-rata lama sekolah cukup tinggi. Cluster ini mencerminkan wilayah yang cukup stabil dan berkembang, mungkin dengan program pengentasan kemiskinan yang efektif atau distribusi kesejahteraan yang lebih merata. Bisa jadi model kebijakan untuk cluster 1.

- Inverse Data

  Setelah melakukan clustering dengan model **KMeans**, kita perlu mengembalikan data yang telah diubah (normalisasi, standarisasi, atau label encoding) ke bentuk aslinya.
  
- Memisahkan fitur (X) dan target (y), hanya digunakan fitur tanpa kolom Cluster, karena Cluster akan dijadikan target (y)
  
- Tahapan ecoding, untuk fitur ketegorical agar model hanya menerima input numerik, dan akan menggunakan metode One Hot Encoding untuk menghindari makna urutan
  
- Melatih model di `X_train` dan `y_train`, untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya (X_test, y_test).
   
- Membuat daftar nama fitur yang numerik, untuk melihat seperti apa bentuk dan isi data numerik yang akan digunakan dalam proses pelatihan model
  
- Standarisasi data pada kolom numerik yang sudah di split, karena akan menggunakan model klasifikasi berbasis jarak seperti seperti KNN, SVM
  
- SMOTE,  untuk menyeimbangkan jumlah data antar klaster agar model klasifikasi tidak bias terhadap klaster yang dominan.
  
- Training model, setiap model dilatih untuk mengenali pola dalam data agar bisa memprediksi cluster (yang sebelumnya diperoleh dari proses clustering). Masing-masing model dilatih menggunakan data hasil oversampling (X_train_resampled, y_train_resampled).

## Modeling

   Terdapat **5 model klasifikasi** yang akan digunakan yakni K-Nearest Neighbor. Decision Tree, Random Forest, Support Vector Machine dan Naive Bayes. Setiap model dilatih untuk mengenali pola dalam data agar bisa memprediksi cluster (yang sebelumnya diperoleh dari proses clustering). Masing-masing model dilatih menggunakan data hasil oversampling `(X_train_resampled, y_train_resampled)`.
   
   - Algoritma K-Nearest Neighbors (KNN) adalah metode supervised learning yang digunakan untuk mengatasi masalah klasifikasi dan regresi. Algoritma ini digunakan untuk mengklasifikasikan data baru berdasarkan kedekatan jarak dengan data yang sudah diberi label pada dataset pelatihan. KNN sering digunakan karena kemudahannya dalam pemahaman dan implementasi meskipun pada praktiknya, ia dapat menjadi sangat efektif untuk berbagai masalah klasifikasi.

      Training Model: 

      `knn = KNeighborsClassifier().fit(X_train_resampled, y_train_resampled)`

     Penjelasan:

     K-Nearest Neighbors (KNN) digunakan dengan nilai parameter default, yaitu `n_neighbors=5`, model mempertimbangkan lima tetangga terdekat untuk menentukan kelas dari suatu sampel. 
     
   - Decision Tree adalah algoritma machine learning yang sering digunakan dalam tugas klasifikasi dan regresi. Struktur dari algoritma ini seperti dengan bentuk pohon dengan setiap cabang mewakili keputusan atau percabangan dari data berdasarkan fitur-fitur yang ada. Struktur dasar dari Decision Tree melibatkan tiga komponen utama, yaitu akar (root node), node (decision node), dan daun (leaf node). Root node mewakili seluruh dataset dan menjadi titik awal untuk pemisahan data. Node-node di sepanjang cabang pohon mewakili keputusan yang diambil berdasarkan fitur tertentu, sedangkan leaf node adalah hasil akhir dari proses klasifikasi atau regresi, seperti label kelas atau nilai numerik.

     Training Model: 

     `dt = DecisionTreeClassifier(random_state=42).fit(X_train_resampled, y_train_resampled)`

     Penjelasan:

     Decision Tree diterapkan dengan parameter `random_state=42` untuk memastikan hasil yang reprodusibel. Decision Tree menggunakan metode pengukuran "impurity" (ketidakmurnian) untuk memutuskan bagaimana membagi data pada tiap node pohon. Parameter criterion mengatur metode mana yang digunakan untuk menilai kualitas suatu split.
    
   - Random Forest adalah algoritma ensemble learning yang menggabungkan beberapa Decision Tree untuk meningkatkan akurasi prediksi dan mengurangi risiko overfitting. Setiap pohon dalam Random Forest dilatih menggunakan subset acak dari data pelatihan dan subset acak dari fitur yang tersedia. Hasil akhir prediksi ditentukan melalui voting (untuk klasifikasi) atau rata-rata (untuk regresi) dari hasil semua pohon dalam model. Tujuan utama Random Forest adalah mengatasi kelemahan Decision Tree yang cenderung overfit terhadap data pelatihan. Dengan menggabungkan prediksi dari banyak pohon, Random Forest mampu menghasilkan model yang lebih stabil, akurat, dan lebih general terhadap data baru.
    
      Training Model: 

      `rf = RandomForestClassifier(random_state=42).fit(X_train_resampled, y_train_resampled)`

     Penjelasan:

     Random Forest menggunakan `random_state=42`, dengan jumlah pohon (`n_estimators`) menggunakan nilai default yaitu 100, serta `criterion='gini'`. 
     
   - Support vector machine (SVM) adalah salah satu algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Namun, SVM lebih sering digunakan pada masalah klasifikasi. SVM bekerja dengan mencari hyperplane yang optimal untuk memisahkan data ke dalam kelas-kelas yang berbeda. Hyperplane adalah garis (pada data dua dimensi) atau bidang (pada data tiga dimensi) yang memisahkan data dari kelas berbeda. Tujuan SVM adalah menemukan hyperplane yang memaksimalkan margin, yaitu jarak antara hyperplane dan titik data terdekat dari setiap kelas.
     
      Training Model: 

      `svm = SVC(random_state=42).fit(X_train_resampled, y_train_resampled)`

     Penjelasan:

     Model Support Vector Machine (SVM) dilatih menggunakan parameter default `kernel='rbf'` dan `C=1.0`, serta random_state=42 untuk memastikan hasil yang konsisten di setiap eksekusi. Kernel 'rbf' dipilih secara default oleh Scikit-learn dan bekerja dengan baik pada data yang tidak linear.
     
   - Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang berdasarkan pada Teorema Bayes, dengan asumsi bahwa fitur-fitur dalam data bersifat independen satu sama lain. Nama "naive" (naif) merujuk pada asumsi independensi ini, yang sering kali tidak realistis. Namun, dalam praktiknya dapat menghasilkan model yang efektif. Naive Bayes menggunakan prinsip probabilitas untuk memprediksi kelas dari data baru berdasarkan pengamatan fitur yang ada. Secara matematis, Naive Bayes bekerja dengan menghitung kemungkinan suatu data termasuk dalam kelas tertentu berdasarkan dua faktor: kemungkinan awal dari setiap kelas (probabilitas prior) dan kemungkinan fitur dalam data jika kelas tersebut benar (probabilitas likelihood). Setelah menghitung kemungkinan untuk setiap kelas, model ini memilih kelas dengan kemungkinan tertinggi sebagai hasil klasifikasi.

      Training Model: 

      ` nb = GaussianNB().fit(X_train_resampled, y_train_resampled)`

     Penjelasan:

     Naive Bayes yang digunakan adalah GaussianNB, yang merupakan varian Naive Bayes yang sesuai untuk fitur kontinu dan menggunakan parameter default tanpa penyesuaian khusus. 
     
## Evaluation Model
   - KNN

     Diperoleh Confusion Matrix:

     ![image](https://github.com/user-attachments/assets/6bb1408f-d898-4088-a58d-6b190be79d05)

      **Interpretasi KNN Confusion Matrix:**

        Model KNN menunjukkan performa dalam mengklasifikasikan data ke dalam empat kelas. Dari confusion matrix, kita dapat melihat beberapa poin penting:
        
        - Kelas 0:
        
          - 95 sampel diprediksi dengan benar sebagai kelas 0 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        - Kelas 1:
          - 2 sampel diprediksi dengan benar sebagai kelas 1 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        - Kelas 2:
          - 6 sampel diprediksi dengan benar sebagai kelas 2 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        Secara keseluruhan, model memiliki performa yang sangat baik, namun berpotensi overfitting.

   - SVM

     Diperoleh Confusion Matrix:
     
     ![image](https://github.com/user-attachments/assets/fd92a375-0073-4727-85f9-d22283069fc9)

     **Interpretasi SVM Confusion Matrix:**

        Model SVM menunjukkan performa dalam mengklasifikasikan data ke dalam tiga kelas. Dari confusion matrix, kita dapat melihat beberapa poin penting:
        
        - Kelas 0:
          - 92 sampel diprediksi dengan benar sebagai kelas 0 (True Positive) dan 3 yang terprediksi sebagai kelas 2 (False Negative)
        
        - Kelas 1:
          - 2 sampel diprediksi dengan benar sebagai kelas 1 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        - Kelas 2:
          - 6 sampel diprediksi dengan benar sebagai kelas 2 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        Secara keseluruhan, model memiliki performa yang baik, hanya terdapat kesalahan memprediksi kelas 0.
     
   - Decision Tree
     
     Diperoleh Confusion Matrix:

     ![image](https://github.com/user-attachments/assets/8f4b4f94-8642-47e3-ba8a-55891769bd60)

     **Interpretasi Decision Tree Confusion Matrix:**
      
      Model Decision Tree menunjukkan performa dalam mengklasifikasikan data ke dalam tiga kelas. Dari confusion matrix, kita dapat melihat beberapa poin penting:
      
      - Kelas 0:
        - 95 sampel diprediksi dengan benar sebagai kelas 0 (True Positive) 
      
      - Kelas 1:
        - 2 sampel diprediksi dengan benar sebagai kelas 1 (True Positive). Tidak ada sampel yang terprediksi salah.
      
      - Kelas 2:
        - 6 sampel diprediksi dengan benar sebagai kelas 2 (True Positive). Tidak ada sampel yang terprediksi salah.
      
      Secara keseluruhan, model memiliki performa yang sangat baik, namun kemungkinan mengalami overfitting
     
   - Random Forest

     Diperoleh Confusion Matrix:

     ![image](https://github.com/user-attachments/assets/61f995e7-8be4-4f73-abdd-7c3b5a4d06c6)
     
     **Interpretasi Random Forest Confusion Matrix:**
        Model Random Forest menunjukkan performa dalam mengklasifikasikan data ke dalam tiga kelas. Dari confusion matrix, kita dapat melihat beberapa poin penting:
        
        - Kelas 0:
          - 95 sampel diprediksi dengan benar sebagai kelas 0 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        - Kelas 1:
          - 1 sampel diprediksi dengan benar sebagai kelas 1 (True Positive). 1 sampel diprediksi salah dengan kelas 2 (False Negative).
        
        - Kelas 2:
          - 6 sampel diprediksi dengan benar sebagai kelas 2 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        Secara keseluruhan, model memiliki performa yang baik
    
   - Naive Bayes
     
     Diperoleh Confusion Matrix:
     
     ![image](https://github.com/user-attachments/assets/b2911008-c50e-4216-a3b3-632303870e61)

     **Interpretasi Naive Bayes Confusion Matrix:**

        Model Naive Bayes menunjukkan performa dalam mengklasifikasikan data ke dalam tiga kelas. Dari confusion matrix, kita dapat melihat beberapa poin penting:
        
        - Kelas 0:
          - 57 sampel diprediksi dengan benar sebagai kelas 0 (True Positive). 38 sampel diprediksi salah dengan kelas 2 (False Negative).
        
        - Kelas 1:
          - 1 sampel diprediksi dengan benar sebagai kelas 1 (True Positive). 1 sampel diprediksi salah dengan kelas 2 (False Negative).
        
        - Kelas 2:
          - 6 sampel diprediksi dengan benar sebagai kelas 2 (True Positive). Tidak ada sampel yang terprediksi salah.
        
        Secara keseluruhan, model memiliki performa yang cukup baik

### **Analisis Hasil Evaluasi Model**
Berdasarkan data evaluasi yang diperbarui, berikut adalah analisis untuk masing-masing model dengan mempertimbangkan metrik Accuracy, Precision, Recall, dan F1-Score:

```python
                             Model  Accuracy  Precision    Recall  F1-Score
    0     K-Nearest Neighbors (KNN)  1.000000   1.000000  1.000000  1.000000
    1            Decision Tree (DT)  1.000000   1.000000  1.000000  1.000000
    2            Random Forest (RF)  0.990291   0.952381  0.833333  0.863248
    3  Support Vector Machine (SVM)  0.970874   0.888889  0.989474  0.927986
    4              Naive Bayes (NB)  0.621359   0.711111  0.700000  0.550654
```
1. **K-Nearest Neighbors (KNN)**
   - **Accuracy**: 100%
   - **Precision**: 100%
   - **Recall**: 100%
   - **F1-Score**: 100%

   **Analisis**:

   K-Nearest Neighbors menunjukkan performa yang sangat tinggi dengan akurasi 100%, serta precision, recall, dan F1-score yang juga sempurna. Hasil ini menunjukkan bahwa model mampu mengklasifikasikan semua data dengan benar, tanpa kesalahan. Namun, nilai yang terlalu sempurna ini kemungkinan besar menunjukkan overfitting, di mana model menghafal pola dalam data latih dan mungkin tidak bekerja sebaik ini pada data baru yang belum pernah dilihat sebelumnya.

2. **Decision Tree (DT)**
   - **Accuracy**: 100%
   - **Precision**: 100%
   - **Recall**: 100%
   - **F1-Score**: 100%

   **Analisis**:

   Decision Tree menunjukkan performa yang sangat tinggi dengan akurasi 100%, serta precision, recall, dan F1-score yang juga sempurna. Hasil ini menunjukkan bahwa model mampu mengklasifikasikan semua data dengan benar, tanpa kesalahan. Namun, nilai yang terlalu sempurna ini kemungkinan besar menunjukkan overfitting, di mana model menghafal pola dalam data latih dan mungkin tidak bekerja sebaik ini pada data baru yang belum pernah dilihat sebelumnya.

3. **Random Forest (RF)**
   - **Accuracy**: 99.02%
   - **Precision**: 95.23%
   - **Recall**: 83.33%
   - **F1-Score**: 86.32%

   **Analisis**:

   Random Forest menunjukkan performa yang sangat tinggi dengan akurasi hampir sempurna (99.02%). Precision-nya (95.23%) juga sangat baik, menunjukkan bahwa model ini sangat akurat dalam memprediksi kasus positif. Recall-nya (83.33%) ini bisa berarti model masih melewatkan beberapa prediksi pada satu atau lebih kelas, meskipun precision-nya bagus.

4. **Support Vector Machine (SVM)**
   - **Accuracy**: 97.08%
   - **Precision**: 88.88%
   - **Recall**: 98.94%
   - **F1-Score**: 92.79%

   **Analisis**:

   Support Vector Machine menunjukkan performa yang sangat tinggi dengan akurasi hampir sempurna (99.02%). Precision-nya (95.23%) juga sangat baik, menandakan kemampuannya menangkap banyak prediksi benar tanpa banyak false positives. Cocok untuk data yang cukup bersih dan terpisah jelas.

5. **Naive Bayes (NB)**
   - **Accuracy**: 62.13%
   - **Precision**: 71.11%
   - **Recall**: 70%
   - **F1-Score**: 55.55%

   **Analisis**:

   Naive Bayes memiliki akurasi yang jauh lebih rendah dibanding model lain (62.13%) dan precision yang rendah (71.11%). dan recall-nya yang rendah (70%) menunjukkan bahwa model ini kurang efektif dalam menangkap kasus positif. F1-Score 55.55% menandakan asumsi independensi antar fitur tidak cocok dengan dataset ini.

### **Kesimpulan:**
  
  Model SVM menjadi pilihan tepat karena tidak hanya memiliki akurasi tinggi, tetapi juga mampu mendeteksi hampir seluruh kelas dengan benar (recall tinggi), sekaligus menjaga kualitas prediksi (precision tinggi). Hal ini sangat penting terutama dalam konteks sosial seperti pendidikan, di mana kesalahan klasifikasi bisa berdampak pada perencanaan kebijakan. Dalam kasus ini, recall yang tinggi sangat penting karena memastikan bahwa kelompok-kelompok wilayah yang membutuhkan perhatian tidak terlewatkan oleh model.
  
  Proyek ini berangkat dari keingintahuan akan bagaimana kondisi ekonomi suatu daerah memengaruhi tingkat pencapaian pendidikan masyarakatnya. Untuk menjawab hal tersebut, dilakukan analisis data ekonomi dan pendidikan dari berbagai wilayah di Indonesia. Dengan menerapkan metode clustering, wilayah-wilayah dikelompokkan berdasarkan kesamaan indikator ekonomi seperti GDP regional, pengeluaran per kapita, persentase penduduk miskin, harapan hidup, dan rata-rata lama sekolah. Hasil klaster ini kemudian dimanfaatkan sebagai label baru untuk membangun model klasifikasi prediktif, sehingga sistem tidak hanya mampu mengenali pola historis, tetapi juga dapat memprediksi pencapaian pendidikan dari data ekonomi wilayah baru. Dengan pendekatan ini, proyek berhasil memberikan gambaran yang lebih terstruktur mengenai keterkaitan ekonomi dan pendidikan, mengidentifikasi kelompok wilayah dengan karakteristik yang serupa, serta menyediakan model prediktif berbasis machine learning yang dapat dijadikan alat bantu dalam perumusan kebijakan pendidikan yang lebih adaptif dan berbasis data.


## **Referensi**

[1]: UNESCO. (2020). Education transforms lives. https://en.unesco.org/themes/education

[2]: Badan Pusat Statistik. (2022). Statistik Pendidikan 2022. https://www.bps.go.id/id/publication/2022/11/25/a80bdf8c85bc28a4e6566661/statistik-pendidikan-2022.html

[^1]: UNESCO. (2020). Education transforms lives. https://en.unesco.org/themes/education
[^2]: Badan Pusat Statistik. (2022). Statistik Pendidikan 2022. https://www.bps.go.id/id/publication/2022/11/25/a80bdf8c85bc28a4e6566661/statistik-pendidikan-2022.html






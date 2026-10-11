# El yazısı rapor için iskelet ve çalışma soruları

Bu dosya teslim edilecek hazır rapor değildir. Sonuç tablolarını ve grafiklerini inceleyip açıklamaları **kendi cümlelerinizle**, el yazısıyla oluşturun. Öğrenci adı/numarası bilinmediği için aşağıdaki boşlukları kendiniz doldurun.

## 1. Öğrenci bilgileri ve veri kaynağı

- Ad soyad: …
- Öğrenci numarası: …
- Ders: BMM4111 Makine Öğrenmesi, Balıkesir Üniversitesi Bilgisayar Mühendisliği.
- Teslim tarihi: … (ödev PDF'sinde son tarih 20 Aralık, 23:59).
- Veri: Online Shoppers Purchasing Intention Dataset, UCI 468; 12.330 oturum, 17 özgün girdi ve Revenue.
- Kaynak/atıf: Sakar & Kastro (2018), https://doi.org/10.24432/C5F88Q; CC BY 4.0.

Kendi açıklamanız için: Bir satır neyi temsil ediyor? Aynı oturumdan üç farklı öğrenme problemi nasıl kuruluyor?

## 2. Veri tanıma ve kalite analizi

Kanıt: `results/tables/dataset_summary.json`, `numeric_summary.csv`, `missing_values.csv`, `outlier_summary.csv`.

- Ham veri 12.330 × 18; eksik değer sayısı 0; tam kopya 125.
- Revenue=False: 10.422; True: 1.908. Satın alma oranı yaklaşık %15,47.
- Ham hedefte sıfır ürün süresi bulunan oturum: 755. MAPE neden uygun değil?
- Modelleme: kopyalar çıkarıldıktan sonra 12.205 satır; 9.764 eğitim, 2.441 test; 15 ortak girdi.
- Sayısal/kategorik ayrımını açıklayın; Browser=2'nin neden Browser=1'in iki katı olmadığını örnekleyin.
- IQR aykırıları silinmedi. IQR=0 olan seyrek/sıfır ağırlıklı alanlarda işaretlenen her kayıt hatalı mı?

Grafikler: `class_distribution.png`, `missing_values.png`, `numeric_distributions.png`, `correlation.png`, `outliers.png` (hepsi `results/figures/`). Korelasyonun nedensellik göstermediğini ve hedefle güçlü ilişki bulmanın tek başına geçerli bir girdi seçimi olmadığını tartışın.

## 3. Sınıflandırma

Gözlem oturum; y=Revenue, iki sınıflı satın alma sonucu. X'te iki görev hedefi ve PageValues yok. CV ölçütü pozitif sınıf F1; 5 katlı stratified CV, seed=42. Ayarlar `classification_selection.json` içinde.

| Model | CV F1 ± std | Eğitim F1 | Test F1 | Test accuracy | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,0000 ± 0,0000 | 0,0000 | 0,0000 | 0,8435 | 0,0000 | 0,0000 |
| Logistic Regression | 0,3899 ± 0,0146 | 0,3960 | 0,4044 | 0,6440 | 0,2739 | 0,7723 |
| ANN (64,32) | 0,0839 ± 0,0820 | 0,1406 | 0,0714 | 0,8402 | 0,3947 | 0,0393 |

ANN (32,) CV F1=0,0553; ANN (64,32) CV F1=0,0839. Mimari seçimi CV üzerinden yapıldı. Genel CV seçimi Logistic Regression. Tam precision/recall/accuracy/F1 eğitim ve CV değerleri `classification_results.csv`, fold ölçümleri `classification_cv_folds.csv` içindedir.

Kendi yorumunuz için: Dummy %84 accuracy ile neden hiçbir satın almayı bulamıyor? Logistic Regression'ın class_weight="balanced" ayarı recall/precision dengesini nasıl etkiliyor? ANN neden baseline benzeri accuracy'ye rağmen düşük recall veriyor? İç early stopping accuracy kullandığında azınlık sınıfı için hangi sınırlılık oluşuyor? ANN'in train/test farkını tek başına büyük aşırı uyum kanıtı saymak doğru mu; iki taraftaki mutlak başarıya da bakın.

Grafikler: seçilmiş modellerin `confusion_matrix_*.png` dosyaları, `classification_comparison.png`, `ann_architectures.png`. Karışıklık matrisinde satır gerçek, sütun tahmin edilen sınıftır; pozitif sınıf satın almadır.

## 4. Regresyon

y=ProductRelated_Duration, saniye biriminde toplam ürün sayfası süresi. X aynı 15 alan; Revenue girdi değildir. 5 katlı KFold ile seçim ölçütü CV RMSE. Tabloda hücreler **eğitim / test** biçimindedir.

| Model | CV RMSE ± std | MAE | MSE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|---|
| Dummy | 1911,99 ± 262,42 | 1104,83 / 1125,21 | 3710595,12 / 3580458,01 | 1926,29 / 1892,21 | 0,0000 / −0,000001 | — |
| Linear Regression | 952,24 ± 158,44 | 495,61 / 494,46 | 874413,78 / 736651,17 | 935,10 / 858,28 | 0,7643 / 0,7943 | — |
| Random Forest | 947,34 ± 141,45 | 288,96 / 488,13 | 277453,88 / 748734,05 | 526,74 / 865,29 | 0,9252 / 0,7909 | — |

Kesin değerleri `regression_results.csv` dosyasından alın; yukarıdaki değerler okunabilirlik için yuvarlatılmıştır. RF ayarı: 80 ağaç, max_depth=16, min_samples_leaf=2. Leaf=5 ve log1p hedefli leaf=2 adayları yalnız CV ile denendi; CV RMSE sırasıyla 992,98 ve 972,76 saniye. Log1p adayı MAE'yi düşürse de birincil RMSE'de seçilmedi.

Kendi yorumunuz için: RMSE büyük hataları MAE'ye göre neden daha fazla etkiler? Negatif R² mümkün mü? CV'de seçilen RF testte Linear Regression'dan biraz daha yüksek RMSE verdiğinde seçimi niçin değiştirmiyoruz? RF'nin düşük eğitim hatasıyla test hatası arasındaki fark ne söylüyor? Yaklaşık 4,90 saniyelik CV üstünlüğü ve fold standart sapmaları düşünülünce kesin üstünlük iddiası kurabilir miyiz?

Grafikler: `actual_predicted_*.png`, `residuals_*.png`, `regression_comparison.png`. Her hata ölçütü orijinal süre biriminde raporlanır; MSE'nin birimi saniye²'dir. Tam tabloyu CSV'den kendi el yazısı tablonuza aktarın.

## 5. Kümeleme

K-Means; y yok; eğitim scaler/encoder ve merkezleri testte yeniden fit edilmez. k=2,3,4,5,6 için inertia/silhouette tablosunu `clustering_results.csv` dosyasından çıkarın. En yüksek silhouette k=3 için 0,3076. Normalize kiriş uzaklığına göre dirsek adayı k=4; k=3 ve k=4 uzaklıkları 0,2115 / 0,2135 ile yakın. k=3 daha yüksek silhouette (k=4: 0,2571) ve daha sade profil nedeniyle seçildi.

| Küme | Eğitim / test sayısı | Eğitim ürün sayfa sayısı ort./medyan | Eğitim BounceRates ort. | Eğitim ExitRates ort. | Veriye dayalı yorum adayı |
|---|---:|---:|---:|---:|---|
| 0 | 7673 / 1909 | 24,94 / 17 | 0,0091 | 0,0322 | Orta etkileşimli çoğunluk |
| 1 | 735 / 197 | 3,33 / 1 | 0,1625 | 0,1780 | Düşük etkileşim ve yüksek çıkış oranı |
| 2 | 1356 / 335 | 88,21 / 61 | 0,0066 | 0,0198 | Yoğun sayfa inceleme |

Kategorik profiller: Returning_Visitor oranları sırasıyla yaklaşık %83,28 / %95,10 / %92,77. Küme 2'de Nov oranı %35,91; küme 1'de May oranı %37,55. Bunları davranışların kesin nedenleri olarak sunmayın. Küme adları sayısal/kategorik profillerden sonra oluşturulmuş yorum adaylarıdır; satın alma etiketi değildir.

Grafikler: `elbow.png`, `silhouette.png`, `cluster_sizes.png`, `cluster_pca.png`. PCA'nın iki bileşeni yaklaşık %37,24 varyansı açıklar. Silhouette 1.500 eğitim satırından hesaplandığından örneklem sınırlılığını belirtin. K-Means'te ölçek ve uç değerler sonucu nasıl etkiler? Burada neden sınıflandırmadaki gibi doğruluk etiketi/baseline yok?

## 6. Veri sızıntısı ve deney tasarımı

`docs/experiment_design.md` içindeki denetimleri okuyup kendi örneğinizle açıklayın: testte scaler fit etmek; Revenue'yu regresyona vermek; PageValues'ın işlem bilgisini taşıması; CV öncesi tüm veride encoder öğrenmek. Her risk için kodda kullanılan önlemi eşleştirin. Sabit seed'in hangi rastgele adımları kontrol ettiğini ve tek split'in bütün genelleme belirsizliğini ortadan kaldırmadığını tartışın.

## 7. Genel sonuç

Kendi cümleleriniz için sorular: Aynı X'ten kategorik sonuç, sürekli süre ve etiketsiz gruplar nasıl çıktı? Hangi model hangi ölçüte göre seçildi? Baseline'a göre kazanım ne? ANN/RF'nin sınırlılıkları ne? Canlı tahmin için hangi ek zaman bilgisi gerekir? Modellerin eğitim/test farkları ve CV değişkenliği ne kadar güvenli yorum yapmanıza izin veriyor?

## 8. GitHub bağlantısı ve teslim kontrolü

Raporun en sonunda tam bağlantıyı yazın: https://github.com/FahriGunduz/bmm4111-machine-learning-project

- Rapor el yazısıyla, öğrencinin kendi ifadeleriyle hazırlanır.
- İlk sayfada ad/soyad, numara, teslim tarihi ve veri kaynağı bulunur.
- Tüm kâğıtlar ve taranan PDF aynı sırada numaralandırılır.
- CamScanner ile tek PDF: `OgrenciNo_Ad_Soyad_ML_F26.pdf`; dosya adında Türkçe karakter/boşluk kullanmayın.
- PDF Teams'e 20 Aralık 23:59'a kadar yüklenir; kâğıt asılları ayrıca elden teslim edilir.
- Kodlar erişilebilir GitHub deposundadır; 3–5 dakikalık bireysel savunmaya hazırlanılır.

PDF oluşturma, Teams yükleme ve elden teslim öğrenci tarafından yapılacaktır; bunlar yazılım projesinin tamamlanmasıyla gerçekleşmiş sayılmaz.

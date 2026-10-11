# BMM4111 – Bir Veri Kümesi, Üç Problem

Balıkesir Üniversitesi Bilgisayar Mühendisliği **BMM4111 Makine Öğrenmesi** dönem ödevi. Tek oturum veri kümesi üzerinde sınıflandırma, regresyon ve kümeleme; gerçek deneyler, eğitim içinde çapraz doğrulama ve veri sızıntısı denetimi içerir. Standart CPU ile çalışır.

Depo: https://github.com/FahriGunduz/bmm4111-machine-learning-project

[Google Colab'de defteri aç](https://colab.research.google.com/github/FahriGunduz/bmm4111-machine-learning-project/blob/main/notebooks/ML_Donem_Odevi.ipynb)

## Veri kaynağı ve atıf

[UCI Online Shoppers Purchasing Intention Dataset, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): 12.330 oturum, 17 özgün girdi, Revenue ile toplam 18 sütun. Lisans: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Atıf:** Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.

Ham veride eksik değer **yoktur**; 125 tam kopya vardır. Kopyalar modelleme öncesinde ayıklanır; ham EDA bu durumu ayrıca gösterir. Modelleme 12.205 farklı kayıt üzerinde yapılır. Aykırı gözlemler topluca silinmez. Revenue dağılımı: 10.422 olumsuz, 1.908 olumlu oturum (%15,47 satın alma). Sayısal ölçümler ve nominal kategoriler birlikte bulunur.

| Görev | Hedef | Baseline | Modeller |
|---|---|---|---|
| Sınıflandırma | Revenue, satın alma var/yok | DummyClassifier, most_frequent | Logistic Regression, MLPClassifier ANN |
| Regresyon | ProductRelated_Duration, saniye | DummyRegressor, mean | Linear Regression, Random Forest Regressor |
| Kümeleme | Hedef yok | Beklenmiyor | K-Means, k=2…6 |

## Kurulum ve çalıştırma

Doğrulanan ortam Python **3.11.9**; ana bağımlılıklar `requirements.txt` içinde sabittir. PyTorch, TensorFlow, GPU, Docker veya web sunucusu gerekmez. Proje kökünde Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ml_coursework.run_all
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_repo_size.py
```

Linux/macOS'ta ortamı `python3 -m venv .venv` ile oluşturup etkinleştirin:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ml_coursework.run_all
python -m pytest -q
python scripts/check_repo_size.py
```

Windows'ta `python` farklı bir kurulum olabilir; yukarıdaki `.venv\Scripts\python.exe` çağrıları doğru ortamı açıkça seçer. Tüm çalıştırmalarda random seed **42** kullanılır. Ana CLI tek BLAS iş parçacığı ve CV n_jobs=1 sınırı koyar. İşlem süresi CPU'ya göre değişir; geniş hiperparametre taraması yapılmaz.

Her aşama tek başına çalıştırılabilir:

```powershell
.\.venv\Scripts\python.exe -m ml_coursework.run_all --stage eda
.\.venv\Scripts\python.exe -m ml_coursework.run_all --stage classification
.\.venv\Scripts\python.exe -m ml_coursework.run_all --stage regression
.\.venv\Scripts\python.exe -m ml_coursework.run_all --stage clustering
```

`--output-dir` ile çıktı konumu değiştirilebilir. Veri otomatik olarak `data/online_shoppers_intention.csv` konumuna indirilir. Öncelik `ucimlrepo.fetch_ucirepo(id=468)`; erişim sorunu olursa resmî UCI ZIP yedeği kullanılır. Var olan önbellek yeniden indirilmez.

**Çevrimdışı kullanım:** Resmî CSV'yi bu konuma koyun veya mevcut dosyanızı belirtin:

```powershell
.\.venv\Scripts\python.exe -m ml_coursework.run_all --data-path 'C:\veriler\online_shoppers_intention.csv'
```

Belirtilen dosya yoksa açık hata verilir; sessizce farklı veri kullanılmaz. Şema, satır sayısı ve hedefler yüklemede doğrulanır. Bağımlılıklar kurulduktan ve veri önbelleğe alındıktan sonra ana CLI internet gerektirmez.

## Deney tasarımı

- Kopyalar ayıklandıktan sonra Revenue ile stratified %80/%20 ayrım: **9.764 eğitim / 2.441 test**. Satır indeksleri bir kez bölünür; üç görev aynı indeksleri kullanır.
- **Revenue, ProductRelated_Duration ve PageValues bütün görevlerin girdilerinden çıkarılır.** 15 ortak girdi kalır. İzin listesi sonradan eklenen hedef türevlerini de dışlar.
- PageValues işlemle ilişkili parasal değer taşır; hesaplanma zamanı ve tahmin anında bulunabilirliği kesin olmadığı için potansiyel sızıntı riski olarak dışlanır. Bu, kanıtlanmış sızıntı iddiası değildir.
- Browser, Region, TrafficType, OperatingSystems, Month, VisitorType ve Weekend kategoriktir. OneHotEncoder bilinmeyen kategorileri ignore eder. Nadir kategorilerin bazı fold'larda görülmemesine ilişkin uyarılar beklenir.
- Sayısal median imputer; kategorik most-frequent imputer; gerekli modellerde StandardScaler; ColumnTransformer ve Pipeline. Her CV fold'unda bütün dönüşümler yalnız o fold'un eğitim verisinde öğrenilir.
- Sınıflandırma: 5 katlı StratifiedKFold, birincil ölçüt **F1**. Regresyon: 5 katlı KFold, birincil ölçüt **RMSE**. CV yalnız eğitim kümesindedir. Raporlanan std, fold ölçümlerinin örnek standart sapmasıdır (ddof=1).
- ANN mimarisi ve RF ayarları CV ile seçilip dondurulduktan sonra baseline ve seçilmiş iki algoritmanın test karşılaştırması yapılır. Test sonraki ayar değişikliğine gerekçe yapılmaz.
- Kümeleme: scaler/encoder/merkezler yalnız eğitimde öğrenilir. Test mevcut merkeze atanır; k seçimi eğitim silhouette'ına dayanır. PCA yalnız görselleştirmedir ve eğitimde öğrenilir.

Ayrıntılar: [Veri sızıntısı ve deney tasarımı](docs/experiment_design.md). Bu, tamamlanmış oturum verisiyle bir ders deneyidir; canlı oturum başı tahmini için zaman damgalı özelliklerin ayrıca incelenmesi gerekir.

## Gerçek sınıflandırma sonuçları

Pozitif sınıf satın almadır. Tablo `results/tables/classification_results.csv` ölçümlerinin yuvarlatılmış özetidir; CSV ayrıca her metrik için CV/eğitim/test değerlerini içerir.

| Model | CV F1 ± std | Eğitim F1 | Test F1 | Test accuracy | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,0000 ± 0,0000 | 0,0000 | 0,0000 | 0,8435 | 0,0000 | 0,0000 |
| **Logistic Regression** | **0,3899 ± 0,0146** | 0,3960 | 0,4044 | 0,6440 | 0,2739 | 0,7723 |
| ANN (64,32) | 0,0839 ± 0,0820 | 0,1406 | 0,0714 | 0,8402 | 0,3947 | 0,0393 |

**CV ile seçilen model Logistic Regression.** Dummy'nin yüksek accuracy'si bütün satın almaları kaçırmasını gizler. LR satın almaları daha fazla yakalarken daha çok yanlış pozitif üretir; balanced sınıf ağırlıklarıyla bu değiş tokuş beklenir. LR eğitim/test F1 yakınlığı bu ayrımda belirgin büyük bir performans düşüşü göstermiyor.

ANN mimarileri: `(32,)` CV F1=0,0553 ± 0,0473; `(64,32)` CV F1=0,0839 ± 0,0820. İkinci mimari CV ile seçildi; her iki mimari de düşük azınlık recall'ı verdi. ANN train/test F1 farkı ve düşük mutlak performans birlikte değerlendirilmelidir. sklearn early stopping iç doğrulamada accuracy kullanır; sınıf dengesizliğinde bu ayar satın alma recall'ını yeterince ödüllendirmeyebilir. Bu açıklama bir olası etkidir; nedenini ayıran ek bir deney yapılmadı.

| Model/ayar | Kullanılan hiperparametreler |
|---|---|
| Logistic Regression | C=1, solver=lbfgs, class_weight=balanced, max_iter=1000, seed=42 |
| ANN | ReLU, Adam, alpha=0,001, learning_rate_init=0,001, max_iter=200, early_stopping=True, validation_fraction=0,15, n_iter_no_change=15, seed=42 |

Tam ayarlar `classification_selection.json`; ANN CV tablosu `ann_architecture_results.csv`; fold ölçümleri `classification_cv_folds.csv`. Karışıklık matrisleri JSON'da ve yerel PNG'lerde saklanır.

## Gerçek regresyon sonuçları

| Model | CV RMSE ± std (s) | Eğitim RMSE (s) | Test MAE (s) | Test MSE (s²) | Test RMSE (s) | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 1911,99 ± 262,42 | 1926,29 | 1125,21 | 3580458,01 | 1892,21 | −0,000001 |
| Linear Regression | 952,24 ± 158,44 | 935,10 | 494,46 | 736651,17 | 858,28 | 0,7943 |
| **Random Forest** | **947,34 ± 141,45** | 526,74 | 488,13 | 748734,05 | 865,29 | 0,7909 |

**CV ile seçilen model Random Forest** (80 ağaç, max_depth=16, min_samples_leaf=2, seed=42, n_jobs=1). Leaf=5 adayı CV RMSE=992,98; eğitim hedefi çarpık olduğundan denenmiş log1p hedefli leaf=2 adayı CV RMSE=972,76 saniye. Log adayının MAE'si daha düşük olsa da seçim ölçütü RMSE olduğu için seçilmedi. Tam aday ölçümleri `regression_candidates.csv` içinde.

RF'nin Linear Regression'a CV üstünlüğü yaklaşık 4,90 saniye; fold değişkenliğine kıyasla küçük, kesin üstünlük iddiası kurmuyoruz. Testte Linear Regression biraz daha düşük RMSE verdi; bu sonuç CV seçimini değiştirmedi. RF eğitim/test RMSE farkı daha büyüktür ve aşırı uyum eğilimine işaret eder. Her iki öğrenen model baseline'ı belirgin aşar.

MAE, MSE, RMSE ve R²'nin tüm eğitim/test ve CV ortalama/std değerleri `regression_results.csv` dosyasında bulunur. Ham hedefte 755 sıfır süre vardır; sıfıra yakın değerler de bulunduğundan MAPE bilinçli olarak boş bırakılır. Log denemesinde bile bütün hata metrikleri geri dönüştürülmüş orijinal saniye birimindedir.

## Gerçek kümeleme sonuçları

| k | Eğitim inertia | Eğitim silhouette |
|---:|---:|---:|
| 2 | 102784,77 | 0,2889 |
| **3** | **87857,67** | **0,3076** |
| 4 | 79708,96 | 0,2571 |
| 5 | 73700,57 | 0,2343 |
| 6 | 70442,81 | 0,1969 |

**k=3 seçildi:** en yüksek silhouette. Dirsek için normalize inertia eğrisinin uçlar arasındaki kirişe uzaklığı k=4'te en yüksek; ancak k=3 ve k=4 uzaklıkları birbirine yakın (0,2115 / 0,2135). k=3'ün daha yüksek silhouette'ı ve daha sade profilleri seçim gerekçesidir. Silhouette aynı seed=42 ile 1.500 eğitim satırı üzerinde hesaplanır; tüm veri için kesin skor olarak sunulmaz. K-Means n_init=10, max_iter=300 ve k-means++ kullanır.

| Küme | Eğitim / test gözlem | Eğitim ürün sayfa sayısı ort./medyan | Eğitim BounceRates ort. | Yorum adayı |
|---|---:|---:|---:|---|
| 0 | 7673 / 1909 | 24,94 / 17 | 0,0091 | Orta etkileşimli çoğunluk |
| 1 | 735 / 197 | 3,33 / 1 | 0,1625 | Düşük etkileşim, yüksek çıkış oranı |
| 2 | 1356 / 335 | 88,21 / 61 | 0,0066 | Yoğun sayfa inceleme |

Bu isimler profil sonuçlarından sonra oluşturulmuş yorumlardır. Küme numaraları satın alma sınıfı değildir. Sayısal ortalama/medyanlar `cluster_numeric_profiles.csv`, kategorik dağılımlar `cluster_categorical_profiles.csv` içinde, eğitim/test ayrı olarak kaydedilir. PCA'nın iki bileşeni yaklaşık %37,24 varyansı açıklar; görsel tüm boyutlardaki küme kalitesini temsil etmez. Kümeleme için baseline veya gözetimli CV üretilmez.

## Dosyalar ve yeniden üretim

```text
ml_coursework/        config, data, eda, preprocessing, classification,
                     regression, clustering, evaluation, run_all
notebooks/           ML_Donem_Odevi.ipynb (çıktıları temiz)
tests/               veri, ön işleme, modeller ve çıktı denetimleri
scripts/             boyut denetimi, notebook oluşturma/yürütme, yeniden üretim
data/                otomatik indirilen CSV (Git dışında)
results/tables/      küçük CSV/JSON ölçümleri ve deney manifesti
results/figures/     rapor için 140 DPI PNG'ler (Git dışında)
results/tmp/         yerel notebook doğrulama çıktıları (Git dışında)
docs/                rapor iskeleti, deney tasarımı, savunma ve doğrulama notları
```

Notebook modülleri doğrudan çağırır; model kodlarının ikinci kopyası değildir. Colab ilk hücrede bu depoyu klonlar ve bağımlılıkları kurar. Yerel notebook doğrulaması:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/verify_reproducibility.py results results/tmp/notebook
```

Çıktılı doğrulama defteri yalnız `results/tmp/notebook/executed.ipynb` içine yazılır; kaynak notebook temiz kalır. `BMM4111_DATA_PATH` notebook'a çevrimdışı CSV, `BMM4111_OUTPUT_DIR` alternatif sonuç klasörü sağlar. Notebook'u Git'e göndermeden önce tüm hücre çıktıları temizlenmelidir. Kaynak defteri yeniden oluşturmak için `python scripts/build_notebook.py` kullanılabilir.

Yerel doğrulamada **35 pytest testi geçti**, tam CLI ve kaynak notebook yürütüldü; ayrıntılı kayıt [docs/validation.md](docs/validation.md) içindedir. Gerçek veri testi yalnız önbellek varsa çalışır; yoksa ağ gerektirmeden skip edilir. Diğer testlerin model eğitimleri küçük sentetik fixture'lardadır.

Veri ve model dosyaları depoyu büyütmemek, notebook çıktıları da büyük tekrarlı içerik göndermemek için Git'te tutulmaz. Ana akış modelleri bellekte kullanır; ağırlık dosyası üretmek gerekmez. Grafikler yeniden oluşturulur; yalnız küçük gerçek ölçüm tabloları depodadır. `check_repo_size.py` Git indeksini (stage edilenler dahil) denetler, 2 MB üstündeki dosyaları uyarır ve toplam 15 MB sınırı aşılırsa başarısız çıkar. Kontrol aşamasında yalnız seçilmiş kaynak/özet dosyaları stage edilmiştir.

## Rapor ve savunma

[El yazısı rapor iskeleti](docs/report_outline.md) gerçek tablo/şekil referansları ve kendi yorumunuzu geliştirecek sorular içerir. [Sözlü savunma notları](docs/oral_defense.md) 3–5 dakikalık bireysel savunmaya hazırlık sağlar.

Ödev PDF'si raporun öğrencinin **kendi cümleleriyle el yazısı** olmasını ister. Hazır rapor metni olarak kopyalamayın. Öğrenci bilgilerini tamamlayıp sayfaları numaralandırın, CamScanner ile tek PDF tarayın: `OgrenciNo_Ad_Soyad_ML_F26.pdf`. PDF Teams'e 20 Aralık 23:59'a kadar; kâğıt asılları ayrıca elden teslim edilir. GitHub bağlantısı raporun en sonunda bulunmalıdır. Bu fiziksel/akademik teslim adımları öğrencinin sorumluluğundadır.

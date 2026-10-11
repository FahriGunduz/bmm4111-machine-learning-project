# Veri sızıntısı ve deney tasarımı

## Ortak bölme

Ham veri 12.330 oturumdur. 125 tam kopya kayıt deterministik olarak, modelleme öncesinde bir kez ayıklanır; ham EDA bunları ayrıca raporlar. Kopya bir oturumun eğitim ve testte bulunması performansı şişirebileceğinden modelleme 12.205 farklı kayıt üzerinde yapılır. Aykırı gözlemler silinmez.

`shared_split` Revenue üzerinden stratified `train_test_split(test_size=0.20, random_state=42)` uygular. 9.764 eğitim ve 2.441 test satırı bütün görevler için aynıdır. `experiment_manifest.json` veri ve indeks SHA-256 özetlerini, sütun listesini ve yazılım sürümlerini kaydeder. Rastgele bölme, eldeki oturum dağılımına benzer yeni oturumları değerlendirir; zaman sırasına göre gelecekteki davranış değişimini ölçmez.

## Model girdileri

`FEATURE_COLUMNS` izin listesi 15 alan içerir. Revenue, ProductRelated_Duration ve PageValues bütün görevlerde çıkarılır. Yeni bir hedef kopyası veya hedef oranı veri çerçevesine eklenirse otomatik olarak girdiye alınmaz. Bu kural, bir görevin cevabını başka göreve taşıma riskini engeller. Ürün sayfası **sayısı** ProductRelated özgün bir girdidir; süre hedefinden hesaplanmış değildir.

UCI açıklamasında PageValues, işlem öncesinde ziyaret edilen sayfalara atanan ortalama parasal değerdir. Nasıl ve hangi zaman penceresiyle hesaplandığına ilişkin yeterli bilgi olmadığı için, aynı işlemin sonucunu veya gelecekteki işlem bilgisini taşıma ve tahmin anında bulunmama riski vardır. Bu veri sızıntısının kanıtlandığı anlamına gelmez; ana deneylerde ihtiyatla dışlanır. Kaynak: [UCI değişken açıklamaları](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

Diğer ziyaret sayıları ve oranlar da oturum özetleridir. Bu çalışma tamamlanmış oturumların tahmini için eğitim amaçlıdır. Canlı, oturum başında satın alma veya süre tahmini yapılacağı iddiası kurulmaz; bunun için zaman damgalı, tahmin anında mevcut özellikler gerekir.

## Dönüşümler ve CV

Sayısal alanlar median imputer, kategorikler most-frequent imputer ve one-hot encoder ile işlenir. Eksik değer olmaması, imputer'ın eksik veri bulunduğunu iddia ettiği anlamına gelmez; yeni veriye karşı koruyucu adımdır. OperatingSystems, Browser, Region ve TrafficType sayıyla kodlanmış nominal kategorilerdir. Sayısal sıralama anlamı yüklenmez.

Scaler istatistikleri testten öğrenilirse eğitim, değerlendirme dağılımına önceden erişir. Encoder tüm veride fit edilirse eğitimde görülmeyen kategoriler önceden öğrenilmiş olur. Bu nedenle ColumnTransformer, sklearn Pipeline içindedir. `cross_validate` her fold için bütün pipeline'ı klonlayarak yalnız o fold'un eğitim satırlarına fit eder. Sınıflandırma için StratifiedKFold, regresyon için KFold kullanılır; ikisi de 5 fold, shuffle=True, seed=42 ayarıyla **yalnız 9.764 eğitim satırına** uygulanır. Test hiçbir CV çağrısına girmez.

Logistic Regression ve ANN için StandardScaler uygulanır. Linear Regression'da sayısal koşullandırma için kullanılır; ağaç modellerinde gerekmez. Gözetimli modellerde tam doğrusal bağımlılığı önlemek için one-hot'ın ilk kategorisi düşürülür. Bilinmeyen kategoriler `handle_unknown="ignore"` ile sıfır vektörü olur. Bazı CV fold'larında nadir kategoriler nedeniyle çıkan uyarı beklenir; bu kategorileri tüm veri üzerinden önceden öğrenmek doğru çözüm değildir.

ANN early stopping, ilgili CV eğitim fold'u içinden %15 iç doğrulama ayırır. sklearn MLPClassifier bu iç doğrulamada accuracy kullanır; birincil **dış CV model seçimi** ise F1'dir. Ön işlemci CV eğitim fold'una fit edildiğinden ANN'in iç doğrulama satırları bu istatistiklere katılır; dış CV validasyon ve nihai test satırları katılmaz. İç early-stopping skorunu bağımsız, tarafsız bir değerlendirme sonucu olarak raporlamıyoruz. ANN'in düşük CV recall'ı ve dengesiz hedefte accuracy temelli durdurmanın sınırlılığı açıkça tartışılır.

## Seçim ve testin rolü

Sınıflandırmada iki ANN mimarisinden en yüksek CV F1'e sahip olan, regresyonda Random Forest adaylarından en düşük CV RMSE'ye sahip olan seçilir. Ardından baseline ve iki algoritmanın seçilmiş ayarları dondurulur. Test değerleri yalnız bu üç nihai model için hesaplanır; denenip elenen ANN/RF ayarlarına test uygulanmaz. Genel model seçimi de CV üzerinden yapılır. Testte farklı bir sıralama görülmesi seçimi geriye dönük değiştirmez.

CV standart sapması fold değerleri üzerinde `ddof=1` ile hesaplanır; bir güven aralığı veya anlamlılık testi değildir. CV RMSE ortalaması, **fold RMSE değerlerinin ortalamasıdır**; CV MSE ortalamasının kareköküne eşit olması gerekmez. Test MSE/RMSE için kare ilişkisi geçerlidir.

Eğitim hedefinin çarpıklığı 1'den büyükse log1p Random Forest adayı denenir. TransformedTargetRegressor her fold'da hedef dönüşümüyle eğitir, tahmini expm1 ile saniyeye döndürür. Bütün seçim/sonuç metrikleri orijinal birimdedir. Sıfır sürelerde MAPE bölme sorunu oluşturduğu için boş bırakılır; sıfırları sessizce atıp farklı bir ölçüm raporlanmaz.

## Kümeleme ve görselleştirme

K-Means, scaler ve encoder yalnız eğitim özelliklerine fit edilir. Kümelemede bütün one-hot kategorileri korunur; böylece bir nominal referans kategorisi diğerlerinden farklı mesafeye sahip olmaz. Sayısal alanlar standartlaştırılarak büyük süre/sayım ölçeklerinin Öklid uzaklığını tek başına belirlemesi önlenir. One-hot alanlarla sayısal alanların göreli ağırlıkları ve uç değerler küme sonucunu etkileyebilir; silhouette=0,308 kusursuz ayrışma değildir.

k=2–6 için silhouette aynı seed ile aynı 1.500 eğitim gözlemi üzerinde hesaplanır. Bu, tüm 9.764 eğitim gözleminin tam silhouette'ı değil, bellek maliyeti sınırlı bir kestirimdir. Birincil seçim maksimum silhouette'tır; inertia dirseği yardımcı tanıdır. Teste aynı merkezler kullanılarak atama yapılır, test yeni k seçiminde kullanılmaz. Baseline veya gözetimli CV uydurulmaz.

PCA eğitim dönüşümlerinde fit edilir. İki bileşen yaklaşık %37,24 varyansı açıklar; kümelerin tüm boyutlardaki kalitesini bu resimden çıkaramayız. PCA kümeleme girdisi veya seçim ölçütü değildir.

## Yeniden üretim ve denetim

Ana bağımlılıklar sabitlenmiştir; seed=42, CV n_jobs=1 ve BLAS iş parçacığı sınırı 1'dir. RF n_jobs=1 ve K-Means n_init=10 kullanılır. Farklı BLAS, işletim sistemi veya Python sürümleri çok küçük sayısal farklar üretebilir. Aynı yerel ortamda CLI ve notebook tabloları `scripts/verify_reproducibility.py` ile karşılaştırılır.

Pytest testleri hedef türevlerini dışlama, ortak/deterministik bölme, train/test ayrıklığı, stratification, kopya ayrıklığı, fold bazında scaler istatistikleri, görülmemiş kategoriler, tahmin boyutları, metrik aralıkları, gerçek önbellek şeması, seed ile tahmin tekrarı, notebook temizliği ve Git dosya sınırlarını denetler.

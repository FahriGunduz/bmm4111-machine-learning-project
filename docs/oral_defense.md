# 3–5 dakikalık sözlü savunma hazırlığı

Notları ezberlemek yerine kodda ve tablolarda ilgili yeri açıp kendi örneğinizle açıklayın.

## Kısa akış

1. **0:00–0:40:** Oturum verisi, Revenue sınıfı, ürün süresi ve etiketsiz ziyaretçi grupları. Ham 12.330 satır, 125 kopya, modellemede 12.205 satır.
2. **0:40–1:30:** Ortak stratified split, eğitimden fit edilen pipeline, iki hedef ve PageValues'ın dışlanması, seed=42.
3. **1:30–2:20:** Sınıflandırma baseline'ının yüksek accuracy/düşük F1 sonucu; LR CV F1=0,390; iki ANN mimarisi ve düşük recall.
4. **2:20–3:10:** CV RMSE ile RF seçimi, test RMSE yaklaşık 865 saniye; Linear Regression yaklaşık 858 saniye. RF train/test farkı. MAPE neden yok?
5. **3:10–4:10:** k=3, silhouette≈0,308, üç davranış profili; PCA'nın sınırlılığı.
6. **4:10–5:00:** Testler, CLI/notebook yeniden üretimi, mevcut sonuçların sınırları ve kendi çıkarımınız.

## Sorular ve kısa hatırlatmalar

| Soru | Açıklamanızda bulunması gereken fikir |
|---|---|
| Model neden seçildi? | Sınıflandırmada eğitim CV F1, regresyonda eğitim CV RMSE; test sıralaması seçim ölçütü değil. |
| ANN nedir? | Ağırlıklı bağlantılar, gizli katmanlarda doğrusal olmayan aktivasyon ve kayıp üzerinden ağırlık öğrenimi; burada MLP, (32,) / (64,32), ReLU ve Adam. |
| ANN neden zayıf? | CV azınlık recall'ı düşük. Dengesizlik ve accuracy kullanan early stopping çoğunluğu ödüllendirebilir; bu tek nedene ilişkin deneysel kanıt iddiası değil. Teste bakıp ayar değiştirilmedi. |
| Baseline niçin gerekli? | Özelliklerden öğrenmeden elde edilen başlangıç düzeyi; çoğunluk sınıfı veya eğitim hedef ortalaması. |
| F1 ile accuracy farkı? | Accuracy bütün doğru tahminlerin oranı; F1 pozitif precision/recall'ın harmonik ortalaması. Dummy %84 accuracy ile bütün satın almaları kaçırır. |
| Precision ve recall? | Precision: satın alır denilenlerin kaçı gerçekten alıyor? Recall: gerçekten alanların kaçı bulunuyor? LR recall≈%77,23; precision≈%27,39. |
| Regresyonda R²? | Ortalama tahmin baseline'ına göre açıklanan değişkenlik; 1 ideal, 0 ortalama düzeyi, negatif baseline'dan kötü olabilir. |
| RMSE neyi ölçer? | Kare hata ortalamasının karekökü; saniye biriminde, büyük hatalara daha duyarlı. MSE saniye², MAE saniye. |
| MAPE niçin yok? | Sıfır/sıfıra yakın sürelerde payda çok küçük; yüzde hata anlamsız şekilde büyür. |
| log1p neden denendi? | Eğitim hedefi çarpık olduğu için. RF-log adayı orijinal birime çevrilerek karşılaştırıldı; MAE iyileşse de RMSE seçimi kazanmadı. |
| K-Means nasıl çalışır? | Noktaları en yakın merkeze atar, merkezleri atanmış noktaların ortalamasıyla günceller; toplam kare uzaklığı azaltır. |
| Neden ölçekleme? | Uzaklıkta büyük sayısal ölçeklerin baskınlığını azaltır. Nominal kodlar one-hot'tır; sıralı sayılar gibi kullanılmaz. |
| k neden 3? | En yüksek eğitim silhouette. Dirsek 4'e işaret etse de 3/4 kiriş sapmaları yakın; k=3 silhouette daha yüksek ve yorum daha sade. |
| Silhouette neyi ölçer? | Aynı küme içindeki yakınlıkla en yakın diğer kümeye uzaklığın göreli farkı; −1…1. Burada sabit 1.500 eğitim örneği kullanılıyor. |
| PCA ne için? | Eğitimde öğrenilen iki boyutlu gösterim; yaklaşık %37,24 varyans, küme modeli veya kalite ölçütü değil. |
| Sızıntı nasıl önlendi? | Hedef izin listesi, PageValues dışlama, kopyaları ayıklama, ortak ayrık indeksler, fold içinde öğrenilen dönüşümler. |
| Test niçin model seçmez? | Çok sayıda test sonucuna göre ayar seçmek testi geliştirme verisine dönüştürür; tarafsız yeni veri değerlendirmesi kaybolur. |
| Aşırı uyum nasıl incelenir? | Eğitim ve test arasındaki fark, mutlak performans ve CV değişkenliği birlikte. RF train RMSE≈527/test≈865; LR≈935/858. |
| Seed ne sağlar? | Aynı bölme ve stokastik başlangıçları tekrar etmeyi sağlar; farklı dağılımlara genellemeyi garanti etmez. |
| Canlı siteye hazır mı? | Bu tamamlanmış oturum verisiyle ders deneyi. Canlı tahmin için zaman damgaları ve tahmin anındaki erişilebilirlik denetlenmeli. |

## Kodda gösterebileceğiniz yerler

- `data.py`: şema, kopya ayıklama ve ortak bölme.
- `preprocessing.py`: izin listesi, ColumnTransformer ve Pipeline.
- `classification.py` / `regression.py`: seçimden önce CV, seçimden sonra test.
- `clustering.py`: eğitim fit, test predict ve eğitim PCA.
- `tests/test_models.py`: her fold'un scaler istatistiklerinin kendi eğitim satırlarından geldiğinin denetimi.
- `results/tables/`: yorumlarınızın gerçek ölçüm dayanağı.

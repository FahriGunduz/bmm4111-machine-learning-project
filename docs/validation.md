# Gerçek çalıştırma ve kalite denetimi kaydı

Bu kayıt Windows üzerinde yeni oluşturulan Python 3.11.9 `.venv` ortamındaki çalıştırmalara dayanır. Ana bağımlılık sürümleri `results/tables/experiment_manifest.json` ve `requirements.txt` içinde bulunur. Google Colab hizmetinde canlı bir oturum açılmadı; Colab giriş hücresi hazırlanmış notebook, yerel gerçek Jupyter kernel'inde uçtan uca yürütüldü.

## Deneyler

1. `python -m ml_coursework.run_all --stage eda`: UCI verisi gerçekten indirildi. Şema 12.330 × 18; 0 eksik, 125 tam kopya. EDA tabloları ve 5 kalite/dağılım grafiği oluşturuldu.
2. `--stage classification`: Dummy, Logistic Regression ve iki ANN mimarisi için 5 katlı eğitim CV'si çalıştı. CV ile seçilen ANN ve LR/baseline için nihai eğitim/test değerlendirmesi yapıldı.
3. `--stage regression`: Dummy, Linear Regression, RF leaf=2/5 ve log1p RF leaf=2 için 5 katlı eğitim CV'si çalıştı. Seçilmiş RF ve LR/baseline için nihai değerlendirme yapıldı.
4. `--stage clustering`: k=2–6 K-Means modelleri yalnız eğitimde fit edildi; inertia, 1.500 eğitim örneğindeki silhouette, test atamaları ve profiller üretildi. PCA eğitimde öğrenildi.
5. `python -m ml_coursework.run_all`: Bütün aşamalar tek CLI çağrısıyla tekrar başarıyla çalıştı.
6. `python scripts/validate_notebook.py`: Gerçek Jupyter kernel'inde **7 kod hücresi** başarıyla yürütüldü; defter ayrıca test komutunu çalıştırdı. Çıktılı notebook yalnız Git dışında tutulan `results/tmp/notebook/executed.ipynb` dosyasına yazıldı.
7. `python scripts/verify_reproducibility.py results results/tmp/notebook`: **18 CSV/JSON dosyası eşleşti.** 13 CSV tablo ve 5 JSON. Sayısal CSV karşılaştırması rtol=1e-10 / atol=1e-8; JSON eşitliği tamdır.

Ana sonuçlar: LR CV F1≈0,3899 ve test F1≈0,4044; RF CV RMSE≈947,34 saniye ve test RMSE≈865,29 saniye. k=3 eğitim silhouette≈0,3076. Test sonuçları yeniden ayar seçmek için kullanılmadı.

## Testler ve çıktılar

- `python -m pytest -q`: **35 test geçti**. Gerçek veri önbellekte olduğundan gerçek şema testi de çalıştı; skip yok. Veri, hedef türevleri, ortak bölme, scaler/encoder eğitim ve CV fold istatistikleri, model tahminleri, sabit seed, kümeleme, tüm modül importları, metrik/ölçüm tutarlılığı, notebook temizliği ve Git dosyaları denetlendi. Git boyut aracı ayrıca geçici bir depoda stage edilen blob ile farklı çalışma kopyası kullanılarak sınandı.
- Notebook'un önceki tam yürütmesinde, son boyut testi eklenmeden önceki **34 test** ayrıca geçti. Son eklenen Git boyut testi modeli veya notebook deney akışını değiştirmedi.
- `python -m pip check`: **No broken requirements found.**
- Kaynak notebook: **0 hücre çıktısı**, execution_count alanları boş.
- Ana sonuç klasörü: **21 PNG** (140 DPI), **13 CSV**, **5 JSON**. Grafikler ve veri Git dışında; modeller bellekte kaldı.
- Aynı seed ile küçük sentetik fixture üzerinde bütün gözetimli adayların ve K-Means'ın tahminleri tekrarlanarak eşleşti.
- Depo boyutu aracı Git indeksindeki dosyaları, yeni stage edilenler dahil denetler. Toplam 15 MB altında; hiçbir takip edilen dosya 2 MB üzerinde değil. Kesin nihai boyut son commit kontrolünde terminalde raporlanır.

Nadir kategorilerin bir CV fold'unda görülmemesi nedeniyle encoder uyarıları çıktı; `handle_unknown="ignore"` beklenen davranışıyla çalıştı. Model başarısını iyileştirmek için test kümesine göre ayar değiştirilmedi. ANN'in düşük recall'ı ve RF'nin eğitim/test farkı gizlenmedi.

## Kalan öğrenci teslim işleri

Kod ve ölçümler hazırlanmıştır. Öğrenci adı/numarası doldurulması, kendi ifadeleriyle el yazısı rapor, CamScanner taraması, Teams yüklemesi ve kâğıt asıllarının teslimi otomatik gerçekleştirilmedi. Ödev PDF'sinin gerektirdiği bu işler öğrenci tarafından tamamlanacaktır.

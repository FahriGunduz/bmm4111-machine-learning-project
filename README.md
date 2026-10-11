# BMM4111 – Makine Öğrenmesi Dönem Ödevi

Balıkesir Üniversitesi Bilgisayar Mühendisliği BMM4111 dersi için tek veri kümesiyle sınıflandırma, regresyon ve kümeleme çalışması.

Veri: [UCI Online Shoppers Purchasing Intention, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), 12.330 oturum ve 17 girdi. Lisans: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Atıf: Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.

## Kurulum

Python 3.11 kullanın. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kaynak kod `ml_coursework/`, deney tabloları `results/tables/`, yerel grafikler `results/figures/` altında bulunur. Veri, grafikler, modeller ve notebook çıktıları Git'e eklenmez. El yazısı rapor için çalışma notları `docs/` altında hazırlanacaktır.

Geliştirme aşamaları: iskelet → veri/ön işleme → sınıflandırma → regresyon → kümeleme → deney denetimi → dokümantasyon. Sonuçlar yalnız çalıştırılan deneylerden doldurulacaktır.

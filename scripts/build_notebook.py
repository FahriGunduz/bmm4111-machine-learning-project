"""Modülleri çağıran, çıktıları temiz Colab defterini oluştur."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# BMM4111 – Bir Veri Kümesi, Üç Problem\n\n"
       "Balıkesir Üniversitesi Makine Öğrenmesi dönem ödevi. Bu defter Python modüllerindeki aynı deneyleri sırayla çalıştırır. "
       "Gerçek sonuçları yorumlayın; el yazısı raporu kendi cümlelerinizle hazırlayın. "
       "Veri: [UCI 468](https://doi.org/10.24432/C5F88Q), CC BY 4.0.\n\n"
       "Colab'de **Çalışma zamanı → Tümünü çalıştır** seçilebilir. Standart CPU yeterlidir. "
       "İlk çalıştırma GitHub ve UCI için internet gerektirir. Yerelde proje kökünden veya notebooks klasöründen açılabilir."),
    md("## 1. Proje ve bağımlılıklar\n\nPython 3.11 veya uyumlu sürüm kullanılır. Colab'de kaynak kod GitHub'dan alınır; yerelde mevcut modüller kullanılır."),
    code('''from pathlib import Path
import os
import subprocess
import sys

repo_url = "https://github.com/FahriGunduz/bmm4111-machine-learning-project.git"
workspace = next((p for p in [Path.cwd(), Path.cwd().parent] if (p / "ml_coursework").is_dir()), None)
if workspace is None:
    workspace = Path("/content/bmm4111-machine-learning-project") if Path("/content").exists() else Path.cwd() / "bmm4111-machine-learning-project"
    if not workspace.exists():
        subprocess.run(["git", "clone", repo_url, str(workspace)], check=True)
    if not (workspace / "ml_coursework").is_dir():
        raise RuntimeError("Proje kaynakları bulunamadı.")
os.chdir(workspace)
sys.path.insert(0, str(workspace))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Python:", sys.version.split()[0], "Proje:", workspace)'''),
    md("## 2. Veri yükleme ve ortak bölme\n\nİlk çalıştırmada ucimlrepo ile UCI verisi indirilir; gerektiğinde resmî ZIP kullanılır. "
       "Sonraki çalıştırmalarda yerel CSV okunur. Tam kopyalar modelleme öncesinde ayıklanır. "
       "Revenue ile stratified %80/%20 bölme bir kez yapılır; üç görev aynı indeksleri kullanır. "
       "Revenue, ProductRelated_Duration ve PageValues hiçbir modele girdi olmaz."),
    code('''from IPython.display import display, Image
import pandas as pd
from threadpoolctl import threadpool_limits
from ml_coursework.run_all import prepare_experiment
from ml_coursework.eda import run_eda
from ml_coursework.classification import run_classification
from ml_coursework.regression import run_regression
from ml_coursework.clustering import run_clustering

output_dir = Path(os.environ.get("BMM4111_OUTPUT_DIR", str(workspace / "results")))
data_path = os.environ.get("BMM4111_DATA_PATH")
raw, frame, train_indices, test_indices = prepare_experiment(data_path, output_dir)
print("Ham veri:", raw.shape, "Modelleme:", frame.shape)
print("Eğitim:", len(train_indices), "Test:", len(test_indices))
display(raw.head(3))'''),
    md("## 3. Veri tanıma ve kalite\n\nHam verinin boyutları, türleri, eksikleri, tekrarları, IQR aykırıları ve dağılımları incelenir. "
       "Bu keşif model seçimi değildir. Aykırı değerler topluca silinmez; çarpıklık ve sıfır IQR nedeniyle işaretlenen değerleri yorumlayın."),
    code('''with threadpool_limits(limits=1):
    summary = run_eda(raw, output_dir)
display(raw.dtypes.to_frame("Tür"))
display(pd.read_csv(output_dir / "tables/missing_values.csv"))
print("Eksik toplamı:", sum(summary["missing_counts"].values()), "Kopya:", summary["duplicate_rows"])
display(Image(filename=str(output_dir / "figures/class_distribution.png")))
display(Image(filename=str(output_dir / "figures/correlation.png")))'''),
    md("## 4. Sınıflandırma\n\nDummy, Logistic Regression ve ANN (32 / 64-32). Yalnız eğitimde 5 katlı StratifiedKFold ve F1 ile seçim yapılır. "
       "Ön işleme her CV fold'unda yeniden öğrenilir. Test sonuçları ayar seçmek için kullanılmaz. "
       "ANN early stopping iç ölçütü accuracy'dir; dengesiz hedefte bu ayarın azınlık recall'ına etkisini CV tablosundan tartışın."),
    code('''with threadpool_limits(limits=1):
    classification_results, classification_models = run_classification(frame, train_indices, test_indices, output_dir)
display(classification_results.round(4))
display(pd.read_csv(output_dir / "tables/ann_architecture_results.csv").round(4))
selected_classifier = classification_results.loc[classification_results.selected_by_cv, "model"].item()
display(Image(filename=str(output_dir / f"figures/confusion_matrix_{selected_classifier}.png")))
display(Image(filename=str(output_dir / "figures/ann_architectures.png")))'''),
    md("## 5. Regresyon\n\nHedef ürün sayfası toplam süresidir. Dummy, Linear Regression ve Random Forest karşılaştırılır. "
       "5 katlı KFold'da birincil ölçüt orijinal saniye biriminde RMSE'dir. "
       "Eğitim hedefi çarpıksa log1p Random Forest adayı da denenir; tahminler expm1 ile geri dönüştürülür. "
       "Sıfır süreler nedeniyle MAPE raporlanmaz. CV ve testte farklı sıralamaların mümkün olduğunu gözlemleyin."),
    code('''with threadpool_limits(limits=1):
    regression_results, regression_models = run_regression(frame, train_indices, test_indices, output_dir)
display(regression_results.round(4))
display(pd.read_csv(output_dir / "tables/regression_candidates.csv").round(4))
selected_regressor = regression_results.loc[regression_results.selected_by_cv, "model"].item()
display(Image(filename=str(output_dir / f"figures/actual_predicted_{selected_regressor}.png")))
display(Image(filename=str(output_dir / "figures/regression_comparison.png")))'''),
    md("## 6. Kümeleme\n\nK-Means yalnız eğitim özelliklerine fit edilir. k=2…6 adayları için inertia ve aynı 1.500 eğitim gözlemi üzerinde silhouette hesaplanır. "
       "Birincil seçim maksimum silhouette'dir; dirsek eğrisi yardımcı tanıdır. Teste yalnız mevcut merkezlerle atama yapılır. "
       "PCA yalnız eğitimde öğrenilir ve görselleştirmedir; küme kalitesini bütünüyle temsil etmez. "
       "Küme numaraları sınıf etiketi değildir; isimleri profil tablosundaki davranışlardan çıkarın."),
    code('''with threadpool_limits(limits=1):
    clustering_results, clustering_model = run_clustering(frame, train_indices, test_indices, output_dir)
display(clustering_results.round(4))
display(pd.read_csv(output_dir / "tables/cluster_sizes.csv"))
display(pd.read_csv(output_dir / "tables/cluster_numeric_profiles.csv").round(3))
for name in ["elbow.png", "silhouette.png", "cluster_sizes.png", "cluster_pca.png"]:
    display(Image(filename=str(output_dir / "figures" / name)))'''),
    md("## 7. Denetim ve rapor hazırlığı\n\n"
       "- CV yalnız eğitim indeksleri üzerinde mi? Scaler ve encoder fold içinde mi?\n"
       "- Hedefler veya PageValues modele girdi mi?\n"
       "- Yüksek accuracy neden iyi satın alma tahmini anlamına gelmeyebilir?\n"
       "- Eğitim/test farkı RF ve ANN için ne söylüyor?\n"
       "- Kümelerin yorumu hangi gerçek profil değerlerine dayanıyor?\n\n"
       "Rapor iskeleti `docs/report_outline.md`, savunma notları `docs/oral_defense.md`, sızıntı denetimi `docs/experiment_design.md` içindedir. "
       "Git'e göndermeden önce defterin tüm hücre çıktılarını temizleyin."),
    code('''subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
print("Tablolar:", output_dir / "tables")
print("Yerel grafikler:", output_dir / "figures")'''),
]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
nbf.validate(nb)
nbf.write(nb, ROOT / "notebooks/ML_Donem_Odevi.ipynb")
print("Created notebooks/ML_Donem_Odevi.ipynb (outputs empty)")

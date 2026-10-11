"""Hedefsiz eğitim özellikleriyle K-Means; test yalnız atama/profil için."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import CATEGORICAL_COLUMNS, RANDOM_STATE, output_dirs
from .evaluation import save_figure, save_json
from .preprocessing import build_pipeline, task_features


def run_clustering(frame, train_indices, test_indices, output_dir: Path):
    tables, figures = output_dirs(output_dir)
    features = task_features(frame, "clustering")
    x_train, x_test = features.loc[train_indices], features.loc[test_indices]
    rows, fitted = [], {}
    # Bütün k adayları için aynı deterministik eğitim alt örneği.
    sample_size = min(1500, len(x_train))
    for k in range(2, 7):
        print(f"Clustering: k={k}", flush=True)
        pipeline = build_pipeline(KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE), drop_first=False)
        pipeline.fit(x_train)
        transformed = pipeline.named_steps["preprocessor"].transform(x_train)
        labels = pipeline.named_steps["model"].labels_
        score = silhouette_score(transformed, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": pipeline.named_steps["model"].inertia_,
                     "silhouette": score, "silhouette_sample_size": sample_size})
        fitted[k] = pipeline
    results = pd.DataFrame(rows)
    chosen_k = int(results.loc[results.silhouette.idxmax(), "k"])
    # Normalize dirsek eğrisini; uç noktalar arasındaki kirişten en büyük sapma.
    x = (results.k - results.k.min()) / (results.k.max() - results.k.min())
    y = (results.inertia - results.inertia.min()) / (results.inertia.max() - results.inertia.min())
    results["elbow_chord_distance"] = (1 - x) - y
    elbow_k = int(results.loc[results.elbow_chord_distance.idxmax(), "k"])
    results["selected"] = results.k.eq(chosen_k)
    results.to_csv(tables / "clustering_results.csv", index=False)
    pipeline = fitted[chosen_k]
    train_labels = pipeline.named_steps["model"].labels_
    test_labels = pipeline.predict(x_test)
    counts = pd.DataFrame({"cluster": np.arange(chosen_k),
                           "train_count": np.bincount(train_labels, minlength=chosen_k),
                           "test_count": np.bincount(test_labels, minlength=chosen_k)})
    counts.to_csv(tables / "cluster_sizes.csv", index=False)
    numeric = [c for c in features if c not in CATEGORICAL_COLUMNS]
    numeric_rows, categorical_rows = [], []
    for split, data, labels in [("train", x_train, train_labels), ("test", x_test, test_labels)]:
        grouped = data.assign(cluster=labels).groupby("cluster")
        profile = grouped[numeric].agg(["mean", "median"])
        profile.columns = [f"{column}_{stat}" for column, stat in profile.columns]
        profile["split"] = split
        numeric_rows.append(profile.reset_index())
        for cluster, subset in grouped:
            for column in CATEGORICAL_COLUMNS:
                for value, count in subset[column].value_counts().items():
                    categorical_rows.append({"split": split, "cluster": int(cluster), "feature": column,
                                             "value": str(value), "count": int(count), "ratio": count / len(subset)})
    pd.concat(numeric_rows, ignore_index=True).to_csv(tables / "cluster_numeric_profiles.csv", index=False)
    pd.DataFrame(categorical_rows).to_csv(tables / "cluster_categorical_profiles.csv", index=False)
    for column, ylabel, filename in [("inertia", "Eğitim inertia", "elbow.png"),
                                     ("silhouette", "Eğitim silhouette (sabit örneklem)", "silhouette.png")]:
        plt.figure(figsize=(6, 4))
        plt.plot(results.k, results[column], "o-")
        plt.axvline(chosen_k, color="red", linestyle="--", label=f"Seçilen k={chosen_k}")
        plt.xlabel("Küme sayısı")
        plt.ylabel(ylabel)
        plt.xticks(results.k)
        plt.legend()
        save_figure(figures / filename)
    counts.set_index("cluster")[["train_count", "test_count"]].plot.bar(figsize=(6, 4))
    plt.ylabel("Oturum sayısı")
    plt.xticks(rotation=0)
    save_figure(figures / "cluster_sizes.png")
    preprocessor = pipeline.named_steps["preprocessor"]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(preprocessor.transform(x_train))
    test_pca = pca.transform(preprocessor.transform(x_test))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, coordinates, labels, title in [(axes[0], train_pca, train_labels, "Eğitim"),
                                           (axes[1], test_pca, test_labels, "Test atamaları")]:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=8, alpha=.4)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    save_figure(figures / "cluster_pca.png")
    save_json({"selected_k": chosen_k, "elbow_chord_k": elbow_k,
               "selection_rule": "Maksimum eğitim silhouette; dirsek eğrisi yardımcı tanı olarak raporlanır.",
               "silhouette_sample_size": sample_size, "silhouette_seed": RANDOM_STATE,
               "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
               "pca_note": "PCA yalnız görselleştirme içindir; iki boyut bütün küme kalitesini göstermez.",
               "categorical_distance_note": "Kümelemede tüm one-hot kategorileri korunur; nominal kodlara sıralı uzaklık atanmaz.",
               "parameters": pipeline.named_steps["model"].get_params()}, tables / "clustering_selection.json")
    print(results.to_string(index=False), flush=True)
    return results, pipeline

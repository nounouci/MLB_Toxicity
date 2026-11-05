# -*- coding: utf-8 -*-
"""Classification de l'inhibition de BSEP avec un SVM et des fingerprints ECFP4.

Depuis la racine du dépôt :
    python -m pip install -r requirements.txt
    python src/svm_bsep.py

Les figures sont enregistrées dans figures/ et les résultats dans outputs/.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
)
import joblib

GRAINE = 42
DOSSIER_FIGURES = Path("figures")
DOSSIER_RESULTATS = Path("outputs")
DOSSIER_FIGURES.mkdir(exist_ok=True)
DOSSIER_RESULTATS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight"
})
print("Environnement prêt.")

URL_CSV = (
    "https://raw.githubusercontent.com/ATOMScience-org/AMPL/master/"
    "atomsci/ddm/examples/BSEP/data/morgan_warner_combined_bsep_data.csv"
)
FICHIER_CSV = Path("data/morgan_warner_combined_bsep_data.csv")

if FICHIER_CSV.exists():
    df = pd.read_csv(FICHIER_CSV)
else:
    df = pd.read_csv(URL_CSV)

print("Dimensions du fichier :", df.shape)
print("\nCinq premières lignes :")
print(df.head().to_string(index=False))
print("\nValeurs manquantes :")
print(df.isna().sum())
print("\nRépartition des classes :")
print(df["active"].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
n_classes = df["active"].value_counts().reindex([0, 1], fill_value=0)
axes[0].bar(["Non-inhibiteurs (0)", "Inhibiteurs (1)"], n_classes,
            color=["#6A9FBA", "#D6A161"], width=0.6)
axes[0].set_title("Répartition des classes")
axes[0].set_ylabel("Nombre de molécules")
for i, nombre in enumerate(n_classes):
    axes[0].text(i, nombre + 4, str(nombre), ha="center", fontweight="bold")

source_table = pd.crosstab(df["source"], df["active"])
source_table = source_table.reindex(columns=[0, 1], fill_value=0)
source_table.plot(kind="bar", stacked=True, ax=axes[1],
                  color=["#6A9FBA", "#D6A161"], rot=0)
axes[1].set_title("Classes par étude d'origine")
axes[1].set_xlabel("Source")
axes[1].set_ylabel("Nombre de molécules")
axes[1].legend(["Non-inhibiteur", "Inhibiteur"])
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "01_classes_et_sources.png", dpi=180)
plt.show()
plt.close(fig)

donnees = df.dropna(subset=["base_rdkit_smiles", "active"]).copy()
donnees["active"] = pd.to_numeric(donnees["active"], errors="coerce")
donnees = donnees[donnees["active"].isin([0, 1])].copy()
donnees["active"] = donnees["active"].astype(int)

def smiles_canonique(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

donnees["canonical_smiles"] = donnees["base_rdkit_smiles"].apply(smiles_canonique)
n_invalides = donnees["canonical_smiles"].isna().sum()
donnees = donnees.dropna(subset=["canonical_smiles"]).copy()

n_labels = donnees.groupby("canonical_smiles")["active"].nunique()
ambigus = set(n_labels[n_labels > 1].index)
donnees = donnees[~donnees["canonical_smiles"].isin(ambigus)]
n_doublons = donnees.duplicated("canonical_smiles").sum()
donnees = donnees.drop_duplicates("canonical_smiles")
donnees = donnees.sort_values("canonical_smiles").reset_index(drop=True)

print("Molécules conservées :", len(donnees))
print("SMILES invalides :", n_invalides)
print("Structures avec labels contradictoires :", len(ambigus))
print("Doublons retirés :", n_doublons)
assert donnees["active"].nunique() == 2

exemples = donnees.sample(6, random_state=GRAINE)
molecules = [Chem.MolFromSmiles(s) for s in exemples["canonical_smiles"]]
legendes = [
    f"{nom[:22]}  |  active={classe}"
    for nom, classe in zip(exemples["compound_name"], exemples["active"])
]
image_molecules = Draw.MolsToGridImage(
    molecules, molsPerRow=3, subImgSize=(290, 210), legends=legendes
)
image_molecules.save(DOSSIER_FIGURES / "02_exemples_molecules.png")
# Dans un notebook, cette dernière expression affiche la grille.
image_molecules

donnees["MolWt"] = [
    Descriptors.MolWt(Chem.MolFromSmiles(s))
    for s in donnees["canonical_smiles"]
]
donnees["LogP"] = [
    Descriptors.MolLogP(Chem.MolFromSmiles(s))
    for s in donnees["canonical_smiles"]
]
donnees["TPSA"] = [
    Descriptors.TPSA(Chem.MolFromSmiles(s))
    for s in donnees["canonical_smiles"]
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, colonne, titre in zip(
    axes,
    ["MolWt", "LogP", "TPSA"],
    ["Masse moléculaire", "Lipophilie (LogP)", "Surface polaire (TPSA)"]
):
    groupes = [
        donnees.loc[donnees["active"] == c, colonne].values
        for c in [0, 1]
    ]
    bp = ax.boxplot(groupes, patch_artist=True, showfliers=False)
    for boite, couleur in zip(bp["boxes"], ["#6A9FBA", "#D6A161"]):
        boite.set_facecolor(couleur)
    ax.set_xticks([1, 2], ["Non-inhib.", "Inhib."])
    ax.set_title(titre)
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("Comparaison descriptive des propriétés moléculaires", fontsize=14)
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "03_proprietes_chimiques.png", dpi=180)
plt.show()
plt.close(fig)

generateur = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=1024
)

X = np.array([
    generateur.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    for s in donnees["canonical_smiles"]
], dtype=np.uint8)
y = donnees["active"].to_numpy()

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Vingt premiers bits :", X[0, :20])
assert X.shape == (len(donnees), 1024)

fig, ax = plt.subplots(figsize=(11, 4))
ax.imshow(X[:24, :90], aspect="auto", cmap="Blues", interpolation="nearest")
ax.set_title("Aperçu des fingerprints : 24 molécules, 90 premiers bits")
ax.set_xlabel("Position du bit")
ax.set_ylabel("Molécule")
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "04_fingerprints.png", dpi=180)
plt.show()
plt.close(fig)

fichier_partage = Path("data/team_split.csv")

if fichier_partage.exists():
    partage = pd.read_csv(fichier_partage)
    etiquettes = partage.set_index("canonical_smiles")["split"]
    attribution = donnees["canonical_smiles"].map(etiquettes)
    if attribution.isna().any() or not attribution.isin(["train", "test"]).all():
        raise ValueError("Le fichier team_split.csv doit attribuer train/test à chaque molécule.")
    train_idx = np.flatnonzero(attribution.to_numpy() == "train")
    test_idx = np.flatnonzero(attribution.to_numpy() == "test")
else:
    train_idx, test_idx = train_test_split(
        np.arange(len(donnees)), test_size=0.20,
        stratify=y, random_state=GRAINE
    )

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

splits = donnees[["canonical_smiles", "active"]].copy()
splits["split"] = "train"
splits.loc[test_idx, "split"] = "test"
splits.to_csv(DOSSIER_RESULTATS / "splits_bsep.csv", index=False)

print("Entraînement :", len(y_train), "molécules", np.bincount(y_train))
print("Test :", len(y_test), "molécules", np.bincount(y_test))
assert set(train_idx).isdisjoint(set(test_idx))

validation = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=GRAINE
)

grilles = {
    "SVM linéaire": {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"]
    },
    "SVM RBF": {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.001, 0.01, 0.1],
        "class_weight": [None, "balanced"]
    }
}

recherches = {}
lignes_cv = []

for nom, grille in grilles.items():
    debut = time.perf_counter()
    recherche = GridSearchCV(
        SVC(cache_size=500), grille,
        cv=validation, scoring="roc_auc",
        n_jobs=-1, refit=True
    )
    recherche.fit(X_train, y_train)
    recherches[nom] = recherche
    i = recherche.best_index_
    lignes_cv.append({
        "model": nom,
        "cv_roc_auc_mean": recherche.cv_results_["mean_test_score"][i],
        "cv_roc_auc_std": recherche.cv_results_["std_test_score"][i],
        "best_params": json.dumps(recherche.best_params_, ensure_ascii=False),
        "search_seconds": round(time.perf_counter() - debut, 2)
    })
    print(nom, ": ROC-AUC CV =", round(recherche.best_score_, 3))
    print("Paramètres :", recherche.best_params_, "\n")

table_cv = pd.DataFrame(lignes_cv).sort_values("cv_roc_auc_mean", ascending=False)
table_cv.to_csv(DOSSIER_RESULTATS / "svm_benchmark_cv.csv", index=False)
print(table_cv[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "best_params"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8.3, 4.2))
ax.barh(table_cv["model"], table_cv["cv_roc_auc_mean"],
        xerr=table_cv["cv_roc_auc_std"],
        color=["#257FA4", "#89B5C8"], capsize=5)
ax.set_xlim(0, 1)
ax.set_xlabel("ROC-AUC moyenne en validation croisée (± écart-type)")
ax.set_title("Benchmark interne : noyau linéaire ou RBF ?")
for i, ligne in enumerate(table_cv.itertuples()):
    ax.text(ligne.cv_roc_auc_mean + 0.015, i,
            f"{ligne.cv_roc_auc_mean:.3f}", va="center")
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "05_comparaison_svm_cv.png", dpi=180)
plt.show()
plt.close(fig)

nom_retenu = table_cv.iloc[0]["model"]
modele = recherches[nom_retenu].best_estimator_
print("Modèle retenu :", nom_retenu)
print("Meilleurs paramètres :", recherches[nom_retenu].best_params_)

def calcul_scores(nom, vraies_classes, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(
        vraies_classes, predictions, labels=[0, 1]
    ).ravel()
    return {
        "model": nom,
        "test_n": len(vraies_classes),
        "roc_auc": roc_auc_score(vraies_classes, scores),
        "pr_auc_ap": average_precision_score(vraies_classes, scores),
        "balanced_accuracy": balanced_accuracy_score(vraies_classes, predictions),
        "accuracy": accuracy_score(vraies_classes, predictions),
        "precision_1": precision_score(vraies_classes, predictions, zero_division=0),
        "recall_1": recall_score(vraies_classes, predictions, zero_division=0),
        "f1_1": f1_score(vraies_classes, predictions, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

dummy = DummyClassifier(strategy="prior")
dummy.fit(X_train, y_train)

pred_dummy = dummy.predict(X_test)
score_dummy = dummy.predict_proba(X_test)[:, 1]

pred_svm = modele.predict(X_test)
score_svm = modele.decision_function(X_test)

table_test = pd.DataFrame([
    calcul_scores("Dummy prior", y_test, pred_dummy, score_dummy),
    calcul_scores(nom_retenu, y_test, pred_svm, score_svm)
])
table_test.to_csv(DOSSIER_RESULTATS / "svm_benchmark_test.csv", index=False)
print(table_test.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(
    y_test, score_svm, name=nom_retenu, ax=axes[0], color="#177FA6"
)
RocCurveDisplay.from_predictions(
    y_test, score_dummy, name="Dummy", ax=axes[0], color="#B7A4A0"
)
axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
axes[0].set_title("ROC — test final")

PrecisionRecallDisplay.from_predictions(
    y_test, score_svm, name=nom_retenu, ax=axes[1], color="#177FA6"
)
PrecisionRecallDisplay.from_predictions(
    y_test, score_dummy, name="Dummy", ax=axes[1], color="#B7A4A0"
)
axes[1].set_title("Précision–rappel — test final")
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "06_roc_et_precision_rappel.png", dpi=180)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.3, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_svm, labels=[0, 1],
    display_labels=["Non-inhibiteur", "Inhibiteur"],
    cmap="Blues", colorbar=False, ax=ax
)
ax.set_title("Matrice de confusion du SVM retenu")
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "07_matrice_confusion.png", dpi=180)
plt.show()
plt.close(fig)

scores_oof = cross_val_predict(
    modele, X_train, y_train,
    cv=validation, method="decision_function", n_jobs=-1
)
p, r, seuils = precision_recall_curve(y_train, scores_oof)

# Le dernier point de la courbe ne possède pas de seuil associé.
p, r = p[:-1], r[:-1]
f2 = np.divide(
    5 * p * r, 4 * p + r,
    out=np.zeros_like(p), where=(4 * p + r) > 0
)
seuil_f2 = float(seuils[np.argmax(f2)])
pred_f2 = (score_svm >= seuil_f2).astype(int)

scores_f2 = calcul_scores(
    nom_retenu + " (seuil F2)", y_test, pred_f2, score_svm
)
pd.DataFrame([scores_f2]).to_csv(
    DOSSIER_RESULTATS / "svm_test_seuil_f2.csv", index=False
)
print("Seuil F2 sélectionné sur le train :", round(seuil_f2, 4))
print(pd.DataFrame([scores_f2]).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.4))
comparaison = [
    recall_score(y_test, pred_svm),
    recall_score(y_test, pred_f2)
]
ax.bar(["Seuil standard", "Seuil F2"], comparaison,
       color=["#6A9FBA", "#27A087"], width=0.55)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Rappel des inhibiteurs")
ax.set_title("Plus de détection, mais davantage de fausses alertes")
for i, valeur in enumerate(comparaison):
    ax.text(i, valeur + 0.025, f"{valeur:.1%}",
            ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(DOSSIER_FIGURES / "08_rappel_et_seuil.png", dpi=180)
plt.show()
plt.close(fig)
print("Faux positifs : standard =", int(((pred_svm == 1) & (y_test == 0)).sum()),
      "; seuil F2 =", int(((pred_f2 == 1) & (y_test == 0)).sum()))

predictions = donnees.iloc[test_idx][[
    "compound_name", "source", "canonical_smiles", "active"
]].copy()
predictions["pred_svm"] = pred_svm
predictions["score_svm"] = score_svm
predictions["pred_seuil_f2"] = pred_f2
predictions.to_csv(DOSSIER_RESULTATS / "svm_test_predictions.csv", index=False)

joblib.dump(
    {"modele": modele, "seuil_f2": seuil_f2,
     "fp_size": 1024, "radius": 2},
    DOSSIER_RESULTATS / "modele_svm_bsep.joblib"
)
print("Fichiers de résultats :", DOSSIER_RESULTATS.resolve())
print("Figures :", DOSSIER_FIGURES.resolve())

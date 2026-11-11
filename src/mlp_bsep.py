# -*- coding: utf-8 -*-
"""Prédiction BSEP avec un réseau de neurones MLP.
Lancer depuis la racine du dépôt : python src/mlp_bsep.py
"""

import importlib.util
import subprocess
import sys
if importlib.util.find_spec("rdkit") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit"])

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import joblib

GRAINE = 42
SORTIE = Path("resultats_nn")
SORTIE.mkdir(exist_ok=True, parents=True)
print("Environnement prêt.")

URL = ("https://raw.githubusercontent.com/ATOMScience-org/AMPL/master/"
       "atomsci/ddm/examples/BSEP/data/morgan_warner_combined_bsep_data.csv")
chemins = [Path("data/morgan_warner_combined_bsep_data.csv"),
           Path("../data/morgan_warner_combined_bsep_data.csv")]
csv_local = next((p for p in chemins if p.exists()), None)
df = pd.read_csv(csv_local if csv_local is not None else URL)
print(df.shape)
display(df.head())
print(df["active"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(6, 4))
n = df["active"].value_counts().reindex([0,1], fill_value=0)
ax.bar(["Non-inhibiteurs", "Inhibiteurs"], n, color=["#72A4BD", "#D6A45E"])
ax.set_title("Répartition des classes BSEP")
ax.set_ylabel("Nombre de molécules")
fig.tight_layout()
fig.savefig(SORTIE / "01_classes.png", dpi=160)
plt.show()

d = df.dropna(subset=["base_rdkit_smiles", "active"]).copy()
d["active"] = pd.to_numeric(d["active"], errors="coerce")
d = d[d["active"].isin([0,1])].copy()
d["active"] = d["active"].astype(int)

def canon(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None

d["canonical_smiles"] = d["base_rdkit_smiles"].apply(canon)
d = d.dropna(subset=["canonical_smiles"])
conflits = d.groupby("canonical_smiles")["active"].nunique()
d = d[~d["canonical_smiles"].isin(conflits[conflits > 1].index)]
d = d.drop_duplicates("canonical_smiles").reset_index(drop=True)
generateur = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
X = np.asarray([generateur.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
                for s in d["canonical_smiles"]], dtype=np.float32)
y = d["active"].to_numpy()
print("Molécules :", len(d), " | X :", X.shape, " | y :", y.shape)
assert X.shape[1] == 1024

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(X[:30,:100], cmap="Blues", aspect="auto", interpolation="nearest")
ax.set_title("30 molécules × 100 premiers bits ECFP4")
ax.set_xlabel("Bit")
ax.set_ylabel("Molécule")
fig.tight_layout()
fig.savefig(SORTIE / "02_fingerprints.png", dpi=160)
plt.show()

fichiers_split = [Path("data/splits_bsep.csv"), Path("../data/splits_bsep.csv"),
                  Path("resultats_svm/splits_bsep.csv")]
f_split = next((p for p in fichiers_split if p.exists()), None)
if f_split is None:
    i_train, i_test = train_test_split(np.arange(len(y)), test_size=.20,
                                      stratify=y, random_state=GRAINE)
else:
    separations = pd.read_csv(f_split)
    separations["canonical_smiles"] = separations["canonical_smiles"].apply(canon)
    assert not separations["canonical_smiles"].duplicated().any()
    repartition = d["canonical_smiles"].map(
        separations.set_index("canonical_smiles")["split"])
    assert repartition.notna().all(), "Des molécules sont absentes du split fourni."
    assert repartition.isin(["train", "test"]).all()
    i_train = np.flatnonzero(repartition.to_numpy() == "train")
    i_test = np.flatnonzero(repartition.to_numpy() == "test")

X_train, X_test = X[i_train], X[i_test]
y_train, y_test = y[i_train], y[i_test]
assert len(set(i_train).intersection(i_test)) == 0
assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2
print("Train :", len(y_train), " | Test :", len(y_test))
split_export = d[["canonical_smiles", "active"]].copy()
split_export["split"] = "train"
split_export.loc[i_test, "split"] = "test"
split_export.to_csv(SORTIE / "splits_bsep.csv", index=False)

validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
mlp = MLPClassifier(
    activation="relu", solver="adam", learning_rate_init=.001,
    batch_size=32, max_iter=250, early_stopping=True,
    validation_fraction=.15, n_iter_no_change=14, random_state=GRAINE)
parametres = {
    "hidden_layer_sizes": [(64,), (128,32)],
    "alpha": [.001, .01]
}
recherche = GridSearchCV(mlp, parametres, cv=validation,
                          scoring="roc_auc", n_jobs=1)
recherche.fit(X_train, y_train)
modele = recherche.best_estimator_
scores_cv = pd.DataFrame(recherche.cv_results_)
cols = ["param_hidden_layer_sizes", "param_alpha", "mean_test_score", "std_test_score"]
scores_cv[cols].to_csv(SORTIE / "benchmark_mlp_cv.csv", index=False)
print("Meilleurs paramètres :", recherche.best_params_)
print("ROC-AUC CV :", round(recherche.best_score_, 3))
print(scores_cv[cols].round(3).to_string(index=False))

r = scores_cv.sort_values("mean_test_score")
fig, ax = plt.subplots(figsize=(8, 4))
labels = [f"{h}, alpha={a}" for h,a in zip(r["param_hidden_layer_sizes"], r["param_alpha"])]
ax.barh(labels, r["mean_test_score"], xerr=r["std_test_score"],
        color="#5D9ABE", capsize=4)
ax.set_xlim(0,1)
ax.set_xlabel("ROC-AUC moyenne (CV)")
ax.set_title("Comparaison des réseaux")
fig.tight_layout()
fig.savefig(SORTIE / "03_reseaux_cv.png", dpi=160)
plt.show()

fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].plot(modele.loss_curve_, color="#287FA5")
axes[0].set_title("Loss d'entraînement")
axes[0].set_xlabel("Époque")
if modele.validation_scores_ is not None:
    axes[1].plot(modele.validation_scores_, color="#399E86")
axes[1].set_title("Accuracy validation interne")
axes[1].set_xlabel("Époque")
fig.tight_layout()
fig.savefig(SORTIE / "04_apprentissage.png", dpi=160)
plt.show()
print("Époques :", modele.n_iter_)

proba = modele.predict_proba(X_test)[:,1]
prediction = (proba >= .5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0,1]).ravel()
bilan = {
    "model": "MLP", "test_n": len(y_test),
    "roc_auc": roc_auc_score(y_test, proba),
    "average_precision": average_precision_score(y_test, proba),
    "accuracy": accuracy_score(y_test, prediction),
    "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
    "precision": precision_score(y_test, prediction, zero_division=0),
    "recall": recall_score(y_test, prediction, zero_division=0),
    "f1": f1_score(y_test, prediction, zero_division=0),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
}
print(pd.DataFrame([bilan]).round(3).to_string(index=False))
pd.DataFrame([bilan]).to_csv(SORTIE/"benchmark_mlp_test.csv", index=False)

dummy = DummyClassifier(strategy="prior").fit(X_train,y_train)
fig, axes = plt.subplots(1,2,figsize=(11,4.6))
RocCurveDisplay.from_predictions(y_test,proba,ax=axes[0],name="MLP")
RocCurveDisplay.from_predictions(y_test,dummy.predict_proba(X_test)[:,1],ax=axes[0],name="Dummy")
PrecisionRecallDisplay.from_predictions(y_test,proba,ax=axes[1],name="MLP")
PrecisionRecallDisplay.from_predictions(y_test,dummy.predict_proba(X_test)[:,1],ax=axes[1],name="Dummy")
axes[0].set_title("ROC test");axes[1].set_title("Précision-rappel test")
fig.tight_layout()
fig.savefig(SORTIE/"05_roc_pr.png",dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(y_test,prediction,
    display_labels=["Non-inhib.","Inhib."],labels=[0,1],
    cmap="Blues",colorbar=False,ax=ax)
ax.set_title("Matrice de confusion du MLP")
fig.tight_layout()
fig.savefig(SORTIE/"06_confusion.png",dpi=160)
plt.show()

predictions = d.iloc[i_test][["compound_name","source","canonical_smiles","active"]].copy()
predictions["mlp_proba"] = proba
predictions["mlp_prediction"] = prediction
predictions["erreur"] = predictions["active"] != prediction
predictions.to_csv(SORTIE/"mlp_predictions_test.csv", index=False)
joblib.dump({"modele":modele,"radius":2,"fpSize":1024},SORTIE/"mlp_bsep.joblib")
print("Fichiers générés dans",SORTIE.resolve())

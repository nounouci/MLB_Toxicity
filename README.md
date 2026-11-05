# MLB – Toxicity of Drugs | Prédiction de l’inhibition de BSEP

## Objectif

L’objectif est de prédire si une molécule inhibe la protéine BSEP (Bile Salt Export Pump, ABCB11) à partir de sa structure chimique. BSEP permet l’export des sels biliaires par les hépatocytes ; son inhibition constitue un mécanisme associé à certaines atteintes hépatiques médicamenteuses. Une inhibition de BSEP **n’équivaut pas** à une prédiction de la toxicité hépatique globale.

Le jeu de données public Morgan–Warner contient **660 molécules**. Les structures sont représentées par des SMILES et l’étiquette `active` vaut 1 pour les inhibiteurs, 0 pour les non-inhibiteurs.

![Démarche du projet](assets/pipeline.svg)

## Partie SVM

Les SMILES sont convertis avec RDKit en fingerprints **Morgan ECFP4 (rayon 2, 1 024 bits)**. Deux Support Vector Machines sont comparés : **SVM linéaire** et **SVM à noyau RBF**. Les hyperparamètres sont choisis par validation croisée stratifiée sur l’entraînement, puis le meilleur modèle est évalué sur un test indépendant.

**Résultats obtenus sur la première séparation aléatoire (graine 42)** :

![Résultats du SVM](assets/resultats.svg)

| Indicateur | Résultat |
|---|---:|
| Jeu d’entraînement / test | 528 / 132 molécules |
| SVM retenu | RBF, C = 10, gamma = scale |
| ROC-AUC (test) | 0,845 |
| Average Precision (test) | 0,822 |
| Accuracy (test) | 78,8 % |
| Rappel des inhibiteurs (test) | 76,3 % |
| F1-score (test) | 0,763 |
| Faux négatifs / faux positifs | 14 / 14 |

Ces chiffres correspondent à **une exécution**, pas à une performance garantie sur de nouvelles molécules. La séparation aléatoire peut être optimiste lorsque des molécules chimiquement proches apparaissent des deux côtés.

## Organisation du dépôt

```text
MLB_Toxicity/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── morgan_warner_combined_bsep_data.csv
├── notebooks/
│   └── 01_SVM_BSEP.ipynb
├── src/
│   └── svm_bsep.py
├── results/
│   ├── README.md
│   ├── svm_benchmark_cv.csv
│   ├── svm_benchmark_test.csv
│   └── svm_test_seuil_f2.csv
└── assets/
    ├── pipeline.svg
    └── resultats.svg
```

## Utilisation

**Dans Google Colab :** ouvrir `notebooks/01_SVM_BSEP.ipynb`, puis exécuter les cellules dans l’ordre. Le notebook explique chaque étape en français et affiche les visualisations. Le CSV est récupéré depuis son URL publique si aucun fichier local n’est présent.

**En local :** après avoir cloné le dépôt, depuis sa racine :

```bash
python -m pip install -r requirements.txt
python src/svm_bsep.py
```

Les images et les fichiers de scores créés à l’exécution sont enregistrés dans les dossiers `figures/` et `outputs/` (ignorés par Git).

## Pour comparer les modèles du groupe

La comparaison entre SVM, Random Forest et les autres classificateurs exige **les mêmes molécules dans le test**. Le script exporte `outputs/splits_bsep.csv`, contenant les SMILES canoniques et la partition train/test. La graine aléatoire utilisée ici est 42. Les colonnes des CSV de résultats sont décrites dans `results/README.md`.

## Métriques et interprétation

- **ROC-AUC** : capacité à classer les inhibiteurs avant les non-inhibiteurs, indépendamment du seuil de décision.
- **Average Precision (AP)** : résumé de la courbe précision–rappel.
- **Rappel** : fraction d’inhibiteurs réellement détectés.
- **Précision** : fraction de véritables inhibiteurs parmi les alertes.
- **F1** : compromis entre précision et rappel.
- **Faux négatif** : inhibiteur non détecté ; **faux positif** : non-inhibiteur signalé à tort.

Un seuil exploratoire sélectionné sur l’entraînement en maximisant F2 a augmenté le rappel sur le test de 76,3 % à 94,9 %, au prix de 14 à 40 faux positifs. Il ne faut pas sélectionner ce seuil en regardant le test.

## Sources

- McLoughlin et al. (2021), *Machine Learning Models to Predict Inhibition of the Bile Salt Export Pump*, [DOI](https://doi.org/10.1021/acs.jcim.0c00950).
- [Jeu public Morgan–Warner, ATOMScience / AMPL](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples/BSEP/data).
- [Documentation RDKit](https://www.rdkit.org/docs/) ; [documentation scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

## Limites

Ce modèle ne remplace pas les mesures biologiques. Le jeu public réunit deux sources expérimentales aux seuils de classification différents. Un test de généralisation par **scaffold split** serait une extension pertinente.

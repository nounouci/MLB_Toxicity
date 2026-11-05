# Résultats SVM

Les CSV de ce dossier proviennent d'une première exécution avec 660 composés et un test aléatoire stratifié (20 %, graine 42).

- `svm_benchmark_cv.csv` : score ROC-AUC moyen (± écart-type) des modèles lors de la validation croisée à 5 plis sur l'entraînement.
- `svm_benchmark_test.csv` : métriques finales sur les 132 molécules du test et comparaison au classificateur naïf.
- `svm_test_seuil_f2.csv` : bilan exploratoire quand le seuil de prédiction est choisi sur l'entraînement pour favoriser le rappel (F2).

**Important** : ces résultats sont comparables aux modèles d'autres personnes uniquement si elles utilisent les mêmes molécules de test, le même label `active`, et le même protocole. Pour un futur modèle, relancer le notebook et conserver le fichier `outputs/splits_bsep.csv`.

TN = vrais négatifs ; FP = faux positifs ; FN = faux négatifs ; TP = vrais positifs.

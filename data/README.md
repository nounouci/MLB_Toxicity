# Données BSEP

Le fichier `morgan_warner_combined_bsep_data.csv` provient du dépôt public ATOMScience/AMPL :
https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples/BSEP/data

Le fichier contient 660 composés issus des études de Morgan et Warner, avec les colonnes :
`compound_name`, `source`, `base_rdkit_smiles`, `relation`, `pIC50` et `active`.

La variable cible de la classification est `active`. La pIC50 est une mesure expérimentale, parfois censurée : elle n'est pas utilisée comme variable prédictive afin d'éviter une fuite d'information.

Les étiquettes peuvent dépendre de seuils différents selon les sources d'origine. Leur présence ne prouve pas qu'un composé est ou n'est pas globalement toxique pour le foie.

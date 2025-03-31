💼 Scoring Project – Predicting Bankruptcy
Ce projet vise à développer un modèle de scoring pour prédire la probabilité de faillite d'une entreprise, à partir de plusieurs jeux de données contenant des variables financières. Il s'agit d'un projet pédagogique de machine learning supervisé, orienté classification binaire.

📂 Fichiers importants
Le fichier to_pkl.py permet de transformet le format des fichiers en pkl afin de pouvoir traiter les données de manière plus rapide
Le fichier process.py permet de faire la jointure entre les différentes source de données, et de générer le fichier processed_set.csv qui sera le fichier utilisé pour la modélisation

📊 Objectif
Détecter les entreprises susceptibles de faire faillite à l’aide d’un modèle de machine learning entraîné sur des données comptables et financières historiques.

🔍 Données
Le projet utilise des jeux de données représentant différentes années. Chaque fichier contient des observations d’entreprises, avec des variables financières et une variable cible provenant de LoPucki indiquant si l'entreprise a fait faillite.

Les étapes incluent :
  - Fusion des datasets.
  - Construction de la variable cible.
  - Nettoyage et normalisation des données.
  - Feature engineering (création des variables explicatives).

🧠 Modèles utilisés
Le projet teste plusieurs approches de classification binaire, notamment :

XGBoost

Logistic Regression

🧪 Évaluation
Les métriques utilisées :

Accuracy

Precision

Recall (prioritaire)

F1-Score

Courbe ROC / AUC


🧰 Technologies
Python

Pandas / Numpy

Scikit-learn

XGBoost

Matplotlib 

Jupyter Notebook

📌 Auteurs
Devynck Tom

Goardet Marie

Rameil Hugo

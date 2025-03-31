# 💼 Scoring Project – Predicting Bankruptcy

Ce projet vise à développer un modèle de scoring pour prédire la probabilité de faillite d'une entreprise à partir de jeux de données contenant des variables financières. Il s'agit d'un projet pédagogique de machine learning supervisé, orienté classification binaire.

---

## 📂 Fichiers importants

- `to_pkl.py` : permet de transformer les fichiers au format `.pkl` pour accélérer les traitements.
- `process.py` : réalise la jointure entre les différentes sources de données et génère le fichier `processed_set.csv` utilisé pour la modélisation.

---

## 📊 Objectif

Détecter les entreprises susceptibles de faire faillite à l’aide d’un modèle de machine learning entraîné sur des données comptables et financières historiques.

---

## 🔍 Données

Le projet utilise des jeux de données représentant différentes années.  
Chaque fichier contient des observations d’entreprises avec :

- Des variables financières.
- Une variable cible (`target`) provenant de la base LoPucki, indiquant si l'entreprise a fait faillite.

### Étapes de traitement :
- Fusion des datasets.
- Construction de la variable cible.
- Nettoyage et normalisation des données.
- Feature engineering (création de nouvelles variables explicatives).

---

## 🧠 Modèles utilisés

Le projet explore plusieurs modèles de classification binaire :

- `XGBoost`
- `Régression logistique`

---

## 🧪 Évaluation

Les modèles sont évalués à l’aide des métriques suivantes :

- Accuracy
- Précision
- Rappel (**prioritaire**)
- F1-Score
- Courbe ROC / AUC

---

## 🧰 Technologies

- Python
- Pandas / Numpy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook

---

## 📌 Auteurs

- **Devynck Tom**  
- **Goardet Marie**  
- **Rameil Hugo**

# 📘 Compte Rendu -- Analyse d'un Projet de Data Science (Churn Client Bancaire)
![WhatsApp Image 2025-12-11 à 11 55 52_0faca475](https://github.com/user-attachments/assets/5b1d9479-1fd4-4a1f-90b6-fee943f66bd2)
## 1. Contexte Métier

Une banque souhaite réduire la défection client (*churn*), phénomène
coûteux en perte de revenus et frais d'acquisition.\
**Objectif principal :** prédire les clients susceptibles de partir.\
**Enjeu prioritaire :** réduire les **Faux Négatifs** --- les clients en
danger que le modèle ne détecte pas.\
**Métrique clé :** le **Recall** sur la classe *Client Parti (1)*.

### Données utilisées

-   **Dataset :** `Churn_Modelling.csv`
-   **Features :** âge, score de crédit, ancienneté, solde, pays, sexe,
    etc.
-   **Target :** `Exited` (1 = Parti, 0 = Retenu)

------------------------------------------------------------------------

## 2. Code Python Utilisé

Le script applique les phases standard d'un projet ML :\
acquisition, nettoyage, encodage, EDA, split, entraînement
(RandomForest), évaluation.

Il inclut notamment : - Encodage One-Hot (`Geography`, `Gender`) -
Imputation des valeurs manquantes (moyenne) - Entraînement d'un
**RandomForestClassifier** - Rapport de classification - Matrice de
confusion

------------------------------------------------------------------------

## 3. Analyse : Data Wrangling

### Encodage catégoriel

`pd.get_dummies()` transforme les variables textuelles en colonnes
binaires : - `Geography_Germany`, `Geography_Spain` - `Gender_Male`

### Imputation

`SimpleImputer(strategy='mean')` comble les valeurs manquantes simulées.

### ⚠️ Attention : Data Leakage

Le nettoyage doit normalement être fait **après** le split Train/Test
pour éviter une fuite d'information.

------------------------------------------------------------------------

## 4. Analyse Exploratoire (EDA)

L'EDA permet d'identifier : - **Outliers** potentiels (ex : âge max
élevé) - **Distributions anormales** - **Variables discriminantes** :
l'âge et l'ancienneté influencent fortement la défection -
**Corrélations** liées à l'encodage

Exemple : les clients allemands ont souvent un solde plus élevé →
corrélations à surveiller.

------------------------------------------------------------------------

## 5. Focus Théorique : Random Forest 🌲

Un Random Forest est un ensemble d'arbres : - **Bootstrapping** : arbres
entraînés sur des échantillons aléatoires - **Feature randomness** :
choix aléatoire des variables à chaque nœud

Cette double source d'aléatoire rend le modèle robuste, stable et
performant.

------------------------------------------------------------------------

## 6. Analyse de Performance

### Matrice de Confusion

                        Prédit Retenu (0)   Prédit Parti (1)
  --------------------- ------------------- ------------------
  **Réel Retenu (0)**   1554 (TN)           34 (FP)
  **Réel Parti (1)**    177 (FN)            235 (TP)

### Interprétation des erreurs

-   **Faux Positifs (FP)** : coût marketing inutile\
-   **Faux Négatifs (FN)** : perte critique --- client en danger non
    détecté

### Métriques clés (classe 1)

-   **Recall ≈ 57%** → le modèle détecte un peu plus de la moitié des
    clients à risque\
-   **Precision ≈ 87%** → lorsqu'il alerte, il a raison dans la grande
    majorité des cas
![WhatsApp Image 2025-12-11 à 11 49 13_655042af](https://github.com/user-attachments/assets/6057c5c4-0ac3-4f94-bea2-c64eb50458c2)

![WhatsApp Image 2025-12-11 à 11 49 13_03fbaf20](https://github.com/user-attachments/assets/6578a51e-2027-4816-8a27-6ff967f9b42e)
------------------------------------------------------------------------

## Conclusion Générale

Le Random Forest offre une **bonne précision** (87%), mais un **Recall
encore insuffisant** pour les besoins métier.\
Pour améliorer la détection des clients à risque, les prochaines étapes
recommandées sont :

-   Oversampling (SMOTE)\
-   Undersampling\
-   Ajustement des poids de classe\
-   Utilisation d'algorithmes optimisés (XGBoost, LightGBM)\
-   Optimisation des hyperparamètres

Le projet fournit une base solide, mais une optimisation approfondie est
indispensable pour maximiser la rétention client.

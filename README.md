#  Exploration du Deep Learning appliqué à la Finance (Prédiction Bitcoin)
##  Présentation du Projet
Ce projet est une initiation personnelle au Deep Learning et au Machine Learning, réalisée après 2 mois d'apprentissage en autonomie. 
L'objectif n'était pas de créer un algorithme de trading fonctionnel, mais de comprendre les mécanismes fondamentaux du traitement des séries temporelles financières avec **PyTorch**.

Le modèle cherche à prédire si le prix du Bitcoin va **monter, descendre ou stagner** le lendemain, en se basant sur la variation des prix des 3 jours précédents.

##  Technologies utilisées
* **Python 3**
* **PyTorch** (nn.Sequential, CrossEntropyLoss, Adam Optimizer)
* **Pandas** (Feature engineering, calcul des rendements logarithmiques/pourcentages)

* 
##  Ce que j'ai appris et expérimenté
* **Préparation des données :** Nettoyage des données financières, gestion des valeurs manquantes (`.dropna()`), et transformation des prix bruts en variations de pourcentage pour rendre les données stationnaires.
* **Architecture Récurrente vs Linéaire :** Conception d'un réseau de neurones multicouches (MLP). Expérimentation sur l'impact de la profondeur du réseau et du nombre de neurones face à la taille des features d'entrée.
* **Problématique du Surapprentissage (Overfitting) :** Compréhension de l'importance de séparer les données en blocs d'entraînement (Train) et de validation/test (Test) pour éviter que le modèle n'apprenne le passé par cœur.
* **Classification Multi-classes :** Mise en place d'un système à 3 classes (Baisse / Stagnation / Hausse) basé sur un seuil de variation ($\pm 0.5\%$).

## Évolutions futures envisagées
* Ajouter des indicateurs techniques (Volume, RSI, Moyennes Mobiles) pour enrichir les features ($X$).
* Implémenter une architecture **LSTM** ou **GRU**, plus adaptée aux dépendances temporelles.
* Mettre en place un vrai système de validation croisée temporelle (Time Series Split).

##  Démarche critique et limites du modèle

> *« Un bon projet de Machine Learning ne se mesure pas seulement à la baisse de sa Loss, mais à la compréhension réelle de ses faiblesses. »*

Bien que la boucle d'entraînement affiche une baisse de la fonction de coût (Loss), une analyse critique approfondie met en évidence plusieurs limites structurelles importantes à comprendre :

1. **Le piège du surapprentissage (Overfitting) :** Dans sa version initiale, le modèle a été entraîné et évalué sur l'intégralité du dataset. Sans séparation stricte entre les données d'entraînement (Train) et de test (Test), le modèle tend à apprendre par cœur les variations passées du Bitcoin au lieu de généraliser des règles applicables au futur.
2. **La dimensionnalité et l'architecture :** Alimenter un réseau de neurones multicouches profond (jusqu'à 64 neurones par couche) avec seulement 3 variables en entrée (`in_features=3`) crée un déséquilibre. Le modèle possède trop de paramètres libres par rapport à la simplicité des données fournies, ce qui amplifie le risque d'overfitting.
3. **La nature des marchés financiers :** Les séries temporelles financières (comme le cours des cryptomonnaies) sont extrêmement bruitées et non-stationnaires. Se baser uniquement sur les 3 jours de rendement précédents offre une fenêtre temporelle trop courte pour capturer des tendances macro-économiques ou des cycles de marché complexes.

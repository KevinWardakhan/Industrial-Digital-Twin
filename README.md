# Industrial-Digital-Twin — PHATE + PGAN

Genération de jumeaux numériques pour une unité industrielle basé sur des méthodes génératives. La pipeline, basée sur le papier de recherche [A virtual sample generation method based on manifold learning and a generative adversarial network for soft sensor models with limited data](https://www.sciencedirect.com/science/article/abs/pii/S1876107023004467), permet de générer des données dans des zones sparses, puis un modèle aval (ex. MLP) est entraîné pour des tâches de prédiction ou de génération. La data augmentation en zone sparse est l'un des objectifs principaux de mon implémentation.

## Contexte
- Unité ciblée : Unité de raffinerie contenant des fours, des réacteurs, des colonnes de distillation, etc...
- Objectif : créer des jumeaux numériques d’équipements pour générer des données virtuelles réalistes, notamment dans les régions de données sparses.

## Méthode et pipeline
Idée générale
1) [PHATE](https://phate.readthedocs.io/en/stable/) : Méthode de réduction de dimensions permettant de préserver à la fois les informations locales et globales de la donnée.
2) PGAN (PHATE GAN) est entraîné dans cet espace pour échantillonner de nouvelles observations, en particulier dans les zones sparses.
3) Les données virtuelles enrichies servent à entraîner un modèle aval (ex. MLP) pour prédiction d’échantillons ou de valeurs.

Pipeline  
[PHATE + PGAN pipeline](docs/pipeline_phate_pgan.jpg)

## Fonctionnalités
- Extraction de features via PHATE.
- Génération de données virtuelles avec PGAN (PHATE GAN) ciblant les zones sparses.
- Orchestration par équipement (sélection, mapping de features).
- Sauvegarde et chargement des modèles et des mappings.
- Scripts pour entraînement, génération et analyse.

## Structure du dépôt

```text
src/
└─ digital_twin/
   ├─ digital_twin_carboptim.py   # orchestration PHATE + PGAN
   ├─ simple_digital_twin.py      # analyse/génération par équipement
   ├─ equipment_features.py       # gestion des features et config
   ├─ load_and_save_models.py     # I/O modèles et mappings
   ├─ carboptim_test.py           # démo/tests internes
   └─ __init__.py
scripts/
└─ quickstart.py                  # exemple d'exécution
docs/
└─ pipeline_phate_pgan.jpg        # schéma du pipeline



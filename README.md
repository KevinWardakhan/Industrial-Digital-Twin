# Industrial-Digital-Twin — PHATE + PGAN

TL;DR
Jumeau numérique pour unité industrielle (DHC) basé sur des méthodes génératives. Le pipeline PHATE -> PGAN (PHATE GAN) densifie les zones clairsemées des données de procédé, puis un modèle aval (ex. MLP) est entraîné pour des tâches de prédiction ou de génération.

Avertissement confidentialité / IP
- Ne pas versionner de données réelles, ni de fichiers de configuration sensibles, ni de modèles entraînés (*.pth, *.pkl) sans autorisation.
- Ce dépôt montre le code et le pipeline réalisés pendant le stage. Utiliser des chemins d’exemple et des données synthétiques.

## Contexte
- Unité ciblée : DHC (raffinerie) avec fours, réacteurs, colonnes de distillation.
- Objectif : créer des jumeaux numériques d’équipements/procédés pour générer des données virtuelles réalistes, notamment dans les régions de données peu denses.

## Méthode et pipeline
Idée générale
1) PHATE extrait des représentations de dimension réduite qui respectent la structure/voisinage.
2) PGAN (PHATE GAN) est entraîné dans cet espace pour échantillonner de nouvelles observations, en particulier dans les zones sparse.
3) Les données virtuelles enrichies servent à entraîner un modèle aval (ex. MLP) pour prédiction d’échantillons ou de valeurs.

Schéma du pipeline
(placer l’image dans docs/pipeline_phate_pgan.png puis laisser le lien ci-dessous)
![PHATE + PGAN pipeline](docs/pipeline_phate_pgan.jpeg)

## Fonctionnalités
- Extraction de features via PHATE.
- Génération de données virtuelles avec PGAN (PHATE GAN) ciblant les zones clairsemées.
- Orchestration par équipement (sélection, mapping de features).
- Sauvegarde et chargement des modèles et des mappings.
- Scripts pour entraînement, génération et analyse.

## Structure du dépôt
src/digital_twin/
  - digital_twin_carboptim.py      (orchestration PHATE + PGAN)
  - simple_digital_twin.py         (analyse/génération par équipement)
  - equipment_features.py          (gestion des features et config)
  - load_and_save_models.py        (I/O modèles et mappings)
  - carboptim_test.py              (démo/tests internes)
  - __init__.py
scripts/
  - quickstart.py                  (exemple d’exécution)
docs/
  - pipeline_phate_pgan.png        (schéma du pipeline)



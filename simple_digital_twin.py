"""
JUMEAU NUMÉRIQUE SIMPLE - UNITÉ DHC
===================================

Interface simple pour utiliser votre jumeau numérique sans optimisation.
Ce code permet de :
1. Simuler le comportement de l'unité DHC
2. Générer des données virtuelles réalistes
3. Comparer avec les données réelles
4. Analyser le comportement de chaque équipement

Usage simple :
    python simple_digital_twin.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from digital_twin_carboptim import digital_twin_carboptim, compare_real_vs_generated
from carboptim_test import DataProcessor
import os


class SimpleDigitalTwin:
    """
    Interface simple pour le jumeau numérique de l'unité DHC
    """

    def __init__(self, csv_path=None, config_path=None):
        """
        Initialise le jumeau numérique

        Args:
            csv_path: Chemin vers les données consolidées
            config_path: Chemin vers la configuration DHC
        """
        # Chemins par défaut
        self.csv_path = (
            csv_path
            or "path_to_file"
        )
        self.config_path = config_path or "dataset.json"

        # Ordre de la chaîne d'équipements
        self.chain_order = [
            "F101",
            "R101",
            "Echangeur_Pre_F101_1",
            "Echangeur_Pre_C101",
            "C101",
            "Echangeur_Pre_F102",
            "C102",
            "F102",
            "C106",
            "Echangeur_Pre_F101_2",
            "C104",
            "Rebouilleur_C102",
            "RCI",
        ]

        self.real_data = None
        self.virtual_data = None

    def load_real_data(self, equipment_name, n_samples):
        """
        Charge les données réelles pour un équipement

        Args:
            equipment_name: Nom de l'équipement
            n_samples: Nombre d'échantillons à charger
        """
        print(f"📊 Chargement des données réelles pour {equipment_name}...")

        data_processor = DataProcessor(
            csv_path=self.csv_path,
            n_output=n_samples,
            equipment_name=equipment_name,
            config_file_path=self.config_path,
        )

        self.real_data = data_processor.load_and_process_data()
        print(f"✅ Données réelles chargées : {self.real_data.shape}")
        return self.real_data

    def generate_virtual_data(self, n_samples):
        """
        Génère des données virtuelles pour tous les équipements

        Args:
            n_samples: Nombre d'échantillons virtuels à générer
        """
        print(f"🔮 Génération de {n_samples} échantillons virtuels...")
        print("⏳ Cela peut prendre quelques minutes...")

        # Utilise votre fonction existante
        virtual_results = digital_twin_carboptim(
            phate_dim=4,
            n_output=n_samples,
            csv_path=self.csv_path,
            config_file_path=self.config_path,
        )

        print("✅ Données virtuelles générées avec succès !")
        return virtual_results

    def analyze_equipment_behavior(self, equipment_name):
        """
        Analyse le comportement d'un équipement spécifique
        """
        print(f"🔍 Analyse du comportement de {equipment_name}")

        # Charge les données
        real_data = self.load_real_data(equipment_name, n_samples=1000)

        print(f"\n📈 Statistiques pour {equipment_name}:")
        print("-" * 50)

        # Statistiques de base sur les données réelles
        print("DONNÉES RÉELLES:")
        print(f"  Nombre de variables : {real_data.shape[1]}")
        print(f"  Nombre d'échantillons : {real_data.shape[0]}")
        print(f"  Variables principales : {list(real_data.columns[:5])}")

        # Affiche quelques statistiques
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n  Moyennes des 5 premières variables :")
            for col in numeric_cols[:5]:
                print(f"    {col}: {real_data[col].mean():.4f}")

        return real_data

    def analyze_single_equipment_only(self, equipment_name):
        """
        Analyse SEULEMENT un équipement spécifique sans générer toute la chaîne
        """
        print(f"🔍 Analyse rapide de {equipment_name}")

        # Charge seulement les données réelles pour cet équipement
        real_data = self.load_real_data(equipment_name, n_samples=1000)

        print(f"\n📈 Statistiques pour {equipment_name}:")
        print("-" * 50)

        # Statistiques de base sur les données réelles
        print("DONNÉES RÉELLES:")
        print(f"  Nombre de variables : {real_data.shape[1]}")
        print(f"  Nombre d'échantillons : {real_data.shape[0]}")
        print(f"  Variables principales : {list(real_data.columns[:5])}")

        # Affiche quelques statistiques
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n  Moyennes des 5 premières variables :")
            for col in numeric_cols[:5]:
                print(f"    {col}: {real_data[col].mean():.4f}")

        return real_data

    def generate_virtual_for_single_equipment(self, equipment_name, n_samples):
        """
        Génère des données virtuelles pour UN SEUL équipement en utilisant carboptim_test.py
        """
        # Vérifie si un modèle existe pour cet équipement
        model_path = f"trained_models/generator_{equipment_name}.pth"
        if not os.path.exists(model_path):
            print(f"❌ Modèle non trouvé pour {equipment_name}")
            print(f"Modèles disponibles :")
            for eq in self.chain_order:
                if os.path.exists(f"trained_models/generator_{eq}.pth"):
                    print(f"  ✅ {eq}")
            return None

        # Utilise carboptim_test.py pour générer UN SEUL équipement
        print(f"🔧 Lancement de carboptim_test.py pour {equipment_name}...")

        # Importe et utilise directement le code de carboptim_test.py
        try:
            from carboptim_test import (
                DataProcessor,
                SparseGeneration,
            )
            import torch
            import pandas as pd

            # Configuration pour cet équipement spécifique
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            phate_dim = 4

            print(f"📊 Chargement des données pour {equipment_name}...")

            # Charge les données pour cet équipement
            data_processor = DataProcessor(
                csv_path=self.csv_path,
                n_output=n_samples,
                equipment_name=equipment_name,
                config_file_path=self.config_path,
            )
            X_ori = data_processor.load_and_process_data_test()
            n_samples_actual, n_features = X_ori.shape

            print(f"✅ Données chargées: {X_ori.shape}")
            # Charge les modèles pré-entraînés
            print(f"🤖 Chargement des modèles pour {equipment_name}...")
            from digital_twin_carboptim import load_models

            models = load_models(
                n_sample=n_samples, equipment_name=equipment_name, phate_dim=phate_dim
            )
            print(f"🧮 Application de PHATE...")

            phate_embedder = models.get("phate_embedder")
            if phate_embedder is None:
                print(f"❌ PHATE embedder non trouvé pour {equipment_name}")
                return None
            Z_ori = phate_embedder.transform(X_ori)

            print(f"✅ PHATE embedding: {Z_ori.shape}")

            if not models:
                print(f"❌ Échec du chargement des modèles")
                return None

            # Génération sparse
            print(f"🎲 Génération sparse...")
            sparse_generator = SparseGeneration(
                Q=1,
                phate_dim=phate_dim,
                n_samples=n_samples_actual,
                W=10,
            )
            Z_vir = sparse_generator.sparse_generation(Z_ori)

            # Génère les données virtuelles
            print(f"🔮 Génération des données virtuelles...")
            generator = models["generator"]
            generator.eval()

            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z_vir).to(device)
                X_vir_tensor = generator(Z_tensor)
                X_vir_scaled = X_vir_tensor.cpu().numpy()[:n_samples]

            # Retourne à l'échelle originale
            X_vir = phate_embedder.scaler.inverse_transform(X_vir_scaled)

            # Crée DataFrame avec les noms de colonnes corrects
            virtual_data = pd.DataFrame(X_vir, columns=X_ori.columns)

            compare_real_vs_generated(
                X_ori[:n_samples], virtual_data, equipment_name=equipment_name
            )

            print(
                f"✅ Données virtuelles générées pour {equipment_name} et {n_samples} échantillons: {virtual_data.shape}"
            )
            return virtual_data

        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")
            import traceback

            traceback.print_exc()
            return None

    def interactive_demo(self):
        """
        Démonstration interactive simple
        """
        print("=" * 60)
        print("🏭 JUMEAU NUMÉRIQUE SIMPLE - UNITÉ DHC")
        print("=" * 60)

        print("\nÉquipements disponibles :")
        for i, eq in enumerate(self.chain_order, 1):
            print(f"  {i:2d}. {eq}")

        while True:
            print("\n" + "=" * 50)
            print("Que voulez-vous faire ?")
            print("1. Analyser un équipement (données réelles)")
            print("2. Simuler l'unité DHC")
            print("3. Génération rapide pour un équipement")
            print("0. Quitter")

            choice = input("\nVotre choix (0-3) : ").strip()

            if choice == "0":
                print("👋 Au revoir !")
                break

            elif choice == "1":
                eq_name = self.ask_equipment_name()
                if eq_name:
                    self.analyze_single_equipment_only(eq_name)

            elif choice == "2":
                n_samples = input(
                    "Combien d'échantillons voulez-vous générer ? : "
                ).strip()
                n_samples = int(n_samples) if n_samples.isdigit() else 1000
                self.generate_virtual_data(n_samples)

            elif choice == "3":
                eq_name = self.ask_equipment_name()
                if eq_name:
                    n_samples = input(
                        "Combien d'échantillons voulez-vous générer ? "
                    ).strip()
                    if n_samples.isdigit():
                        n_samples = int(n_samples)
                    else:
                        n_samples = 3000
                    self.generate_virtual_for_single_equipment(eq_name, n_samples)

            else:
                print("❌ Choix invalide")

    def ask_equipment_name(self):
        """
        Demande à l'utilisateur de choisir un équipement
        """
        print("\nÉquipements disponibles :")
        for i, eq in enumerate(self.chain_order, 1):
            print(f"  {i:2d}. {eq}")

        print("  0. Annuler")

        while True:
            choice = input(
                f"\nChoisissez un équipement (1-{len(self.chain_order)}) ou nom direct : "
            ).strip()

            if choice == "0":
                return None

            # Si c'est un nombre, utilise l'index
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self.chain_order):
                    selected = self.chain_order[idx]
                    print(f"✅ Équipement sélectionné : {selected}")
                    return selected
                else:
                    print(
                        f"❌ Numéro invalide. Choisissez entre 1 et {len(self.chain_order)}"
                    )
                    continue

            # Si c'est un nom direct, vérifie qu'il existe
            elif choice in self.chain_order:
                print(f"✅ Équipement sélectionné : {choice}")
                return choice

            # Sinon cherche une correspondance partielle
            else:
                matches = [
                    eq for eq in self.chain_order if choice.lower() in eq.lower()
                ]
                if len(matches) == 1:
                    selected = matches[0]
                    print(f"✅ Équipement trouvé : {selected}")
                    return selected
                elif len(matches) > 1:
                    print(f"🤔 Plusieurs correspondances trouvées : {matches}")
                    print("Soyez plus précis.")
                else:
                    print(f"❌ Équipement '{choice}' non trouvé.")
                    print("Utilisez le numéro ou le nom exact.")


if __name__ == "__main__":
    # Choix du mode
    dt = SimpleDigitalTwin()
    dt.interactive_demo()

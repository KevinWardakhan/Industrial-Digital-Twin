#!/usr/bin/env python3
"""
Simple script to train and save PGAN models for all equipment using carboptim_test.py
"""

import os
import sys
import torch
import pickle
import json
import pandas as pd
from equipment_features import get_all_equipment_names, get_equipment_feature_names
from carboptim_test import DataProcessor, PHATEEmbedder, PGANtraining, device


def get_dropped_features_for_equipment(equipment_name):
    """
    Get the features that were dropped for a given equipment (features that exist in original but not in modified config)

    Args:
        equipment_name (str): Name of the equipment

    Returns:
        dict: Dictionary with 'kept_features', 'dropped_features', and 'feature_mapping'
    """
    try:
        # Get features from original config
        original_features = get_equipment_feature_names(
            equipment_name, "dataset/carboptim/DHC_unit_config.json"
        )

        # Get features from modified config
        modified_features = get_equipment_feature_names(
            equipment_name, "dataset/carboptim/DHC_unit_config_modified.json"
        )

        # Find dropped features
        dropped_features = [f for f in original_features if f not in modified_features]

        # Create mapping for reconstruction
        feature_mapping = {
            "original_features": original_features,
            "kept_features": modified_features,
            "dropped_features": dropped_features,
            "original_indices": {
                feat: idx for idx, feat in enumerate(original_features)
            },
            "kept_indices": {feat: idx for idx, feat in enumerate(modified_features)},
        }

        return feature_mapping

    except Exception as e:
        print(f"Error getting dropped features for {equipment_name}: {e}")
        return None


def save_feature_mapping(equipment_name, feature_mapping):
    """Save the feature mapping to a JSON file for later reconstruction"""
    try:
        os.makedirs("trained_models_final", exist_ok=True)
        mapping_path = f"trained_models_final/feature_mapping_{equipment_name}.json"
        with open(mapping_path, "w") as f:
            json.dump(feature_mapping, f, indent=2)
        print(f"   ✓ Feature mapping saved: {mapping_path}")
        return True
    except Exception as e:
        print(f"   ❌ Error saving feature mapping: {e}")
        return False


def reconstruct_full_feature_matrix(
    equipment_name, generated_data, previous_equipment_data=None
):
    """
    Reconstruct the full feature matrix by adding back dropped features

    Args:
        equipment_name (str): Name of the equipment
        generated_data (pd.DataFrame): Generated data with only kept features
        previous_equipment_data (dict): Dictionary with previous equipment data {equipment_name: DataFrame}

    Returns:
        pd.DataFrame: Full feature matrix with dropped features added back
    """
    try:
        # Load feature mapping
        mapping_path = f"trained_models_final/feature_mapping_{equipment_name}.json"
        with open(mapping_path, "r") as f:
            feature_mapping = json.load(f)

        original_features = feature_mapping["original_features"]
        kept_features = feature_mapping["kept_features"]
        dropped_features = feature_mapping["dropped_features"]

        # Initialize full matrix with NaN
        n_samples = len(generated_data)
        full_data = pd.DataFrame(index=range(n_samples), columns=original_features)

        # Fill in the generated features
        for feat in kept_features:
            if feat in generated_data.columns:
                full_data[feat] = generated_data[feat].values

        # Fill in dropped features from previous equipment data
        if previous_equipment_data:
            for feat in dropped_features:
                # Find which previous equipment generated this feature
                for prev_equipment, prev_data in previous_equipment_data.items():
                    if feat in prev_data.columns:
                        # Use the same number of samples
                        if len(prev_data) >= n_samples:
                            full_data[feat] = prev_data[feat].iloc[:n_samples].values
                        else:
                            # If not enough samples, repeat the last value
                            values = prev_data[feat].values
                            extended_values = list(values) + [values[-1]] * (
                                n_samples - len(values)
                            )
                            full_data[feat] = extended_values[:n_samples]
                        break

                # If feature not found in previous data, use a default strategy
                if full_data[feat].isna().all():
                    print(
                        f"   ⚠️  Feature {feat} not found in previous equipment data, using zeros"
                    )
                    full_data[feat] = 0.0

        return full_data

    except Exception as e:
        print(f"Error reconstructing full feature matrix for {equipment_name}: {e}")
        return generated_data  # Return original if reconstruction fails


def load_and_display_feature_mappings():
    """
    Load and display all feature mappings for analysis
    """
    print("\n" + "=" * 60)
    print("FEATURE MAPPING ANALYSIS")
    print("=" * 60)

    try:
        equipment_names = get_all_equipment_names(
            "dataset/carboptim/DHC_unit_config_modified.json"
        )

        total_original = 0
        total_kept = 0
        total_dropped = 0

        for equipment in equipment_names:
            try:
                mapping_path = (
                    f"trained_models_new_new/feature_mapping_{equipment}.json"
                )
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        mapping = json.load(f)

                    print(f"\n{equipment}:")
                    print(f"  Original: {len(mapping['original_features'])} features")
                    print(f"  Kept: {len(mapping['kept_features'])} features")
                    print(f"  Dropped: {len(mapping['dropped_features'])} features")

                    if mapping["dropped_features"]:
                        print(f"  Dropped features: {mapping['dropped_features']}")

                    total_original += len(mapping["original_features"])
                    total_kept += len(mapping["kept_features"])
                    total_dropped += len(mapping["dropped_features"])

            except Exception as e:
                print(f"  Error loading mapping for {equipment}: {e}")

        print(f"\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Total original features: {total_original}")
        print(f"Total kept features: {total_kept}")
        print(f"Total dropped features: {total_dropped}")
        print(f"Compression ratio: {total_kept/total_original*100:.1f}%")

    except Exception as e:
        print(f"Error in feature mapping analysis: {e}")


def train_and_save_equipment_model(equipment_name):
    """Train and save PGAN model for a single equipment"""

    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR {equipment_name}")
    print(f"{'='*60}")

    # Training parameters (same as carboptim_test.py)
    BATCH_SIZE = 64
    EPOCHS = 1000
    phate_dim = 5
    n_output = 1000
    lr = 0.001

    try:
        # Step 1: Load data
        print(f"Step 1: Loading data for {equipment_name}...")
        data_processor = DataProcessor(
            "/Users/kevinwardakhan/test/dt-stage-GenAI/dataset/carboptim/DHC_consolidated.parquet",
            n_output=n_output,
            equipment_name=equipment_name,
            config_file_path="dataset/carboptim/DHC_unit_config.json",
        )
        X_ori = data_processor.load_and_process_data()

        if X_ori is None or X_ori.empty:
            print(f"   ❌ No data available for {equipment_name}")
            return False

        n_samples, n_features = X_ori.shape
        print(f"   ✓ Data loaded: {X_ori.shape}")

        # Get and save feature mapping info
        print(f"Step 1.5: Analyzing feature mapping...")
        feature_mapping = get_dropped_features_for_equipment(equipment_name)
        if feature_mapping:
            print(
                f"   • Original features: {len(feature_mapping['original_features'])}"
            )
            print(f"   • Kept features: {len(feature_mapping['kept_features'])}")
            print(f"   • Dropped features: {len(feature_mapping['dropped_features'])}")
            if feature_mapping["dropped_features"]:
                print(f"   • Dropped: {feature_mapping['dropped_features']}")
            save_feature_mapping(equipment_name, feature_mapping)
        else:
            print(f"   ⚠️  Could not analyze feature mapping")

        # Step 2: PHATE embedding
        phate_dim = min(phate_dim, n_features - 1)
        phate_dim = max(1, phate_dim)  # Ensure at least 1 dimension
        print(f"Step 2: Computing PHATE embedding with {phate_dim} dimensions...")
        phate_embedder = PHATEEmbedder(n_components=phate_dim, knn=100)
        Z_ori = phate_embedder.fit_transform(X_ori)
        X_scaled = phate_embedder.scaler.transform(X_ori)
        print(f"   ✓ PHATE embedding computed: {Z_ori.shape}")

        # Step 3: Train PGAN
        print(f"Step 3: Training PGAN...")
        pgan = PGANtraining(
            lambda_l1=0.001,
            phate_dim=phate_dim,
            n_samples=n_samples,
            n_features=n_features,
            lr=lr,
        )
        pgan.train(X_scaled, Z_ori, batch_size=BATCH_SIZE, epochs=EPOCHS)
        print(f"   ✓ PGAN training completed")

        # Step 4: Save models
        print(f"Step 4: Saving models...")

        # Create trained_models_final directory if it doesn't exist
        os.makedirs("trained_models_final", exist_ok=True)

        # Save generator model
        generator_path = f"trained_models_final/generator_{equipment_name}.pth"
        torch.save(pgan.Generator.state_dict(), generator_path)
        print(f"   ✓ Generator saved: {generator_path}")

        # Save discriminator model
        discriminator_path = f"trained_models_final/discriminator_{equipment_name}.pth"
        torch.save(pgan.Discriminator.state_dict(), discriminator_path)
        print(f"   ✓ Discriminator saved: {discriminator_path}")

        # Save PHATE embedder
        embedder_path = f"trained_models_final/phate_embedder_{equipment_name}.pkl"
        with open(embedder_path, "wb") as f:
            pickle.dump(phate_embedder, f)
        print(f"   ✓ PHATE embedder saved: {embedder_path}")

        print(
            f"   ✅ {equipment_name} model training and saving completed successfully!"
        )
        return True

    except Exception as e:
        print(f"   ❌ Error training {equipment_name}: {e}")
        return False


def main():
    """Train and save models for all equipment"""

    print("🏭 TRAINING AND SAVING PGAN MODELS FOR ALL EQUIPMENT (MODIFIED CONFIG)")
    print("=" * 60)

    # Get all equipment names from modified config
    try:
        equipment_names = get_all_equipment_names(
            "dataset/carboptim/DHC_unit_config.json"
        )
        print(f"Found {len(equipment_names)} equipment: {equipment_names}")
    except Exception as e:
        print(f"❌ Error getting equipment names: {e}")
        return

    # Train models for each equipment
    successful_models = []
    failed_models = []

    for i, equipment in enumerate(equipment_names):
        print(f"\n[{i+1}/{len(equipment_names)}] Processing {equipment}...")

        if train_and_save_equipment_model(equipment):
            successful_models.append(equipment)
        else:
            failed_models.append(equipment)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully trained: {len(successful_models)} models")
    print(f"❌ Failed: {len(failed_models)} models")

    if successful_models:
        print(f"\n✅ Successfully trained models:")
        for model in successful_models:
            print(f"   • {model}")

    if failed_models:
        print(f"\n❌ Failed models:")
        for model in failed_models:
            print(f"   • {model}")

    print(f"\n🎉 Process completed!")
    print(f"All models are saved in the 'trained_models_new_new' directory")
    print(f"Models trained on unique features from DHC_unit_config_modified.json")

    # Display feature mapping analysis
    # load_and_display_feature_mappings()


if __name__ == "__main__":
    main()

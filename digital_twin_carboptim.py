import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import phate
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from equipment_features import (
    get_all_equipment_names,
    get_equipment_feature_names,
    get_shared_features_across_equipment,
)
from carboptim_test import (
    DataProcessor,
    SparseGeneration,
    PHATEGenerator,
    PHATEDiscriminator,
    PHATEEmbedder,
)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import warnings
from scipy.stats import ttest_ind
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import os
import traceback

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
warnings.filterwarnings("ignore")


def load_feature_mapping(equipment_name, mapping_dir="trained_models"):
    """
    Load feature mapping for the specified equipment.

    Args:
        equipment_name (str): Name of the equipment.
        mapping_dir (str): Directory containing feature mapping files.

    Returns:
        dict: Dictionary containing feature mapping information.
    """
    mapping_path = os.path.join(mapping_dir, f"feature_mapping_{equipment_name}.json")

    try:
        with open(mapping_path, "r") as f:
            feature_mapping = json.load(f)
        return feature_mapping
    except FileNotFoundError:
        print(f"Warning: Feature mapping file not found for {equipment_name}")
        return None
    except Exception as e:
        print(f"Error loading feature mapping for {equipment_name}: {e}")
        return None


def load_models(n_sample, equipment_name, phate_dim):
    """
    Load pre-trained models for the specified equipment.

    Args:
        equipment_name (str): Name of the equipment.
        phate_dim (int): PHATE embedding dimensions.

    Returns:
        dict: Dictionary containing the loaded models.
    """
    try:
        # Load feature mapping to get the correct feature count
        feature_mapping = load_feature_mapping(equipment_name)
        if feature_mapping is None:
            raise ValueError(f"Feature mapping not found for {equipment_name}")

        # Use ALL original features for model training (not just kept features)
        original_features = feature_mapping.get("original_features", [])
        n_features = len(original_features)

        # Loading models silently

        # Load the saved state dictionaries
        generator_state = torch.load(
            f"trained_models/generator_{equipment_name}.pth",
            map_location=device,
        )
        discriminator_state = torch.load(
            f"trained_models/discriminator_{equipment_name}.pth",
            map_location=device,
        )

        # Create model instances with the correct dimensions
        generator = PHATEGenerator(
            phate_dim=phate_dim,
            n_samples=n_sample,
            n_features=n_features,
            hidden_dim=32,
        )

        discriminator = PHATEDiscriminator(
            phate_dim=phate_dim,
            n_samples=n_sample,
            n_features=n_features,
            hidden_dim=32,
        )

        # Load the state dictionaries into the models
        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)

        # Move models to device
        generator.to(device)
        discriminator.to(device)

        # Set to evaluation mode
        generator.eval()
        discriminator.eval()

        # Load the saved PHATE embedder from training
        phate_embedder_path = f"trained_models/phate_embedder_{equipment_name}.pkl"
        try:
            with open(phate_embedder_path, "rb") as f:
                phate_embedder = pickle.load(f)
        except FileNotFoundError:
            # Creating new PHATE embedder (fallback)
            phate_embedder = phate.PHATE(phate_dim, n_jobs=-1)

        models = {
            "generator": generator,
            "discriminator": discriminator,
            "phate_embedder": phate_embedder,
        }

        return models

    except Exception as e:
        print(f"Error loading models for {equipment_name}: {e}")
        return None


def evaluate_pc_similarity(real_data, generated_data, n_pca=4):
    """
    Quick evaluation of PC similarity between real and generated data without plots.
    Returns the average component similarity score.
    """
    try:
        # Ensure both datasets are DataFrames
        if isinstance(real_data, np.ndarray):
            real_data = pd.DataFrame(real_data)
        if isinstance(generated_data, np.ndarray):
            generated_data = pd.DataFrame(generated_data, columns=real_data.columns)

        # Adjust n_pca based on available features
        max_components = min(
            n_pca,
            len(real_data.columns),
            real_data.shape[0] - 1,
            generated_data.shape[0] - 1,
        )
        n_pca_actual = max(1, max_components)

        # Scale datasets separately
        scaler_real = StandardScaler()
        scaler_gen = StandardScaler()

        X_real_scaled = scaler_real.fit_transform(real_data)
        X_gen_scaled = scaler_gen.fit_transform(generated_data)

        # Apply separate PCAs
        pca_real = PCA(n_components=n_pca_actual)
        pca_gen = PCA(n_components=n_pca_actual)

        pca_real.fit(X_real_scaled)
        pca_gen.fit(X_gen_scaled)

        # Compare component loadings
        component_similarity_scores = []

        for i in range(n_pca_actual):
            real_loadings = pca_real.components_[i]
            gen_loadings = pca_gen.components_[i]

            # Calculate cosine similarity
            dot_product = np.dot(real_loadings, gen_loadings)
            norm_product = np.linalg.norm(real_loadings) * np.linalg.norm(gen_loadings)
            cosine_similarity = (
                abs(dot_product / norm_product) if norm_product > 0 else 0
            )

            component_similarity_scores.append(cosine_similarity)

        return np.mean(component_similarity_scores)

    except Exception as e:
        # If evaluation fails, return low score
        return 0.0


def digital_twin_carboptim(
    phate_dim,
    n_output,
    csv_path="dataset/carboptim/processed_data.parquet",
    config_file_path="dataset/carboptim/DHC_unit_config.json",
    Q=1,
):
    """
    Main function to run the digital twin for the CarbOptim project.
    It loads the necessary data, processes it, and applies the models.
    """
    chain_order = [
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

    # Dictionary to store all predicted features across equipment
    X_virs = {}

    # Initialize X_all with zeros - this will store cumulative predictions
    # Get all possible features from any equipment to initialize properly
    all_features = set()
    for eq_name in chain_order:
        mapping = load_feature_mapping(eq_name)
        if mapping:
            all_features.update(mapping.get("original_features", []))

    all_features = sorted(list(all_features))
    X_all = pd.DataFrame(0.0, index=range(n_output), columns=all_features)

    for equipment_name in chain_order:
        print(f"\n=== Processing {equipment_name} ===")
        try:
            # Load feature mapping for current equipment
            feature_mapping = load_feature_mapping(equipment_name)
            if not feature_mapping:
                print(f"Skipping {equipment_name} - missing feature mapping")
                continue

            dropped_features = feature_mapping.get("dropped_features", [])
            kept_features = feature_mapping.get("kept_features", [])
            original_features = feature_mapping.get("original_features", [])

            print(
                f"📦 {equipment_name}: {len(kept_features)} new, {len(dropped_features)} reused"
            )

            models = load_models(
                n_sample=n_output, equipment_name=equipment_name, phate_dim=phate_dim
            )
            if not models:
                print(f"Skipping {equipment_name} - model loading failed")
                continue

            # Load original data for this equipment (with ALL its features)
            data_processor = DataProcessor(
                csv_path=csv_path,
                n_output=n_output,
                equipment_name=equipment_name,
                config_file_path=config_file_path,
            )

            X_ori = data_processor.load_and_process_data_test()
            X_ori_compare = X_ori.copy()

            # Replace dropped features with previously predicted values
            replaced_count = 0
            for dropped_feature in dropped_features:
                if (
                    dropped_feature in X_all.columns
                    and dropped_feature in X_ori.columns
                ):
                    X_ori[dropped_feature] = X_all[dropped_feature].values
                    replaced_count += 1

            if replaced_count > 0:
                print(f"   ↻ Replaced {replaced_count} features")

            # Apply PHATE embedding using the SAVED embedder from training
            phate_embedder = models["phate_embedder"]

            # Transform data using the saved embedder (don't fit again!)
            try:
                Z_ori = phate_embedder.transform(X_ori.values)
            except Exception as e:
                # Error with saved embedder, using fallback
                # Fallback: create new embedder if saved one fails
                phate_embedder = PHATEEmbedder(
                    n_components=phate_dim, knn=100, n_jobs=-1
                )
                Z_ori = phate_embedder.fit_transform(X_ori.values)

            # Generate multiple sparse virtual samples for diversity
            sparse_generator = SparseGeneration(
                Q=Q,  # Increase number of generation cycles for more diversity
                phate_dim=phate_dim,
                n_samples=n_output,
                W=10,  # Increase candidates per original sample
            )

            # Generate virtual embeddings
            Z_vir = sparse_generator.sparse_generation(Z_ori)

            # Generating virtual data...

            with torch.no_grad():  # Disable gradient computation for inference
                # Generate virtual features from the embedding
                Z_tensor = torch.FloatTensor(Z_vir).to(device)
                X_vir_tensor = models["generator"](Z_tensor)
                X_vir_generated_scaled = X_vir_tensor.cpu().detach().numpy()

            # Transform back to original scale
            if hasattr(phate_embedder, "scaler") and phate_embedder.scaler is not None:
                try:
                    X_vir_generated = phate_embedder.scaler.inverse_transform(
                        X_vir_generated_scaled
                    )
                except Exception as e:
                    print(f"ERROR: Could not apply inverse scaling: {e}")
                    X_vir_generated = X_vir_generated_scaled
            else:
                X_vir_generated = X_vir_generated_scaled

            # Convert to DataFrame
            X_vir_generated = pd.DataFrame(X_vir_generated, columns=original_features)

            # Evaluate PC similarity
            similarity_score = evaluate_pc_similarity(X_ori, X_vir_generated)

            # Update X_all with the newly predicted kept features
            updated_count = 0
            for kept_feature in kept_features:
                if kept_feature in X_vir_generated.columns:
                    X_all[kept_feature] = X_vir_generated[kept_feature].values
                    updated_count += 1

            # For dropped features, keep the previously predicted values in X_vir_generated
            for dropped_feature in dropped_features:
                if (
                    dropped_feature in X_all.columns
                    and dropped_feature in X_vir_generated.columns
                ):
                    X_vir_generated[dropped_feature] = X_all[dropped_feature].values

            # Store the complete virtual data for this equipment
            X_virs[equipment_name] = X_vir_generated

            # Save the results
            output_dir = "/Users/kevinwardakhan/test/dt-stage-GenAI/dataset/carboptim/virtual_data"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/virtual_data_{equipment_name}.csv"
            X_vir_generated.to_csv(output_path, index=False)

            # Compare real vs generated data with enhanced metrics
            _, quality_score = compare_real_vs_generated_enhanced(
                X_ori_compare, X_vir_generated, equipment_name, kept_features
            )
            print(
                f"   ✅ Quality: {quality_score:.0f}/100 | PC: {similarity_score:.3f} | Updated: {updated_count}"
            )

        except Exception as e:
            print(f"ERROR processing {equipment_name}: {e}")
            if "traceback" in str(e).lower():
                traceback.print_exc()

    return X_virs


def compact_column_wise_comparison(real_data, generated_data, equipment_name):
    """
    Compact comparison of distributions for each column pair without PCA.

    Args:
        real_data: Real data DataFrame
        generated_data: Generated data DataFrame
        equipment_name: Equipment name for logging

    Returns:
        dict: Comprehensive distribution comparison metrics
    """
    # Ensure DataFrames
    if isinstance(real_data, np.ndarray):
        real_data = pd.DataFrame(real_data)
    if isinstance(generated_data, np.ndarray):
        generated_data = pd.DataFrame(generated_data, columns=real_data.columns)

    # Compact column-wise analysis (suppressed detailed output)
    results = {"equipment": equipment_name, "column_metrics": {}, "summary": {}}

    js_scores = []
    qq_correlations = []

    for col in real_data.columns:
        if col in generated_data.columns:
            real_vals = real_data[col].values
            gen_vals = generated_data[col].values

            # 1. Jensen-Shannon Divergence (requires histograms)
            # Create histograms for JS divergence
            combined_min = min(real_vals.min(), gen_vals.min())
            combined_max = max(real_vals.max(), gen_vals.max())
            bins = np.linspace(combined_min, combined_max, 50)

            real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
            gen_hist, _ = np.histogram(gen_vals, bins=bins, density=True)

            # Normalize to probabilities
            real_hist = (
                real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            )
            gen_hist = gen_hist / gen_hist.sum() if gen_hist.sum() > 0 else gen_hist

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            real_hist = real_hist + epsilon
            gen_hist = gen_hist + epsilon

            js_div = jensenshannon(real_hist, gen_hist)

            # 2. Q-Q Plot Correlation
            # Sort both arrays and compute correlation
            real_sorted = np.sort(real_vals)
            gen_sorted = np.sort(gen_vals)

            # Ensure same length for correlation
            min_len = min(len(real_sorted), len(gen_sorted))
            real_qq = real_sorted[:min_len]
            gen_qq = gen_sorted[:min_len]

            qq_corr, _ = pearsonr(real_qq, gen_qq)

            # Store metrics
            results["column_metrics"][col] = {
                "jensen_shannon_div": js_div,
                "qq_correlation": qq_corr,
                "real_mean": real_vals.mean(),
                "gen_mean": gen_vals.mean(),
                "real_std": real_vals.std(),
                "gen_std": gen_vals.std(),
            }

            # Collect for summary
            js_scores.append(js_div)
            qq_correlations.append(qq_corr)

    # Calculate summary metrics
    results["summary"] = {
        "avg_jensen_shannon": np.mean(js_scores),
        "avg_qq_correlation": np.mean(qq_correlations),
        "n_columns_compared": len(js_scores),
    }

    # Suppressed detailed column analysis output for readability

    # Calculate quality score silently
    quality_score = 0

    # Jensen-Shannon divergence (closer to 0 is better)
    if results["summary"]["avg_jensen_shannon"] < 0.1:
        quality_score += 50
    elif results["summary"]["avg_jensen_shannon"] < 0.3:
        quality_score += 30

    # Q-Q correlation (closer to 1 is better)
    if results["summary"]["avg_qq_correlation"] > 0.9:
        quality_score += 50
    elif results["summary"]["avg_qq_correlation"] > 0.7:
        quality_score += 30

    results["compact_quality_score"] = quality_score

    return results


def compare_real_vs_generated_enhanced(
    real_data, generated_data, equipment_name, kept_features, save_plots=True, n_pca=4
):
    """
    Enhanced comparison between real and generated data with focus on newly predicted features.
    """
    # 1. Compact column-wise comparison (NEW!)
    compact_results = compact_column_wise_comparison(
        real_data, generated_data, equipment_name
    )

    # 2. Overall comparison using existing function
    comparison_stats, base_quality_score = compare_real_vs_generated(
        real_data, generated_data, equipment_name, save_plots, n_pca
    )

    # 2. Focus on newly predicted features
    kept_feature_scores = []

    for feature in kept_features:
        if feature in real_data.columns and feature in generated_data.columns:
            real_vals = real_data[feature].values
            gen_vals = generated_data[feature].values

            # Statistical metrics
            mean_diff_pct = (
                abs(gen_vals.mean() - real_vals.mean()) / abs(real_vals.mean()) * 100
            )
            std_diff_pct = (
                abs(gen_vals.std() - real_vals.std()) / abs(real_vals.std()) * 100
            )

            # Range similarity
            real_range = real_vals.max() - real_vals.min()
            gen_range = gen_vals.max() - gen_vals.min()
            range_diff_pct = (
                abs(gen_range - real_range) / real_range * 100 if real_range > 0 else 0
            )

            # Calculate feature-specific quality score
            feature_score = 0
            if mean_diff_pct < 10:
                feature_score += 33
            elif mean_diff_pct < 20:
                feature_score += 20
            elif mean_diff_pct < 30:
                feature_score += 7

            if std_diff_pct < 15:
                feature_score += 33
            elif std_diff_pct < 30:
                feature_score += 20
            elif std_diff_pct < 50:
                feature_score += 7

            if range_diff_pct < 20:
                feature_score += 34
            elif range_diff_pct < 40:
                feature_score += 20
            elif range_diff_pct < 60:
                feature_score += 7

            kept_feature_scores.append(feature_score)

    # Calculate overall enhanced quality score
    if kept_feature_scores:
        avg_kept_feature_score = np.mean(kept_feature_scores)
        # Weight: 50% base score + 25% newly predicted features + 25% compact column-wise
        enhanced_quality_score = (
            0.5 * base_quality_score
            + 0.25 * avg_kept_feature_score
            + 0.25 * compact_results["compact_quality_score"]
        )
    else:
        # Weight: 70% base score + 30% compact column-wise
        enhanced_quality_score = (
            0.7 * base_quality_score + 0.3 * compact_results["compact_quality_score"]
        )

    return comparison_stats, enhanced_quality_score


def compare_real_vs_generated(
    real_data, generated_data, equipment_name, save_plots=True, n_pca=4
):
    """
    Comprehensive comparison between real and generated data using PCA projection.
    """
    # Ensure both datasets are DataFrames for consistent handling
    if isinstance(real_data, np.ndarray):
        real_data = pd.DataFrame(real_data)
    if isinstance(generated_data, np.ndarray):
        generated_data = pd.DataFrame(generated_data, columns=real_data.columns)

    # Ensure both have same columns
    if list(real_data.columns) != list(generated_data.columns):
        raise ValueError(
            f"Column mismatch between real and generated data for {equipment_name}"
        )

    # =====================================
    # STEP 1.5: COMPONENT STRUCTURE COMPARISON
    # =====================================
    # Use existing evaluate_pc_similarity function to avoid duplication
    avg_component_similarity = evaluate_pc_similarity(real_data, generated_data, n_pca)

    # Display detailed sensor information for understanding
    max_components = min(
        n_pca,
        len(real_data.columns),
        real_data.shape[0] - 1,
        generated_data.shape[0] - 1,
    )
    n_pca_actual = max(1, max_components)

    # Scale datasets and fit PCA for sensor analysis only
    scaler_real = StandardScaler()
    X_real_scaled = scaler_real.fit_transform(real_data)
    pca_real = PCA(n_components=n_pca_actual)
    pca_real.fit(X_real_scaled)

    # Detailed sensor information suppressed for cleaner output

    # Create DataFrames with PCA component names for continued analysis
    scaler_gen = StandardScaler()
    X_gen_scaled = scaler_gen.fit_transform(generated_data)
    pca_gen = PCA(n_components=n_pca_actual)

    X_real_pca = pca_real.fit_transform(X_real_scaled)
    X_gen_pca = pca_gen.fit_transform(X_gen_scaled)

    pca_columns = [f"PC{i+1}" for i in range(n_pca_actual)]
    X_real_pca_df = pd.DataFrame(X_real_pca, columns=pca_columns)
    X_gen_pca_df = pd.DataFrame(X_gen_pca, columns=pca_columns)

    # Get explained variance ratios
    real_explained_var = pca_real.explained_variance_ratio_
    gen_explained_var = pca_gen.explained_variance_ratio_

    # print(f"Real explained variance: {real_explained_var}")
    # print(f"Gen explained variance:  {gen_explained_var}")

    # =====================================
    # STEP 2: STATISTICAL ANALYSIS
    # =====================================
    comparison_stats = []

    for feature in pca_columns:
        # Basic statistics
        real_std = X_real_pca_df[feature].std()
        gen_std = X_gen_pca_df[feature].std()
        real_min = X_real_pca_df[feature].min()
        gen_min = X_gen_pca_df[feature].min()
        real_mean = X_real_pca_df[feature].mean()
        gen_mean = X_gen_pca_df[feature].mean()
        real_max = X_real_pca_df[feature].max()
        gen_max = X_gen_pca_df[feature].max()

        # T-test: tests if two samples have same mean
        t_stat, t_pvalue = ttest_ind(
            X_real_pca_df[feature], X_gen_pca_df[feature], equal_var=False
        )

        # Calculate percentage differences
        std_diff_pct = abs(gen_std - real_std) / real_std * 100 if real_std != 0 else 0
        mean_diff_pct = (
            abs(gen_mean - real_mean) / real_mean * 100 if real_mean != 0 else 0
        )
        # Store comprehensive statistics
        stat_dict = {
            "feature": feature,
            "real_std": real_std,
            "gen_std": gen_std,
            "mean_diff_%": mean_diff_pct,
            "std_diff_%": std_diff_pct,
            "real_range": [real_min, real_max],
            "gen_range": [gen_min, gen_max],
            "t_stat": t_stat,
            "t_pvalue": t_pvalue,
        }

        comparison_stats.append(stat_dict)

    # =====================================
    # STEP 3: CORRELATION ANALYSIS
    # =====================================
    real_corr = X_real_pca_df.corr()
    gen_corr = X_gen_pca_df.corr()
    corr_diff = np.abs(gen_corr - real_corr)

    max_corr_diff = corr_diff.max().max()
    mean_corr_diff = corr_diff.mean().mean()

    # =====================================
    # STEP 4: VISUALIZATIONS (simplified)
    # =====================================
    if save_plots:
        # Create directories if they don't exist
        os.makedirs(
            "/Users/kevinwardakhan/test/dt-stage-GenAI/dataset_comparison_dt",
            exist_ok=True,
        )
        os.makedirs(
            "/Users/kevinwardakhan/test/dt-stage-GenAI/pca_scatter_dt", exist_ok=True
        )

        # Calculate optimal number of bins based on EFFECTIVE data size
        # Use the minimum of real and generated data sizes for fair comparison
        n_real_samples = len(real_data)
        n_gen_samples = len(generated_data)
        effective_samples = min(n_real_samples, n_gen_samples)

        # Calculate optimal binning silently
        n_real_samples = len(real_data)
        n_gen_samples = len(generated_data)
        effective_samples = min(n_real_samples, n_gen_samples)

        # Adaptive binning based on effective sample size
        if effective_samples <= 100:
            n_bins = max(8, int(np.log2(effective_samples)) + 2)
        elif effective_samples <= 500:
            n_bins = max(12, int(np.log2(effective_samples)) + 3)
        elif effective_samples <= 1500:
            n_bins = max(15, int(np.log2(effective_samples)) + 4)
        elif effective_samples <= 3000:
            n_bins = max(20, int(np.log2(effective_samples)) + 5)
        else:
            n_bins = max(25, int(np.log2(effective_samples)) + 6)

        # Additional adjustment for clipped data: reduce bins if data variance is low
        combined_data = pd.concat([X_real_pca_df, X_gen_pca_df])
        data_range_ratio = (
            combined_data.std().mean() / combined_data.mean().abs().mean()
        )
        if data_range_ratio < 0.1:  # Low variance relative to mean
            n_bins = max(5, n_bins // 2)  # Reduce bins for low-variance data

        # ORIGINAL FEATURES HISTOGRAMS - Maximum screen utilization
        original_features = list(real_data.columns)
        n_features = len(original_features)

        # Calculate optimal grid layout for maximum screen usage
        # Aim for roughly square layout, but prioritize more columns for wide screens
        if n_features <= 6:
            n_cols = min(3, n_features)
            n_rows = int(np.ceil(n_features / n_cols))
        elif n_features <= 12:
            n_cols = 4
            n_rows = int(np.ceil(n_features / n_cols))
        elif n_features <= 20:
            n_cols = 5
            n_rows = int(np.ceil(n_features / n_cols))
        elif n_features <= 30:
            n_cols = 6
            n_rows = int(np.ceil(n_features / n_cols))
        else:
            # For very many features, limit to first 30 and use 6 columns
            original_features = original_features[:30]
            n_features = 30
            n_cols = 6
            n_rows = 5
            print(f"⚠️  Showing only first {n_features} features (too many to display)")

        # Calculate figure size for optimal screen usage
        fig_width = min(20, max(12, n_cols * 3))  # 3 inches per column, max 20
        fig_height = min(16, max(8, n_rows * 2.5))  # 2.5 inches per row, max 16

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Creating histograms (display info suppressed for clean output)

        # Create histograms for each ORIGINAL feature
        for i, feature_name in enumerate(original_features):
            plt.subplot(n_rows, n_cols, i + 1)

            # Get data for this original feature
            real_feature_data = real_data[feature_name]
            gen_feature_data = generated_data[feature_name]

            # Calculate common range for consistent binning
            combined_min = min(real_feature_data.min(), gen_feature_data.min())
            combined_max = max(real_feature_data.max(), gen_feature_data.max())

            # Use common bins range to ensure fair comparison
            bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

            # Create histograms with consistent binning
            plt.hist(
                real_feature_data,
                bins=bin_edges,
                alpha=0.6,
                label="Real",
                density=True,
                color="blue",
                edgecolor="black",
                linewidth=0.3,
            )
            plt.hist(
                gen_feature_data,
                bins=bin_edges,
                alpha=0.6,
                label="Generated",
                density=True,
                color="red",
                edgecolor="black",
                linewidth=0.3,
            )

            # Add compact title and labels
            plt.title(f"{feature_name}", fontsize=9, pad=5)
            plt.ylabel("Density", fontsize=8)
            plt.legend(fontsize=7, loc="upper right")

            plt.grid(True, alpha=0.2)
            plt.tick_params(axis="both", labelsize=7)

            # Rotate x-axis labels if they're too long
            plt.xticks(rotation=45 if len(str(combined_max)) > 6 else 0)

        plt.tight_layout(pad=1.0)
        plot_path = f"/Users/kevinwardakhan/test/dt-stage-GenAI/dataset_comparison_dt/original_features_comparison_{equipment_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()  # Close plot instead of showing
        # Plot saved silently

        # =====================================
        # PC HISTOGRAMS AND SCATTER PLOTS
        # =====================================

        # 1. PC Histograms (PC1 to PC4)
        n_pc_to_show = min(4, n_pca_actual)
        if n_pc_to_show > 0:
            fig_pc, axes_pc = plt.subplots(2, 2, figsize=(12, 10))
            axes_pc = axes_pc.flatten()

            for i in range(n_pc_to_show):
                pc_name = f"PC{i+1}"

                # Get data for this PC
                real_pc_data = (
                    X_real_pca_df[pc_name] if pc_name in X_real_pca_df.columns else []
                )
                gen_pc_data = (
                    X_gen_pca_df[pc_name] if pc_name in X_gen_pca_df.columns else []
                )

                if len(real_pc_data) > 0 and len(gen_pc_data) > 0:
                    # Calculate common range for consistent binning
                    combined_min = min(real_pc_data.min(), gen_pc_data.min())
                    combined_max = max(real_pc_data.max(), gen_pc_data.max())

                    # Create histogram
                    axes_pc[i].hist(
                        real_pc_data,
                        bins=n_bins,
                        alpha=0.6,
                        label="Real",
                        density=True,
                        color="blue",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    axes_pc[i].hist(
                        gen_pc_data,
                        bins=n_bins,
                        alpha=0.6,
                        label="Generated",
                        density=True,
                        color="red",
                        edgecolor="black",
                        linewidth=0.5,
                    )

                    # Add explained variance in title
                    real_var = (
                        real_explained_var[i] if i < len(real_explained_var) else 0
                    )
                    gen_var = gen_explained_var[i] if i < len(gen_explained_var) else 0

                    axes_pc[i].set_title(
                        f"{pc_name} - Real: {real_var:.1%}, Gen: {gen_var:.1%}",
                        fontsize=10,
                        pad=8,
                    )
                    axes_pc[i].set_ylabel("Density", fontsize=9)
                    axes_pc[i].legend(fontsize=8)
                    axes_pc[i].grid(True, alpha=0.3)
                else:
                    # Hide empty subplot
                    axes_pc[i].set_visible(False)

            # Hide unused subplots
            for j in range(n_pc_to_show, 4):
                axes_pc[j].set_visible(False)

            plt.suptitle(
                f"Principal Components Comparison - {equipment_name}",
                fontsize=14,
                y=0.98,
            )
            plt.tight_layout()

            # Save PC histograms
            pc_plot_path = f"/Users/kevinwardakhan/test/dt-stage-GenAI/pca_scatter_dt/pc_histograms_{equipment_name}.png"
            plt.savefig(pc_plot_path, dpi=300, bbox_inches="tight")
            plt.show()

        # 2. Scatter Plots (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3, PC3 vs PC4)
        if n_pca_actual >= 2:
            fig_scatter, axes_scatter = plt.subplots(2, 2, figsize=(15, 12))
            axes_scatter = axes_scatter.flatten()

            # Define PC pairs for scatter plots
            pc_pairs = [
                ("PC1", "PC2"),
                ("PC1", "PC3") if n_pca_actual >= 3 else ("PC1", "PC2"),
                ("PC2", "PC3") if n_pca_actual >= 3 else ("PC1", "PC2"),
                (
                    ("PC3", "PC4")
                    if n_pca_actual >= 4
                    else ("PC2", "PC3") if n_pca_actual >= 3 else ("PC1", "PC2")
                ),
            ]

            for idx, (pc_x, pc_y) in enumerate(pc_pairs):
                if pc_x in X_real_pca_df.columns and pc_y in X_real_pca_df.columns:
                    # Real data scatter
                    axes_scatter[idx].scatter(
                        X_real_pca_df[pc_x],
                        X_real_pca_df[pc_y],
                        alpha=0.6,
                        s=20,
                        color="blue",
                        label="Real",
                        edgecolors="navy",
                        linewidth=0.3,
                    )

                    # Generated data scatter
                    axes_scatter[idx].scatter(
                        X_gen_pca_df[pc_x],
                        X_gen_pca_df[pc_y],
                        alpha=0.6,
                        s=20,
                        color="red",
                        label="Generated",
                        edgecolors="darkred",
                        linewidth=0.3,
                    )

                    axes_scatter[idx].set_xlabel(f"{pc_x}", fontsize=10)
                    axes_scatter[idx].set_ylabel(f"{pc_y}", fontsize=10)
                    axes_scatter[idx].set_title(f"{pc_x} vs {pc_y}", fontsize=11, pad=8)
                    axes_scatter[idx].legend(fontsize=9)
                    axes_scatter[idx].grid(True, alpha=0.3)

                else:
                    # Hide subplot if PCs don't exist
                    axes_scatter[idx].set_visible(False)

            plt.suptitle(f"PC Scatter Plots - {equipment_name}", fontsize=14, y=0.98)
            plt.tight_layout()

            # Save scatter plots
            scatter_plot_path = f"/Users/kevinwardakhan/test/dt-stage-GenAI/pca_scatter_dt/pc_scatter_{equipment_name}.png"
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
            plt.show()

    # =====================================
    # STEP 5: ENHANCED QUALITY ASSESSMENT
    # =====================================
    # Calculate aggregate quality metrics
    mean_diffs = [stat["mean_diff_%"] for stat in comparison_stats]
    std_diffs = [stat["std_diff_%"] for stat in comparison_stats]

    avg_mean_diff = np.mean(mean_diffs)
    avg_std_diff = np.mean(std_diffs)

    # Enhanced quality score calculation (0-100 scale)
    quality_score = 0

    # Component structure similarity (40 points): MOST IMPORTANT
    # This is the key insight - if PC structures are different, it's a bad sign
    if avg_component_similarity > 0.8:
        quality_score += 40
    elif avg_component_similarity > 0.6:
        quality_score += 25
    elif avg_component_similarity > 0.4:
        quality_score += 10
    else:
        quality_score += 0

    # Mean similarity (30 points): good if mean difference < 15%
    if avg_mean_diff < 15:
        quality_score += 30
    elif avg_mean_diff < 30:
        quality_score += 15

    # Standard deviation similarity (30 points): good if std difference < 25%
    if avg_std_diff < 25:
        quality_score += 30
    elif avg_std_diff < 50:
        quality_score += 15

    # Add component similarity to comparison stats for detailed analysis
    comparison_stats.append(
        {
            "metric": "avg_component_similarity",
            "value": avg_component_similarity,
            "real_explained_var": real_explained_var.tolist(),
            "gen_explained_var": gen_explained_var.tolist(),
        }
    )

    return comparison_stats, quality_score


if __name__ == "__main__":
    X_virs = digital_twin_carboptim(
        phate_dim=4,
        n_output=1500,
        csv_path="/Users/kevinwardakhan/test/dt-stage-GenAI/dataset/carboptim/DHC_consolidated.parquet",
        config_file_path="dataset/carboptim/DHC_unit_config.json",
        Q=1,
    )

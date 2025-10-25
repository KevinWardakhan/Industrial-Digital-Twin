# ============================================================================
# PHATE+PGAN VSG (Virtual Sample Generation) Implementation for CarbOptim
# ============================================================================
# This implementation combines PHATE (Potential of Heat Diffusion for
# Affinity Transition Embedding) with Progressive GAN for generating
# synthetic industrial process data from the CarbOptim dataset.
#
# Key Components:
# 1. PHATE: Non-linear dimensionality reduction preserving data topology
# 2. PGAN: Progressive GAN that generates realistic synthetic samples
# 3. Sparse Generation: Algorithm to create diverse virtual samples
# 4. Statistical Validation: Comprehensive comparison between real and synthetic data
# ============================================================================

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
    get_equipment_features,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import warnings
from scipy.stats import ks_2samp, ttest_ind

# Configure computing device 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
warnings.filterwarnings("ignore")


class DataProcessor:
    """
    Processes raw Parquet files from CarbOptim dataset into usable data for ML models.

    This class handles:
    - Loading parquet files containing industrial process data
    - Column name normalization (removing timestamp suffixes)
    - Equipment-specific feature extraction using DHC_unit_config.json
    - Data cleaning (missing value imputation using backward fill)
    - Sample size limiting for computational efficiency

    Attributes:
        csv_path (str): Path to the parquet file containing the dataset
        n_output (int): Maximum number of samples to extract from the dataset
        equipment_name (str): Name of the equipment to extract features for (e.g., "F101", "C106")
        config_file_path (str): Path to the equipment configuration JSON file
    """

    def __init__(self, csv_path, n_output, equipment_name, config_file_path):
        self.csv_path = csv_path
        self.n_output = (
            n_output  # Limit samples for computational efficiency and memory management
        )
        self.equipment_name = equipment_name
        self.config_file_path = config_file_path

    def load_and_process_data_test(self):
        """
        Loads and preprocesses the industrial process data.

        Processing steps:
        1. Load parquet file (more efficient than CSV for large datasets)
        2. Normalize column names by removing timestamp suffixes (e.g., "90A1920-2023-01-01" -> "90A1920")
        3. Extract equipment-specific features using the configuration file
        4. Handle missing values using backward fill (appropriate for time series data)
        5. Limit to specified number of samples

        Returns:
            pd.DataFrame: Processed equipment data with shape (n_output, n_features)
                         where n_features is the number of features for the specified equipment
        """
        # Load the industrial process data (parquet format for efficiency)
        data = pd.read_parquet(self.csv_path)

        # Normalize column names: remove timestamp suffixes to match equipment config
        # Example: "90A1920-2023-01-01" becomes "90A1920"
        new_col_names = [col.split("-")[0] for col in data.columns]
        data.columns = new_col_names

        # Extract only the features relevant to the specified equipment
        # This uses the DHC_unit_config.json to identify which columns belong to which equipment
        equipment_columns = get_equipment_feature_names(
            self.equipment_name, config_file_path=self.config_file_path
        )
        equipment_data = data[equipment_columns]

        # Handle missing values using backward fill
        # This is appropriate for industrial time series where missing values
        # can be reasonably interpolated from future observations
        equipment_data = equipment_data.fillna(method="bfill")

        # Return only the first n_output samples to manage computational complexity
        return equipment_data.iloc[self.n_output + 1 : 2 * self.n_output + 1]

    def load_and_process_data(self):
        """
        Loads and preprocesses the industrial process data.

        Processing steps:
        1. Load parquet file (more efficient than CSV for large datasets)
        2. Normalize column names by removing timestamp suffixes (e.g., "90A1920-2023-01-01" -> "90A1920")
        3. Extract equipment-specific features using the configuration file
        4. Handle missing values using backward fill (appropriate for time series data)
        5. Limit to specified number of samples

        Returns:
            pd.DataFrame: Processed equipment data with shape (n_output, n_features)
                         where n_features is the number of features for the specified equipment
        """
        # Load the industrial process data (parquet format for efficiency)
        data = pd.read_parquet(self.csv_path)

        # Normalize column names: remove timestamp suffixes to match equipment config
        # Example: "90A1920-2023-01-01" becomes "90A1920"
        new_col_names = [col.split("-")[0] for col in data.columns]
        data.columns = new_col_names

        # Extract only the features relevant to the specified equipment
        # This uses the DHC_unit_config.json to identify which columns belong to which equipment
        equipment_columns = get_equipment_feature_names(
            self.equipment_name, config_file_path=self.config_file_path
        )
        equipment_data = data[equipment_columns]

        # Handle missing values using backward fill
        # This is appropriate for industrial time series where missing values
        # can be reasonably interpolated from future observations
        equipment_data = equipment_data.fillna(method="bfill")

        # Return only the first n_output samples to manage computational complexity
        return equipment_data.iloc[: self.n_output]


class PHATEEmbedder:
    """
    PHATE (Potential of Heat Diffusion for Affinity Transition Embedding) wrapper.

    PHATE is a dimensionality reduction technique that preserves both local and global
    structure in high-dimensional data. Unlike PCA or t-SNE, PHATE:
    - Preserves continuous progressions and trajectories in data
    - Maintains both local neighborhoods and global geometry
    - Is particularly effective for time series and process data
    - Handles non-linear manifolds better than linear methods

    This class provides a scikit-learn style interface with data scaling integration.

    Attributes:
        n_components (int): Target dimensionality for the embedding
        phate_operator: The fitted PHATE transformer
        scaler: StandardScaler for data normalization
        is_fitted (bool): Flag to track if the embedder has been fitted
    """

    def __init__(self, n_components, knn, n_pca=None, n_jobs=-1):
        """
        Initialize PHATE embedder with specified parameters.

        Args:
            n_components (int): Number of dimensions for the embedding (typically 2-4)
            knn (int): Number of nearest neighbors for graph construction (affects local structure)
            n_pca (int, optional): Number of PCA components for preprocessing (reduces noise)
            n_jobs (int): Number of parallel jobs (-1 uses all available cores)
        """
        self.n_components = n_components
        self.phate_operator = phate.PHATE(
            n_components=n_components,
            n_pca=n_pca,  # Optional PCA preprocessing to reduce noise
            knn=knn,  # Controls local neighborhood size
            n_jobs=n_jobs,
            random_state=24,  # For reproducible results
        )
        self.scaler = StandardScaler()  # Essential for PHATE to work properly
        self.is_fitted = False

    def fit_transform(self, X):
        """
        Fit PHATE on data and return the embedding.

        The process:
        1. Standardize features (mean=0, std=1) - critical for PHATE performance
        2. Apply PHATE transformation to reduce dimensionality
        3. Return low-dimensional embedding that preserves data topology

        Args:
            X (array-like): Input data of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: PHATE embedding of shape (n_samples, n_components)
        """
        # Standardization is crucial for PHATE as it's distance-based
        X_scaled = self.scaler.fit_transform(X)

        # Apply PHATE transformation - this captures the intrinsic geometry
        Z_ori = self.phate_operator.fit_transform(X_scaled)
        self.is_fitted = True
        return Z_ori

    def transform(self, X_new):
        """
        Transform new data using the already fitted PHATE embedder.

        This allows embedding new samples into the same low-dimensional space
        as the training data, maintaining consistency for virtual sample generation.

        Args:
            X_new (array-like): New data to transform

        Returns:
            numpy.ndarray: PHATE embedding of the new data

        Raises:
            ValueError: If the embedder hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("PHATEEmbedder must be fitted first before transform!")

        # Use the same scaler and PHATE transformation as training
        X_scaled = self.scaler.transform(X_new)
        Z_new = self.phate_operator.transform(X_scaled)
        return Z_new


class PHATEGenerator(nn.Module):
    """
    Neural network generator for the PGAN architecture.

    This generator takes low-dimensional PHATE embeddings as input and generates
    high-dimensional synthetic data that mimics the original industrial process data.

    Architecture rationale:
    - Input: PHATE embedding (typically 2-4 dimensions)
    - Progressive expansion through fully connected layers
    - LeakyReLU activations for stable gradient flow
    - No final activation to allow full range of values

    The generator learns the mapping from PHATE space (which captures data topology)
    back to the original feature space, enabling realistic synthetic data generation.
    """

    def __init__(self, phate_dim, n_samples, n_features, hidden_dim=32):
        """
        Generator with complex architecture matching existing trained models:
        phate_dim -> 32 -> 64 -> 128 -> n_features
        """
        super(PHATEGenerator, self).__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.model = nn.Sequential(
            nn.Linear(phate_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, n_features),
        )

    def forward(self, Z_ori):
        """
        Forward pass through the generator.

        Args:
            Z_ori (torch.Tensor): PHATE embedding batch of shape (batch_size, phate_dim)

        Returns:
            torch.Tensor: Generated synthetic data of shape (batch_size, n_features)
        """
        # Generate synthetic data from PHATE embedding
        output = self.model(Z_ori)
        return output


class PHATEDiscriminator(nn.Module):
    """
    Discriminator network for the PGAN architecture.

    This discriminator evaluates pairs of (PHATE_embedding, feature_data) to determine
    whether the feature data is real or generated. It implements a dual-branch architecture:

    1. PHATE Branch: Processes the low-dimensional PHATE embedding
    2. Feature Branch: Processes the high-dimensional feature data
    3. Fusion Layer: Combines both branches to make the final decision

    This design allows the discriminator to consider both the topology (via PHATE)
    and the actual feature values when making authenticity judgments.
    """

    def __init__(self, phate_dim, n_samples, n_features, hidden_dim=32):
        """
        Discriminator with dual-branch architecture matching existing trained models
        """
        super(PHATEDiscriminator, self).__init__()
        self.n_samples = n_samples
        self.n_features = n_features

        # PHATE branch: processes PHATE embeddings
        self.phate_branche = nn.Sequential(
            nn.Linear(phate_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
        )

        # Feature branch: processes feature data
        # Architecture must match exactly: x_branch.0, x_branch.3, x_branch.6
        self.x_branch = nn.Sequential(
            nn.Linear(n_features, 256),  # 0: x_branch.0
            nn.LeakyReLU(0.2),  # 1
            nn.LeakyReLU(0.2),  # 2
            nn.Linear(256, 128),  # 3: x_branch.3
            nn.LeakyReLU(0.2),  # 4
            nn.LeakyReLU(0.2),  # 5
            nn.Linear(128, 128),  # 6: x_branch.6
        )

        # Fusion layer: combines both branches
        # Architecture must match exactly: fusion.0, fusion.3, fusion.6, fusion.9
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 0: fusion.0 (128 from phate + 128 from features)
            nn.LeakyReLU(0.2),  # 1
            nn.LeakyReLU(0.2),  # 2
            nn.Linear(128, 64),  # 3: fusion.3
            nn.LeakyReLU(0.2),  # 4
            nn.LeakyReLU(0.2),  # 5
            nn.Linear(64, 32),  # 6: fusion.6
            nn.LeakyReLU(0.2),  # 7
            nn.LeakyReLU(0.2),  # 8
            nn.Linear(32, 1),  # 9: fusion.9
            nn.Sigmoid(),  # 10
        )

    def forward(self, Z_ori, X_ori):
        """
        Forward pass through the dual-branch discriminator.
        Args:
            Z_ori (torch.Tensor): PHATE embedding (batch_size, phate_dim)
            X_ori (torch.Tensor): Feature data (batch_size, n_features)
        Returns:
            torch.Tensor: Probability that the (Z_ori, X_ori) pair is real, shape (batch_size,)
        """
        # Process PHATE embedding through PHATE branch
        phate_out = self.phate_branche(Z_ori)

        # Process features through feature branch
        x_out = self.x_branch(X_ori)

        # Concatenate and pass through fusion layer
        combined = torch.cat([phate_out, x_out], dim=1)
        prob = self.fusion(combined)
        return prob.squeeze()


class SparseGeneration:
    """
    Implements the Sparse Generation algorithm for creating diverse virtual samples.

    This algorithm generates virtual samples in the PHATE embedding space that:
    1. Are sufficiently far from existing samples (maintains diversity)
    2. Cover unexplored regions of the embedding space
    3. Follow a principled approach to maximize spatial distribution

    The algorithm is based on the paper's sparse generation method, which ensures
    that synthetic samples don't just interpolate between existing points but
    explore new regions of the data manifold.

    Key Parameters:
    - Q: Number of generation cycles (more cycles = more virtual samples)
    - W: Number of candidate samples per original sample (exploration breadth)
    - R: Radius parameter controlling the generation distance
    """

    def __init__(self, Q, phate_dim, n_samples, W):
        """
        Initialize the sparse generation algorithm.

        Args:
            Q (int): Number of generation cycles - each cycle generates n_samples new points
            phate_dim (int): Dimensionality of the PHATE embedding space
            n_samples (int): Number of original samples
            W (int): Number of candidate samples generated per original sample
        """
        self.n_samples = n_samples
        self.Q = Q  # Number of cycles for the original feature set
        self.phate_dim = phate_dim  # Dimension of PHATE features
        self.W = W  # Number of candidate features per original feature

    def sparse_generation(self, Z_ori):
        """
        Generate virtual features using the sparse generation algorithm.

        Algorithm overview:
        1. For each cycle Q:
           2. For each original sample i:
              3. Generate W candidate samples around sample i
              4. Select the candidate that maximizes minimum distance to all existing samples
              5. Add the best candidate to the virtual set

        This ensures that virtual samples are:
        - Diverse (far from existing samples)
        - Well-distributed across the embedding space
        - Not clustered around any particular region

        Args:
            Z_ori (numpy.ndarray): Original PHATE embedding, shape (N, phate_dim)

        Returns:
            numpy.ndarray: Virtual PHATE embedding, shape (Q*N, phate_dim)
        """
        N = Z_ori.shape[0]  # Number of original features

        # Initialize empty virtual feature set
        Z_vir = np.array([]).reshape(0, self.phate_dim)
        # Z_all tracks both original and virtual features for distance calculations
        Z_all = Z_ori.copy()  # Z_all = Z_ori ∪ Z_vir (initially only Z_ori)

        print(f"Starting sparse generation: Q={self.Q}, N={N}, W={self.W}")

        # === STEP 1: Compute distance statistics ===
        # Calculate pairwise distances to determine generation radius
        D = euclidean_distances(Z_ori)
        D_sorted = np.sort(D, axis=1)
        # Use average minimum distance as base radius for generation
        R = np.mean(D_sorted[:, 1])  # D_sorted[:, 0] would be distance to self (0)

        print(f"Computed generation radius R = {R}")

        # === STEP 2: Main generation loop ===
        num = 0  # Number of completed cycles

        while num < self.Q:
            print(f"Cycle {num + 1}/{self.Q}: Processing {N} samples...")

            # For each original sample, generate one virtual sample
            for i in range(1, N + 1):  # 1-indexed to match paper notation
                # Show progress for long-running generations
                if i % 50 == 0 or i == N:
                    print(f"  Progress: {i}/{N} samples processed")

                # === STEP 3: Generate W candidate features ===
                C = []  # List to store candidate features

                for j in range(1, self.W + 1):
                    # Generate random spherical coordinates for candidate placement
                    s = np.random.uniform(0, R)  # Distance from original point
                    alpha = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
                    beta = np.random.uniform(0, 2 * np.pi)  # Polar angle

                    # Generate candidate based on dimensionality
                    if self.phate_dim == 2:
                        # 2D case: use polar coordinates
                        cj = np.array(
                            [
                                Z_ori[i - 1, 0] + s * np.cos(alpha),
                                Z_ori[i - 1, 1] + s * np.sin(alpha),
                            ]
                        )
                    elif self.phate_dim == 3:
                        # 3D case: use spherical coordinates
                        cj = np.array(
                            [
                                Z_ori[i - 1, 0] + s * np.sin(alpha) * np.cos(beta),
                                Z_ori[i - 1, 1] + s * np.sin(alpha) * np.sin(beta),
                                Z_ori[i - 1, 2] + s * np.cos(alpha),
                            ]
                        )
                    else:
                        # Higher dimensions: generate random direction
                        cj = Z_ori[i - 1].copy()
                        for dim in range(self.phate_dim):
                            angle = np.random.uniform(0, 2 * np.pi)
                            cj[dim] += s * np.sin(angle)

                    C.append(cj)

                # === STEP 4: Select best candidate ===
                # Choose candidate that maximizes minimum distance to existing points
                # This ensures maximum diversity in the virtual sample set
                best_idx = 0
                max_min_dist = 0

                for j, cj in enumerate(C):
                    # Calculate distances from candidate to all existing points
                    distances = [np.linalg.norm(cj - z_point) for z_point in Z_all]
                    min_dist = min(distances)  # Minimum distance to any existing point

                    # Keep track of candidate with maximum minimum distance
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = j

                # === STEP 5: Add best candidate to virtual set ===
                if best_idx < len(C):
                    best_candidate = C[best_idx]
                    Z_vir = np.vstack([Z_vir, best_candidate.reshape(1, -1)])
                    Z_all = np.vstack([Z_all, best_candidate.reshape(1, -1)])

            num += 1

        print(f"Sparse generation completed! Generated {len(Z_vir)} virtual samples")
        print(f"Virtual samples shape: {Z_vir.shape}")

        return Z_vir


class PGANtraining:
    """
    Training class for PHATE-GAN (PGAN) architecture.

    This class implements the training procedure for a GAN that operates in PHATE space:
    1. Generator: Maps PHATE embeddings → Original feature space
    2. Discriminator: Evaluates (PHATE embedding, features) pairs for authenticity

    The training uses adversarial loss combined with L1 reconstruction loss to ensure:
    - Generated samples fool the discriminator (adversarial training)
    - Generated samples resemble the original data distribution (L1 loss)
    - Training stability through appropriate learning rates and label smoothing

    Key innovations:
    - Point-to-point correspondence between PHATE embeddings and features
    - Dual-loss training for better reconstruction quality
    - Regularization to prevent mode collapse
    """

    def __init__(self, lambda_l1, phate_dim, n_samples, n_features, lr):
        """
        Initialize PGAN training components.

        Args:
            lambda_l1 (float): Weight for L1 reconstruction loss (balances adversarial vs reconstruction)
            phate_dim (int): Dimension of PHATE embedding
            n_samples (int): Number of training samples
            n_features (int): Number of features in original data
            lr (float): Base learning rate
        """
        self.lambda_l1 = lambda_l1

        # Initialize networks and move to appropriate device
        self.Discriminator = PHATEDiscriminator(
            phate_dim=phate_dim, n_samples=n_samples, n_features=n_features
        ).to(device)
        self.Generator = PHATEGenerator(
            phate_dim=phate_dim, n_samples=n_samples, n_features=n_features
        ).to(device)

        # Optimizers with different learning rates for stability
        # Generator learns faster to catch up with discriminator
        self.opt_G = optim.Adam(
            self.Generator.parameters(),
            lr=lr * 5,  # Higher learning rate for generator
            betas=(0.5, 0.99),  # β1=0.5 is common for GANs
        )
        self.opt_D = optim.Adam(
            self.Discriminator.parameters(), lr=lr, betas=(0.5, 0.99)
        )

        # Loss functions
        self.bce_loss = nn.BCELoss()  # Binary cross-entropy for adversarial loss
        self.l1_loss = nn.L1Loss()  # L1 loss for reconstruction quality

        # Training history tracking
        self.history = {"G_loss": [], "D_loss": []}

    def train(self, X_ori, Z_ori, batch_size, epochs):
        """
        Train the PGAN using alternating generator and discriminator updates.

        Training procedure:
        1. Train Discriminator: Learn to distinguish real vs fake (PHATE, feature) pairs
        2. Train Generator: Learn to fool discriminator while reconstructing features

        Args:
            X_ori (pd.DataFrame or np.array): Original feature data
            Z_ori (np.array): PHATE embeddings corresponding to X_ori
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
        """
        # Ensure networks are on correct device
        self.Generator.to(device)
        self.Discriminator.to(device)

        # Convert data to tensors and move to device
        if isinstance(X_ori, pd.DataFrame):
            X_ori_tensor = torch.FloatTensor(X_ori.values).to(device)
        else:
            X_ori_tensor = torch.FloatTensor(X_ori).to(device)
        Z_tensor = torch.FloatTensor(Z_ori).to(device)

        # Create dataset with point-to-point correspondence
        # Each PHATE embedding Z[i] corresponds to feature vector X[i]
        dataset = TensorDataset(Z_tensor, X_ori_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training diversity
            num_workers=0,  # Single-threaded for stability
        )

        print("Starting PGAN training...")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        print(f"Dataset size: {len(dataset)} samples")

        for epoch in range(epochs):
            d_losses = []
            g_losses = []

            for batch_idx, (z_batch, x_batch) in enumerate(dataloader):
                current_batch_size = z_batch.shape[0]

                # =====================================
                # TRAIN DISCRIMINATOR
                # =====================================
                self.opt_D.zero_grad()

                # Generate fake samples
                X_vir = self.Generator(z_batch)

                # Label smoothing for training stability
                # Real labels: 0.9 instead of 1.0 (reduces overconfidence)
                # Fake labels: 0.1 instead of 0.0 (adds noise for robustness)
                real_labels = torch.ones(current_batch_size).to(device) * 0.9
                fake_labels = torch.zeros(current_batch_size).to(device) + 0.1

                # Discriminator predictions
                real_output = self.Discriminator(z_batch, x_batch)  # Real pairs
                fake_output = self.Discriminator(
                    z_batch, X_vir.detach()
                )  # Fake pairs (.detach() prevents generator gradients)

                # Discriminator loss: maximize log(D(real)) + log(1-D(fake))
                real_loss = self.bce_loss(real_output, real_labels)
                fake_loss = self.bce_loss(fake_output, fake_labels)
                d_loss = real_loss + fake_loss

                # Update discriminator
                d_loss.backward()
                self.opt_D.step()

                # =====================================
                # TRAIN GENERATOR
                # =====================================
                self.opt_G.zero_grad()

                # Generate samples (no .detach() this time - we want gradients)
                x_fake = self.Generator(z_batch)
                d_fake = self.Discriminator(z_batch, x_fake)

                # Generator adversarial loss: maximize log(D(G(z)))
                # (Equivalent to minimizing log(1-D(G(z))))
                g_loss_adv = self.bce_loss(d_fake, real_labels)

                # L1 reconstruction loss: encourages generated samples to match real ones
                l1_loss = self.l1_loss(x_fake, x_batch)

                # Regularization loss: prevents extreme values
                reg_loss = 0.01 * torch.mean(torch.abs(x_fake))

                # Combined generator loss
                g_loss = g_loss_adv + self.lambda_l1 * l1_loss + reg_loss

                # Update generator
                g_loss.backward()
                self.opt_G.step()

                # Record losses for monitoring
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            # Calculate average losses for this epoch
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)

            # Store in history for later analysis
            self.history["D_loss"].append(avg_d_loss)
            self.history["G_loss"].append(avg_g_loss)

            # Print progress periodically
            if epoch % 25 == 0:
                print(
                    f"  Epoch {epoch}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}"
                )

            # Additional progress info for long batches
            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch}/{epochs} - Batch index: {batch_idx} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}"
                )

        print("Training completed!")
        print(
            f"Final - G_loss: {self.history['G_loss'][-1]:.4f}, D_loss: {self.history['D_loss'][-1]:.4f}"
        )

    def generate_virtual_data(self, Z_vir):
        """
        Generate virtual feature data from virtual PHATE embeddings.

        This method uses the trained generator to map virtual PHATE embeddings
        (created by sparse generation) back to the original feature space.

        Args:
            Z_vir (np.array): Virtual PHATE embeddings from sparse generation

        Returns:
            np.array: Generated virtual feature data in original space
        """
        self.Generator.eval()  # Set to evaluation mode (disables dropout, etc.)

        with torch.no_grad():  # Disable gradient computation for inference
            Z_tensor = torch.FloatTensor(Z_vir).to(device)
            X_virtual = self.Generator(Z_tensor)  # Generate virtual features

        self.Generator.train()  # Return to training mode
        return X_virtual.cpu().numpy()  # Move back to CPU and convert to numpy


def compare_datasets(X_ori, X_vir, save_plots=True, n_pca=4):
    """
    Comprehensive comparison between original and virtual datasets using PCA projection.

    This function performs statistical validation of the generated synthetic data by:
    1. Applying PCA to reduce dimensionality and focus on main variations
    2. Computing statistical similarity metrics (means, std deviations, distributions)
    3. Performing statistical tests (Kolmogorov-Smirnov, t-tests)
    4. Analyzing correlation structures
    5. Generating visualizations for manual inspection
    6. Computing an overall quality score

    The PCA projection helps by:
    - Reducing noise and focusing on main data patterns
    - Making visualization more manageable
    - Capturing the most important variations in fewer dimensions

    Args:
        X_ori (pd.DataFrame): Original dataset
        X_vir (pd.DataFrame or np.array): Virtual/synthetic dataset
        save_plots (bool): Whether to save comparison visualizations
        n_pca (int): Number of PCA components for analysis (default: 4)

    Returns:
        tuple: (comparison_stats, quality_score)
            - comparison_stats: Detailed statistics for each PCA component
            - quality_score: Overall quality score (0-100)
    """

    # Ensure both datasets are DataFrames for consistent handling
    if isinstance(X_vir, np.ndarray):
        X_vir = pd.DataFrame(X_vir, columns=X_ori.columns)

    print("\n" + "=" * 60)
    print("DATASET COMPARISON ANALYSIS (PCA PROJECTED)")
    print("=" * 60)

    # =====================================
    # STEP 1: PCA PROJECTION
    # =====================================
    print(f"\nApplying PCA with {n_pca} components...")

    # Combine datasets for consistent scaling and PCA fitting
    # This ensures both datasets are transformed using the same parameters
    X_combined = pd.concat([X_ori, X_vir], axis=0)
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    # Apply PCA to capture main variations
    pca = PCA(n_components=n_pca)
    X_combined_pca = pca.fit_transform(X_combined_scaled)

    # Split back into original and virtual datasets
    n_ori = len(X_ori)
    X_ori_pca = X_combined_pca[:n_ori]
    X_vir_pca = X_combined_pca[n_ori:]

    # Create DataFrames with PCA component names
    pca_columns = [f"PC{i+1}" for i in range(n_pca)]
    X_ori_pca_df = pd.DataFrame(X_ori_pca, columns=pca_columns)
    X_vir_pca_df = pd.DataFrame(X_vir_pca, columns=pca_columns)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # =====================================
    # STEP 2: STATISTICAL ANALYSIS
    # =====================================
    print("\nBASIC STATISTICS (PCA COMPONENTS)")
    print("-" * 30)

    feature_names = pca_columns
    comparison_stats = []

    for feature in feature_names:
        # Basic statistics
        ori_mean = X_ori_pca_df[feature].mean()
        vir_mean = X_vir_pca_df[feature].mean()
        ori_std = X_ori_pca_df[feature].std()
        vir_std = X_vir_pca_df[feature].std()
        ori_min = X_ori_pca_df[feature].min()
        vir_min = X_vir_pca_df[feature].min()
        ori_max = X_ori_pca_df[feature].max()
        vir_max = X_vir_pca_df[feature].max()

        # Statistical tests for distribution similarity
        # Kolmogorov-Smirnov test: tests if two samples come from same distribution
        ks_stat, ks_pvalue = ks_2samp(
            X_ori_pca_df[feature], X_vir_pca_df[feature], method="exact"
        )

        # T-test: tests if two samples have same mean (assuming different variances)
        t_stat, t_pvalue = ttest_ind(
            X_ori_pca_df[feature], X_vir_pca_df[feature], equal_var=False
        )

        # Store comprehensive statistics
        comparison_stats.append(
            {
                "feature": feature,
                "ori_mean": ori_mean,
                "vir_mean": vir_mean,
                "mean_diff_%": (
                    abs(vir_mean - ori_mean) / abs(ori_mean) * 100
                    if ori_mean != 0
                    else 0
                ),
                "ori_std": ori_std,
                "vir_std": vir_std,
                "std_diff_%": (
                    abs(vir_std - ori_std) / ori_std * 100 if ori_std != 0 else 0
                ),
                "ori_range": [ori_min, ori_max],
                "vir_range": [vir_min, vir_max],
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "t_stat": t_stat,
                "t_pvalue": t_pvalue,
            }
        )

    # =====================================
    # STEP 3: CORRELATION ANALYSIS
    # =====================================
    print(f"\nCORRELATIONS (PCA COMPONENTS)")
    print("-" * 30)

    # Analyze how well correlation structure is preserved
    ori_corr = X_ori_pca_df.corr()
    vir_corr = X_vir_pca_df.corr()
    corr_diff = np.abs(vir_corr - ori_corr)

    print(f"Max correlation difference: {corr_diff.max().max():.3f}")
    print(f"Mean correlation difference: {corr_diff.mean().mean():.3f}")

    # =====================================
    # STEP 4: VISUALIZATIONS
    # =====================================
    if save_plots:
        print(f"\nGenerating comparison plots...")

        n_features = len(feature_names)
        fig = plt.figure(figsize=(16, 8))

        # Distribution comparisons (top row)
        for i, feature in enumerate(feature_names):
            plt.subplot(2, n_features, i + 1)

            # Density histograms for distribution comparison
            plt.hist(
                X_ori_pca_df[feature],
                bins=30,
                alpha=0.7,
                label="Original",
                density=True,
                color="blue",
            )
            plt.hist(
                X_vir_pca_df[feature],
                bins=30,
                alpha=0.7,
                label="Virtual",
                density=True,
                color="red",
            )

            plt.title(f"{feature} Distribution")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Box plots for quartile comparison (bottom row)
        for i, feature in enumerate(feature_names):
            plt.subplot(2, n_features, n_features + i + 1)

            data_to_plot = [X_ori_pca_df[feature], X_vir_pca_df[feature]]
            plt.boxplot(data_to_plot, labels=["Original", "Virtual"])
            plt.title(f"{feature} Box Plot")
            plt.ylabel(feature)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"/Users/kevinwardakhan/test/dt-stage-GenAI/dataset_comparison/dataset_comparison_carboptim_pca_{equipment_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(f"Plots saved to: dataset_comparison_carboptim_pca_{equipment_name}.png")

        # Additional scatter plots for PCA space visualization
        if n_pca >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # PC1 vs PC2 scatter plot
            axes[0].scatter(
                X_ori_pca_df["PC1"],
                X_ori_pca_df["PC2"],
                alpha=0.6,
                label="Original",
                s=50,
            )
            axes[0].scatter(
                X_vir_pca_df["PC1"],
                X_vir_pca_df["PC2"],
                alpha=0.6,
                label="Virtual",
                s=50,
            )
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            axes[0].set_title("PCA Projection: PC1 vs PC2")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # PC3 vs PC4 scatter plot (if available)
            if n_pca >= 4:
                axes[1].scatter(
                    X_ori_pca_df["PC3"],
                    X_ori_pca_df["PC4"],
                    alpha=0.6,
                    label="Original",
                    s=50,
                )
                axes[1].scatter(
                    X_vir_pca_df["PC3"],
                    X_vir_pca_df["PC4"],
                    alpha=0.6,
                    label="Virtual",
                    s=50,
                )
                axes[1].set_xlabel("PC3")
                axes[1].set_ylabel("PC4")
                axes[1].set_title("PCA Projection: PC3 vs PC4")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"/Users/kevinwardakhan/test/dt-stage-GenAI/pca_scatter/pca_scatter_comparison_{equipment_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

            print(
                f"PCA scatter plots saved to: pca_scatter_comparison_{equipment_name}.png"
            )

    # # =====================================
    # # STEP 5: QUALITY ASSESSMENT
    # # =====================================

    # Calculate aggregate quality metrics
    mean_diffs = [stat["mean_diff_%"] for stat in comparison_stats]
    std_diffs = [stat["std_diff_%"] for stat in comparison_stats]
    ks_pvalues = [stat["ks_pvalue"] for stat in comparison_stats]

    avg_mean_diff = np.mean(mean_diffs)
    avg_std_diff = np.mean(std_diffs)
    avg_ks_pvalue = np.mean(ks_pvalues)

    # Quality score calculation (0-100 scale)
    quality_score = 0

    # Mean similarity (25 points): good if mean difference < 15%
    if avg_mean_diff < 15:
        quality_score += 25

    # Standard deviation similarity (25 points): good if std difference < 25%
    if avg_std_diff < 25:
        quality_score += 25

    # Distribution similarity (25 points): good if KS test p-value > 0.05
    if avg_ks_pvalue > 0.05:
        quality_score += 25

    # Correlation preservation (25 points): good if max correlation difference < 0.3
    if corr_diff.max().max() < 0.3:
        quality_score += 25

    print(f"\nOverall Quality Score: {quality_score}/100")

    # Quality interpretation
    if quality_score >= 80:
        print("EXCELLENT: Virtual data closely matches original!")
    elif quality_score >= 60:
        print("GOOD: Virtual data shows good similarity.")
    elif quality_score >= 40:
        print("FAIR: Virtual data shows moderate similarity.")
    else:
        print("POOR: Virtual data significantly differs.")

    # =====================================
    # STEP 6: PCA-SPECIFIC ANALYSIS
    # =====================================
    print(f"\nPCA-SPECIFIC ANALYSIS")
    print("-" * 30)

    # Check if PCA components preserve the explained variance structure
    ori_pca_var = np.var(X_ori_pca, axis=0)
    vir_pca_var = np.var(X_vir_pca, axis=0)

    print("Component variance preservation:")
    for i, (ori_var, vir_var) in enumerate(zip(ori_pca_var, vir_pca_var)):
        var_diff = abs(vir_var - ori_var) / ori_var * 100
        print(f"  PC{i+1}: {var_diff:.1f}% difference")

    return comparison_stats, quality_score


if __name__ == "__main__":
    print("PHATE+PGAN VSG Implementation for CarbOptim")
    print("=" * 50)

    # =====================================
    # HYPERPARAMETERS AND CONFIGURATION
    # =====================================
    # Training parameters
    BATCH_SIZE = 64  # Batch size for GAN training (balance between stability and speed)
    EPOCHS = 1000  # Number of training epochs (adjust based on convergence)
    Q = 1  # Number of sparse generation cycles
    phate_dim = 4  # PHATE embedding dimensionality
    n_components = phate_dim  # Alias for clarity
    n_output = 10  # Number of samples to use (approx 4 months)
    lr = 0.001  # Base learning rate
    equipment_name = "R101"  # Target equipment for analysis

    print(f"Configuration:")
    print(f"  Equipment: {equipment_name}")
    print(f"  Samples: {n_output}")
    print(f"  PHATE dimensions: {phate_dim}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")

    # =====================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # =====================================
    print(f"\nSTEP 1: Data Loading")
    print("-" * 30)

    # Load and preprocess industrial process data
    data_processor = DataProcessor(
        "/Users/kevinwardakhan/test/dt-stage-GenAI/dataset/carboptim/DHC_consolidated.parquet",
        n_output=n_output,
        equipment_name=equipment_name,
        config_file_path="dataset/carboptim/DHC_unit_config.json",
    )
    X_ori = data_processor.load_and_process_data()
    n_samples, n_features = X_ori.shape

    print(f"Loaded data shape for equipment {equipment_name}: {X_ori.shape}")
    print(f"   Features: {list(X_ori.columns)}")
    print(f"   Data range: {X_ori.min().min():.2f} to {X_ori.max().max():.2f}")

    # =====================================
    # STEP 2: PHATE EMBEDDING
    # =====================================
    print(f"\nSTEP 2: PHATE Embedding")
    print("-" * 30)

    # Apply PHATE for dimensionality reduction while preserving topology
    phate_embedder = PHATEEmbedder(n_components=n_components, knn=5)
    Z_ori = phate_embedder.fit_transform(X_ori)
    X_scaled = phate_embedder.scaler.transform(X_ori)  # Get scaled version for training

    print(f"PHATE embedding shape: {Z_ori.shape}")
    print(f"   Embedding range: [{Z_ori.min():.2f}, {Z_ori.max():.2f}]")
    print(f"   Original data scaled for training")

    # =====================================
    # STEP 3: PGAN TRAINING
    # =====================================
    print(f"\nSTEP 3: PGAN Training")
    print("-" * 30)

    # Initialize and train the PHATE-GAN
    pgan = PGANtraining(
        lambda_l1=0.001,  # L1 loss weight (balances adversarial vs reconstruction)
        phate_dim=phate_dim,
        n_samples=n_samples,
        n_features=n_features,
        lr=lr,
    )

    print(f"Training GAN...")
    pgan.train(X_scaled, Z_ori, batch_size=BATCH_SIZE, epochs=EPOCHS)
    print(f"PGAN training completed!")

    # =====================================
    # STEP 4: SPARSE GENERATION
    # =====================================
    print(f"\nSTEP 4: Sparse Generation")
    print("-" * 30)

    # Generate virtual samples in PHATE space using sparse generation algorithm
    sparse_generator = SparseGeneration(
        Q=Q,  # Number of generation cycles
        phate_dim=phate_dim,
        n_samples=n_samples,
        W=5,  # Number of candidates per original sample
    )
    Z_vir = sparse_generator.sparse_generation(Z_ori)

    print(f"Generated {len(Z_vir)} virtual PHATE embeddings")

    # =====================================
    # STEP 5: VIRTUAL DATA GENERATION
    # =====================================
    print(f"\nSTEP 5: Virtual Data Generation")
    print("-" * 30)

    # Use trained generator to map virtual PHATE embeddings back to feature space
    X_vir_scaled = pgan.generate_virtual_data(Z_vir)
    X_vir = phate_embedder.scaler.inverse_transform(
        X_vir_scaled
    )  # Unscale to original range

    print(f"Generated virtual data shape: {X_vir.shape}")
    print(f"   Virtual data range: [{X_vir.min():.2f}, {X_vir.max():.2f}]")

    # =====================================
    # STEP 6: SAVE RESULTS
    # =====================================
    print(f"\nSTEP 6: Saving Results")
    print("-" * 30)

    # Save virtual data with proper column names
    virtual_df = pd.DataFrame(
        X_vir,
        columns=get_equipment_features(equipment_name),
    )
    output_path = f"/Users/kevinwardakhan/test/dt-stage-GenAI/X_virtuals/X_virtual_carboptim_{equipment_name}.csv"
    virtual_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # =====================================
    # STEP 7: VALIDATION AND COMPARISON
    # =====================================
    print(f"\nSTEP 7: Dataset Validation")
    print("=" * 50)

    # Load saved data for comparison (ensures consistency)
    X_virtual_loaded = pd.read_csv(output_path)

    # Comprehensive statistical comparison
    comparison_stats, quality_score = compare_datasets(
        X_ori, X_virtual_loaded, save_plots=True
    )

    # =====================================
    # STEP 8: COVERAGE ANALYSIS
    # =====================================
    print(f"\nSTEP 8: Coverage Analysis")
    print("-" * 30)

    print("Feature coverage (how well virtual data spans original data range):")
    total_coverage = 0

    for feature in X_ori.columns:
        ori_range = X_ori[feature].max() - X_ori[feature].min()
        ori_min, ori_max = X_ori[feature].min(), X_ori[feature].max()
        vir_min, vir_max = (
            X_virtual_loaded[feature].min(),
            X_virtual_loaded[feature].max(),
        )

        # Calculate overlap percentage
        overlap_min = max(ori_min, vir_min)
        overlap_max = min(ori_max, vir_max)
        coverage = (
            max(0, overlap_max - overlap_min) / ori_range * 100
            if ori_range > 0
            else 100
        )
        total_coverage += coverage

        # Visual indicator based on coverage quality
        status = "GOOD" if coverage > 80 else "FAIR" if coverage > 50 else "POOR"
        print(f"{feature[:15]:15} | Coverage: {coverage:5.1f}% {status}")

    avg_coverage = total_coverage / len(X_ori.columns)
    print(f"\nAverage coverage: {avg_coverage:.1f}%")

    # =====================================
    # FINAL SUMMARY
    # =====================================
    print(f"\nPROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Summary:")
    print(f"   Original samples: {len(X_ori)}")
    print(f"   Virtual samples: {len(X_virtual_loaded)}")
    print(f"   Quality Score: {quality_score}/100")
    print(f"   Average Coverage: {avg_coverage:.1f}%")
    print(f"   Equipment: {equipment_name}")
    print(f"   Output file: X_virtual_carboptim_{equipment_name}.csv")
    print(f"\nThe synthetic data can now be used for:")
    print(f"   • Data augmentation for machine learning models")
    print(f"   • Process simulation and what-if analysis")
    print(f"   • Rare event modeling and edge case testing")
    print(f"   • Privacy-preserving data sharing")

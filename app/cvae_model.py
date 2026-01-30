"""
Conditional VAE model for learning p(X|Z) distribution
Adapted from R torch implementation to Python PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CVAEEncoder(nn.Module):
    """Encoder network: q(u | x, z)"""

    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, z):
        h = torch.cat([x, z], dim=1)
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """Decoder network: p(x | u, z)"""

    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, u, z):
        h = torch.cat([u, z], dim=1)
        h = F.relu(self.fc1(h))
        x_recon = self.fc2(h)
        return x_recon


class ConditionalVAE(nn.Module):
    """Conditional VAE for learning p(X|Z)"""

    def __init__(self, input_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = 1  # Binary treatment

        self.encoder = CVAEEncoder(input_dim, self.condition_dim, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(latent_dim, self.condition_dim, input_dim, hidden_dim)

        # Store scaling parameters
        self.scaler = StandardScaler()
        self.is_fitted = False

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, z):
        """Forward pass through encoder and decoder"""
        mu, logvar = self.encoder(x, z)
        u = self.reparameterize(mu, logvar)
        x_recon = self.decoder(u, z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        """ELBO loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    def fit(self, X, Z, epochs=30, lr=1e-3, batch_size=None, verbose=True):
        """Train the CVAE model"""
        # Fit and transform data
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        Z_tensor = torch.FloatTensor(Z).reshape(-1, 1)

        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Training loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_recon, mu, logvar = self(X_tensor, Z_tensor)
            loss = self.loss_function(x_recon, X_tensor, mu, logvar)

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.eval()
        return self

    def sample(self, z_values, n_samples=None):
        """Sample X given Z values"""
        self.eval()

        if n_samples is None:
            n_samples = len(z_values)

        with torch.no_grad():
            # Sample from latent space
            u = torch.randn(n_samples, self.latent_dim)
            z_tensor = torch.FloatTensor(z_values).reshape(-1, 1)

            # Decode
            x_scaled = self.decoder(u, z_tensor).numpy()

            # Inverse transform to original scale
            x_original = self.scaler.inverse_transform(x_scaled)

        return x_original


def encode_data(df, outcome_col='y', treatment_col='z', exclude_cols=None):
    """
    Encode dataframe for CVAE training

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    outcome_col : str
        Name of outcome column
    treatment_col : str
        Name of treatment column
    exclude_cols : list
        Additional columns to exclude

    Returns:
    --------
    X : np.ndarray
        Encoded covariates (one-hot for categoricals)
    Z : np.ndarray
        Treatment assignments
    feature_names : list
        Names of encoded features
    """
    if exclude_cols is None:
        exclude_cols = []

    exclude = [outcome_col, treatment_col] + exclude_cols

    # Get covariate columns
    X_df = df.drop(columns=[col for col in exclude if col in df.columns])

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X_df, drop_first=False)

    # Get treatment
    Z = df[treatment_col].values

    return X_encoded.values, Z, X_encoded.columns.tolist()


def decode_samples(X_samples, feature_names, original_df, treatment_col='z', outcome_col='y'):
    """
    Decode sampled data back to original format with proper categorical handling

    Parameters:
    -----------
    X_samples : np.ndarray
        Sampled encoded features
    feature_names : list
        Names of encoded features
    original_df : pd.DataFrame
        Original dataframe to extract metadata
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column

    Returns:
    --------
    pd.DataFrame
        Decoded samples with original column types
    """
    # Start with encoded samples
    X_df = pd.DataFrame(X_samples, columns=feature_names)

    # Identify categorical variables (those that were one-hot encoded)
    exclude = [outcome_col, treatment_col, 'id'] if 'id' in original_df.columns else [outcome_col, treatment_col]
    original_covariates = original_df.drop(columns=[col for col in exclude if col in original_df.columns])

    decoded_df = pd.DataFrame()

    for col in original_covariates.columns:
        if original_covariates[col].dtype == 'object' or isinstance(original_covariates[col].dtype, pd.CategoricalDtype):
            # This was a categorical variable - find its one-hot columns
            cat_cols = [c for c in feature_names if c.startswith(f"{col}_")]

            if cat_cols:
                # Get probabilities from one-hot encoded columns
                probs = X_df[cat_cols].values
                # Softmax to get valid probabilities
                probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
                # Choose category with highest probability
                categories = [c.replace(f"{col}_", "") for c in cat_cols]
                decoded_df[col] = [categories[i] for i in probs.argmax(axis=1)]
        else:
            # Numeric variable - keep as is
            if col in feature_names:
                decoded_df[col] = X_df[col].values

    return decoded_df

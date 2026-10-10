"""
Conditional VAE for learning p(X | Z)  --  the "realism" half of Credence.

WHAT THIS FILE IS FOR (read this before the code)
-------------------------------------------------
The Credence framework (Parikh et al., ICML 2022) separates two things that
hand-written simulations normally fuse together:

  realism  -- where does the covariate structure come from?  ANSWER: learn it
              from real data, here with a conditional VAE that models p(X | Z).
  truth    -- what is the true causal effect?  ANSWER: we choose it (see dgp.py),
              which is the only reason we can score an estimator at all.

This file is the *realism* half only. It never sees the outcome Y and it never
knows what tau is. Keeping that boundary sharp is the single most important
teaching point in the module: if the generator knew the effect, "recovering" the
effect would prove nothing.

WHAT A VAE IS DOING, IN ONE PARAGRAPH
-------------------------------------
An encoder q(u | x, z) squeezes each row of covariates down to a low-dimensional
latent code u. A decoder p(x | u, z) expands a latent code back into covariates.
Train the pair so that decode(encode(x)) ~ x, while forcing the codes u to look
like standard normal noise. Once trained, you can throw away the encoder, draw
u ~ N(0, I), and decode it: out comes a *new* row of covariates that never
existed but is statistically like the ones you trained on.

The "conditional" part is the z: we feed the treatment arm into both networks, so
the model learns how the treated and control covariate distributions DIFFER.
That difference is the observed confounding, learned rather than invented.

WHY THE TRAINING LOOP LOOKS THE WAY IT DOES (a fixed bug worth teaching)
-----------------------------------------------------------------------
The previous version of this file ran ONE optimizer step per epoch on the whole
dataset at once. At the app's default of 50 epochs, that is 50 gradient steps in
total -- nowhere near convergence. The generated covariates were therefore closer
to the untrained prior than to the data, while the app told the user they were
"near-indistinguishable from the observed sample".

Two lessons, and both are the point of the exercise:
  1. Minibatching matters. n/batch_size steps per epoch instead of 1.
  2. An unverified claim is a claim you should not make. We now plot the training
     curve (below) and measure realism directly (realism.py), so the claim is
     something the participant can check instead of something we assert.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from schema import CovariateSchema, FeatureNames, column_kind, fitted_schema


# ---------------------------------------------------------------------------
# Reproducibility.
#
# Three separate random number generators can influence a run: Python's, NumPy's,
# and PyTorch's. Model weights are initialized from torch's GLOBAL generator at
# construction time, so seeding has to happen BEFORE the model object is built --
# passing a seed to .fit() afterwards is already too late for the weights.
#
# Elsewhere (dgp.py) we prefer a LOCAL generator, np.random.default_rng(seed),
# which is the modern NumPy pattern: it cannot be disturbed by unrelated library
# code that happens to draw from the global stream. We use a global seed here
# only because torch's module initialization gives us no choice.
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Seed every global RNG that can affect model construction and training."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


class CVAEEncoder(nn.Module):
    """Encoder network: q(u | x, z). Covariates + arm -> latent mean and log-variance."""

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
    """Decoder network: p(x | u, z). Latent code + arm -> covariates."""

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
    """Conditional VAE for learning p(X | Z)."""

    def __init__(self, input_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = 1  # binary treatment

        self.encoder = CVAEEncoder(input_dim, self.condition_dim, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(latent_dim, self.condition_dim, input_dim, hidden_dim)

        # Covariates are standardized before training and un-standardized after
        # sampling, so the network works on a scale where every column matters
        # equally. Without this, a column measured in dollars would dominate the
        # reconstruction loss and a column measured in years would be ignored.
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.history_ = []          # per-epoch training trace, filled by fit()

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick.

        We need to sample u ~ N(mu, sigma^2), but you cannot backpropagate through
        a random draw. So we draw the randomness OUTSIDE the parameters --
        eps ~ N(0, 1) -- and write u = mu + sigma * eps. Now u is a differentiable
        function of mu and sigma, and the gradient flows.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, z):
        mu, logvar = self.encoder(x, z)
        u = self.reparameterize(mu, logvar)
        x_recon = self.decoder(u, z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        """
        Negative ELBO, returned as (total, reconstruction, KL) so the caller can
        plot the two competing pressures separately.

        reconstruction : how badly the decoder recreated the input. Pushes the
                         model to memorize.
        KL divergence  : how far the latent codes drift from standard normal.
                         Pushes the model to generalize.

        Their sum is what we minimize. Watching them trade off against each other
        is the most instructive plot in the whole app: reconstruction falls fast
        at first, then the KL term starts to bite and progress slows.

        Both are summed over the batch (not averaged); fit() divides by the number
        of rows at the end of each epoch so the reported number is per-observation
        and therefore comparable across different batch sizes.
        """
        recon = F.mse_loss(x_recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld, recon, kld

    def fit(self, X, Z, epochs=100, lr=1e-3, batch_size=128, seed=None, verbose=True):
        """
        Train the CVAE with minibatch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Encoded covariates (numeric columns plus one-hot dummies).
        Z : np.ndarray, shape (n,)
            Binary treatment assignment.
        epochs : int
            Passes over the data. Each pass now takes ceil(n / batch_size)
            gradient steps, not one.
        batch_size : int or None
            Rows per gradient step. None means full-batch, which is what the old
            code did on every epoch -- kept available so the workshop can SHOW the
            failure mode rather than just describe it.
        seed : int or None
            Seeds the shuffling order. Model weights must be seeded before the
            object is constructed; see set_seed().

        Returns
        -------
        self, with `history_` populated: one dict per epoch holding the
        per-observation negative ELBO and its two components.
        """
        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        self.is_fitted = True

        X_tensor = torch.FloatTensor(X_scaled)
        Z_tensor = torch.FloatTensor(np.asarray(Z, dtype=np.float64)).reshape(-1, 1)
        n = X_tensor.shape[0]

        # A LOCAL torch generator for the shuffle. Local rather than global so that
        # re-fitting a model does not perturb any other torch randomness in the app.
        gen = torch.Generator()
        gen.manual_seed(int(seed) if seed is not None else 0)

        if batch_size is None or batch_size <= 0 or batch_size > n:
            batch_size = n

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.history_ = []

        self.train()
        for epoch in range(epochs):
            # Reshuffle every epoch: minibatches should not see the same partners
            # each pass, or the gradient noise becomes systematic instead of random.
            order = torch.randperm(n, generator=gen)
            totals = np.zeros(3)  # accumulates (total, recon, kl) over the epoch

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, zb = X_tensor[idx], Z_tensor[idx]

                optimizer.zero_grad()
                x_recon, mu, logvar = self(xb, zb)
                loss, recon, kld = self.loss_function(x_recon, xb, mu, logvar)
                loss.backward()
                optimizer.step()

                totals += np.array([loss.item(), recon.item(), kld.item()])

            # Per-observation, so the y-axis means the same thing at any batch size.
            self.history_.append({
                'epoch': epoch + 1,
                'neg_elbo': totals[0] / n,
                'recon': totals[1] / n,
                'kl': totals[2] / n,
                'steps': int(np.ceil(n / batch_size)),
            })

            if verbose and (epoch + 1) % 10 == 0:
                h = self.history_[-1]
                print("Epoch {}/{}  neg-ELBO/obs {:.4f}  (recon {:.4f} + KL {:.4f})".format(
                    epoch + 1, epochs, h['neg_elbo'], h['recon'], h['kl']))

        self.eval()
        return self

    def sample(self, z_values, seed=None):
        """
        Draw synthetic covariates for a given vector of treatment assignments.

        The encoder plays no part here. We draw latent codes straight from the
        prior, u ~ N(0, I), and decode them conditional on z. That is what makes
        these rows genuinely new data rather than reconstructions of real people --
        which also matters for privacy, since a generator that merely echoed its
        training rows would leak them.

        Parameters
        ----------
        z_values : array-like, shape (m,)
            Treatment arm for each synthetic row.
        seed : int or None
            Seeds the latent draws, so the same seed gives the same cohort.
        """
        self.eval()
        z_values = np.asarray(z_values, dtype=np.float64).reshape(-1)
        m = len(z_values)

        gen = torch.Generator()
        gen.manual_seed(int(seed) if seed is not None else 0)

        with torch.no_grad():
            u = torch.randn(m, self.latent_dim, generator=gen)
            z_tensor = torch.FloatTensor(z_values).reshape(-1, 1)
            x_scaled = self.decoder(u, z_tensor).numpy()
            x_original = self.scaler.inverse_transform(x_scaled)

        return x_original


def encode_data(df, outcome_col='y', treatment_col='z', exclude_cols=None, schema=None):
    """Encode using one fitted schema; reuse it for held-out and generated rows."""
    excluded = [outcome_col, treatment_col] + list(exclude_cols or [])
    covariates = df.drop(columns=[c for c in excluded if c in df.columns])
    schema = CovariateSchema.fit(covariates) if schema is None else schema
    return schema.encode(covariates), df[treatment_col].to_numpy(), FeatureNames(schema)


def decode_samples(X_samples, feature_names, original_df, treatment_col='z',
                   outcome_col='y', rng=None, sample_categories=True):
    """Decode MSE reconstructions with the fitted column and category mapping.

    Categorical blocks are probability estimates, not logits. This remains a
    post-processing approximation to a Gaussian/MSE decoder; fidelity still needs
    to be measured. A typed likelihood is a separate future model improvement.
    """
    covariates = original_df.drop(columns=[c for c in (outcome_col, treatment_col) if c in original_df])
    schema = fitted_schema(feature_names, covariates)
    return schema.decode(X_samples, rng=rng, sample_categories=sample_categories)

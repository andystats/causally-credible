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


def encode_data(df, outcome_col='y', treatment_col='z', exclude_cols=None):
    """
    Turn a raw data frame into the numeric matrix the network consumes.

    Categorical columns become one-hot dummy columns named "<col>_<level>"; that
    naming convention is what decode_samples() below relies on to put them back
    together, so the two functions must be read as a pair.

    Note that Y is dropped here and never reaches the model. See the file header:
    the generator learns p(X | Z) only.

    Returns
    -------
    X : np.ndarray (float64), shape (n, p)
    Z : np.ndarray, shape (n,)
    feature_names : list of str, the column names of X
    """
    if exclude_cols is None:
        exclude_cols = []

    exclude = [outcome_col, treatment_col] + exclude_cols

    X_df = df.drop(columns=[col for col in exclude if col in df.columns])
    X_encoded = pd.get_dummies(X_df, drop_first=False)

    Z = df[treatment_col].values

    # Cast explicitly. pandas returns uint8 dummies on older versions and bool on
    # pandas >= 2.0; mixing either with float columns makes .values come back as
    # dtype object, which torch cannot consume. Being explicit means the app
    # behaves identically on the workshop laptop and on the deployment server.
    return X_encoded.values.astype(np.float64), Z, X_encoded.columns.tolist()


def column_kind(series):
    """
    Classify an original covariate so the decoder's output can be mapped back onto
    the values that variable is actually allowed to take.

    The decoder emits a real number for EVERY column, because the network only
    ever sees floats. What we do with that number depends on what the column was:

      'categorical' -- was one-hot expanded; the dummy block holds scores.
      'binary'      -- exactly two distinct values (0/1 indicators, and also
                       things like 1/2 coding).
      'integer'     -- all values whole numbers (counts, years, ordinal scales).
      'continuous'  -- anything else; the raw decoder output is already valid.

    Returns a dict describing the column, including the information needed to
    invert it (the two levels, or the observed range).
    """
    if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
        return {'kind': 'categorical'}

    values = pd.to_numeric(series, errors='coerce').dropna().values
    if len(values) == 0:
        return {'kind': 'continuous'}

    unique = np.unique(values)
    if len(unique) == 2:
        return {'kind': 'binary', 'low': float(unique[0]), 'high': float(unique[1])}
    if np.allclose(unique, np.round(unique)):
        return {'kind': 'integer', 'min': float(unique.min()), 'max': float(unique.max())}
    return {'kind': 'continuous'}


def decode_samples(X_samples, feature_names, original_df, treatment_col='z',
                   outcome_col='y', rng=None, sample_categories=True):
    """
    Turn the network's numeric output back into a data frame shaped like the input.

    WHY THIS FUNCTION MATTERS (it was previously written but never called)
    ----------------------------------------------------------------------
    The decoder emits a real number for every column. Left alone:

      * a categorical variable comes out as a handful of continuous scores like
        (0.62, 0.31, 0.07) instead of a category;
      * a 0/1 indicator comes out as -0.30, or 0.47, or 1.35.

    Exporting that hands participants a file that no longer matches their own
    schema. It also makes the realism check unwinnable in a completely
    uninteresting way: a discriminator does not need to understand your data to
    notice that real `priorPneumonia` is always exactly 0 or 1 while the synthetic
    version is a smear of decimals. That is how this very bug was found -- the
    Tier 0 realism diagnostic scored AUC 1.000 and the reason turned out to be
    binary columns, not anything subtle about the generator.

    SAMPLING, NOT ROUNDING (a subtle statistical point, worth pausing on)
    ---------------------------------------------------------------------
    Given scores implying category probabilities (0.62, 0.31, 0.07), there are two
    ways to pick a category:

      argmax   -- always take the most likely one. Every row with these scores
                  becomes category 1. Rare categories vanish entirely and the
                  synthetic data ends up LESS variable than the real data.
      sampling -- draw with probability 0.62 / 0.31 / 0.07. Across many rows the
                  category frequencies come out right.

    The same choice arises for binary columns: thresholding at 0.5 is the argmax
    move, Bernoulli sampling is the honest one. We sample in both cases, because
    what we need to preserve is the MARGINAL DISTRIBUTION, not the most likely
    value of any single row. `sample_categories` is exposed so the workshop can
    switch it off and watch the realism diagnostic get worse.

    A HONEST LIMITATION, AND A FORWARD REFERENCE
    --------------------------------------------
    Everything here is POST-HOC repair: the network was trained as if every column
    were continuous, and we discretize its output afterwards. That is why the
    generator still gets the joint structure between a binary column and a
    continuous one only approximately. The architectural fix is to give the model
    type-specific decoders and train it with the right likelihood per column --
    which is exactly what CausalMix (Zhang, Parikh et al., 2026) does, and the
    main reason Tier 2 of the plan contemplates adopting it.

    Parameters
    ----------
    rng : np.random.Generator or None
        Local generator for the category and Bernoulli draws, so decoding is
        reproducible.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_df = pd.DataFrame(X_samples, columns=feature_names)

    exclude = [outcome_col, treatment_col]
    if 'id' in original_df.columns:
        exclude.append('id')
    original_covariates = original_df.drop(
        columns=[col for col in exclude if col in original_df.columns])

    decoded_df = pd.DataFrame()

    for col in original_covariates.columns:
        spec = column_kind(original_covariates[col])

        if spec['kind'] == 'categorical':
            # Recover the dummy block for this column. Caveat worth knowing: if one
            # column is named "x" and another "x_y", this prefix match can pick up
            # the wrong block. Fine for the workshop datasets; a production version
            # would carry an explicit column map out of encode_data().
            cat_cols = [c for c in feature_names if c.startswith("{}_".format(col))]
            if not cat_cols:
                continue

            scores = X_df[cat_cols].values
            # Softmax, computed after subtracting the row max. Mathematically this
            # changes nothing (the shift cancels), but it stops exp() overflowing
            # when the decoder emits a large score.
            shifted = scores - scores.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / probs.sum(axis=1, keepdims=True)

            categories = [c[len(col) + 1:] for c in cat_cols]
            if sample_categories:
                # One categorical draw per row, using that row's probabilities.
                picks = [rng.choice(len(categories), p=p) for p in probs]
            else:
                picks = probs.argmax(axis=1)
            decoded_df[col] = [categories[i] for i in picks]

        elif col not in feature_names:
            continue

        elif spec['kind'] == 'binary':
            # Read the decoder output as a position between the two levels, clip it
            # into [0, 1] so it can act as a probability, then draw. Sampling (not
            # thresholding) is what keeps the synthetic prevalence close to the real
            # one when the decoder's output sits near the middle.
            low, high = spec['low'], spec['high']
            span = high - low if high != low else 1.0
            p = np.clip((X_df[col].values - low) / span, 0.0, 1.0)
            if sample_categories:
                draws = rng.binomial(1, p)
            else:
                draws = (p >= 0.5).astype(int)
            decoded_df[col] = np.where(draws == 1, high, low)

        elif spec['kind'] == 'integer':
            # Round to the nearest whole number and keep it inside the range the
            # variable was actually observed to take, so we never invent an age of
            # -3 or a comorbidity count of 97.
            values = np.round(X_df[col].values)
            decoded_df[col] = np.clip(values, spec['min'], spec['max'])

        else:
            decoded_df[col] = X_df[col].values

    return decoded_df

# Probabilistic Matrix Factorization for Movie Recommendations

A from-scratch Bayesian recommender that predicts unseen movie ratings by learning
latent "taste" vectors for users and movies. Built in [Pyro](https://pyro.ai/),
it compares MAP estimation against mean-field variational inference (SVI) and
quantifies how predictive accuracy responds to model capacity, training-set size,
and choice of inference method.

## The problem

A movie-ratings table is mostly empty — every user has only rated a tiny fraction
of movies. The task is to fill in the blanks: given the ratings we *do* observe,
predict the ones we don't. This is the core of collaborative-filtering
recommenders.

## Approach

Each user `i` and movie `j` is assigned a `K`-dimensional latent vector. The model
treats these vectors as Gaussian-distributed and a predicted rating as the noisy
inner product of the two:

```
U_i  ~ Normal(0, σ_p² I)        # user latent vector
V_j  ~ Normal(0, σ_p² I)        # movie latent vector
r_ij ~ Normal(U_i · V_j, σ²)    # observed rating
```

Inference is done two ways and compared head-to-head:

- **MAP** (`AutoDelta`) — a single point estimate of the latent factors;
  equivalent to L2-regularized matrix factorization.
- **SVI** (`AutoNormal`) — a mean-field Gaussian *posterior* over the factors,
  capturing parameter uncertainty rather than a single guess.

## Results

Evaluated by held-out RMSE on a 2000×500 dense subset of MovieLens-1M
(~364K ratings, 1–5 star scale):

| Method | Held-out RMSE |
| --- | --- |
| Baseline — global mean | 1.025 |
| Baseline — per-user mean | 0.966 |
| PMF, MAP (K=10) | 0.863 |
| **PMF, SVI (K=10)** | **0.849** |

Both factorization models clearly beat the mean baselines, and the Bayesian SVI
variant edges out the MAP point estimate. The full study also sweeps the latent
dimension `K` (capacity vs. overfitting) and the training-set size (a learning
curve) — see `results/`.

## Data

MovieLens-1M (`ratings.dat`, `UserID::MovieID::Rating::Timestamp`). `build_matrix()`
selects the 500 most-rated movies and the 2000 most-active users to form a dense
matrix, then caches it to `data/subset_matrix.npy`.

## Run

```bash
pip install pyro-ppl matplotlib numpy torch
python experiments.py
```

Outputs three CSVs and three figures (RMSE vs. K, the learning curve, and the
method comparison) into `results/`.

## Project layout

```
.
├── pmf.py            # data subsetting, PMF model, MAP/SVI guides, training, metrics, baselines
├── experiments.py    # the three sweeps; writes CSVs + figures
├── results/          # generated CSVs and figures
└── mlens-1m/         # MovieLens-1M data
```

## Stack

Python · Pyro · PyTorch · NumPy · Matplotlib
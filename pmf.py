import os
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.optim import Adam
 
RATING_MIN, RATING_MAX = 1.0, 5.0
 
 
# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_device():
    return torch.device("cpu")
 
 
def build_matrix(ratings_path="mlens-1m/ratings.dat", n_movies=500, n_users=2000,
                 cache="data/subset_matrix.npy"):
    """Build the dense (n_users x n_movies) ratings matrix (NaN = unobserved)."""
    if cache and os.path.exists(cache):
        return np.load(cache)
 
    u, m, r = [], [], []
    with open(ratings_path) as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) < 3:
                continue
            u.append(int(p[0])); m.append(int(p[1])); r.append(float(p[2]))
    u, m, r = np.array(u), np.array(m), np.array(r, dtype=np.float32)
 
    # keep the n_movies most-rated movies
    mv, mc = np.unique(m, return_counts=True)
    top_movies = mv[np.argsort(-mc)[:n_movies]]
    keep = np.isin(m, top_movies)
    u, m, r = u[keep], m[keep], r[keep]
 
    # among those, keep the n_users most-active users
    uv, uc = np.unique(u, return_counts=True)
    top_users = uv[np.argsort(-uc)[:n_users]]
    keep = np.isin(u, top_users)
    u, m, r = u[keep], m[keep], r[keep]
 
    # remap ids to 0..N-1 / 0..M-1 and fill the matrix
    umap = {uid: i for i, uid in enumerate(np.unique(u))}
    mmap = {mid: j for j, mid in enumerate(np.unique(m))}
    X = np.full((len(umap), len(mmap)), np.nan, dtype=np.float32)
    for uid, mid, rr in zip(u, m, r):
        X[umap[uid], mmap[mid]] = rr
 
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, X)
    return X
 
 
def get_observed(X):
    ui, mi = np.where(~np.isnan(X))
    return ui.astype(np.int64), mi.astype(np.int64), X[ui, mi].astype(np.float32)
 
 
def train_test_split(ui, mi, r, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(r))
    n_test = int(round(test_frac * len(r)))
    te, tr = perm[:n_test], perm[n_test:]
    return (ui[tr], mi[tr], r[tr]), (ui[te], mi[te], r[te])
 
 
def subsample_train(train, frac, seed=0):
    ui, mi, r = train
    if frac >= 1.0:
        return train
    rng = np.random.default_rng(seed)
    keep = rng.permutation(len(r))[: int(round(frac * len(r)))]
    return ui[keep], mi[keep], r[keep]
 
 
# --------------------------------------------------------------------------- #
# Model + inference
# --------------------------------------------------------------------------- #
def pmf_model(user_idx, movie_idx, ratings, N, M, K, prior_scale, obs_scale):
    with pyro.plate("users", N):
        U = pyro.sample("U", dist.Normal(0.0, prior_scale).expand([N, K]).to_event(1))
    with pyro.plate("movies", M):
        V = pyro.sample("V", dist.Normal(0.0, prior_scale).expand([M, K]).to_event(1))
    pred = (U[user_idx] * V[movie_idx]).sum(-1)
    with pyro.plate("data", ratings.shape[0]):
        pyro.sample("obs", dist.Normal(pred, obs_scale), obs=ratings)
 
 
def fit(train, N, M, K, method="svi", prior_scale=1.0, obs_scale=1.0,
        num_steps=1500, lr=0.01, seed=0, device=None, verbose=False):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    device = device or get_device()
 
    ui, mi, r = train
    global_mean = float(r.mean())
    user_idx = torch.tensor(ui, dtype=torch.long, device=device)
    movie_idx = torch.tensor(mi, dtype=torch.long, device=device)
    ratings = torch.tensor(r - global_mean, dtype=torch.float32, device=device)
 
    def model():
        return pmf_model(user_idx, movie_idx, ratings, N, M, K, prior_scale, obs_scale)
 
    guide = AutoDelta(model) if method == "map" else AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
 
    losses = []
    for step in range(num_steps):
        loss = svi.step()
        losses.append(loss)
        if verbose and step % 200 == 0:
            print(f"    step {step:4d}   elbo loss {loss:,.0f}")
 
    med = guide.median()
    return {
        "U": med["U"].detach().cpu().numpy(),
        "V": med["V"].detach().cpu().numpy(),
        "global_mean": global_mean,
        "losses": losses, "method": method, "K": K,
    }
 
 
# --------------------------------------------------------------------------- #
# Prediction + metrics
# --------------------------------------------------------------------------- #
def predict(fit_out, ui, mi, clip=True):
    pred = (fit_out["U"][ui] * fit_out["V"][mi]).sum(axis=1) + fit_out["global_mean"]
    return np.clip(pred, RATING_MIN, RATING_MAX) if clip else pred
 
 
def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))
 
 
def baseline_global_mean(train, test):
    return rmse(np.full_like(test[2], train[2].mean()), test[2])
 
 
def baseline_user_mean(train, test, N):
    ui_tr, _, r_tr = train
    g = r_tr.mean()
    sums = np.zeros(N); cnts = np.zeros(N)
    np.add.at(sums, ui_tr, r_tr); np.add.at(cnts, ui_tr, 1)
    user_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), g)
    return rmse(user_mean[test[0]], test[2])
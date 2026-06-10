import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pmf

RESULTS = "results"
FIGS = os.path.join(RESULTS, "figs")
SEED = 0
SEEDS = (0, 1, 2)
NUM_STEPS = 1500
TEST_FRAC = 0.2
FIXED_K = 10
SPARSITY_DENS = (0.05, 0.10, 0.20, 0.29)   # target training densities (fraction of cells observed)
METHOD_SEEDS = (0, 1, 2, 3, 4)             # 5 seeds


def setup():
    os.makedirs(FIGS, exist_ok=True)
    X = pmf.build_matrix()
    N, M = X.shape
    ui, mi, r = pmf.get_observed(X)
    train, test = pmf.train_test_split(ui, mi, r, test_frac=TEST_FRAC, seed=SEED)
    print(f"matrix {N}x{M} | {len(r):,} observed | "
          f"train {len(train[2]):,} / test {len(test[2]):,}")
    return N, M, train, test, (ui, mi, r)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def sweep_k(N, M, train, test, Ks=(1, 2, 5, 10, 20, 40)):
    print("\n=== sweep 1: latent dimension K (SVI) ===")
    rows = []
    for K in Ks:
        out = pmf.fit(train, N, M, K, method="svi", num_steps=NUM_STEPS, seed=SEED)
        e = pmf.rmse(pmf.predict(out, test[0], test[1]), test[2])
        rows.append((K, round(e, 4)))
        print(f"  K={K:3d}   test RMSE = {e:.4f}")
    write_csv(f"{RESULTS}/sweep_k.csv", ["K", "test_rmse"], rows)

    plt.figure(figsize=(5, 3.5))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-")
    plt.xscale("log")
    plt.xlabel("latent dimension K")
    plt.ylabel("held-out RMSE (stars)")
    plt.title("RMSE vs latent dimension (SVI)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGS}/sweep_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rows


def sweep_trainsize(N, M, train, test, fracs=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    print("\n=== sweep 2: training-set size (SVI, K=%d) ===" % FIXED_K)
    rows = []
    for fr in fracs:
        sub = pmf.subsample_train(train, fr, seed=SEED)
        out = pmf.fit(sub, N, M, FIXED_K, method="svi", num_steps=NUM_STEPS, seed=SEED)
        e = pmf.rmse(pmf.predict(out, test[0], test[1]), test[2])
        rows.append((fr, len(sub[2]), round(e, 4)))
        print(f"  frac={fr:.2f}  (n={len(sub[2]):>7,})   test RMSE = {e:.4f}")
    write_csv(f"{RESULTS}/sweep_trainsize.csv",
              ["train_frac", "n_train", "test_rmse"], rows)

    plt.figure(figsize=(5, 3.5))
    plt.plot([r[1] for r in rows], [r[2] for r in rows], "o-")
    plt.xlabel("number of training ratings")
    plt.ylabel("held-out RMSE (stars)")
    plt.title(f"Learning curve (SVI, K={FIXED_K})")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGS}/sweep_trainsize.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rows


def sweep_method(N, M, obs, seeds=SEEDS):
    print("\n=== sweep 3: MAP vs SVI vs baselines (K=%d, %d splits) ==="
          % (FIXED_K, len(seeds)))
    ui, mi, r = obs
    methods = ["baseline: global mean", "baseline: per-user mean",
               f"PMF MAP (K={FIXED_K})", f"PMF SVI (K={FIXED_K})"]
    per_seed = {name: [] for name in methods}
    for s in seeds:
        train, test = pmf.train_test_split(ui, mi, r, test_frac=TEST_FRAC, seed=s)
        per_seed[methods[0]].append(pmf.baseline_global_mean(train, test))
        per_seed[methods[1]].append(pmf.baseline_user_mean(train, test, N))
        for method, name in (("map", methods[2]), ("svi", methods[3])):
            out = pmf.fit(train, N, M, FIXED_K, method=method,
                          num_steps=NUM_STEPS, seed=s)
            per_seed[name].append(
                pmf.rmse(pmf.predict(out, test[0], test[1]), test[2]))

    rows = []
    for name in methods:
        vals = per_seed[name]
        rows.append((name, round(float(np.mean(vals)), 4),
                     round(float(np.std(vals)), 4)))
    for name, m, sd in rows:
        print(f"  {name:28s}  RMSE = {m:.4f} +/- {sd:.4f}")
    write_csv(f"{RESULTS}/sweep_method.csv",
              ["method", "mean_rmse", "std_rmse"], rows)

    plt.figure(figsize=(6, 3.5))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = ["#B4B2A9", "#888780", "#7F77DD", "#1D9E75"]
    plt.barh(range(len(rows)), vals, xerr=errs, color=colors[:len(rows)],
             error_kw=dict(ecolor="#333", capsize=3))
    plt.yticks(range(len(rows)), names, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("held-out RMSE (stars)")
    plt.title("Method comparison")
    for i, v in enumerate(vals):
        plt.text(v - 0.02, i, f"{v:.3f}", va="center", ha="right",
                 color="white", fontsize=8)
    plt.grid(True, axis="x", alpha=0.3)
    plt.savefig(f"{FIGS}/sweep_method.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rows
def sweep_sparsity(N, M, obs, densities=SPARSITY_DENS, seeds=METHOD_SEEDS):
    print("\n=== sweep 4: sparsity (SVI vs MAP vs training density, K=%d) ===" % FIXED_K)
    ui, mi, r = obs
    total_cells = N * M
    rows = []
    for d in densities:
        map_errs, svi_errs, achieved = [], [], []
        for s in seeds:
            train, test = pmf.train_test_split(ui, mi, r, test_frac=TEST_FRAC, seed=s)
            frac = min(1.0, d * total_cells / len(train[2]))
            sub = pmf.subsample_train(train, frac, seed=s)
            achieved.append(len(sub[2]) / total_cells)
            out_map = pmf.fit(sub, N, M, FIXED_K, method="map", num_steps=NUM_STEPS, seed=s)
            out_svi = pmf.fit(sub, N, M, FIXED_K, method="svi", num_steps=NUM_STEPS, seed=s)
            map_errs.append(pmf.rmse(pmf.predict(out_map, test[0], test[1]), test[2]))
            svi_errs.append(pmf.rmse(pmf.predict(out_svi, test[0], test[1]), test[2]))
        dens = float(np.mean(achieved))
        m_map, s_map = float(np.mean(map_errs)), float(np.std(map_errs))
        m_svi, s_svi = float(np.mean(svi_errs)), float(np.std(svi_errs))
        gap = m_map - m_svi
        rows.append((round(dens, 4), round(m_map, 4), round(s_map, 4),
                     round(m_svi, 4), round(s_svi, 4), round(gap, 4)))
        print(f"  density={dens*100:4.1f}%  MAP={m_map:.4f}+/-{s_map:.4f}  "
              f"SVI={m_svi:.4f}+/-{s_svi:.4f}  gap={gap:+.4f}")
    write_csv(f"{RESULTS}/sweep_sparsity.csv",
              ["density", "map_rmse", "map_std", "svi_rmse", "svi_std", "gap"], rows)

    dens  = [r[0] * 100 for r in rows]
    map_r = [r[1] for r in rows]; map_e = [r[2] for r in rows]
    svi_r = [r[3] for r in rows]; svi_e = [r[4] for r in rows]
    gap   = [r[5] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3))
    ax1.errorbar(dens, map_r, yerr=map_e, marker="o", label="MAP",
                 color="#7F77DD", capsize=3)
    ax1.errorbar(dens, svi_r, yerr=svi_e, marker="o", label="SVI",
                 color="#1D9E75", capsize=3)
    ax1.set_xlabel("training density (% of cells observed)")
    ax1.set_ylabel("held-out RMSE (stars)")
    ax1.set_title("RMSE vs density")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(dens, gap, "o-", color="#444")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("training density (% of cells observed)")
    ax2.set_ylabel("MAP RMSE - SVI RMSE")
    ax2.set_title("SVI advantage vs density")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/sweep_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rows

if __name__ == "__main__":
    N, M, train, test, obs = setup()
    sweep_k(N, M, train, test)
    sweep_trainsize(N, M, train, test)
    sweep_method(N, M, obs)
    print("\nDone. CSVs in results/, figures in results/figs/")
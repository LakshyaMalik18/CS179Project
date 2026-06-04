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
NUM_STEPS = 1500
TEST_FRAC = 0.2
FIXED_K = 10
 
 
def setup():
    os.makedirs(FIGS, exist_ok=True)
    X = pmf.build_matrix()
    N, M = X.shape
    ui, mi, r = pmf.get_observed(X)
    train, test = pmf.train_test_split(ui, mi, r, test_frac=TEST_FRAC, seed=SEED)
    print(f"matrix {N}x{M} | {len(r):,} observed | "
          f"train {len(train[2]):,} / test {len(test[2]):,}")
    return N, M, train, test
 
 
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
 
 
def sweep_method(N, M, train, test):
    print("\n=== sweep 3: MAP vs SVI vs baselines (K=%d) ===" % FIXED_K)
    rows = [
        ("baseline: global mean", round(pmf.baseline_global_mean(train, test), 4)),
        ("baseline: per-user mean", round(pmf.baseline_user_mean(train, test, N), 4)),
    ]
    for method in ("map", "svi"):
        out = pmf.fit(train, N, M, FIXED_K, method=method, num_steps=NUM_STEPS, seed=SEED)
        e = pmf.rmse(pmf.predict(out, test[0], test[1]), test[2])
        rows.append((f"PMF {method.upper()} (K={FIXED_K})", round(e, 4)))
    for name, e in rows:
        print(f"  {name:28s}  test RMSE = {e:.4f}")
    write_csv(f"{RESULTS}/sweep_method.csv", ["method", "test_rmse"], rows)
 
    plt.figure(figsize=(6, 3.5))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#B4B2A9", "#888780", "#7F77DD", "#1D9E75"]
    plt.barh(range(len(rows)), vals, color=colors[:len(rows)])
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
 
 
if __name__ == "__main__":
    N, M, train, test = setup()
    sweep_k(N, M, train, test)
    sweep_trainsize(N, M, train, test)
    sweep_method(N, M, train, test)
    print("\nDone. CSVs in results/, figures in results/figs/")
 
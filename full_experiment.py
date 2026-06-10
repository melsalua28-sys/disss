"""
=============================================================
Effective Methods of Data Visualization and Their Statistical Analysis
Master's Thesis — Melsova Alua Erbolovna
L.N. Gumilyov Eurasian National University
Department of Fundamental Mathematics, 2025
=============================================================

This script implements the full experimental validation of the
finite-sample stability theorem for visualization operators.

Theorem (Main Result):
    If T: R^D -> R^m is L-Lipschitz, x_i = gamma_i + xi_i,
    E[xi_i] = 0, E[||xi_i||^2] <= sigma^2, then for all eps > 0:
        P(||T(x_i) - T(gamma_i)|| >= eps) <= L^2 * sigma^2 / eps^2

Methods tested: PCA, Diffusion Maps, Laplacian Eigenmaps, UMAP, t-SNE
Manifolds:      Sphere S^2, Torus T^2, Linear Subspace
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# ── Set random seed for reproducibility ─────────────────────
np.random.seed(42)


# ============================================================
# PART 1: GENERATE SYNTHETIC MANIFOLDS
# ============================================================

def generate_sphere(n=1000, noise=0.0):
    """
    Generate n points on the unit sphere S^2 in R^3.
    Optionally embed in higher-dimensional space R^D.
    
    Parameters:
        n     : number of points
        noise : standard deviation of additive Gaussian noise
    Returns:
        gamma : clean points on manifold (n x 3)
        x     : noisy observations (n x 3)
    """
    theta = np.random.uniform(0, np.pi, n)
    phi   = np.random.uniform(0, 2 * np.pi, n)
    
    gamma = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    xi = np.random.normal(0, noise, gamma.shape)
    x  = gamma + xi
    return gamma, x


def generate_torus(n=1000, R=2.0, r=1.0, noise=0.0):
    """
    Generate n points on the torus T^2 embedded in R^3.
    R = distance from center of tube to center of torus
    r = radius of the tube
    
    Parameters:
        n     : number of points
        R, r  : torus parameters
        noise : standard deviation of additive Gaussian noise
    Returns:
        gamma : clean points on manifold (n x 3)
        x     : noisy observations (n x 3)
    """
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi   = np.random.uniform(0, 2 * np.pi, n)
    
    gamma = np.column_stack([
        (R + r * np.cos(theta)) * np.cos(phi),
        (R + r * np.cos(theta)) * np.sin(phi),
        r * np.sin(theta)
    ])
    
    xi = np.random.normal(0, noise, gamma.shape)
    x  = gamma + xi
    return gamma, x


def generate_linear_subspace(n=1000, d=3, D=50, noise=0.0):
    """
    Generate n points on a random linear subspace of dimension d
    embedded in R^D.
    
    Parameters:
        n     : number of points
        d     : intrinsic dimension of subspace
        D     : ambient dimension
        noise : standard deviation of additive Gaussian noise
    Returns:
        gamma : clean points on manifold (n x D)
        x     : noisy observations (n x D)
    """
    # Random orthonormal basis for the subspace
    A = np.random.randn(D, d)
    A, _ = np.linalg.qr(A)  # orthonormalize
    A    = A[:, :d]
    
    # Generate points in d-dimensional subspace
    Z     = np.random.randn(n, d)
    gamma = Z @ A.T  # embed in R^D
    
    xi = np.random.normal(0, noise, gamma.shape)
    x  = gamma + xi
    return gamma, x


def embed_in_high_dim(X, D=50):
    """
    Embed low-dimensional data X (n x d) into R^D
    by padding with zeros and adding a small random rotation.
    """
    n, d = X.shape
    if d >= D:
        return X
    # Pad with zeros
    X_padded = np.zeros((n, D))
    X_padded[:, :d] = X
    return X_padded


# ============================================================
# PART 2: APPLY VISUALIZATION METHODS
# ============================================================

def apply_pca(X, n_components=2):
    """
    Principal Component Analysis (PCA).
    Linear projection onto top-2 principal components.
    Lipschitz constant L = 1 (orthogonal projection).
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def apply_diffusion_maps(X, n_components=2, n_neighbors=15, sigma=1.0):
    """
    Diffusion Maps via spectral embedding of the heat kernel.
    Approximated here using SpectralEmbedding with RBF kernel.
    """
    se = SpectralEmbedding(
        n_components=n_components,
        affinity='rbf',
        gamma=1.0 / (2 * sigma**2),
        random_state=42
    )
    return se.fit_transform(X)


def apply_laplacian_eigenmaps(X, n_components=2, n_neighbors=10):
    """
    Laplacian Eigenmaps via spectral embedding of graph Laplacian.
    """
    se = SpectralEmbedding(
        n_components=n_components,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42
    )
    return se.fit_transform(X)


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    UMAP: Uniform Manifold Approximation and Projection.
    Topological graph embedding.
    """
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    return reducer.fit_transform(X)


def apply_tsne(X, n_components=2, perplexity=30):
    """
    t-SNE: t-Distributed Stochastic Neighbor Embedding.
    Non-Lipschitz method — L = infinity.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000
    )
    return tsne.fit_transform(X)


# Dictionary of all methods
METHODS = {
    'PCA':                apply_pca,
    'Diffusion Maps':     apply_diffusion_maps,
    'Laplacian Eigenmaps': apply_laplacian_eigenmaps,
    'UMAP':               apply_umap,
    't-SNE':              apply_tsne,
}

COLORS = {
    'PCA':                '#2563EB',   # AzureBlue
    'Diffusion Maps':     '#0D9488',   # TealMid
    'Laplacian Eigenmaps':'#1a6b62',   # darker teal
    'UMAP':               '#D97706',   # GoldHigh
    't-SNE':              '#EF4444',   # CoralRed
}

MARKERS = {
    'PCA':                'o',
    'Diffusion Maps':     's',
    'Laplacian Eigenmaps':'v',
    'UMAP':               'D',
    't-SNE':              'p',
}


# ============================================================
# PART 3: COMPUTE DEVIATION
# ============================================================

def compute_mean_deviation(T_clean, T_noisy):
    """
    Compute mean ||T(x_i) - T(gamma_i)|| over all points.
    
    Note: we align embeddings using Procrustes analysis
    to remove rotational/reflective ambiguity.
    """
    from scipy.spatial import procrustes
    try:
        _, T_noisy_aligned, _ = procrustes(T_clean, T_noisy)
        deviations = np.linalg.norm(T_noisy_aligned - T_clean, axis=1)
    except Exception:
        deviations = np.linalg.norm(T_noisy - T_clean, axis=1)
    return np.mean(deviations)


def theoretical_bound(L, sigma, eps):
    """
    Theoretical upper bound from the main theorem:
    P(||T(x) - T(gamma)|| >= eps) <= L^2 * sigma^2 / eps^2
    
    Here we use the bound as a function of sigma for fixed L, eps.
    """
    return (L**2 * sigma**2) / (eps**2)


# ============================================================
# PART 4: RUN EXPERIMENTS
# ============================================================

def run_experiment(manifold_name, gamma_fn, sigma_values,
                   methods=None, n_points=500, embed_dim=None):
    """
    Run full stability experiment on a given manifold.
    
    Parameters:
        manifold_name : name of the manifold (for display)
        gamma_fn      : function to generate clean manifold points
        sigma_values  : list of noise levels to test
        methods       : dict of {name: function} for methods to test
        n_points      : number of data points
        embed_dim     : if set, embed data in higher dimension
    
    Returns:
        results : dict {method_name: [mean_deviation per sigma]}
    """
    if methods is None:
        methods = METHODS
    
    print(f"\n{'='*50}")
    print(f"Manifold: {manifold_name}")
    print(f"Points: {n_points}, Noise levels: {sigma_values}")
    print(f"{'='*50}")
    
    # Generate clean data
    gamma, _ = gamma_fn(n=n_points, noise=0.0)
    
    if embed_dim is not None and gamma.shape[1] < embed_dim:
        gamma = embed_in_high_dim(gamma, D=embed_dim)
    
    results = {name: [] for name in methods}
    
    for sigma in sigma_values:
        print(f"  sigma = {sigma:.3f}", end="  |  ")
        
        # Add noise
        xi = np.random.normal(0, sigma, gamma.shape)
        x  = gamma + xi
        
        for method_name, method_fn in methods.items():
            try:
                T_clean = method_fn(gamma)
                T_noisy = method_fn(x)
                dev = compute_mean_deviation(T_clean, T_noisy)
                results[method_name].append(dev)
                print(f"{method_name}: {dev:.3f}", end="  ")
            except Exception as e:
                print(f"{method_name}: ERROR({e})", end="  ")
                results[method_name].append(np.nan)
        print()
    
    return results


# ============================================================
# PART 5: PLOT RESULTS
# ============================================================

def plot_results(all_results, sigma_values, L=1.0, eps=0.5,
                 save_path=None):
    """
    Plot deviation vs noise level for all manifolds.
    
    Parameters:
        all_results  : dict {manifold_name: {method_name: [deviations]}}
        sigma_values : list of noise levels
        L, eps       : parameters for theoretical bound
        save_path    : if set, save figure to this path
    """
    manifold_names = list(all_results.keys())
    n_manifolds    = len(manifold_names)
    
    fig, axes = plt.subplots(1, n_manifolds,
                              figsize=(5 * n_manifolds, 4.5),
                              sharey=True)
    
    if n_manifolds == 1:
        axes = [axes]
    
    # Compute theoretical bound
    bound = [theoretical_bound(L, s, eps) for s in sigma_values]
    
    for ax, manifold_name in zip(axes, manifold_names):
        results = all_results[manifold_name]
        
        # Plot theoretical bound
        ax.plot(sigma_values, bound, 'k--', linewidth=2.5,
                label='Theoretical bound', zorder=5)
        
        # Plot each method
        for method_name, deviations in results.items():
            ax.plot(sigma_values, deviations,
                    marker=MARKERS[method_name],
                    color=COLORS[method_name],
                    linewidth=2, markersize=6,
                    label=method_name)
        
        ax.set_title(f'{manifold_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Noise level σ', fontsize=11)
        ax.set_ylabel('Mean deviation ||T(x) - T(γ)||', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)
        ax.set_xlim(0, max(sigma_values) * 1.05)
        ax.set_ylim(0)
    
    fig.suptitle(
        'Experimental Validation of Stability Theorem\n'
        'Deviation vs Noise Level across Manifolds',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_combined(all_results, sigma_values, L=1.0, eps=0.5,
                  save_path=None):
    """
    Plot averaged results across all manifolds in one figure.
    """
    # Average across manifolds
    method_names = list(list(all_results.values())[0].keys())
    avg_results  = {}
    
    for method_name in method_names:
        all_devs = []
        for manifold_results in all_results.values():
            all_devs.append(manifold_results[method_name])
        avg_results[method_name] = np.nanmean(all_devs, axis=0)
    
    bound = [theoretical_bound(L, s, eps) for s in sigma_values]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(sigma_values, bound, 'k--', linewidth=2.5,
            label='Theoretical bound', zorder=5)
    
    for method_name, deviations in avg_results.items():
        ax.plot(sigma_values, deviations,
                marker=MARKERS[method_name],
                color=COLORS[method_name],
                linewidth=2.5, markersize=7,
                label=method_name)
    
    ax.set_title('Deviation vs Noise Level — All Manifolds (averaged)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Noise level σ', fontsize=12)
    ax.set_ylabel('Mean deviation ||T(x) - T(γ)||', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(sigma_values) * 1.05)
    ax.set_ylim(0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return avg_results


# ============================================================
# PART 6: MAIN
# ============================================================

if __name__ == '__main__':
    
    # ── Experimental parameters ──────────────────────────────
    N_POINTS     = 500          # points per manifold
    SIGMA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    L            = 1.0          # Lipschitz constant (PCA = 1)
    EPS          = 0.5          # threshold for bound computation
    
    # ── Define manifolds ─────────────────────────────────────
    manifolds = {
        'Sphere $S^2$':         generate_sphere,
        'Torus $T^2$':          generate_torus,
        'Linear Subspace':      lambda n, noise:
                                generate_linear_subspace(
                                    n=n, d=3, D=10, noise=noise),
    }
    
    # ── Run experiments ───────────────────────────────────────
    all_results = {}
    
    for manifold_name, manifold_fn in manifolds.items():
        results = run_experiment(
            manifold_name = manifold_name,
            gamma_fn      = manifold_fn,
            sigma_values  = SIGMA_VALUES,
            methods       = METHODS,
            n_points      = N_POINTS,
        )
        all_results[manifold_name] = results
    
    # ── Plot per-manifold results ─────────────────────────────
    plot_results(
        all_results  = all_results,
        sigma_values = SIGMA_VALUES,
        L            = L,
        eps          = EPS,
        save_path    = 'results_per_manifold.png'
    )
    
    # ── Plot combined (averaged) results ──────────────────────
    avg = plot_combined(
        all_results  = all_results,
        sigma_values = SIGMA_VALUES,
        L            = L,
        eps          = EPS,
        save_path    = 'results_combined.png'
    )
    
    # ── Print summary table ───────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY: Mean deviation at sigma=0.5")
    print("="*60)
    print(f"{'Method':<25} {'Avg deviation':>15} {'Below bound?':>15}")
    print("-"*60)
    
    bound_at_05 = theoretical_bound(L, 0.5, EPS)
    
    for method_name in METHODS:
        dev = avg[method_name][-1]
        below = "YES ✓" if dev <= bound_at_05 else "NO  ✗"
        print(f"{method_name:<25} {dev:>15.4f} {below:>15}")
    
    print(f"\nTheoretical bound at sigma=0.5: {bound_at_05:.4f}")
    print("="*60)
    print("\nDone! Check results_per_manifold.png and results_combined.png")

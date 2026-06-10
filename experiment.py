import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from umap import UMAP

# ── Generate manifolds ──────────────────────────────────────

def generate_sphere(n=1000):
    theta = np.random.uniform(0, np.pi, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    X = np.column_stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    return X

def generate_torus(n=1000, R=2, r=1):
    theta = np.random.uniform(0, 2*np.pi, n)
    phi   = np.random.uniform(0, 2*np.pi, n)
    X = np.column_stack([
        (R + r*np.cos(theta))*np.cos(phi),
        (R + r*np.cos(theta))*np.sin(phi),
        r*np.sin(theta)
    ])
    return X

def generate_linear(n=1000, d=3, D=50):
    A = np.random.randn(D, d)
    A, _ = np.linalg.qr(A)
    Z = np.random.randn(n, d)
    return Z @ A.T

# ── Add noise ───────────────────────────────────────────────

def add_noise(X, sigma):
    return X + np.random.normal(0, sigma, X.shape)

# ── Apply visualization methods ─────────────────────────────

def apply_method(name, X):
    if name == 'PCA':
        return PCA(n_components=2).fit_transform(X)
    elif name == 'tSNE':
        return TSNE(n_components=2, random_state=42).fit_transform(X)
    elif name == 'UMAP':
        return UMAP(n_components=2, random_state=42).fit_transform(X)
    elif name == 'LaplacianEigenmaps':
        return SpectralEmbedding(n_components=2).fit_transform(X)

# ── Compute deviation ───────────────────────────────────────

def compute_deviation(T_true, T_noisy):
    return np.mean(np.linalg.norm(T_noisy - T_true, axis=1))

# ── Main experiment ─────────────────────────────────────────

manifolds = {
    'Sphere':          generate_sphere(1000),
    'Torus':           generate_torus(1000),
    'Linear Subspace': generate_linear(1000)
}

methods = ['PCA', 'LaplacianEigenmaps', 'UMAP', 'tSNE']
sigmas  = [0.01, 0.05, 0.1, 0.2, 0.5]

for manifold_name, gamma in manifolds.items():
    results = {m: [] for m in methods}

    for sigma in sigmas:
        x_noisy = add_noise(gamma, sigma)
        for method in methods:
            T_true  = apply_method(method, gamma)
            T_noisy = apply_method(method, x_noisy)
            dev = compute_deviation(T_true, T_noisy)
            results[method].append(dev)

    # ── Plot ────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    colors = {
        'PCA':               'blue',
        'LaplacianEigenmaps':'green',
        'UMAP':              'orange',
        'tSNE':              'red'
    }
    for method in methods:
        plt.plot(sigmas, results[method],
                 marker='o', label=method,
                 color=colors[method])

    # Theoretical bound
    L, eps = 1, 0.5
    bound = [L**2 * s**2 / eps**2 for s in sigmas]
    plt.plot(sigmas, bound, 'k--', linewidth=2,
             label='Theoretical bound')

    plt.xlabel('Noise level σ')
    plt.ylabel('Mean deviation ||T(x) - T(γ)||')
    plt.title(f'Stability Validation — {manifold_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'results_{manifold_name}.png', dpi=150)
    plt.show()

print("Done! Results saved as PNG files.")

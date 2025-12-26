# utils/latent_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class LatentCollector:
    """
    Collects encoder outputs z0 over many batches and analyzes:
      - mean & std per dimension
      - covariance & eigenvalues (anisotropy)
      - PCA spectrum
      - correlation matrix
      - histograms (optional)
    """

    def __init__(self, latent_dim, max_batches=200):
        self.latent_dim = latent_dim
        self.max_batches = max_batches
        self.storage = []

    @torch.no_grad()
    def add_batch(self, z0):
        """
        z0: [B, latent_dim]
        """
        if len(self.storage) < self.max_batches:
            self.storage.append(z0.detach().cpu())
        # else ignore (enough data collected)

    def stacked(self):
        return torch.cat(self.storage, dim=0)  # [N, D]

    def compute_basic_stats(self):
        Z = self.stacked()
        mean = Z.mean(0)
        std = Z.std(0)
        return mean, std

    def compute_covariance(self):
        Z = self.stacked()
        cov = torch.cov(Z.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        return cov, eigvals, eigvecs

    def compute_pca(self, n_components=20):
        Z = self.stacked().numpy()
        pca = PCA(n_components=n_components)
        pca.fit(Z)
        return pca.explained_variance_ratio_, pca

    def correlation_matrix(self):
        Z = self.stacked().numpy()
        return np.corrcoef(Z.T)

    def plot_pca_spectrum(self, explained_ratio):
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(explained_ratio), marker='o')
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA Spectrum of Encoder Latents")
        plt.grid(True)
        plt.show()

    def plot_histograms(self, num_dims=8):
        Z = self.stacked().numpy()
        dims = min(num_dims, self.latent_dim)
        plt.figure(figsize=(12, 8))
        for i in range(dims):
            plt.subplot(2, dims // 2, i + 1)
            plt.hist(Z[:, i], bins=40, density=True)
            plt.title(f"Dim {i}")
        plt.tight_layout()
        plt.show()
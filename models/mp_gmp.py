"""Memory Polynomial and Generalized Memory Polynomial models.

Both expose:
  build_features(x) -> complex feature matrix Phi[N, F]
  fit(x, y)         -> coeffs via least squares
  predict(x)        -> y_hat
"""
import numpy as np


class MemoryPolynomial:
    """y[n] = sum_{k=1..K} sum_{q=0..Q} a_kq * x[n-q] * |x[n-q]|^(k-1)"""

    def __init__(self, K: int = 7, Q: int = 4):
        self.K = K
        self.Q = Q
        self.coeffs: np.ndarray | None = None

    def build_features(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        feats = []
        for q in range(self.Q + 1):
            xq = np.zeros(N, dtype=np.complex64)
            if q == 0:
                xq[:] = x
            else:
                xq[q:] = x[:-q]
            for k in range(1, self.K + 1):
                feats.append(xq * np.abs(xq) ** (k - 1))
        return np.stack(feats, axis=1).astype(np.complex64)  # [N, (K)*(Q+1)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        Phi = self.build_features(x)
        # ignore the first Q samples to avoid zero-padding bias
        Phi_t = Phi[self.Q:]
        y_t = y[self.Q:]
        self.coeffs, *_ = np.linalg.lstsq(Phi_t, y_t, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.coeffs is not None, "fit first"
        Phi = self.build_features(x)
        return (Phi @ self.coeffs).astype(np.complex64)


class GMP:
    """Generalized MP with aligned, lagging and leading envelope terms.

    Aligned:  x[n-q] * |x[n-q]|^(k-1)
    Lagging:  x[n-q] * |x[n-q-m]|^(k-1), m=1..L
    Leading:  x[n-q] * |x[n-q+m]|^(k-1), m=1..L
    """

    def __init__(self, K: int = 5, Q: int = 3, L: int = 2):
        self.K = K
        self.Q = Q
        self.L = L
        self.coeffs: np.ndarray | None = None

    @staticmethod
    def _shift(x: np.ndarray, q: int) -> np.ndarray:
        N = len(x)
        out = np.zeros(N, dtype=np.complex64)
        if q == 0:
            out[:] = x
        elif q > 0:
            out[q:] = x[:-q]
        else:
            out[:q] = x[-q:]
        return out

    def build_features(self, x: np.ndarray) -> np.ndarray:
        feats = []
        # Aligned
        for q in range(self.Q + 1):
            xq = self._shift(x, q)
            for k in range(1, self.K + 1):
                feats.append(xq * np.abs(xq) ** (k - 1))
        # Lagging
        for q in range(self.Q + 1):
            xq = self._shift(x, q)
            for m in range(1, self.L + 1):
                xqm = self._shift(x, q + m)
                for k in range(2, self.K + 1):
                    feats.append(xq * np.abs(xqm) ** (k - 1))
        # Leading
        for q in range(self.Q + 1):
            xq = self._shift(x, q)
            for m in range(1, self.L + 1):
                xqm = self._shift(x, q - m)
                for k in range(2, self.K + 1):
                    feats.append(xq * np.abs(xqm) ** (k - 1))
        return np.stack(feats, axis=1).astype(np.complex64)

    def fit(self, x: np.ndarray, y: np.ndarray):
        Phi = self.build_features(x)
        skip = self.Q + self.L
        Phi_t = Phi[skip:]
        y_t = y[skip:]
        self.coeffs, *_ = np.linalg.lstsq(Phi_t, y_t, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.coeffs is not None, "fit first"
        Phi = self.build_features(x)
        return (Phi @ self.coeffs).astype(np.complex64)

    def n_features(self) -> int:
        return ((self.Q + 1) * self.K
                + (self.Q + 1) * self.L * (self.K - 1)
                + (self.Q + 1) * self.L * (self.K - 1))

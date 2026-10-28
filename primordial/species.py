"""Automatic species detection: cluster particles by body + behaviour, then name clusters.

Features are a mix of genes (what a particle *is*) and slow moving averages of its
actions (what it *does*), plus diet (how much energy came from killing). Clustering is
k-means, warm-started from the previous centroids so labels stay stable over time.
"""

import torch

FEATURES = ["size", "speed", "sense", "moving", "attacking", "signalling", "scenting", "bonds", "predation"]


def features(world, idx: torch.Tensor) -> torch.Tensor:
    g = world.genome
    diet = world.prey_gained[idx] / (world.prey_gained[idx] + world.food_gained[idx] + 1e-6)
    return torch.cat([g.body[idx], world.ema[idx], diet[:, None]], dim=1)


def name_cluster(c: dict) -> str:
    if c["predation"] > 0.35:
        return "hunter"
    if c["bonds"] > 1.0:
        return "colonial"
    if c["moving"] < 0.15:
        return "drifter"
    if c["speed"] > 0.6 and c["sense"] > 0.6:
        return "scout"
    if c["signalling"] > 0.35:
        return "signaller"
    if c["scenting"] > 0.35:
        return "trail-layer"
    return "grazer"


class SpeciesTracker:
    def __init__(self, k: int = 5, seed: int = 0, iters: int = 15):
        self.k, self.iters = k, iters
        self.gen = torch.Generator().manual_seed(seed + 12345)
        self.centroids: torch.Tensor | None = None   # raw feature space
        self.names: list[str] = []

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """x (n, F) raw features -> labels (n,). Updates centroids and names."""
        n = len(x)
        if n == 0:
            return torch.zeros(0, dtype=torch.long)
        k = min(self.k, n)
        x = x.cpu()
        mu, sd = x.mean(0), x.std(0).clamp_min(1e-3) if n > 1 else torch.ones(x.shape[1])
        z = (x - mu) / sd

        if self.centroids is None or len(self.centroids) != k:
            c = self._kmeans_pp(z, k)
        else:
            c = (self.centroids - mu) / sd

        for _ in range(self.iters):
            labels = torch.cdist(z, c).argmin(1)
            for j in range(k):
                members = z[labels == j]
                if len(members):
                    c[j] = members.mean(0)
                else:  # empty cluster: re-seed at the point worst served
                    c[j] = z[torch.cdist(z, c).min(1).values.argmax()]
        labels = torch.cdist(z, c).argmin(1)

        self.centroids = c * sd + mu
        self.names = self._unique_names([name_cluster(dict(zip(FEATURES, row.tolist()))) for row in self.centroids])
        return labels

    def _kmeans_pp(self, z: torch.Tensor, k: int) -> torch.Tensor:
        first = torch.randint(0, len(z), (1,), generator=self.gen)
        c = [z[first[0]]]
        for _ in range(1, k):
            d2 = torch.cdist(z, torch.stack(c)).min(1).values ** 2
            p = d2 / d2.sum() if d2.sum() > 0 else torch.full_like(d2, 1 / len(d2))
            c.append(z[torch.multinomial(p, 1, generator=self.gen)[0]])
        return torch.stack(c).clone()

    @staticmethod
    def _unique_names(names: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out = []
        for name in names:
            seen[name] = seen.get(name, 0) + 1
            out.append(name if seen[name] == 1 else f"{name} {seen[name]}")
        return out

    def state_dict(self) -> dict:
        return {"centroids": self.centroids, "names": self.names, "rng": self.gen.get_state()}

    def load_state(self, s: dict) -> None:
        self.centroids, self.names = s["centroids"], list(s["names"])
        self.gen.set_state(s["rng"])

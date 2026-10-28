"""Neighbour search on a torus via a uniform grid, in pure tensor ops.

`neighbours(q, t, radius, world)` returns, for every query point, a padded list
of the target indices within `radius`:

    cand    (nq, K) long   index into t (0 where padded)
    valid   (nq, K) bool   False for padding (and for self, if exclude_self)
    dx, dy  (nq, K)        shortest torus displacement q -> t[cand]
    dist    (nq, K)        |(dx, dy)|, +inf where not valid

x and y are kept as separate arrays on purpose: broadcasting over a trailing
dimension of size 2 is ~15x slower on CPU than two plain 2-D ops.

Large queries use a cell grid (sort points by cell, then gather the 9 cells around
each query); small ones use all pairs. Either way the result is compacted to the
largest number of in-range neighbours any query actually has.
"""

import math
from dataclasses import dataclass

import torch

OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
DENSE_PAIRS = 60_000  # below this many query x target pairs, all-pairs beats building a grid


@dataclass
class Neighbours:
    cand: torch.Tensor
    valid: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    dist: torch.Tensor


def wrap(delta: torch.Tensor, world: float) -> torch.Tensor:
    return delta - torch.round(delta / world) * world


def _grid_candidates(q: torch.Tensor, t: torch.Tensor, radius: float, world: float):
    """(nq, 9*M) candidate indices (-1 = empty), or None when a grid would not help."""
    nq, nt, dev = len(q), len(t), q.device
    g = int(world / radius)
    if g < 3 or nq * nt <= DENSE_PAIRS:
        return None
    cell = (t / world * g).long().clamp(0, g - 1)
    cid = cell[:, 0] * g + cell[:, 1]
    counts = torch.bincount(cid, minlength=g * g)
    m = int(counts.max())
    if 9 * m >= nt:  # heavily clumped: the grid would scan more than all pairs
        return None
    order = torch.argsort(cid, stable=True)
    sorted_cid = cid[order]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(nt, device=dev) - starts[sorted_cid]
    table = torch.full((g * g, m), -1, dtype=torch.long, device=dev)
    table[sorted_cid, rank] = order
    qcell = (q / world * g).long().clamp(0, g - 1)
    off = torch.tensor(OFFSETS, device=dev)
    ncell = (qcell[:, None, :] + off[None]) % g
    return table[ncell[..., 0] * g + ncell[..., 1]].reshape(nq, 9 * m)


def neighbours(q: torch.Tensor, t: torch.Tensor, radius: float, world: float,
               exclude_self: bool = False) -> Neighbours:
    nq, nt, dev = len(q), len(t), q.device
    if nt == 0 or nq == 0:
        z = torch.zeros(nq, 0, device=dev)
        return Neighbours(z.long(), z.bool(), z, z, z)

    qx, qy, tx, ty = q[:, 0].contiguous(), q[:, 1].contiguous(), t[:, 0].contiguous(), t[:, 1].contiguous()
    cand = _grid_candidates(q, t, radius, world)
    if cand is None:  # all pairs: plain outer differences, no gather
        dx = wrap(tx[None, :] - qx[:, None], world)
        dy = wrap(ty[None, :] - qy[:, None], world)
        cand = torch.arange(nt, device=dev).expand(nq, nt)
        in_range = dx * dx + dy * dy <= radius * radius
    else:
        present = cand >= 0
        cand = cand.clamp_min(0)
        dx = wrap(tx[cand] - qx[:, None], world)
        dy = wrap(ty[cand] - qy[:, None], world)
        in_range = present & (dx * dx + dy * dy <= radius * radius)
    if exclude_self:
        in_range &= cand != torch.arange(nq, device=dev)[:, None]

    # Compact: pack each row's in-range entries to the left.
    rows, cols = in_range.nonzero(as_tuple=True)
    k = int(in_range.sum(1).max()) if len(rows) else 0
    slot = (in_range.cumsum(1) - 1)[rows, cols]
    out_cand = torch.zeros(nq, k, dtype=torch.long, device=dev)
    out_dx = torch.zeros(nq, k, device=dev)
    out_dy = torch.zeros(nq, k, device=dev)
    valid = torch.zeros(nq, k, dtype=torch.bool, device=dev)
    out_cand[rows, slot] = cand[rows, cols]
    out_dx[rows, slot] = dx[rows, cols]
    out_dy[rows, slot] = dy[rows, cols]
    valid[rows, slot] = True
    dist = torch.sqrt(out_dx * out_dx + out_dy * out_dy).masked_fill(~valid, math.inf)
    return Neighbours(out_cand, valid, out_dx, out_dy, dist)

"""Append-only family record: every particle ever born gets a unique id (uid).

Indexed by uid, so lookups are O(1). Births are rare compared to steps, so plain
Python lists are both simple and fast enough.
"""

from collections import defaultdict


class Lineage:
    def __init__(self):
        self.parent: list[int] = []   # -1 for spontaneously spawned particles
        self.mate: list[int] = []     # -1 for asexual births
        self.born: list[int] = []
        self.died: list[int] = []     # -1 while alive
        self.color: list[tuple] = []
        self.size: list[float] = []

    def __len__(self) -> int:
        return len(self.parent)

    def register(self, parents, mates, step, colors, sizes) -> list[int]:
        start = len(self.parent)
        n = len(parents)
        self.parent += list(parents)
        self.mate += list(mates)
        self.born += [step] * n
        self.died += [-1] * n
        self.color += [tuple(c) for c in colors]
        self.size += list(sizes)
        return list(range(start, start + n))

    def record_deaths(self, uids, step) -> None:
        for u in uids:
            self.died[u] = step

    def ancestors(self, uid: int, depth: int = 50) -> list[int]:
        """uid, parent, grandparent, ... up to `depth` entries."""
        chain = []
        while uid >= 0 and len(chain) < depth:
            chain.append(uid)
            uid = self.parent[uid]
        return chain

    def living_tree(self, alive_uids) -> list[dict]:
        """The tree connecting all living particles, with unbranched chains collapsed.

        Returns nodes in parent-before-child order:
            {uid, parent (uid of nearest kept ancestor, or -1), born, died, color, alive, leaves}
        A node is kept if it is alive, a root, or a branch point (2+ surviving child lines).
        """
        alive = set(alive_uids)
        children = defaultdict(list)
        seen = set()
        roots = []
        for u in alive:
            v = u
            while v not in seen:
                seen.add(v)
                p = self.parent[v]
                if p < 0:
                    roots.append(v)
                    break
                children[p].append(v)
                v = p

        nodes = []
        leaves_of = {}

        def count_leaves(root):
            # iterative post-order count of living descendants (including self)
            stack, order = [root], []
            while stack:
                v = stack.pop()
                order.append(v)
                stack.extend(children[v])
            for v in reversed(order):
                leaves_of[v] = (v in alive) + sum(leaves_of[c] for c in children[v])

        for r in roots:
            count_leaves(r)
            stack = [(r, -1)]
            while stack:
                v, kept_parent = stack.pop()
                keep = v in alive or kept_parent == -1 or len(children[v]) >= 2
                if keep:
                    nodes.append({
                        "uid": v, "parent": kept_parent, "born": self.born[v], "died": self.died[v],
                        "color": self.color[v], "alive": v in alive, "leaves": leaves_of[v],
                    })
                    kept_parent = v
                for c in children[v]:
                    stack.append((c, kept_parent))
        return nodes

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("parent", "mate", "born", "died", "color", "size")}

    @classmethod
    def from_state(cls, state: dict) -> "Lineage":
        lin = cls()
        for k, v in state.items():
            setattr(lin, k, list(v) if k != "color" else [tuple(c) for c in v])
        return lin

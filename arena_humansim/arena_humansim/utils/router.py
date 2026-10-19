from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from arena_humansim.utils.funnel import funnel, polyline
from arena_humansim.utils.mesh import CELL, TOL, Mesh, SegIndex, _cross, _expand, _pt_seg, _seg_seg

VERIFY_EPS = 5e-3  # accepted shortfall below r
MAX_INSERT = 4
MAX_RESEARCH = 4
ASTAR_CAP = 300  # expansions before a one-off query switches to a scipy Dijkstra tree
TREE_CACHE = 64
GAP_SHARE = 0.45  # share of a passage width that one corner disc may take
ROOM = 0.8  # share of the distance to a terminal that a corner disc may take
SLOPE = 0.5  # max growth of the corner radius per metre along a channel side
SNAP_MARGIN = 0.01  # a snapped point ends up this far outside the inflation band [m]
SNAP_TRIES = 8
REACH_NUDGE = 1e-3  # candidates start this far inside their triangle so the snap pushes them to its side of a wall [m]
REACH_TRIES = 12

Route = tuple[np.ndarray, np.ndarray]


class Router:
    """Shortest paths with clearance radius over a Mesh. Corners keep comfort_radius where the passage allows."""

    def __init__(self, mesh: Mesh, radius: float, comfort_radius: float) -> None:
        if 2 * radius > mesh.reach:
            raise ValueError("radius above the width cap of this mesh")
        self.mesh = mesh
        self.r = radius
        self.comfort = max(comfort_radius, radius)
        self._rho: np.ndarray | None = None
        if comfort_radius > radius:
            gap = np.where(mesh.vertex_gap < mesh.reach, mesh.vertex_gap, np.inf)
            self._rho = np.clip(GAP_SHARE * gap, radius, comfort_radius)
        self._mx, self._my = mesh.e_mid[:, 0].tolist(), mesh.e_mid[:, 1].tolist()
        self._trees: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self._index, self._walls = mesh.index, mesh.con_edges
        self._blocked = np.zeros(0, np.int64)
        self._graph()

    def set_blocked(self, edges: np.ndarray) -> None:
        """Treat exactly these free mesh edges as walls."""
        mesh = self.mesh
        self._blocked = np.asarray(edges, dtype=np.int64)
        self._index, self._walls = mesh.index, mesh.con_edges
        if len(self._blocked):
            self._walls = np.vstack([mesh.con_edges, np.column_stack([mesh.ea[self._blocked], mesh.eb[self._blocked]])])
            self._index = SegIndex(mesh.P[self._walls[:, 0]], mesh.P[self._walls[:, 1]])
        self._trees.clear()
        self._graph()

    def clearance(self, pts: np.ndarray) -> np.ndarray:
        """Exact distance to the nearest wall, inf beyond the comfort radius."""
        return self._index.point_clearance(np.atleast_2d(np.asarray(pts, dtype=np.float64)), self.comfort)

    def _graph(self) -> None:
        mesh = self.mesh
        E = len(mesh.ea)
        self._open = ~mesh.e_con & (mesh.e_len >= 2 * self.r - 1e-9)
        self._open[self._blocked] = False
        m = (mesh.arc_w >= 2 * self.r - 1e-9) & self._open[mesh.arc_a] & self._open[mesh.arc_b]
        a, b = mesh.arc_a[m], mesh.arc_b[m]
        c = mesh.arc_c[m]
        src, dst, cst = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([c, c])
        order = np.argsort(src, kind="stable")
        src, dst, cst = src[order], dst[order], cst[order]
        indptr = np.searchsorted(src, np.arange(E + 1))
        rank = np.arange(len(src)) - indptr[src]
        nbr = np.full((E, 4), -1, np.int64)
        nc = np.zeros((E, 4))
        nbr[src, rank] = dst
        nc[src, rank] = cst
        _, comp = connected_components(csr_matrix((cst, dst, indptr), shape=(E, E)), directed=False)
        self._indptr, self._indices, self._data, self._comp = indptr, dst, cst, comp
        self._nbr, self._cst = nbr.tolist(), nc.tolist()

    def _seg_clear(self, A: np.ndarray, B: np.ndarray, owner: np.ndarray, n_owner: int, pad: float) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, int, float] | None]:
        """Exact min wall distance per owner for path segments A->B (inf where nothing is within pad)."""
        index = self._index
        ln = np.hypot(*(B - A).T)
        npc = np.maximum(np.ceil(ln / CELL).astype(np.int64), 1)
        seg = np.repeat(np.arange(len(A)), npc)
        j = _expand(np.zeros(len(A), np.int64), npc)
        step = ((B - A) / npc[:, None])[seg]
        p0 = A[seg] + j[:, None] * step
        p1 = p0 + step
        q, wj = index.pairs(np.minimum(p0, p1) - pad, np.maximum(p0, p1) + pad)
        out = np.full(n_owner, np.inf)
        if not len(q):
            return out, None
        d = _seg_seg(p0[q], p1[q], index.A[wj], index.B[wj])
        np.minimum.at(out, owner[seg[q]], d)
        k = int(np.argmin(d))
        return out, (p0[q[k]], p1[q[k]], int(wj[k]), float(d[k]))

    def _violation(self, poly: np.ndarray) -> dict | None:
        P = self.mesh.P
        out, worst = self._seg_clear(poly[:-1], poly[1:], np.zeros(len(poly) - 1, np.int64), 1, self.r)
        if out[0] >= self.r - VERIFY_EPS:
            return None
        assert worst is not None
        p0, p1, wj, _ = worst
        a, b = self._walls[wj]
        da, ta = _pt_seg(P[a], p0, p1)
        db, tb = _pt_seg(P[b], p0, p1)
        dp = min(float(_pt_seg(p0, P[a], P[b])[0]), float(_pt_seg(p1, P[a], P[b])[0]))
        vid, dv, tv = (int(a), float(da), float(ta)) if da <= db else (int(b), float(db), float(tb))
        q = p0 + tv * (p1 - p0)
        crossing = _seg_seg(p0[None], p1[None], P[a][None], P[b][None])[0] == 0.0
        if crossing or dv > dp + 1e-9:
            return dict(q=q, vid=None)
        side = -1 if _cross(p1 - p0, P[vid] - q) > 0 else 1
        return dict(q=q, vid=vid, side=side)

    def _events(self, ch: list[int] | np.ndarray, ts: int, tg: int) -> tuple[dict | None, int | None]:
        """Channel as funnel events. On a channel that turns back on itself returns (None, portal index to ban)."""
        mesh = self.mesh
        ch = np.asarray(ch, dtype=np.int64)
        if not len(ch):
            if ts != tg:
                return None, 0
            return dict(sides=[], ids=[], xs=[], ys=[], forced=[], portal=[], L=ch, R=ch, tris=np.array([ts]), ch=ch), None
        t0, t1 = mesh.e_t0[ch], mesh.e_t1[ch]
        leave = np.empty(len(ch), np.int64)
        leave[0] = ts
        leave[1:] = np.where((t0[:-1] == t0[1:]) | (t0[:-1] == t1[1:]), t0[:-1], t1[:-1])
        flip = leave != t0
        enter = np.where(flip, t0, t1)
        uturn = np.nonzero(enter[:-1] != leave[1:])[0]
        if len(uturn):
            return None, int(uturn[0]) + 1
        if enter[-1] != tg:
            return None, len(ch) - 1
        Lv = np.where(flip, mesh.ea[ch], mesh.eb[ch])
        Rv = np.where(flip, mesh.eb[ch], mesh.ea[ch])
        ids = np.column_stack([Lv, Rv]).ravel()
        xy = mesh.P[ids]
        k = len(ch)
        ev = dict(
            sides=[-1, 1] * k,
            ids=ids.tolist(),
            xs=xy[:, 0].tolist(),
            ys=xy[:, 1].tolist(),
            forced=[False] * (2 * k),
            portal=np.repeat(np.arange(1, k + 1), 2).tolist(),
            L=Lv,
            R=Rv,
            tris=np.concatenate([[ts], enter]),
            ch=ch,
        )
        return ev, None

    def _locate(self, ev: dict, q: np.ndarray) -> int:
        """Index of the channel triangle holding q, else the one after the nearest portal."""
        tq = int(self.mesh.tri.find_simplex(q))
        at = np.nonzero(ev["tris"] == tq)[0]
        if len(at) or not len(ev["ch"]):
            return int(at[0]) if len(at) else 0
        return int(np.argmin(np.hypot(*(self.mesh.e_mid[ev["ch"]] - q).T)))

    def _insert(self, ev: dict, viol: dict) -> bool:
        """Add an off-channel vertex whose circle intrudes into the channel as an extra funnel event."""
        P, V = self.mesh.P, self.mesh.V
        vid = viol["vid"]
        if vid is None:
            return False
        side = viol["side"]
        if vid in ev["ids"]:
            for e, (i, sd) in enumerate(zip(ev["ids"], ev["sides"], strict=False)):
                if i == vid and sd == side:
                    if ev["forced"][e]:
                        return False
                    ev["forced"][e] = True
                    return True
            return False
        k = len(ev["ch"])
        i = self._locate(ev, viol["q"])
        w = P[vid]
        best, best_d = None, np.inf
        for j in range(max(0, i - 3), min(k, i + 3) + 1):
            if j in (0, k):
                tv = V[ev["tris"][j]]
                d = min(float(_pt_seg(w, P[tv[m]], P[tv[(m + 1) % 3]])[0]) for m in range(3))
            else:
                left = ev["R"][j - 1] == ev["R"][j]
                if left != (side == -1):
                    continue
                arr = ev["L"] if left else ev["R"]
                d = float(_pt_seg(w, P[arr[j - 1]], P[arr[j]])[0])
            if d < best_d:
                best, best_d = j, d
        if best is None:
            return False
        pos = 0
        while pos < len(ev["portal"]) and ev["portal"][pos] <= best:
            pos += 1
        ev["sides"].insert(pos, side)
        ev["ids"].insert(pos, vid)
        ev["xs"].insert(pos, float(w[0]))
        ev["ys"].insert(pos, float(w[1]))
        ev["portal"].insert(pos, best)
        ev["forced"].insert(pos, False)
        return True

    def _pull(self, s: np.ndarray, g: np.ndarray, ev: dict) -> np.ndarray:
        return polyline(funnel(s, g, ev["sides"], ev["ids"], ev["xs"], ev["ys"], ev["forced"], [self.r] * len(ev["sides"])))

    def _radii(self, s: np.ndarray, g: np.ndarray, ev: dict) -> np.ndarray:
        """Comfort radius per event: the two ends of a portal share its length, terminals stay outside, no disc swallows its chain neighbour."""
        ends = np.stack([ev["L"], ev["R"]])
        v = self.mesh.P[ends]
        x, y = v[..., 0], v[..., 1]
        assert self._rho is not None
        rho = self._rho[ends]
        span = 2 * GAP_SHARE * np.hypot(x[0] - x[1], y[0] - y[1])
        rho = np.minimum(rho, np.maximum(span / 2, span - rho[::-1]))
        room = np.minimum(np.hypot(x - s[0], y - s[1]), np.hypot(x - g[0], y - g[1]))
        rho = np.maximum(np.minimum(rho, ROOM * room), self.r)
        run = np.zeros_like(rho)
        np.cumsum(SLOPE * np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)), axis=1, out=run[:, 1:])
        rho = np.minimum(run + np.minimum.accumulate(rho - run, axis=1), np.minimum.accumulate((rho + run)[:, ::-1], axis=1)[:, ::-1] - run)
        return rho.T.ravel()

    def _comfort_pull(self, s: np.ndarray, g: np.ndarray, ev: dict) -> np.ndarray | None:
        """Pull with per-corner comfort radii, unverified. None when the funnel degenerates or ev carries inserted events."""
        if len(ev["ids"]) != 2 * len(ev["ch"]):
            return None
        return polyline(funnel(s, g, ev["sides"], ev["ids"], ev["xs"], ev["ys"], ev["forced"], self._radii(s, g, ev).tolist(), nearest_first=True), strict=True)

    def _string_pull(self, s: np.ndarray, g: np.ndarray, ev: dict, first: np.ndarray | None = None) -> tuple[np.ndarray | None, int | None]:
        """Funnel, verify, repair by vertex insertion. Returns (path, None) or (None, channel triangle index to ban)."""
        if self._rho is not None and first is None:
            poly = self._comfort_pull(s, g, ev)
            if poly is not None and self._violation(poly) is None:
                return poly, None
        viol = None
        for rep in range(MAX_INSERT + 1):
            poly = first if rep == 0 and first is not None else self._pull(s, g, ev)
            viol = self._violation(poly)
            if viol is None:
                return poly, None
            if not self._insert(ev, viol):
                break
        assert viol is not None
        return None, self._locate(ev, viol["q"])

    def _terminals(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validity, triangle and usable edge nodes (n,3) with costs for query points."""
        mesh = self.mesh
        pts = np.atleast_2d(pts)
        t = mesh.tri.find_simplex(pts).astype(np.int64)
        ok = (t >= 0) & (self._index.point_clearance(pts, self.r) >= self.r - 1e-9)
        nodes = mesh.he[np.maximum(t, 0)]
        use = ok[:, None] & self._open[nodes]
        reach = np.hypot(mesh.e_mid[nodes, 0] - pts[:, None, 0], mesh.e_mid[nodes, 1] - pts[:, None, 1])
        return ok, t, nodes, np.where(use, reach, np.inf)

    def _astar(self, g: np.ndarray, snodes: dict[int, float], gnodes: dict[int, float], banned: set[tuple[int, int]] | None, cap: int = 0) -> list[int] | None:
        nbr, cst, mx, my = self._nbr, self._cst, self._mx, self._my
        gx, gy = float(g[0]), float(g[1])
        best, parent, closed, heap = {}, {}, set(), []
        for n, c in snodes.items():
            best[n] = c
            parent[n] = None
            heapq.heappush(heap, (c + math.hypot(mx[n] - gx, my[n] - gy), c, n))
        while heap:
            _, gc, n = heapq.heappop(heap)
            if n == -1:
                out = []
                n = parent[-1]
                while n is not None:
                    out.append(n)
                    n = parent[n]
                return out[::-1]
            if n in closed:
                continue
            closed.add(n)
            if cap and len(closed) > cap:
                return None
            if n in gnodes:
                c2 = gc + gnodes[n]
                if c2 < best.get(-1, math.inf):
                    best[-1] = c2
                    parent[-1] = n
                    heapq.heappush(heap, (c2, c2, -1))
            row, crow = nbr[n], cst[n]
            for j in range(4):
                m = row[j]
                if m < 0:
                    break
                if banned and (n, m) in banned:
                    continue
                c2 = gc + crow[j]
                if c2 < best.get(m, math.inf):
                    best[m] = c2
                    parent[m] = n
                    heapq.heappush(heap, (c2 + math.hypot(mx[m] - gx, my[m] - gy), c2, m))
        return None

    def _route(self, s: np.ndarray, g: np.ndarray, ts: int, tg: int, snodes: dict[int, float], gnodes: dict[int, float], ch: list[int] | None = None) -> Route | None:
        """String-pull a channel. On an unrepairable violation ban the offending traversal and search again."""
        comp = self._comp
        gcomp = {int(comp[n]) for n in gnodes}
        snodes = {n: c for n, c in snodes.items() if int(comp[n]) in gcomp}
        banned: set[tuple[int, int]] = set()
        for _ in range(MAX_RESEARCH + 1):
            if not snodes or not gnodes:
                return None
            if ch is None:
                ch = self._astar(g, snodes, gnodes, banned)
            if ch is None:
                return None
            ev, hint = self._events(ch, ts, tg)
            if ev is not None:
                poly, hint = self._string_pull(s, g, ev)
                if poly is not None:
                    return poly, ev["ch"]
            if len(ch) == 1 and ts == tg:
                return None
            assert hint is not None
            if hint <= 0:
                snodes.pop(ch[0], None)
            elif hint >= len(ch):
                gnodes.pop(ch[-1], None)
            else:
                banned.add((ch[hint - 1], ch[hint]))
                banned.add((ch[hint], ch[hint - 1]))
            ch = None
        return None

    def _tree(self, g: np.ndarray, gnodes: np.ndarray, gcost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shortest-path tree towards goal g over the edge graph, cached per goal."""
        key = (float(g[0]), float(g[1]))
        hit = self._trees.pop(key, None)
        if hit is None:
            E = len(self.mesh.ea)
            gm = np.isfinite(gcost)
            indptr = np.append(self._indptr, self._indptr[-1] + gm.sum())
            graph = csr_matrix((np.append(self._data, gcost[gm]), np.append(self._indices, gnodes[gm]), indptr), shape=(E + 1, E + 1))
            hit = dijkstra(graph, directed=True, indices=E, return_predecessors=True)
            if len(self._trees) >= TREE_CACHE:
                self._trees.pop(next(iter(self._trees)))
        self._trees[key] = hit
        return hit

    def _walk(self, pred: np.ndarray, first: int) -> list[int]:
        ch, m, E = [], int(first), len(self.mesh.ea)
        while m != E:
            ch.append(m)
            m = int(pred[m])
        return ch

    def snap(self, p: np.ndarray) -> np.ndarray | None:
        """p itself when it keeps the radius, else p pushed off the nearest walls, None when that fails."""
        for flip in (1.0, -1.0):
            q = p
            for _ in range(SNAP_TRIES):
                hit = self._index.nearest(q, self.r)
                if hit is None or hit[0] >= self.r:
                    return q
                d, wall, along = hit
                away = (q - wall) / d if d > 1e-9 else flip * np.array([-along[1], along[0]]) / np.hypot(*along)
                q = wall + away * (self.r + SNAP_MARGIN)
        return None

    def nearest_reachable(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        """Point of the free space connected to start that lies closest to goal, None when start is walled in."""
        s, g = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        ok, t, nodes, cost = self._terminals(s[None])
        use = np.isfinite(cost[0])
        if not ok[0] or not use.any():
            return None
        mesh = self.mesh
        e = np.nonzero(np.isin(self._comp, self._comp[nodes[0][use]]) & self._open)[0]
        tris = np.unique(np.concatenate([mesh.e_t0[e], mesh.e_t1[e], t]))
        V = mesh.P[mesh.V[tris[tris >= 0]]]
        A, D = V, V[:, [1, 2, 0]] - V
        u = np.clip(((g - A) * D).sum(2) / np.maximum((D * D).sum(2), TOL**2), 0.0, 1.0)
        C = A + u[..., None] * D
        inward = V.mean(1)[:, None] - C
        cand = (C + inward / np.maximum(np.hypot(inward[..., 0], inward[..., 1]), TOL)[..., None] * REACH_NUDGE).reshape(-1, 2)
        best, best_d = s, float(np.hypot(*(s - g)))
        for q0 in cand[np.argsort(np.hypot(*(cand - g).T), kind="stable")[:REACH_TRIES]]:
            q = self.snap(q0)
            if q is not None and np.hypot(*(q - g)) < best_d and self.plan(s, q) is not None:
                best, best_d = q, float(np.hypot(*(q - g)))
        return best

    def line_of_sight(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Per segment A->B: clear of walls by the comfort radius."""
        n = len(A)
        clear, _ = self._seg_clear(A, B, np.arange(n), n, self.comfort)
        return clear >= self.comfort - VERIFY_EPS

    def plan(self, start: np.ndarray, goal: np.ndarray, try_line_of_sight: bool = True) -> Route | None:
        s, g = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        ok, t, nodes, cost = self._terminals(np.array([s, g]))
        if not ok.all():
            return None
        none = np.zeros(0, np.int64)
        if try_line_of_sight and self.line_of_sight(s[None], g[None])[0]:
            return np.array([s, g]), none
        if t[0] == t[1]:
            ev, _ = self._events([], int(t[0]), int(t[1]))
            assert ev is not None
            poly, _ = self._string_pull(s, g, ev)
            if poly is not None:
                return poly, none
        snodes = {int(n): float(c) for n, c in zip(nodes[0], cost[0], strict=False) if np.isfinite(c)}
        gnodes = {int(n): float(c) for n, c in zip(nodes[1], cost[1], strict=False) if np.isfinite(c)}
        comp = self._comp
        if not {int(comp[n]) for n in snodes} & {int(comp[n]) for n in gnodes}:
            return None
        ch = None
        if (float(g[0]), float(g[1])) not in self._trees:
            ch = self._astar(g, snodes, gnodes, None, cap=ASTAR_CAP)
        if ch is None:
            dist, pred = self._tree(g, nodes[1], cost[1])
            total = cost[0] + dist[nodes[0]]
            if not np.isfinite(total).any():
                return None
            ch = self._walk(pred, nodes[0][np.argmin(total)])
        return self._route(s, g, int(t[0]), int(t[1]), snodes, gnodes, ch)

    def _clear_many(self, polys: list[np.ndarray]) -> np.ndarray:
        cnt = np.array([len(p) - 1 for p in polys])
        A = np.vstack([p[:-1] for p in polys])
        B = np.vstack([p[1:] for p in polys])
        return self._seg_clear(A, B, np.repeat(np.arange(len(polys)), cnt), len(polys), self.r)[0]

    def plan_many(self, starts: np.ndarray, goal: np.ndarray) -> list[Route | None]:
        starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
        g = np.asarray(goal, dtype=np.float64)
        n = len(starts)
        out: list[Route | None] = [None] * n
        gok, gt, gnodes, gcost = self._terminals(g[None])
        if not gok[0]:
            return out
        ok, ts, nodes, cost = self._terminals(starts)
        dist, pred = self._tree(g, gnodes[0], gcost[0])
        total = cost + dist[nodes]
        pick = np.argmin(total, axis=1)
        reach = np.min(total, axis=1)
        first = nodes[np.arange(n), pick]
        euclid = np.hypot(*(starts - g).T)

        cand = np.nonzero(ok & (~np.isfinite(reach) | (reach <= 1.5 * euclid + 1.0) | (ts == gt[0])))[0]
        if len(cand):
            for i in cand[self.line_of_sight(starts[cand], np.repeat(g[None], len(cand), 0))]:
                out[i] = (np.array([starts[i], g]), np.zeros(0, np.int64))

        todo, evs = [], []
        for i in np.nonzero(ok & np.isfinite(reach))[0]:
            if out[i] is not None:
                continue
            if ts[i] == gt[0]:
                out[i] = self.plan(starts[i], g, try_line_of_sight=False)
                continue
            ev, _ = self._events(self._walk(pred, first[i]), int(ts[i]), int(gt[0]))
            if ev is None:
                out[i] = self.plan(starts[i], g)
                continue
            todo.append(i)
            evs.append(ev)
        if todo and self._rho is not None:
            soft = [self._comfort_pull(starts[i], g, ev) for i, ev in zip(todo, evs, strict=True)]
            have = [k for k, poly in enumerate(soft) if poly is not None]
            if have:
                clear = self._clear_many([poly for poly in soft if poly is not None])
                for k, c in zip(have, clear, strict=True):
                    if c >= self.r - VERIFY_EPS:
                        out[todo[k]] = (soft[k], evs[k]["ch"])
            keep = [k for k, i in enumerate(todo) if out[i] is None]
            todo, evs = [todo[k] for k in keep], [evs[k] for k in keep]
        if todo:
            polys = [self._pull(starts[i], g, ev) for i, ev in zip(todo, evs, strict=True)]
            clear = self._clear_many(polys)
            for i, ev, poly, c in zip(todo, evs, polys, clear, strict=True):
                if c >= self.r - VERIFY_EPS:
                    out[i] = (poly, ev["ch"])
                    continue
                fixed, _ = self._string_pull(starts[i], g, ev, first=poly)
                out[i] = (fixed, ev["ch"]) if fixed is not None else self.plan(starts[i], g)
        return out

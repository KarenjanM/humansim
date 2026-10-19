from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree  # pyright: ignore[reportAttributeAccessIssue]

TOL = 1e-5  # point merge and intersection tolerance [m]
MIN_LEN = 2e-3  # sub-edges shorter than this are never split
SHELL_UNIT = 1e-3  # concentric shell unit for splits next to an input vertex
PIECE = 1.0  # candidate-pair piece length for the intersection pass
CELL = 1.0  # segment index cell size
FRAME = 3.0  # margin of the free frame points around the map
FOOT_END = 0.05  # feet closer than this to a sub-edge end are dropped


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _expand(start: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Concatenated aranges start[i] .. start[i] + count[i]."""
    tot = int(count.sum())
    off = np.cumsum(count) - count
    return np.repeat(start - off, count) + np.arange(tot)


def split_segments(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split at crossings, T-junctions and collinear overlaps. Returns points (M,2), unique sub-edges (K,2)."""
    P, Q = segs[:, :2], segs[:, 2:]
    keep = np.hypot(*(Q - P).T) > TOL
    P, Q = P[keep], Q[keep]
    D = Q - P
    L = np.hypot(D[:, 0], D[:, 1])
    n = len(P)
    npc = np.maximum(np.ceil(L / PIECE).astype(np.int64), 1)
    owner = np.repeat(np.arange(n), npc)
    tm = (_expand(np.zeros(n, np.int64), npc) + 0.5) / npc[owner]
    mids = P[owner] + tm[:, None] * D[owner]
    pairs = cKDTree(mids).query_pairs(PIECE + 2 * TOL, output_type="ndarray")
    i, k = owner[pairs[:, 0]], owner[pairs[:, 1]]
    key = np.unique(np.minimum(i, k) * n + np.maximum(i, k))
    i, k = key // n, key % n
    m = i != k
    i, k = i[m], k[m]

    r, s, w = D[i], D[k], P[k] - P[i]
    den = _cross(r, s)
    par = np.abs(den) <= TOL * np.minimum(L[i], L[k])
    sid, st = [np.arange(n), np.arange(n)], [np.zeros(n), np.ones(n)]

    npar = ~par
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(w, s) / den
        u = _cross(w, r) / den
    ti, tk = TOL / L[i], TOL / L[k]
    hit = npar & (t >= -ti) & (t <= 1 + ti) & (u >= -tk) & (u <= 1 + tk)
    sid += [i[hit], k[hit]]
    st += [np.clip(t[hit], 0, 1), np.clip(u[hit], 0, 1)]

    col = par & (np.abs(_cross(w, r)) / L[i] <= TOL)
    ic, kc, wc, rc, sc = i[col], k[col], w[col], r[col], s[col]
    for seg, num in ((ic, (wc * rc).sum(1)), (ic, ((wc + sc) * rc).sum(1)), (kc, (-wc * sc).sum(1)), (kc, ((rc - wc) * sc).sum(1))):
        tt = num / L[seg] ** 2
        ok = (tt > 0) & (tt < 1)
        sid.append(seg[ok])
        st.append(tt[ok])

    sid = np.concatenate(sid)
    st = np.concatenate(st)
    X = P[sid] + st[:, None] * D[sid]
    near = cKDTree(X).query_pairs(TOL, output_type="ndarray")
    g = csr_matrix((np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(len(X), len(X)))
    _, lab = connected_components(g, directed=False)
    _, first = np.unique(lab, return_index=True)
    pts = X[first]

    order = np.lexsort((st, sid))
    a, b = lab[order[:-1]], lab[order[1:]]
    same = (sid[order[:-1]] == sid[order[1:]]) & (a != b)
    a, b = a[same], b[same]
    npt = len(pts)
    ek = np.unique(np.minimum(a, b) * npt + np.maximum(a, b))
    return pts, np.column_stack([ek // npt, ek % npt])


def add_feet(pts: np.ndarray, edges: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Split walls at the perpendicular foot of every vertex closer than reach and in line of sight.

    The gate between vertex and foot becomes a short mesh edge, so a gap between a corner and a wall interior is an edge length.
    """
    A, B = pts[edges[:, 0]], pts[edges[:, 1]]
    idx = SegIndex(A, B)
    v, e = idx.pairs(pts - reach, pts + reach)
    key = np.unique(v * len(edges) + e)
    v, e = key // len(edges), key % len(edges)
    d, t = _pt_seg(pts[v], A[e], B[e])
    ln = np.hypot(*(B[e] - A[e]).T)
    ok = (d > TOL) & (d < reach) & (t * ln > FOOT_END) & ((1 - t) * ln > FOOT_END)
    v, e, t = v[ok], e[ok], t[ok]
    F = A[e] + t[:, None] * (B[e] - A[e])
    C = pts[v]
    q, w = idx.pairs(np.minimum(C, F), np.maximum(C, F))
    other = (w != e[q]) & (edges[w, 0] != v[q]) & (edges[w, 1] != v[q])
    q, w = q[other], w[other]
    hidden = np.zeros(len(v), bool)
    hidden[q[_seg_seg(C[q], F[q], A[w], B[w]) < TOL]] = True
    e, t, F = e[~hidden], t[~hidden], F[~hidden]
    if not len(e):
        return pts, edges
    fid = len(pts) + np.arange(len(e))
    sid = np.concatenate([np.arange(len(edges)), np.arange(len(edges)), e])
    st = np.concatenate([np.zeros(len(edges)), np.ones(len(edges)), t])
    pid = np.concatenate([edges[:, 0], edges[:, 1], fid])
    order = np.lexsort((st, sid))
    a, b = pid[order[:-1]], pid[order[1:]]
    same = sid[order[:-1]] == sid[order[1:]]
    return np.vstack([pts, F]), np.column_stack([a[same], b[same]])


def _split_edges(pts: np.ndarray, is_input: np.ndarray, edges: np.ndarray, which: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the selected sub-edges. Shell split next to a lone input vertex, midpoint otherwise."""
    a, b = edges[which, 0], edges[which, 1]
    A, B = pts[a], pts[b]
    ln = np.hypot(*(B - A).T)
    frac = np.full(len(a), 0.5)
    ia, ib = is_input[a], is_input[b]
    shell = SHELL_UNIT * 2.0 ** np.round(np.log2(np.maximum(ln, 1e-12) / (2 * SHELL_UNIT)))
    fs = np.clip(shell / ln, 0.25, 0.75)
    frac = np.where(ia & ~ib, fs, frac)
    frac = np.where(ib & ~ia, 1 - fs, frac)
    new = A + frac[:, None] * (B - A)
    nid = len(pts) + np.arange(len(a))
    pts = np.vstack([pts, new])
    is_input = np.concatenate([is_input, np.zeros(len(a), bool)])
    rest = np.delete(edges, which, axis=0)
    fresh = np.vstack([np.column_stack([a, nid]), np.column_stack([nid, b])])
    return pts, is_input, np.vstack([rest, fresh]), len(rest)


def refine_gabriel(pts: np.ndarray, is_input: np.ndarray, edges: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split until every sub-edge has an empty diametral circle, which forces it into the Delaunay triangulation."""
    n_clean = 0
    new_from = 0
    for _ in range(80):
        A, B = pts[edges[:, 0]], pts[edges[:, 1]]
        mid, rad = (A + B) / 2, np.hypot(*(B - A).T) / 2
        enc = np.zeros(len(edges), bool)
        if n_clean < len(edges):
            d, _ = cKDTree(np.vstack([pts, extra])).query(mid[n_clean:], k=3)
            enc[n_clean:] = d[:, 2] <= rad[n_clean:] * (1 + 1e-9)
        if n_clean and new_from < len(pts):
            d, _ = cKDTree(pts[new_from:]).query(mid[:n_clean], k=1)
            enc[:n_clean] = d <= rad[:n_clean] * (1 + 1e-9)
        enc &= rad * 2 > MIN_LEN
        which = np.nonzero(enc)[0]
        if not len(which):
            break
        new_from = len(pts)
        pts, is_input, edges, n_clean = _split_edges(pts, is_input, edges, which)
    return pts, is_input, edges


class SegIndex:
    """Sparse uniform-cell index over wall sub-edges. Memory follows wall length."""

    def __init__(self, A: np.ndarray, B: np.ndarray, h: float = CELL) -> None:
        self.A, self.B, self.h = A, B, h
        lo = np.minimum(A, B).min(0)
        self.lo = lo
        c0 = np.floor((np.minimum(A, B) - lo) / h).astype(np.int64)
        c1 = np.floor((np.maximum(A, B) - lo) / h).astype(np.int64)
        self.nmax = c1.max(0)
        self.NX = int(self.nmax[0]) + 1
        nx = c1[:, 0] - c0[:, 0] + 1
        cnt = nx * (c1[:, 1] - c0[:, 1] + 1)
        seg = np.repeat(np.arange(len(A)), cnt)
        loc = _expand(np.zeros(len(A), np.int64), cnt)
        cx = c0[seg, 0] + loc % nx[seg]
        cy = c0[seg, 1] + loc // nx[seg]
        big = cnt[seg] > 4
        if big.any():
            ctr = lo + (np.column_stack([cx, cy]) + 0.5) * h
            d = _pt_seg(ctr, A[seg], B[seg])[0]
            ok = ~big | (d <= h * 0.7072)
            seg, cx, cy = seg[ok], cx[ok], cy[ok]
        key = cy * self.NX + cx
        order = np.argsort(key, kind="stable")
        self.keys, self.start = np.unique(key[order], return_index=True)
        self.end = np.append(self.start[1:], len(key))
        self.items = seg[order]

    def pairs(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(query index, wall index) candidates for query boxes lo..hi."""
        c0 = np.clip(np.floor((lo - self.lo) / self.h).astype(np.int64), 0, self.nmax)
        c1 = np.clip(np.floor((hi - self.lo) / self.h).astype(np.int64), 0, self.nmax)
        nx = c1[:, 0] - c0[:, 0] + 1
        cnt = nx * (c1[:, 1] - c0[:, 1] + 1)
        q = np.repeat(np.arange(len(lo)), cnt)
        loc = _expand(np.zeros(len(lo), np.int64), cnt)
        key = (c0[q, 1] + loc // nx[q]) * self.NX + c0[q, 0] + loc % nx[q]
        pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
        hit = self.keys[pos] == key
        q, pos = q[hit], pos[hit]
        c2 = self.end[pos] - self.start[pos]
        return np.repeat(q, c2), self.items[_expand(self.start[pos], c2)]

    def nearest(self, p: np.ndarray, r: float) -> tuple[float, np.ndarray, np.ndarray] | None:
        """Distance, closest wall point and wall direction for one point, None when no wall is within r."""
        _, j = self.pairs(p[None] - r, p[None] + r)
        if not len(j):
            return None
        d, t = _pt_seg(p[None], self.A[j], self.B[j])
        k = int(np.argmin(d))
        a, b = self.A[j[k]], self.B[j[k]]
        return float(d[k]), a + t[k] * (b - a), b - a

    def point_clearance(self, pts: np.ndarray, r: float) -> np.ndarray:
        """Exact distance to the nearest wall, capped at inf beyond r."""
        out = np.full(len(pts), np.inf)
        q, j = self.pairs(pts - r, pts + r)
        if len(q):
            np.minimum.at(out, q, _pt_seg(pts[q], self.A[j], self.B[j])[0])
        return out


def _pt_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = b - a
    l2 = np.maximum((d * d).sum(-1), 1e-300)
    t = np.clip(((p - a) * d).sum(-1) / l2, 0, 1)
    c = a + t[..., None] * d
    return np.hypot(p[..., 0] - c[..., 0], p[..., 1] - c[..., 1]), t


def _seg_seg(p0: np.ndarray, p1: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    d = np.minimum(np.minimum(_pt_seg(p0, w0, w1)[0], _pt_seg(p1, w0, w1)[0]), np.minimum(_pt_seg(w0, p0, p1)[0], _pt_seg(w1, p0, p1)[0]))
    o1, o2 = _cross(p1 - p0, w0 - p0), _cross(p1 - p0, w1 - p0)
    o3, o4 = _cross(w1 - w0, p0 - w0), _cross(w1 - w0, p1 - w0)
    return np.where((o1 * o2 < 0) & (o3 * o4 < 0), 0.0, d)


def _clip_dist(C: np.ndarray, A: np.ndarray, B: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Distance from C to the part of segment UW inside the wedge at C from A to B, counter-clockwise."""
    ca, cb, u, w = A - C, B - C, U - C, W - C
    lo = np.zeros(len(C))
    hi = np.ones(len(C))
    empty = np.zeros(len(C), bool)
    for fu, fw in ((_cross(ca, u), _cross(ca, w)), (_cross(u, cb), _cross(w, cb))):
        df = fw - fu
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = -fu / df
        lo = np.where(df > 0, np.maximum(lo, tc), lo)
        hi = np.where(df < 0, np.minimum(hi, tc), hi)
        empty |= (df == 0) & (fu < 0)
    d = w - u
    l2 = np.maximum((d * d).sum(1), 1e-300)
    ts = np.clip(-(u * d).sum(1) / l2, lo, np.maximum(hi, lo))
    q = u + ts[:, None] * d
    return np.where(empty | (lo > hi), np.inf, np.hypot(q[:, 0], q[:, 1]))


class Mesh:
    """Conforming Delaunay mesh over wall segments with exact passage widths up to reach."""

    n_frame = 4

    def __init__(self, tri: Delaunay, con_edges: np.ndarray, missing: np.ndarray, reach: float) -> None:
        self.tri = tri
        self.P: np.ndarray = tri.points
        self.con_edges = con_edges
        self.reach = reach
        self._topology(missing)
        self._widths()
        self.index = SegIndex(self.P[con_edges[:, 0]], self.P[con_edges[:, 1]])
        self._vertices: cKDTree | None = None
        tri.find_simplex(self.P[:1])

    @classmethod
    def build(cls, segs: np.ndarray, reach: float) -> Mesh:
        pts, edges = split_segments(np.asarray(segs, dtype=np.float64))
        pts, edges = add_feet(pts, edges, reach)
        lo, hi = pts.min(0) - FRAME, pts.max(0) + FRAME
        frame = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        is_input = np.ones(len(pts), bool)
        pts, is_input, edges = refine_gabriel(pts, is_input, edges, frame)
        for it in range(60):
            tri = Delaunay(np.vstack([pts, frame]))
            if len(tri.coplanar):
                remap = np.arange(len(pts) + 4)
                remap[tri.coplanar[:, 0]] = tri.coplanar[:, 2]
                edges = remap[edges]
                edges = edges[edges[:, 0] != edges[:, 1]]
            npt = len(pts) + 4
            s = tri.simplices.astype(np.int64)
            tk = np.unique(np.concatenate([np.minimum(s[:, i], s[:, j]) * npt + np.maximum(s[:, i], s[:, j]) for i, j in ((0, 1), (1, 2), (0, 2))]))
            ck = np.minimum(edges[:, 0], edges[:, 1]) * npt + np.maximum(edges[:, 0], edges[:, 1])
            miss = ~np.isin(ck, tk)
            ln = np.hypot(*(pts[edges[:, 0]] - pts[edges[:, 1]]).T)
            which = np.nonzero(miss & (ln > MIN_LEN))[0]
            if not len(which) or it == 59:
                break
            pts, is_input, edges, _ = _split_edges(pts, is_input, edges, which)
        else:
            raise AssertionError("unreachable")
        return cls(tri, edges, edges[miss], reach)

    def _topology(self, missing: np.ndarray) -> None:
        P = self.P
        V = self.tri.simplices.astype(np.int64)
        NB = self.tri.neighbors.astype(np.int64)
        neg = _cross(P[V[:, 1]] - P[V[:, 0]], P[V[:, 2]] - P[V[:, 0]]) < 0
        V[neg, 1], V[neg, 2] = V[neg, 2], V[neg, 1].copy()
        NB[neg, 1], NB[neg, 2] = NB[neg, 2], NB[neg, 1].copy()
        T, npt = len(V), len(P)
        ar = np.arange(T)
        NBK = np.zeros((T, 3), np.int64)
        for k in range(3):
            for j in range(3):
                NBK[(NB[:, k] >= 0) & (NB[NB[:, k], j] == ar), k] = j
        hu, hv = V[:, [1, 2, 0]], V[:, [2, 0, 1]]
        uniq, inv = np.unique((np.minimum(hu, hv) * npt + np.maximum(hu, hv)).ravel(), return_inverse=True)
        he = inv.reshape(T, 3)
        E = len(uniq)
        self._edge_key = uniq
        ea, eb = uniq // npt, uniq % npt
        ce = self.con_edges
        con = np.isin(uniq, np.minimum(ce[:, 0], ce[:, 1]) * npt + np.maximum(ce[:, 0], ce[:, 1]))
        con[he[NB < 0]] = True
        if len(missing):
            free = np.nonzero(~con)[0]
            for a, b in missing:
                con[free[_seg_seg(P[ea[free]], P[eb[free]], P[a][None], P[b][None]) < TOL]] = True
        t0 = np.full(E, -1, np.int64)
        t1 = np.full(E, -1, np.int64)
        fwd = hu < hv
        tt = np.repeat(ar, 3).reshape(T, 3)
        t0[he[fwd]] = tt[fwd]
        t1[he[~fwd]] = tt[~fwd]
        self.V, self.NB, self.NBK, self.he = V, NB, NBK, he
        self.ea, self.eb, self.e_con, self.e_t0, self.e_t1 = ea, eb, con, t0, t1
        self.e_len = np.hypot(*(P[ea] - P[eb]).T)
        self.e_mid = (P[ea] + P[eb]) / 2

    def free_edges(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Id of the free mesh edge joining the vertices at A[i] and B[i], -1 where there is none."""
        if not len(A):
            return np.zeros(0, np.int64)
        vertices = self._vertices
        if vertices is None:
            vertices = self._vertices = cKDTree(self.P)
        da, a = vertices.query(A)
        db, b = vertices.query(B)
        key = np.minimum(a, b) * len(self.P) + np.maximum(a, b)
        e = np.minimum(np.searchsorted(self._edge_key, key), len(self._edge_key) - 1)
        found = (da <= TOL) & (db <= TOL) & (self._edge_key[e] == key) & ~self.e_con[e]
        return np.where(found, e, -1)

    def _widths(self) -> None:
        """Passage width of every traversal (triangle t, around vertex k), exact up to reach."""
        P, V, NB, NBK, he, con = self.P, self.V, self.NB, self.NBK, self.he, self.e_con
        T = len(V)
        free = ~con[he]
        tt, kk = np.nonzero(free[:, [1, 2, 0]] & free[:, [2, 0, 1]])
        C, A, B = P[V[tt, kk]], P[V[tt, (kk + 1) % 3]], P[V[tt, (kk + 2) % 3]]
        w = np.minimum(np.minimum(np.hypot(*(A - C).T), np.hypot(*(B - C).T)), self.reach)
        ft, t, ke = np.arange(len(tt)), tt, kk
        visited = ft * T + t
        for _ in range(200):
            d = _clip_dist(C[ft], A[ft], B[ft], P[V[t, (ke + 1) % 3]], P[V[t, (ke + 2) % 3]])
            act = d < w[ft]
            ec = con[he[t, ke]]
            m = act & ec
            np.minimum.at(w, ft[m], d[m])
            go = act & ~ec
            ft, nt, nk = ft[go], NB[t[go], ke[go]], NBK[t[go], ke[go]]
            key, first = np.unique(ft * T + nt, return_index=True)
            new = first[~np.isin(key, visited)]
            if not len(new):
                break
            ft, nt, nk = ft[new], nt[new], nk[new]
            visited = np.union1d(visited, ft * T + nt)
            X = P[V[nt, nk]] - C[ft]
            inw = (_cross(A[ft] - C[ft], X) >= 0) & (_cross(X, B[ft] - C[ft]) >= 0)
            np.minimum.at(w, ft[inw], np.hypot(X[inw, 0], X[inw, 1]))
            ft = np.concatenate([ft, ft])
            t = np.concatenate([nt, nt])
            ke = np.concatenate([(nk + 1) % 3, (nk + 2) % 3])
        self.arc_a = he[tt, (kk + 1) % 3]
        self.arc_b = he[tt, (kk + 2) % 3]
        self.arc_w = w
        self.arc_c = np.hypot(*(self.e_mid[self.arc_a] - self.e_mid[self.arc_b]).T)
        self.vertex_gap = np.full(len(P), self.reach)
        np.minimum.at(self.vertex_gap, V[tt, kk], w)

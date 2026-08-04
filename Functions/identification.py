"""Dictionary identification: match the observed honeycomb structure against the spherical code dictionary."""

import json
import math
import os
import time

import cv2
import numpy as np

DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "Material",
                         "sphereModel.json")  # coordinates measured by the registration algorithm
STAR_MAX_HAM = 0       # max Hamming distance for star state comparison (0 = exact match)
# Reflection sign: the sphere is only viewed from outside, so no physical
# mirror ambiguity exists; -1 absorbs the convention difference between
# image coordinates (y down) and the dictionary cyclic order.
STAR_REFLECTION = -1
ACCEPT_MIN_MAP = 1     # minimum mapped nodes to accept a hypothesis
SEED_CAP = 400         # max candidate seeds grown per frame


# ---- observation grid reconstruction: triangulation, edge removal, cells ----

def ekey(a, b):
    """Canonical edge key (unordered endpoints)."""
    return (a, b) if a <= b else (b, a)


def triangulate(dots, w, h):
    """Delaunay triangulation -> (all edges, longest edge per triangle).

    The marker lattice is honeycomb, so each triangle's longest edge is a
    cross-lattice "non-honeycomb edge" to be removed.
    """
    if len(dots) < 3:
        return [], []
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for cx, cy, _ in dots:
        try:
            subdiv.insert((float(cx), float(cy)))
        except cv2.error:
            continue  # out-of-range or duplicate point
    # Subdiv2D keeps virtual points outside the image; clip their edges
    edges = [(x1, y1, x2, y2) for x1, y1, x2, y2 in subdiv.getEdgeList()
             if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h]
    longest = []
    for t in subdiv.getTriangleList():
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if not all(0 <= x < w and 0 <= y < h for x, y in pts):
            continue
        tri_edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]
        e = max(tri_edges, key=lambda e: (e[0][0] - e[1][0]) ** 2
                + (e[0][1] - e[1][1]) ** 2)
        longest.append((e[0][0], e[0][1], e[1][0], e[1][1]))
    return edges, longest


def mesh_remaining_edges(dots, w, h):
    """Edges left after deleting each triangle's longest edge (the
    honeycomb grid)."""
    edges, longest = triangulate(dots, w, h)
    removed = {ekey((e[0], e[1]), (e[2], e[3])) for e in longest}
    return edges, longest, [e for e in edges
                            if ekey((e[0], e[1]), (e[2], e[3])) not in removed]


def find_cells_and_structure(edges):
    """DCEL half-edge face traversal -> (cells, struct_edges, struct_nodes).

    At b after a->b, take the outgoing edge just past the reverse edge b->a
    (smallest turn); this makes the successor map a permutation so each
    face is walked once. Rings that backtrack into a dangling stub hide an
    edge inside and are discarded. 5/6-edged clean rings are the cells.
    struct = union of triad faces (a degree-3 vertex whose 3 incident
    faces are 6-6-6 or 6-6-5).
    """
    adj = {}
    for x1, y1, x2, y2 in edges:
        a, b = (x1, y1), (x2, y2)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    visited = set()
    faces = []
    for u in adj:
        for v in adj[u]:
            if (u, v) in visited:
                continue
            face = []
            a, b = u, v
            backtracked = False
            while (a, b) not in visited and len(face) < 100:  # 100 guards against infinite loops
                visited.add((a, b))
                face.append(a)
                ang_rev = math.atan2(a[1] - b[1], a[0] - b[0])  # direction b->a
                nxt, nxt_d = None, None
                for c in adj[b]:
                    if c == a:
                        continue
                    d = (math.atan2(c[1] - b[1], c[0] - b[0]) - ang_rev) \
                        % (2 * math.pi)
                    if nxt_d is None or d < nxt_d:
                        nxt, nxt_d = c, d
                if nxt is None:
                    nxt = a  # dead end: backtrack along the same edge
                    backtracked = True
                a, b = b, nxt
            if (a, b) == (u, v) and len(face) >= 3 and not backtracked:
                faces.append(face)

    def signed_area(f):
        s = 0.0
        for i in range(len(f)):
            x1, y1 = f[i]
            x2, y2 = f[(i + 1) % len(f)]
            s += x1 * y2 - x2 * y1
        return s / 2

    # the area bound removes the outer face wrapping the whole grid
    cells = [f for f in faces
             if len(f) in (5, 6) and 0 < abs(signed_area(f)) < 100000]

    v2faces = {}
    for i, f in enumerate(cells):
        for v in f:
            v2faces.setdefault(v, []).append(i)

    # triad test: degree 3, incident faces 6-6-6 / 6-6-5
    struct_edges = set()
    struct_nodes = set()
    for v, fs in v2faces.items():
        if len(adj.get(v, ())) != 3 or len(fs) != 3:
            continue
        sizes = sorted(len(cells[i]) for i in fs)
        if sizes in ([5, 6, 6], [6, 6, 6]):
            for i in fs:
                f = cells[i]
                for j in range(len(f)):
                    struct_edges.add(ekey(f[j], f[(j + 1) % len(f)]))
                struct_nodes.update(f)
    return cells, struct_edges, struct_nodes


class SphereDict:
    """Dictionary graph preprocessing: adjacency, cyclic order, faces, star library (once at startup)."""

    def __init__(self, path=DICT_PATH):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        self.state = {n["id"]: int(n["state"]) for n in doc["nodes"]}
        self.xyz = {n["id"]: np.asarray(n["xyz"], dtype=float)
                    for n in doc["nodes"]}
        xyz = self.xyz
        self.adj = {i: [] for i in self.state}
        for i, j in doc["edges"]:
            self.adj[i].append(j)
            self.adj[j].append(i)
        # cyclic order: neighbors projected onto the tangent plane, sorted by atan2 (CCW)
        self.cyc = {}
        for i, nbrs in self.adj.items():
            n = xyz[i] / np.linalg.norm(xyz[i])
            ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 \
                else np.array([0.0, 1, 0])
            t1 = np.cross(n, ref)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            self.cyc[i] = sorted(
                nbrs, key=lambda j: math.atan2((xyz[j] - xyz[i]) @ t2,
                                               (xyz[j] - xyz[i]) @ t1))
        # DCEL half-edge traversal; take the direction that yields 162 faces
        self.faces = self._find_faces(+1)
        if len(self.faces) != 162:
            self.faces = self._find_faces(-1)
        assert len(self.faces) == 162, f"Unexpected dictionary face count: {len(self.faces)}"
        self.node_faces = {i: [] for i in self.state}
        for fi, fvs in enumerate(self.faces):
            for v in fvs:
                self.node_faces[v].append(fi)
        # (node, ring index i) -> edge count of the face between ring neighbors i and i+1
        self.face_between = {}
        for A in self.state:
            ring = self.cyc[A]
            for i in range(3):
                b1, b2 = ring[i], ring[(i + 1) % 3]
                for fi in self.node_faces[A]:
                    fvs = self.faces[fi]
                    if b1 in fvs and b2 in fvs:
                        self.face_between[(A, i)] = len(fvs)
                        break
        # ---- stars: vertex union of the 3 incident faces (13/12 points); the source of seed uniqueness ----
        self.star = {}
        for A in self.state:
            ring = self.cyc[A]
            r2 = []       # per ring index t: neighbors after/before A in the ring of B=ring[t]
            anti = []     # antipodal vertex of each face (None for pentagons)
            face_sz = []  # edge count of each incident face
            for t in range(3):
                B = ring[t]
                rb = self.cyc[B]
                p = rb.index(A)
                r2.append((rb[(p + 1) % 3], rb[(p - 1) % 3]))
                b1, b2 = ring[t], ring[(t + 1) % 3]
                fvs = None
                for fi in self.node_faces[A]:
                    f = self.faces[fi]
                    if b1 in f and b2 in f:
                        fvs = set(f)
                        break
                face_sz.append(len(fvs))
                B2 = ring[(t + 1) % 3]
                rb2 = self.cyc[B2]
                p2 = rb2.index(A)
                in_face = [x for x in (r2[t][0], r2[t][1],
                                       rb2[(p2 + 1) % 3], rb2[(p2 - 1) % 3])
                           if x in fvs]
                rest = fvs - {A, b1, b2} - set(in_face)
                assert len(rest) <= 1, f"Unexpected star structure: center {A} face {t}"
                anti.append(rest.pop() if rest else None)
            nodes = {A, *ring}
            for af, bf in r2:
                nodes.update((af, bf))
            nodes.update(x for x in anti if x is not None)
            # canonical state vector: [center, ring x3, second ring (after,before) x3, hexagon antipodes]
            vec = [self.state[A]] + [self.state[b] for b in ring]
            for af, bf in r2:
                vec += [self.state[af], self.state[bf]]
            vec += [self.state[x] for x in anti if x is not None]
            self.star[A] = {
                "nodes": nodes,
                "face_sz": face_sz,
                "face_key": tuple(sorted(face_sz)),
                "vec": np.array(vec, dtype=np.uint8),
                "hollow": int(sum(vec)),
            }
        # hash index for exact matching (STAR_MAX_HAM=0)
        self.star_hash = {}
        for A, dst in self.star.items():
            key = (tuple(dst["face_sz"]), dst["vec"].tobytes())
            self.star_hash.setdefault(key, []).append(A)
        # edge list + pentagon centers, for wireframe visualization
        self.edge_list = sorted({(i, j) for i in self.adj
                                 for j in self.adj[i] if i < j})
        self.pent_centers = [np.mean([xyz[v] for v in fvs], axis=0)
                             for fvs in self.faces if len(fvs) == 5]

    def _find_faces(self, sign):
        """Half-edge face traversal: after reaching b, take the sign-th neighbor after a in b's cyclic order."""
        visited = set()
        faces = []
        for u in self.adj:
            for v in self.adj[u]:
                if (u, v) in visited:
                    continue
                face = []
                a, b = u, v
                while (a, b) not in visited and len(face) < 200:  # upper bound prevents infinite loops
                    visited.add((a, b))
                    face.append(a)
                    ring = self.cyc[b]
                    a, b = b, ring[(ring.index(a) + sign) % 3]
                if (a, b) == (u, v) and len(face) >= 3:
                    faces.append(face)
        return faces


def build_obs_graph(struct_nodes, struct_edges, cells, hollow_map):
    """Observation graph from struct nodes/edges/cells (image y is down; the mirrored
    atan2 cyclic order is absorbed by STAR_REFLECTION). Missing states are None."""
    def nid(x, y):
        return (round(float(x), 1), round(float(y), 1))

    nodes = {nid(x, y) for x, y in struct_nodes}
    adj = {n: set() for n in nodes}
    for (x1, y1), (x2, y2) in struct_edges:
        a, b = nid(x1, y1), nid(x2, y2)
        if a in nodes and b in nodes:
            adj[a].add(b)
            adj[b].add(a)
    cyc = {}
    for n, ns in adj.items():
        if len(ns) == 3:
            cyc[n] = sorted(ns, key=lambda c: math.atan2(c[1] - n[1],
                                                         c[0] - n[0]))
    cell_lists = []
    cells_all = []  # includes cells extending beyond struct, for anchored completion
    node_cells = {n: [] for n in nodes}
    extra_nodes = set()
    for fvs in cells:
        vs = [nid(x, y) for x, y in fvs]
        cells_all.append(vs)
        if all(v in nodes for v in vs):
            idx = len(cell_lists)
            cell_lists.append(vs)
            for v in vs:
                node_cells[v].append(idx)
        else:
            extra_nodes.update(v for v in vs if v not in nodes)
    state = {n: hollow_map.get(n) for n in nodes}
    state.update({n: hollow_map.get(n) for n in extra_nodes})
    return {"nodes": nodes, "adj": adj, "cyc": cyc, "cells": cell_lists,
            "cells_all": cells_all, "node_cells": node_cells,
            "state": state}


def obs_star(obs, a):
    """Observation star (vertex union of the 3 incident cells); None if unreliable
    (degree != 3, missing cells, or any of the 13/12 points unknown)."""
    cyc = obs["cyc"]
    st = obs["state"]
    if a not in cyc or st[a] is None:
        return None
    ring = cyc[a]
    if any(st[b] is None for b in ring):
        return None
    ncells = obs["node_cells"][a]
    if len(ncells) != 3:
        return None
    r2 = []
    for B in ring:
        rb = cyc.get(B)
        if rb is None:
            return None
        p = rb.index(a)
        af, bf = rb[(p + 1) % 3], rb[(p - 1) % 3]
        if af == a or bf == a or af == bf or af in ring or bf in ring:
            return None
        if st.get(af) is None or st.get(bf) is None:
            return None
        r2.append((af, bf))
    r2flat = [x for pair in r2 for x in pair]
    if len(set(r2flat)) != 6:  # second-ring points must be pairwise distinct
        return None
    face_sz, anti = [], []
    for t in range(3):
        b1, b2 = ring[t], ring[(t + 1) % 3]
        fvs = None
        for ci in ncells:
            vs = set(obs["cells"][ci])
            if b1 in vs and b2 in vs:
                fvs = vs
                break
        if fvs is None:
            return None
        face_sz.append(len(fvs))
        r2in = [x for x in (*r2[t], *r2[(t + 1) % 3]) if x in fvs]
        rest = fvs - {a, b1, b2} - set(r2in)
        if len(rest) == 1:
            x = rest.pop()
            if x in ring or x in r2flat or st.get(x) is None:
                return None
            anti.append(x)
        elif len(rest) == 0:
            anti.append(None)
        else:
            return None
    nodes = {a, *ring, *r2flat}
    nodes.update(x for x in anti if x is not None)
    return {"center": a, "ring": ring, "r2": r2, "face_sz": face_sz,
            "anti": anti, "nodes": nodes,
            "face_key": tuple(sorted(face_sz))}


def star_seeds(sd, obs):
    """Collect seed alignments [(hamming, obs center, dict center, k, s)] by matching
    reliable observation stars against dictionary stars."""
    st = obs["state"]
    seeds = []
    n_reliable = 0
    exact = STAR_MAX_HAM == 0  # exact matching goes through the hash index
    for a in obs["nodes"]:
        ost = obs_star(obs, a)
        if ost is None:
            continue
        n_reliable += 1
        cand = None
        if not exact:
            oh = sum(st[x] for x in ost["nodes"])
            cand = [A for A in sd.state
                    if sd.star[A]["face_key"] == ost["face_key"]
                    and abs(sd.star[A]["hollow"] - oh) <= STAR_MAX_HAM]
            if not cand:
                continue
        ring = ost["ring"]
        s = STAR_REFLECTION
        for k in range(3):
            # aligned face sequences must match, otherwise a pentagon would be displaced
            seq = [ost["face_sz"][(t - k) % 3] if s == 1
                   else ost["face_sz"][(k - t - 1) % 3]
                   for t in range(3)]
            # obs state vector in the dictionary canonical order
            vec = [st[a]]
            for t in range(3):
                vec.append(st[ring[(s * (t - k)) % 3]])
            for t in range(3):
                af, bf = ost["r2"][(s * (t - k)) % 3]
                vec += [st[af], st[bf]] if s == 1 else [st[bf], st[af]]
            for t in range(3):
                if seq[t] == 5:
                    continue
                f = (t - k) % 3 if s == 1 else (k - t - 1) % 3
                vec.append(st[ost["anti"][f]])
            vec = np.array(vec, dtype=np.uint8)
            if exact:
                for A in sd.star_hash.get((tuple(seq), vec.tobytes()),
                                          ()):
                    seeds.append((0, a, A, k, s))
            else:
                for A in cand:
                    dst = sd.star[A]
                    if dst["face_sz"] != seq:
                        continue
                    hd = int((vec != dst["vec"]).sum())
                    if hd <= STAR_MAX_HAM:
                        seeds.append((hd, a, A, k, s))
    seeds.sort()
    return seeds, n_reliable


def obs_face_sizes(obs, a):
    """Edge counts of the 3 incident faces of node a (cyclic order); None if a face is missing."""
    ring = obs["cyc"].get(a)
    ncells = obs["node_cells"][a]
    if ring is None or len(ncells) != 3:
        return None
    out = []
    for i in range(3):
        b1, b2 = ring[i], ring[(i + 1) % 3]
        sz = None
        for ci in ncells:
            vs = obs["cells"][ci]
            if b1 in vs and b2 in vs:
                sz = len(vs)
                break
        out.append(sz)
    return out


def grow_match(sd, obs, seed_o, seed_d, k, s):
    """BFS growth from (seed_o -> seed_d, alignment k/s).

    Alignment convention: observation ring index i -> dictionary ring position
    (off + s*i) % 3, with off=k at the seed; s is uniform for the whole frame.
    topo/cell/state conflicts are only counted, never terminate growth.
    """
    om = {seed_o: seed_d}     # observation -> dictionary
    dm = {seed_d: seed_o}     # dictionary -> observation (occupancy check)
    off = {seed_o: k}
    topo = cell = state = 0
    conflict_edges = []       # edges where topo conflicts occur, for defect localization
    queue = [seed_o]
    while queue:
        a = queue.pop()
        A = om[a]
        ring_o = obs["cyc"][a]
        for i, c in enumerate(ring_o):
            T = sd.cyc[A][(off[a] + s * i) % 3]
            if c in om:
                if om[c] != T:
                    topo += 1  # c already mapped to another dictionary node
                    conflict_edges.append((a, c))
                continue
            if T in dm:
                topo += 1      # target dictionary node already occupied
                conflict_edges.append((a, c))
                continue
            om[c] = T
            dm[T] = c
            # derive c's offset from the positions of the incoming edge in both rings
            if c in obs["cyc"]:
                p = obs["cyc"][c].index(a)
                q = sd.cyc[T].index(A)
                off[c] = (q - s * p) % 3
                queue.append(c)
            st = obs["state"].get(c)
            if st is not None and st != sd.state[T]:
                state += 1
            fsz = obs_face_sizes(obs, c)
            if fsz is not None and c in off:
                for i2, sz in enumerate(fsz):
                    if sz is None:
                        continue
                    j = (off[c] + s * i2) % 3
                    if sz != sd.face_between.get((T, j)):
                        cell += 1
    mapped = len(om)
    score = mapped - 3 * topo - 2 * cell - 1 * state
    return {"map": om, "topo": topo, "cell": cell, "state": state,
            "score": score, "mapped": mapped, "reflection": s,
            "conflict_edges": conflict_edges,
            "seed": (seed_o, seed_d, k, s)}


def accept_match(r):
    """Accept any hypothesis reaching ACCEPT_MIN_MAP; always output the best-scoring one."""
    return (r is not None and r["mapped"] >= ACCEPT_MIN_MAP)


REPAIR_MIN_TOPO = 3  # defect excision repair is attempted only at this many topo conflicts


def repair_match(sd, obs, r):
    """Excise the defective cell and regrow.

    Topo conflicts mean growth took a wrong turn (typically one misdetected
    cell); the error propagates along growth paths and surfaces only where two
    paths meet, so the defect lies upstream of the conflict points. The cut is
    adopted only if the weighted conflict sum drops below half of the original.
    """
    cedges = r.get("conflict_edges")
    if not cedges:
        return r, []
    suspect = set()
    for a, c in cedges:
        suspect.add(a)
        suspect.add(c)
    # candidate cells: cells containing a suspect node plus adjacent cells sharing a node
    nb = set(suspect)
    for cell in obs["cells"]:
        if any(n in suspect for n in cell):
            nb.update(cell)
    cand, seen = [], set()
    for cell in obs["cells"]:
        key = tuple(sorted(cell))
        if key not in seen and any(n in nb for n in cell):
            seen.add(key)
            cand.append(cell)
    if not cand:
        return r, []

    def cell_edges(cell):
        return {(cell[i], cell[(i + 1) % len(cell)])
                if cell[i] <= cell[(i + 1) % len(cell)]
                else (cell[(i + 1) % len(cell)], cell[i])
                for i in range(len(cell))}

    def regrow_without(cut):
        adj = {n: set() for n in obs["nodes"]}
        for a, bs in obs["adj"].items():
            for b in bs:
                e = (a, b) if a <= b else (b, a)
                if e not in cut:
                    adj[a].add(b)
        cyc = {}
        for n, ns in adj.items():
            if len(ns) == 3:
                cyc[n] = sorted(ns, key=lambda c: math.atan2(
                    c[1] - n[1], c[0] - n[0]))
        so, sdd, k, s = r["seed"]
        if so not in cyc:
            return None  # seed node lost its cyclic order; cannot regrow
        return grow_match(sd, dict(obs, adj=adj, cyc=cyc), so, sdd, k, s)

    def conf_of(h):
        return 3 * h["topo"] + 2 * h["cell"] + h["state"]

    best_r, best_cut, best_conf = r, [], conf_of(r)
    for cell in cand:
        r2 = regrow_without(cell_edges(cell))
        if r2 is not None and conf_of(r2) < best_conf:
            best_r, best_cut, best_conf = r2, cell_edges(cell), conf_of(r2)
    if len(cand) > 1:
        cut_all = set().union(*(cell_edges(c) for c in cand))
        r2 = regrow_without(cut_all)
        if r2 is not None and conf_of(r2) < best_conf:
            best_r, best_cut, best_conf = r2, cut_all, conf_of(r2)
    if best_cut and best_conf <= 0.5 * conf_of(r):
        return best_r, sorted(best_cut)
    return r, []


def complete_cells(sd, result, obs):
    """Assign IDs to complete cells extending beyond the structure.

    An adjacent anchor pair fixes the dictionary face (state conflicts over
    unmapped nodes arbitrate between the two faces of an edge); remaining
    nodes get the face's remaining ids. Iterates to a fixed point.
    """
    om = result["map"]
    st = obs["state"]
    dm = {d: n for n, d in om.items()}
    added = 0

    def nid(p):
        return (round(float(p[0]), 1), round(float(p[1]), 1))

    changed = True
    while changed:
        changed = False
        for cell in obs["cells_all"]:
            ring = [nid(p) for p in cell]
            L = len(ring)
            anch = [(i, om[n]) for i, n in enumerate(ring) if n in om]
            if len(anch) < 2 or len(anch) == L:
                continue
            pair = None
            for i, d in anch:
                j = (i + 1) % L
                if ring[j] in om:
                    pair = (i, d, om[ring[j]])
                    break
            if pair is None:
                continue
            i0, d0, d1 = pair
            feasible = []  # (state conflict count, alignment function fid)
            for fi in sd.node_faces[d0]:
                fvs = sd.faces[fi]
                if len(fvs) != L or d1 not in fvs:
                    continue
                p0 = fvs.index(d0)
                step = 1 if fvs[(p0 + 1) % L] == d1 else -1
                if fvs[(p0 + step) % L] != d1:
                    continue

                def fid(t, p0=p0, step=step, fvs=fvs):
                    return fvs[(p0 + step * t) % L]

                if any(fid(i - i0) != d for i, d in anch):
                    continue  # contradicts other anchors
                conf, ok = 0, True
                for i, n in enumerate(ring):
                    if n in om:
                        continue
                    D = fid(i - i0)
                    if D in dm:
                        ok = False  # target id already occupied
                        break
                    s0 = st.get(n)
                    if s0 is not None and s0 != sd.state[D]:
                        conf += 1
                if ok:
                    feasible.append((conf, fid))
            if not feasible:
                continue
            min_conf = min(c for c, _ in feasible)
            winners = [(c, f) for c, f in feasible if c == min_conf]
            if len(winners) > 1:
                continue  # evidence cannot discriminate between faces; give up
            fid = winners[0][1]
            for i, n in enumerate(ring):
                if n not in om:
                    D = fid(i - i0)
                    om[n] = D
                    dm[D] = n
                    added += 1
                    changed = True
    result["completed"] = added
    result["mapped"] = len(om)


def match_frame(sd, obs, prof=None):
    """Full-frame matching: seeding -> growth -> two-stage selection -> repair/completion.

    Honeycomb symmetry admits "ghost alignments" that are topologically
    self-consistent but state-scrambled and may score higher, so selection is
    two-stage: among hypotheses mapping >= 80% of the max nodes, take the
    lowest state conflict rate, then the highest score.
    prof keys: seed, grow (growth+selection), repair (excision+completion).
    """
    def _mark(name, t_):
        if prof is not None:
            prof[name] = prof.get(name, 0.0) + time.perf_counter() - t_
        return time.perf_counter()

    t0 = time.perf_counter()
    seeds, n_reliable = star_seeds(sd, obs)
    t1 = _mark("seed", t0)
    # seeds with Hamming <= 1 first, then the rest if needed (capped at SEED_CAP)
    h01 = [x for x in seeds if x[0] <= 1]
    h2 = [x for x in seeds if x[0] > 1]
    hyps = []
    for rnd in (h01, h2[:max(0, SEED_CAP - len(h01))]):
        if not rnd:
            continue
        for hd, n, A, k, s in rnd:
            r = grow_match(sd, obs, n, A, k, s)
            if r["mapped"] >= ACCEPT_MIN_MAP:
                hyps.append(r)
        if any(accept_match(h) for h in hyps):
            break
    t1 = _mark("grow", t1)

    best = None
    if hyps:
        mmax = max(h["mapped"] for h in hyps)
        cands = [h for h in hyps if h["mapped"] >= 0.8 * mmax]
        cands.sort(key=lambda h: (h["state"] / h["mapped"], -h["score"]))
        best = cands[0]
    if best is None:
        return {"accepted": False, "map": {}, "score": 0, "mapped": 0,
                "topo": 0, "cell": 0, "state": 0, "reflection": 0,
                "n_reliable": n_reliable, "n_seeds": len(seeds),
                "ms": (time.perf_counter() - t0) * 1000}
    best["accepted"] = accept_match(best)
    if best["accepted"] and best["topo"] >= REPAIR_MIN_TOPO:
        best, defect = repair_match(sd, obs, best)
        best["accepted"] = accept_match(best)
        if defect:
            best["defect_edges"] = defect
    if best["accepted"]:
        complete_cells(sd, best, obs)
    _mark("repair", t1)
    best["n_reliable"] = n_reliable
    best["n_seeds"] = len(seeds)
    best["ms"] = (time.perf_counter() - t0) * 1000
    return best

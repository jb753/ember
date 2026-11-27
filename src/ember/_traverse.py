"""Structured traverse cuts built from a grid's own lines.

The work behind :func:`ember.cut.structured_traverse`. Every block the cut
crosses cleanly, with one family of its grid lines each crossing the cut the
same number of times, gives an exact structured patch: each node a linear
interpolation along one grid edge. So does each run of lines in a block
crossed only part-way that crosses once from endwall to endwall. Patches that
share a side, across an interface or a periodic face, are stitched into
chains. What lies between chains is a hole, filled from the marching cubes
triangles of the ragged blocks along rows that start and end on the chains'
exact nodes; with no chain at all, the hole runs the whole pitch between two
copies of the periodic seam.

Anything that scales with the grid is done in the kernels of
``_fortran/traverse.f90``; this module handles only the patches and their
sides.
"""

import numpy as np
from pykdtree.kdtree import KDTree

import ember.cut
import ember.fortran
from ember.periodic import PeriodicPatch

F32 = np.float32


def _distance(xr_cut, block):
    """The fields and coefficients the kernels form the cut's distance from."""
    args = ember.cut._distance_args(xr_cut, block)
    if len(args) == 1:
        a = np.asfortranarray(args[0], dtype=F32)
        return a, a, (1.0, 0.0, 0.0)
    a, b, coeffs = args
    return (np.asfortranarray(a, dtype=F32), np.asfortranarray(b, dtype=F32), coeffs)


def _counts(dist, shape):
    """Signs at the nodes, crossings per line of each axis, and positive nodes."""
    ni, nj, nk = shape
    sgn = np.empty(shape, dtype=np.int8, order="F")
    cnt = [
        np.empty(s, dtype=np.int32, order="F") for s in ((nj, nk), (ni, nk), (ni, nj))
    ]
    prev_j = np.empty(ni, dtype=np.int8)
    prev_k = np.empty((ni, nj), dtype=np.int8, order="F")
    a, b, (na, nb, c0) = dist
    n_pos = ember.fortran.crossing_counts(a, b, na, nb, c0, sgn, *cnt, prev_j, prev_k)
    return sgn, cnt, n_pos


def _patch(block, dist, sgn, ax, nth, p=None, q=None):
    """Block data at the nth crossing of every line along axis `ax`.

    The lines are indexed by the other two axes in order; `p` and `q` are
    (start, stop) ranges along them, the whole axis by default.
    """
    others = [s for s in range(3) if s != ax]
    p = p or (0, block.shape[others[0]])
    q = q or (0, block.shape[others[1]])
    n1, n2 = p[1] - p[0], q[1] - q[0]
    nvar = block._data.shape[-1]
    out = np.empty((n1, n2, nvar), dtype=F32, order="F")
    edge = np.empty((n1, n2), dtype=np.int32, order="F")
    cnt = np.empty((n1, n2), dtype=np.int32, order="F")
    a, b, (na, nb, c0) = dist
    data = np.asfortranarray(block._data, dtype=F32)
    ember.fortran.crossing_patch(
        a, b, na, nb, c0, sgn, data, ax + 1, nth, p[0], q[0], out, edge, cnt
    )
    return out


def block_patches(block, xr_cut):
    """Exact patches of the cut through one block, and how it was crossed.

    Returns (patches, status), status "miss" where the cut does not cross
    the block, "clean" where one axis's lines all cross it the same number
    of times, one patch per crossing, and "ragged" otherwise, the patches
    then those of the runs that do cross cleanly.
    """
    dist = _distance(xr_cut, block)
    sgn, cnt, n_pos = _counts(dist, block.shape)
    if n_pos == 0 or n_pos == sgn.size:
        return [], "miss"
    uniform = [
        (int(c.flat[0]), ax)
        for ax, c in enumerate(cnt)
        if c.size and c.flat[0] > 0 and c.min() == c.max()
    ]
    if not uniform:
        return _sub_patches(block, dist, sgn, cnt), "ragged"
    m, ax = min(uniform)
    return [_patch(block, dist, sgn, ax, n) for n in range(1, m + 1)], "clean"


def _ends_on_walls(block, axis):
    """Whether neither face at the ends of `axis` carries a periodic patch,
    interface or periodic face alike: both are walls."""
    n = block.shape[axis]
    for patch in block.patches:
        if not isinstance(patch, PeriodicPatch):
            continue
        lim = patch.ijk_lim_abs[axis]
        if lim[0] == lim[1] and lim[0] in (0, n - 1):
            return False
    return True


def _runs(mask, min_len=2):
    """(start, stop) of the runs of True in `mask` at least `min_len` long."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def _sub_patches(block, dist, sgn, cnt):
    """Exact patches from the runs of a ragged block's lines that cross once.

    For an axis whose lines cross at most once, take one of the other two
    axes whole, one that ends on walls at both faces so a patch runs endwall
    to endwall, and find the stations along the last axis where every line
    crosses exactly once. The choice covering most stations wins: no axis is
    assumed to be spanwise.
    """
    best = (0, None)
    for ax, c in enumerate(cnt):
        if c.size == 0 or c.max() > 1:
            continue
        others = [s for s in range(3) if s != ax]
        for whole in (0, 1):
            if not _ends_on_walls(block, others[whole]):
                continue
            runs = _runs((c == 1).all(axis=whole))
            cover = sum(b - a for a, b in runs)
            if cover > best[0]:
                best = (cover, (ax, whole, runs))
    if best[1] is None:
        return []
    ax, whole, runs = best[1]
    if whole == 0:
        return [_patch(block, dist, sgn, ax, 1, q=run) for run in runs]
    return [_patch(block, dist, sgn, ax, 1, p=run) for run in runs]


def _sides(P):
    """A patch's four sides: first and last rows, first and last columns."""
    return (P[0], P[-1], P[:, 0], P[:, -1])


def _side_last(P, side):
    """`P` turned so that its side `side` is its last row, in order."""
    return (P[::-1], P, np.swapaxes(P, 0, 1)[::-1], np.swapaxes(P, 0, 1))[side]


def _side_first(P, side):
    """`P` turned so that its side `side` is its first row, in order."""
    return (P, P[::-1], np.swapaxes(P, 0, 1), np.swapaxes(P, 0, 1)[::-1])[side]


def _side_match(sa, sb, ix, pitch, tol):
    """Whole pitches to add to side `sb` to bring it onto side `sa`, or None."""
    if sa.shape[0] != sb.shape[0]:
        return None
    xr_a, xr_b = sa[:, ix[:2]], sb[:, ix[:2]]
    if np.abs(xr_a - xr_b).max() > tol[0]:
        return None
    dt = sa[:, ix[2]].astype(float) - sb[:, ix[2]].astype(float)
    n = np.round(dt[0] / pitch)
    if np.abs(dt - n * pitch).max() > tol[1]:
        return None
    return n


def _find_join(A, B, ix, pitch, tol):
    """(side of A, side of B, whether B's runs the other way, pitches) of a
    side the two patches share, or None."""
    for sa, side_a in enumerate(_sides(A)):
        for sb, side_b in enumerate(_sides(B)):
            for flip in (False, True):
                n = _side_match(
                    side_a, side_b[::-1] if flip else side_b, ix, pitch, tol
                )
                if n is not None:
                    return sa, sb, flip, n
    return None


def stitch(patches, ix, pitch):
    """Join patches that share a whole side, a whole number of pitches apart.

    Each join puts one patch after the other along the first axis, theta of
    the second shifted by the pitches between them, the shared side kept
    once. Returns the joined arrays and the (x and r, theta) tolerances
    the sides were matched to: a few single precision steps.
    """
    if not patches:
        return [], (0.0, 0.0)
    allx = np.concatenate([np.abs(p[..., ix].reshape(-1, 3)) for p in patches])
    big = allx.max(axis=0)
    tol = (
        4 * float(np.spacing(F32(big[:2].max()))),
        4 * float(np.spacing(F32(big[2]))),
    )
    items = list(patches)
    joined = True
    while joined:
        joined = False
        for a in range(len(items)):
            for b in range(len(items)):
                if a == b:
                    continue
                join = _find_join(items[a], items[b], ix, pitch, tol)
                if join is None:
                    continue
                sa, sb, flip, n = join
                A = _side_last(items[a], sa)
                B = _side_first(items[b], sb)
                B = B[:, ::-1] if flip else B
                B = B.copy()
                B[..., ix[2]] += F32(n * pitch)
                items[a] = np.concatenate([A, B[1:]])
                del items[b]
                joined = True
                break
            if joined:
                break
    return items, tol


def orient(chain, ix, xr_cut_nd, pitch):
    """Turn a chain to (span, pitch): hub first, theta increasing.

    Returns the chain and its arc length along the cut and theta in pitches.
    """
    n1, n2 = chain.shape[:2]
    xq = np.asfortranarray(chain[..., ix[:2]].reshape(-1, 2, order="F"), dtype=F32)
    arc = np.empty(n1 * n2, dtype=F32)
    ember.fortran.polyline_arc(np.asfortranarray(xr_cut_nd, dtype=F32), xq, arc)
    Z = arc.reshape(n1, n2, order="F").astype(float)
    T = chain[..., ix[2]].astype(float) / pitch
    if np.abs(np.diff(Z, axis=1)).mean() > np.abs(np.diff(Z, axis=0)).mean():
        chain, Z, T = (np.swapaxes(v, 0, 1) for v in (chain, Z, T))
    if Z[0].mean() > Z[-1].mean():
        chain, Z, T = chain[::-1], Z[::-1], T[::-1]
    if T[:, 0].mean() > T[:, -1].mean():
        chain, Z, T = chain[:, ::-1], Z[:, ::-1], T[:, ::-1]
    return chain, Z, T


def _roll_to_start(sheet, T, ix, pitch):
    """Start a closed sheet at the column whose hub theta is least modulo
    the pitch, whatever patch the stitching began from: the columns before
    it move to the end, a pitch on."""
    c0 = int(np.argmin(np.mod(T[0, :-1], 1.0)))
    if c0:
        tail = sheet[:, 1 : c0 + 1].copy()
        tail[..., ix[2]] += F32(pitch)
        sheet = np.concatenate([sheet[:, c0:], tail], axis=1)
    # Whole pitches off, so the first hub node lies in the first pitch
    n = np.floor(float(sheet[0, 0, ix[2]]) / pitch + 1e-6)
    if n:
        sheet = sheet.copy()
        sheet[..., ix[2]] -= F32(n * pitch)
    return sheet


def _arc(xr_nd, xq):
    """Arc length along the cut of points (n, 2), by the kernel."""
    arc = np.empty(len(xq), dtype=F32)
    ember.fortran.polyline_arc(
        np.asfortranarray(xr_nd, dtype=F32), np.asfortranarray(xq, dtype=F32), arc
    )
    return arc


def _embed(xrt, pitch):
    """Points in which copies a whole pitch apart in theta coincide."""
    phi = xrt[:, 2].astype(float) * (2 * np.pi / pitch)
    R = xrt[:, 1].astype(float) * pitch / (2 * np.pi)
    return np.ascontiguousarray(
        np.column_stack([xrt[:, 0], xrt[:, 1], R * np.cos(phi), R * np.sin(phi)])
    )


def _triangles(blocks, xr_ext, ix, pitch, xr_nd):
    """Marching cubes triangles of the ragged blocks, merged and described.

    Vertex copies within a few single precision steps of each other, modulo
    a pitch, share a merged id: the grid is snapped, so copies across an
    interface or a periodic face coincide to that. The merged mesh's edges
    give its Euler characteristic and its boundary loops, in order.
    """
    verts, faces, n = [], [], 0
    for block in blocks:
        out = ember.cut._marching_cubes(
            block._data, *ember.cut._distance_args(xr_ext, block)
        )
        if out is None:
            continue
        v, f = out
        verts.append(np.asarray(v, dtype=F32))
        faces.append(f + n)
        n += len(v)
    V = np.concatenate(verts)
    faces = np.concatenate(faces).astype(np.int32)
    xrt = V[:, ix]
    big = np.abs(xrt).max(axis=0).astype(F32)
    tol = 4.0 * float(
        max(np.spacing(big[0]), np.spacing(big[1]), big[1] * np.spacing(big[2]))
    )

    emb = _embed(xrt, pitch)
    k = min(8, len(emb))
    dist, idx = KDTree(emb).query(emb, k=k)
    near = (dist <= tol) & (idx != np.arange(len(emb))[:, None])
    a = np.repeat(np.arange(len(emb), dtype=np.int32), k)[near.ravel()]
    b = idx[near].astype(np.int32)
    if len(a):
        mid = np.empty(len(emb), dtype=np.int32)
        parent = np.empty(len(emb), dtype=np.int32)
        n_mid = ember.fortran.union_labels(a, b, mid, parent)
    else:
        mid, n_mid = np.arange(len(emb), dtype=np.int32), len(emb)

    fm = np.asfortranarray(mid[faces].T, dtype=np.int32)
    nf = fm.shape[1]
    cap = 3 * nf
    start = np.empty(n_mid + 1, dtype=np.int32)
    bucket = np.empty(3 * nf, dtype=np.int32)
    uses = np.empty(3 * nf, dtype=np.int32)
    nbr = np.empty((2, n_mid), dtype=np.int32, order="F")
    mark = np.empty(n_mid, dtype=np.int32)
    loops = np.empty(cap, dtype=np.int32)
    loop_start = np.empty(cap + 1, dtype=np.int32)
    n_edge, _, n_loop, n_used, status = ember.fortran.edge_topology(
        fm, start, bucket, uses, nbr, mark, loops, loop_start, n_mid, nf, cap
    )
    degenerate = (fm[0] == fm[1]) | (fm[1] == fm[2]) | (fm[2] == fm[0])
    first = np.full(n_mid, -1)
    first[mid[::-1]] = np.arange(len(mid))[::-1]
    bnd_ids = loops[: max(loop_start[n_loop] - 1, 0)]
    return dict(
        V=V,
        faces=faces,
        z=_arc(xr_nd, xrt[:, :2]).astype(float),
        tn=xrt[:, 2].astype(float) / pitch,
        mid=mid,
        first=first,
        tol=tol,
        bnd_ids=bnd_ids,
        bnd_emb=emb[first[bnd_ids]],
        chi=n_used - n_edge + int((~degenerate).sum()),
        status=status,
        loops=[loops[loop_start[q] - 1 : loop_start[q + 1] - 1] for q in range(n_loop)],
    )


def _corner(tri, xrt, pitch):
    """Merged id of the boundary vertex at a side's end node."""
    d = np.linalg.norm(tri["bnd_emb"] - _embed(xrt[None], pitch), axis=1)
    i = int(np.argmin(d))
    if d[i] > 1e-4 * max(1.0, float(np.abs(xrt[:2]).max())):
        raise ValueError(
            "A side of a hole does not meet the triangles: are the grid's "
            "periodic pairs snapped? See Grid.snap_periodic."
        )
    return int(tri["bnd_ids"][i])


def _wall(tri, c0, c1, t0, t1, z_tol):
    """The boundary between two corners along one endwall, in order.

    Walks the boundary loop through merged corner `c0` both ways to corner
    `c1`, keeping theta between `t0` and `t1`, and takes the way that stays
    nearest one arc length, which it checks is an endwall. Returns the pitchwise
    fractions, arc lengths and data of its vertices, corners excluded.
    """
    loop = next((lp for lp in tri["loops"] if c0 in lp), None)
    if loop is None or c1 not in loop:
        raise ValueError("A hole's corners do not lie on one boundary loop.")
    i0 = int(np.nonzero(loop == c0)[0][0])
    n = len(loop)
    best = None
    for step in (1, -1):
        walk = loop[(i0 + step * np.arange(1, n + 1)) % n]
        path = walk[: int(np.argmax(walk == c1))]
        v = tri["first"][path]
        tn = tri["tn"][v]
        # Unwrap along the walk from the first corner
        steps = np.diff(np.concatenate([[t0], tn]))
        t = t0 + np.cumsum(steps - np.round(steps))
        prev = t[-1] if len(t) else t0
        # The far corner, unwrapped from the last vertex, must be the side's
        tc = tri["tn"][tri["first"][c1]]
        tc += np.round(prev - tc)
        if abs(tc - t1) > 1e-4 or (
            len(t) and (t.min() < t0 - 1e-4 or t.max() > t1 + 1e-4)
        ):
            continue
        o = np.argsort(t, kind="stable")
        spread = float(np.ptp(tri["z"][v])) if len(v) else 0.0
        # Both ways round can keep theta between the corners: the endwall is
        # the way that stays at one arc length, not the one up the sides
        if best is None or spread < best[2]:
            best = (v[o], t[o], spread)
    if best is None:
        raise ValueError("No endwall path joins a hole's corners.")
    v, t, spread = best
    z = tri["z"][v]
    if spread > z_tol:
        raise ValueError(
            "A boundary of the cut between chains is not an endwall: the cut "
            "meets a blade, tip gap or other solid."
        )
    return (t - t0) / (t1 - t0), z, tri["V"][v]


def _drop_slivers(u, frac=0.2):
    """Ascending `u`, ends kept, without points that make a gap under `frac`
    of the larger neighbouring gap: where the cut runs nearly along grid
    lines their crossings bunch into slivers that say nothing of the
    resolution."""
    u = u.copy()
    while u.size > 3:
        gap = np.diff(u)
        nb = np.maximum(
            np.concatenate([gap[1:], [gap[-1]]]), np.concatenate([[gap[0]], gap[:-1]])
        )
        k = int(np.argmin(gap / nb))
        if gap[k] >= frac * nb[k]:
            break
        u = np.delete(u, k + 1 if k + 1 < u.size - 1 else k)
    return u


def _resample(u, m):
    return np.interp(np.linspace(0, 1, m), np.linspace(0, 1, u.size), u)


def _fill(tri, L, R, factor, ix, pitch):
    """A hole's grid and data, its first and last columns its exact sides.

    `L` and `R` are dicts of the sides' arc lengths "z", theta in pitches
    "t" and data "d", hub first. Rows join node j of each side; their
    spacing is that of the endwall vertices between the corners, slivers
    merged, blended along the span; the endwall rows take their values
    along the walls, the rest from the triangles.
    """
    n = L["z"].size
    z_tol = 0.5 * min(
        L["z"][1] - L["z"][0],
        R["z"][1] - R["z"][0],
        L["z"][-1] - L["z"][-2],
        R["z"][-1] - R["z"][-2],
    )
    walls = []
    for j in (0, -1):
        c0 = _corner(tri, L["d"][j, ix], pitch)
        c1 = _corner(tri, R["d"][j, ix] - np.array([0, 0, pitch]), pitch)
        u, z, d = _wall(tri, c0, c1, L["t"][j], R["t"][j], z_tol)
        walls.append(
            (
                np.concatenate([[0.0], u, [1.0]]),
                np.concatenate([[L["z"][j]], z, [R["z"][j]]]),
                np.concatenate([L["d"][j][None], d, R["d"][j][None]]),
            )
        )
    u_hub, u_cas = _drop_slivers(walls[0][0]), _drop_slivers(walls[1][0])
    m = int(factor) * (max(u_hub.size, u_cas.size) - 1) + 1
    s = ((L["z"] - L["z"][0]) / (L["z"][-1] - L["z"][0]))[:, None]
    U = (1 - s) * _resample(u_hub, m)[None] + s * _resample(u_cas, m)[None]
    T = L["t"][:, None] + U * (R["t"] - L["t"])[:, None]

    nvar = tri["V"].shape[1]
    out = np.zeros((n, m, nvar), dtype=F32, order="F")
    hits = np.zeros((n, m), dtype=np.int32, order="F")
    ember.fortran.fill_hole(
        np.asfortranarray(tri["z"], dtype=F32),
        np.asfortranarray(tri["tn"], dtype=F32),
        np.asfortranarray(tri["V"], dtype=F32),
        np.asfortranarray(tri["faces"].T, dtype=np.int32),
        *(np.asfortranarray(v, dtype=F32) for v in (L["z"], L["t"], R["z"], R["t"])),
        np.asfortranarray(T.T, dtype=F32),
        F32(1e-5),
        out,
        hits,
    )
    if (hits[1:-1, 1:-1] == 0).any():
        raise ValueError(
            f"{int((hits[1:-1, 1:-1] == 0).sum())} nodes of a hole lie on no "
            "triangle: the cut meets a solid between chains, or the grid's "
            "periodic pairs are not snapped (see Grid.snap_periodic)."
        )
    for j, (u, z, d) in zip((0, -1), walls):
        out[j] = np.stack([np.interp(U[j], u, d[:, q]) for q in range(nvar)], axis=-1)
    out[:, 0], out[:, -1] = L["d"], R["d"]
    out[1:-1, 1:-1, ix[2]] = T[1:-1, 1:-1] * pitch
    out[[0, -1], 1:-1, ix[2]] = T[[0, -1], 1:-1] * pitch
    return out


def _side(chain, Z, T, col, shift=0.0, pitch=0.0, ix=None):
    """A chain's column as a hole side, `shift` pitches on."""
    d = chain[:, col].astype(F32).copy()
    if shift:
        d[:, ix[2]] += F32(shift * pitch)
    return dict(z=Z[:, col], t=T[:, col] + shift, d=d)


def _seam(tri, ix, pitch):
    """The periodic seam as a hole side: merged vertices with copies a pitch
    apart, hub to casing, theta made continuous along it, near ones merged."""
    tn, mid, z = tri["tn"], tri["mid"], tri["z"]
    lo = np.full(mid.max() + 1, np.inf)
    hi = np.full(mid.max() + 1, -np.inf)
    np.minimum.at(lo, mid, tn)
    np.maximum.at(hi, mid, tn)
    groups = np.nonzero(hi - lo > 0.5)[0]
    if groups.size < 2:
        raise ValueError("The cut has no periodic seam from hub to casing.")
    v = tri["first"][groups]
    v = v[np.argsort(z[v], kind="stable")]
    zs, ts = z[v], tn[v].copy()
    for q in range(1, len(ts)):
        ts[q] += np.round(ts[q - 1] - ts[q])
    # Merge vertices nearer than a twentieth of the neighbouring gaps
    keep = np.ones(len(zs), dtype=bool)
    gap = np.diff(zs)
    for q in range(1, len(zs) - 1):
        nb = max(gap[q - 1] if q > 0 else 0.0, gap[q] if q < len(gap) else 0.0)
        if gap[q - 1] < 0.05 * nb:
            keep[q] = False
    v, zs, ts = v[keep], zs[keep], ts[keep]
    d = tri["V"][v].astype(F32).copy()
    d[:, ix[2]] = (ts * pitch).astype(F32)
    L = dict(z=zs, t=ts, d=d)
    dR = d.copy()
    dR[:, ix[2]] += F32(pitch)
    R = dict(z=zs, t=ts + 1.0, d=dR)
    return L, R


def traverse(grid, xr_cut, factor=1):
    """One structured sheet of the cut through `grid`, as a 2D Block."""
    xr_cut = np.asarray(xr_cut, dtype=float)
    xr_ext = ember.cut._extend_ends(xr_cut, ember.cut.CUT_EXTEND)

    patches, ragged, touched = [], [], []
    for bid, block in enumerate(grid):
        if not ember.cut._cut_reaches_block(xr_ext, block):
            continue
        ps, status = block_patches(block, xr_ext)
        if status == "miss":
            continue
        touched.append(block)
        if status == "ragged":
            ragged.append(block)
        patches += ps
    if not touched:
        raise ValueError("The cut does not cross the grid.")
    if len({b.Nb for b in touched}) > 1:
        raise ValueError("The cut crosses blocks of different blade counts.")
    tmpl = touched[0]
    pitch = float(tmpl.pitch)
    ix = [tmpl._data_inds[k] for k in ("x", "r", "t")]
    xr_nd = np.ascontiguousarray(xr_ext / tmpl.L_ref)

    chains, tol = stitch(patches, ix, pitch)
    chains = [orient(c, ix, xr_nd, pitch) for c in chains]

    tri = _triangles(ragged, xr_ext, ix, pitch, xr_nd) if ragged else None
    if tri is not None:
        if tri["status"] != 0:
            raise ValueError("The cut's triangles are not a manifold surface.")
        rings = not chains
        if (rings and (tri["chi"] != 0 or len(tri["loops"]) != 2)) or (
            not rings and tri["chi"] != len(tri["loops"])
        ):
            raise ValueError(
                "The cut is not a blade-free annulus: it meets a blade or other "
                "solid, or the grid's periodic pairs are not snapped (see "
                "Grid.snap_periodic)."
            )

    if not chains:
        L, R = _seam(tri, ix, pitch)
        hole = _fill(tri, L, R, factor, ix, pitch)
        sheet = hole
        T = hole[..., ix[2]].astype(float) / pitch
        return _block(tmpl, _roll_to_start(sheet, T, ix, pitch))

    # Chains in order round the pitch, each starting within the first pitch
    order = []
    for chain, Z, T in chains:
        n = np.floor(T[0, 0])
        chain = chain.copy()
        chain[..., ix[2]] -= F32(n * pitch)
        order.append((chain, Z, T - n))
    order.sort(key=lambda c: c[2][0, 0])
    if len({c[0].shape[0] for c in order}) > 1:
        raise ValueError(
            "The cut's pieces have different numbers of spanwise nodes: "
            "a non-matching interface."
        )

    pieces = []
    for q, (chain, Z, T) in enumerate(order):
        pieces.append(chain)
        nxt = order[(q + 1) % len(order)]
        shift = 1.0 if q + 1 == len(order) else 0.0
        L = _side(chain, Z, T, -1)
        R = _side(nxt[0], nxt[1], nxt[2], 0, shift, pitch, ix)
        if _side_match(L["d"], R["d"], ix, pitch, tol) is not None:
            continue
        if tri is None:
            raise ValueError(
                "The cut does not close round the pitch: it meets a blade or "
                "other solid, which structured_meridional can cut."
            )
        pieces.append(_fill(tri, L, R, factor, ix, pitch)[:, 1:-1])
    first = order[0][0][:, :1].copy()
    first[..., ix[2]] += F32(pitch)
    sheet = np.concatenate(pieces + [first], axis=1)
    T = sheet[..., ix[2]].astype(float) / pitch
    # A chain that closed on itself repeats its first column at the end
    if np.abs(T[:, -1] - T[:, -2]).max() < 1e-6:
        sheet, T = sheet[:, :-1], T[:, :-1]
    if _side_match(sheet[:, -1], sheet[:, 0], ix, pitch, tol) != 1:
        raise ValueError(
            "The cut does not close round the pitch: it meets a blade or other "
            "solid, which structured_meridional can cut."
        )
    return _block(tmpl, _roll_to_start(sheet, T, ix, pitch))


def _block(tmpl, sheet):
    out = tmpl.empty(shape=sheet.shape[:2])
    out._data = np.asfortranarray(sheet, dtype=F32)
    return out

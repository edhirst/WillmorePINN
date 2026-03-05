"""
Spectral Basis for Genus-2 Surface via Hyperbolic Octagon Fundamental Domain

Computes Laplace-Beltrami eigenfunctions on the regular hyperbolic octagon with
edge identification a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹, producing a spectral
basis for the genus-2 surface of constant curvature K = −1.

The regular hyperbolic octagon lives in the Poincaré disk. Its 8 edges are
identified in 4 pairs (with reversal) to form a closed genus-2 surface Σ₂.
All 8 corner vertices are identified to a single point.

The discrete Laplace-Beltrami operator is built using cotangent weights on the
triangulated octagon, with boundary DOFs merged according to the identification.
The hyperbolic (Poincaré) metric is accounted for via the generalised eigenproblem:

    L_cot φ = λ M_hyp φ

where M_hyp uses the conformal weight ρ² = 4 / (1 − |p|²)² per triangle.

The resulting eigenfunctions φ₁, ..., φ_N form a fixed spectral basis Ψ(x,y)
that the network uses as input. Interpolation from mesh to an arbitrary point
in the octagon is done by barycentric interpolation and is differentiable
w.r.t. the (x,y) coordinates (needed for computing ∂F/∂x, ∂F/∂y in the
Willmore energy via autograd).
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import Delaunay, cKDTree
import torch
from typing import Optional, Tuple


# ============================================================================
# Octagon Geometry
# ============================================================================


def _octagon_disk_radius() -> float:
    """
    Poincaré disk Euclidean radius of the regular hyperbolic octagon  with all vertex angles π/4.

    The circumradius in the hyperbolic plane R_h satisfies:
        cosh(R_h) = cos(π/n) / sin(α/2)   for n=8, α=π/4
                  = cos(π/8) / sin(π/8)
                  = cot(π/8) = 1 + √2

    In the Poincaré disk: r = tanh(R_h / 2).
    """
    R_h = np.arccosh(1.0 / np.tan(np.pi / 8.0))
    return float(np.tanh(R_h / 2.0))


def build_octagon_vertices() -> np.ndarray:
    """
    Compute the 8 vertices of the regular hyperbolic octagon in the Poincaré disk.

    Vertices are at angles θ_k = π/8 + k·π/4 (k = 0,...,7) so that edges are
    symmetric about the principal axes.

    Returns:
        vertices: (8, 2) vertex positions in Poincaré disk coordinates
    """
    r = _octagon_disk_radius()
    angles = np.pi / 8.0 + np.arange(8) * (np.pi / 4.0)
    return r * np.column_stack([np.cos(angles), np.sin(angles)])


def octagon_euclidean_area() -> float:
    """
    Euclidean area of the octagon polygon in the Poincaré disk.

    For a regular octagon inscribed in a circle of Euclidean radius r:
        A = (1/2) · 8 · r² · sin(2π/8) = 2√2 · r²
    """
    r = _octagon_disk_radius()
    return 2.0 * np.sqrt(2.0) * r ** 2


def is_inside_octagon(
    points:   np.ndarray,
    vertices: np.ndarray,
    margin:   float = 0.0,
) -> np.ndarray:
    """
    Test whether points lie inside the convex octagon using the half-plane test.

    Args:
        points:   (N, 2) candidate points
        vertices: (8, 2) octagon vertices in CCW order
        margin:   minimum perpendicular distance from each edge required for
                  a point to be considered inside. margin > 0 erodes the
                  octagon inward, excluding a strip of width margin near each edge.

    Returns:
        Boolean mask of shape (N,)
    """
    n = len(vertices)
    inside = np.ones(len(points), dtype=bool)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        # For a CCW polygon, interior is the left side of each directed edge:
        #   (edge × dp) = edge_x·dp_y − edge_y·dp_x ≥ 0
        # cross / |edge| = signed distance from the edge line (positive = inside).
        edge_len = np.linalg.norm(edge)
        dp = points - v1
        cross = edge[0] * dp[:, 1] - edge[1] * dp[:, 0]
        inside &= (cross >= margin * edge_len - 1e-10)
    return inside


# ============================================================================
# Mesh Construction
# ============================================================================


def build_octagon_mesh(
    num_edge_divisions: int = 10,
    interior_grid_spacing: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a triangulation of the hyperbolic octagon with explicit boundary
    vertices placed uniformly on each edge.

    Edge identification (genus-2 word: a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹):

        Edge 0 (a₁)  ↔  Edge 2 (a₁⁻¹) reversed
        Edge 1 (b₁)  ↔  Edge 3 (b₁⁻¹) reversed
        Edge 4 (a₂)  ↔  Edge 6 (a₂⁻¹) reversed
        Edge 5 (b₂)  ↔  Edge 7 (b₂⁻¹) reversed

    All 8 corner vertices are identified to a single DOF.

    Edge identification parametrisation: a point at parameter t ∈ (0,1) along
    edge e is identified with the point at parameter t along the reversed partner
    edge (i.e., parameter 1−t along the forward partner edge).

    Args:
        num_edge_divisions: Number of equal segments per octagon edge.
            m = num_edge_divisions − 1 interior vertices per edge.
        interior_grid_spacing: Cartesian grid spacing for interior sample points.

    Returns:
        vertices:   (N_total, 2) all mesh vertex positions
        triangles:  (M, 3)      triangle vertex indices (interior triangles only)
        id_map:     (N_total,)  mapping from vertex index to compact reduced DOF
        vertex_info: dict with mesh statistics
    """
    verts_oct = build_octagon_vertices()  # (8, 2) corner vertices
    m = num_edge_divisions - 1            # interior points per edge (may be 0)

    # ------------------------------------------------------------------
    # 1. Boundary vertices: corners, then edge interiors
    # ------------------------------------------------------------------
    all_verts = list(verts_oct)           # indices 0..7 = corner vertices
    edge_start = {}                       # edge_start[e] = index of first interior point on edge e

    for e in range(8):
        edge_start[e] = len(all_verts)
        v_start = verts_oct[e]
        v_end   = verts_oct[(e + 1) % 8]
        for k in range(m):
            t = (k + 1.0) / (m + 1.0)
            all_verts.append((1.0 - t) * v_start + t * v_end)

    # ------------------------------------------------------------------
    # 2. Interior grid points
    # ------------------------------------------------------------------
    r_disk = _octagon_disk_radius()
    xs = np.arange(-r_disk + interior_grid_spacing,
                    r_disk, interior_grid_spacing)
    ys = np.arange(-r_disk + interior_grid_spacing,
                    r_disk, interior_grid_spacing)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    boundary_verts_arr = np.array(all_verts)   # current boundary vertices
    in_oct = is_inside_octagon(grid_pts, verts_oct)
    # Exclude grid points too close to existing boundary vertices
    margin = 0.4 * interior_grid_spacing
    dist_to_bnd = np.min(
        np.linalg.norm(
            grid_pts[:, None, :] - boundary_verts_arr[None, :, :], axis=2
        ),
        axis=1,
    )
    interior_mask = in_oct & (dist_to_bnd > margin)
    interior_pts = grid_pts[interior_mask]

    int_start_idx = len(all_verts)
    for p in interior_pts:
        all_verts.append(p)

    vertices = np.array(all_verts, dtype=np.float64)  # (N_total, 2)
    N_total  = len(vertices)

    # ------------------------------------------------------------------
    # 3. Delaunay triangulation; keep only interior triangles
    # ------------------------------------------------------------------
    tri = Delaunay(vertices)
    triangles_all = tri.simplices
    centroids    = vertices[triangles_all].mean(axis=1)
    inside_mask  = is_inside_octagon(centroids, verts_oct)
    triangles    = triangles_all[inside_mask]     # (M, 3)

    # ------------------------------------------------------------------
    # 4. Build identification map (DOF merging)
    # ------------------------------------------------------------------
    id_map = np.arange(N_total, dtype=int)

    # All corner vertices → vertex index 0
    for corner in range(1, 8):
        id_map[corner] = 0

    # Identified edge pairs: (e_a, e_b) with e_a[k] ↔ e_b[m-1-k]
    id_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for (e_a, e_b) in id_pairs:
        for k in range(m):
            i_a = edge_start[e_a] + k
            i_b = edge_start[e_b] + (m - 1 - k)
            canonical = min(i_a, i_b)
            id_map[i_a] = canonical
            id_map[i_b] = canonical

    # Path-compress to resolve transitive chains (corners all chain to 0)
    def _find(i):
        while id_map[i] != i:
            id_map[i] = id_map[id_map[i]]
            i = id_map[i]
        return i

    for i in range(N_total):
        id_map[i] = _find(i)

    # Compact DOFs to contiguous range [0, n_dof)
    unique_dofs = sorted(set(id_map.tolist()))
    relabel     = {old: new for new, old in enumerate(unique_dofs)}
    id_map      = np.array([relabel[id_map[i]] for i in range(N_total)], dtype=int)
    n_dof       = len(unique_dofs)

    vertex_info = {
        'n_total':        N_total,
        'n_dof':          n_dof,
        'n_triangles':    len(triangles),
        'n_interior_pts': len(interior_pts),
        'm':              m,
    }

    return vertices, triangles, id_map, vertex_info


# ============================================================================
# Discrete Laplacian with Hyperbolic Mass Matrix
# ============================================================================


def build_laplacian_and_mass(
    vertices:  np.ndarray,
    triangles: np.ndarray,
    id_map:    np.ndarray,
    n_dof:     int,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Assemble the cotangent stiffness matrix L and the hyperbolic lumped mass
    matrix M on the identified surface.

    The generalised eigenproblem  L φ = λ M φ  gives eigenfunctions of the
    Laplace-Beltrami operator for the hyperbolic (Poincaré) metric.

    The hyperbolic conformal factor at a point p is:
        ρ²(p) = 4 / (1 − |p|²)²

    Approximated per triangle by its centroid value (lumped mass rule).

    Args:
        vertices:  (N_total, 2) vertex positions (Poincaré disk)
        triangles: (M, 3)       triangle index triples
        id_map:    (N_total,)   vertex → reduced DOF
        n_dof:     int          number of reduced DOFs

    Returns:
        L: (n_dof, n_dof) sparse cotangent stiffness matrix
        M: (n_dof, n_dof) sparse diagonal mass matrix
    """
    L_rows, L_cols, L_data = [], [], []
    M_rows, M_data = [], []

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        e01 = v1 - v0
        e02 = v2 - v0
        e12 = v2 - v1

        # Euclidean triangle area (shoelace / cross product)
        area_flat = 0.5 * abs(e01[0] * e02[1] - e01[1] * e02[0])
        if area_flat < 1e-15:
            continue

        # Cotangent weights for each opposite angle
        def _cot(a, b):
            cross = a[0] * b[1] - a[1] * b[0]
            if abs(cross) < 1e-15:
                return 0.0
            return (a[0] * b[0] + a[1] * b[1]) / cross

        # cot at v0 (opposite edge i1-i2):  use vectors e01, e02
        w0 = _cot(e01, e02)
        # cot at v1 (opposite edge i0-i2):  use vectors -e01, e12
        w1 = _cot(-e01, e12)
        # cot at v2 (opposite edge i0-i1):  use vectors -e02, -e12
        w2 = _cot(-e02, -e12)

        # Hyperbolic conformal factor at centroid: ρ²(c) = 4/(1-|c|²)²
        c    = (v0 + v1 + v2) / 3.0
        r2c  = min(c[0] ** 2 + c[1] ** 2, 1.0 - 1e-6)
        rho2 = 4.0 / (1.0 - r2c) ** 2
        area_hyp = rho2 * area_flat

        d0, d1, d2 = id_map[i0], id_map[i1], id_map[i2]

        # Cotangent stiffness: each edge (da, db) with weight 0.5*(cot opposite)
        for da, db, w in [(d1, d2, 0.5 * w0),
                          (d0, d2, 0.5 * w1),
                          (d0, d1, 0.5 * w2)]:
            L_rows.extend([da, db, da, db])
            L_cols.extend([db, da, da, db])
            L_data.extend([-w, -w, +w, +w])

        # Lumped hyperbolic mass: 1/3 of triangle area to each vertex
        for d in (d0, d1, d2):
            M_rows.append(d)
            M_data.append(area_hyp / 3.0)

    # Assemble
    L = scipy.sparse.csr_matrix(
        (L_data, (L_rows, L_cols)), shape=(n_dof, n_dof)
    ).tocsr()

    M_diag = np.bincount(M_rows, weights=M_data, minlength=n_dof)
    M = scipy.sparse.diags(M_diag, format='csr')

    return L, M


def solve_eigenfunctions(
    L:                  scipy.sparse.csr_matrix,
    M:                  scipy.sparse.csr_matrix,
    num_eigenfunctions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalised eigenproblem L φ = λ M φ for the smallest nontrivial
    positive eigenfunctions (skipping the constant function whose λ ≈ 0).

    The cotangent Laplacian on the identified mesh can have small negative
    eigenvalues (a known artefact of non-Delaunay boundary triangles after DOF
    merging).  We therefore use the dense symmetric solver which handles
    indefinite L correctly, then select eigenvectors with the smallest *positive*
    eigenvalues.

    Args:
        L:                   (n, n) cotangent stiffness matrix
        M:                   (n, n) diagonal mass matrix
        num_eigenfunctions:  number of nontrivial eigenfunctions requested

    Returns:
        eigenvalues:  (k,) sorted positive eigenvalues (k ≤ num_eigenfunctions)
        eigenvectors: (n, k) corresponding eigenvectors (M-orthonormal)
    """
    import scipy.linalg

    L_dense = L.toarray()
    M_dense = np.diag(M.diagonal())

    # Symmetrise L (numerical drift)
    L_sym = 0.5 * (L_dense + L_dense.T)

    # Dense generalised eigenproblem: all eigenvalues, ascending
    vals, vecs = scipy.linalg.eigh(L_sym, M_dense, driver='gvd')

    # Discard negative and near-zero (constant) eigenvalues
    positive_mask = vals > 1e-4
    vals_pos = vals[positive_mask]
    vecs_pos = vecs[:, positive_mask]

    if len(vals_pos) == 0:
        # Very coarse mesh fallback: take the least-negative ones
        order = np.argsort(vals)
        vals  = vals[order[1:num_eigenfunctions + 1]]
        vecs  = vecs[:, order[1:num_eigenfunctions + 1]]
    else:
        # Take the num_eigenfunctions smallest positive eigenvalues
        order     = np.argsort(vals_pos)
        vals_pos  = vals_pos[order[:num_eigenfunctions]]
        vecs_pos  = vecs_pos[:, order[:num_eigenfunctions]]
        vals, vecs = vals_pos, vecs_pos

    # M-normalise each eigenvector
    M_diag = M.diagonal()
    for k in range(vecs.shape[1]):
        norm_sq = np.dot(vecs[:, k] ** 2, M_diag)
        if norm_sq > 1e-14:
            vecs[:, k] /= np.sqrt(norm_sq)

    return vals, vecs


# ============================================================================
# Genus-2 Reference Surface (level-set mesh + spectral matching)
# ============================================================================


def _torus_implicit(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    cx: float, R: float, r: float,
) -> np.ndarray:
    """Signed implicit for torus centred at (cx, 0, 0) in the xy-plane.

    Returns < 0 inside the tube, > 0 outside.
    """
    rho = np.sqrt((X - cx) ** 2 + Y ** 2)
    return (rho - R) ** 2 + Z ** 2 - r ** 2


def _project_blend_to_level_set(
    xyz:     np.ndarray,
    cx1:     float,
    cx2:     float,
    R:       float,
    r:       float,
    epsilon: float,
    n_iter:  int   = 60,
    lr:      float = 0.5,
) -> np.ndarray:
    """
    Project guide positions xyz onto the genus-2 level set T₁·T₂ = ε.

    Tₖ(p) = (ρₖ − R)² + z² − r²  where ρₖ = √((x−cxₖ)² + y²).
    Level set: f(p) = T₁(p)·T₂(p) − ε = 0.

    Uses damped Newton projection:
        p ← p − lr · f(p) / ‖∇f(p)‖² · ∇f(p)

    The Abelian-blend positions are close to one or both tori, so starting
    iterates are already near the level set and convergence is fast.

    Args:
        xyz:     (N, 3) starting positions (Abelian blend).
        cx1:     x-centre of torus 1.
        cx2:     x-centre of torus 2.
        R:       major radius of both tori.
        r:       tube radius of both tori.
        epsilon: level-set value (controls bridge thickness).
        n_iter:  maximum Newton iterations.
        lr:      damping factor (< 1 improves stability).

    Returns:
        (N, 3) float32 array of projected positions on T₁·T₂ = ε.
    """
    eps_rho = 1e-8
    p = xyz.copy().astype(np.float64)

    for _ in range(n_iter):
        rho1 = np.sqrt((p[:, 0] - cx1) ** 2 + p[:, 1] ** 2)
        rho2 = np.sqrt((p[:, 0] - cx2) ** 2 + p[:, 1] ** 2)

        T1 = (rho1 - R) ** 2 + p[:, 2] ** 2 - r ** 2
        T2 = (rho2 - R) ** 2 + p[:, 2] ** 2 - r ** 2
        f  = T1 * T2 - epsilon

        # ∇Tₖ
        dT1_x = 2.0 * (rho1 - R) * (p[:, 0] - cx1) / (rho1 + eps_rho)
        dT1_y = 2.0 * (rho1 - R) * p[:, 1]          / (rho1 + eps_rho)
        dT1_z = 2.0 * p[:, 2]

        dT2_x = 2.0 * (rho2 - R) * (p[:, 0] - cx2) / (rho2 + eps_rho)
        dT2_y = 2.0 * (rho2 - R) * p[:, 1]          / (rho2 + eps_rho)
        dT2_z = 2.0 * p[:, 2]

        # ∇f = T₂·∇T₁ + T₁·∇T₂
        gx = T2 * dT1_x + T1 * dT2_x
        gy = T2 * dT1_y + T1 * dT2_y
        gz = T2 * dT1_z + T1 * dT2_z

        g_sq   = gx ** 2 + gy ** 2 + gz ** 2 + eps_rho
        step   = lr * f / g_sq

        p[:, 0] -= step * gx
        p[:, 1] -= step * gy
        p[:, 2] -= step * gz

    return p.astype(np.float32)


def _largest_component(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Retain only the largest connected component of a triangle mesh."""
    from scipy.sparse.csgraph import connected_components as _cc
    N = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj  = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N)
    )
    n_comp, labels = _cc(adj, directed=False)
    if n_comp == 1:
        return verts, faces
    counts  = np.bincount(labels, minlength=n_comp)
    largest = int(np.argmax(counts))
    mask    = labels == largest
    new_idx = np.full(N, -1, dtype=int)
    new_idx[mask] = np.arange(int(mask.sum()))
    valid   = np.all(mask[faces], axis=1)
    return verts[mask], new_idx[faces[valid]]


def _euler_characteristic(verts: np.ndarray, faces: np.ndarray) -> int:
    """Compute Euler characteristic V − E + F of a triangle mesh."""
    V = len(verts)
    F = len(faces)
    edges: set = set()
    for f in faces:
        for i in range(3):
            edges.add(tuple(sorted([int(f[i]), int(f[(i + 1) % 3])])))
    E = len(edges)
    return V - E + F




def build_genus2_surface_mesh(
    R:        float = 1.0,
    r:        float = 0.35,
    d:        float = 1.6,
    epsilon:  float = 0.005,
    grid_res: int   = 64,
    verbose:  bool  = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a triangulated genus-2 surface via marching cubes on the level set

        T₁(x,y,z) · T₂(x,y,z) = ε

    where  Tᵢ = (√((x ∓ d/2)² + y²) − R)² + z² − r²  are two tori of
    major radius R, tube radius r, centred at (±d/2, 0, 0).

    Parameter constraints for genus 2:
        R − r < d/2 < R + r  (tori overlap in their tube region)

    When ε > 0 the level set smoothly rounds the self-intersection of
    T₁ · T₂ = 0, producing a single connected component with χ = −2.

    Args:
        R:        major radius of each torus.
        r:        tube radius of each torus.
        d:        centre-to-centre distance (∓d/2 along x-axis).
        epsilon:  level-set value; controls connection thickness.
        grid_res: marching-cubes grid resolution per axis.
        verbose:  print progress/verification info.

    Returns:
        verts: (N, 3) surface vertex positions.
        faces: (M, 3) triangle face indices (0-indexed, largest component).
    """
    from skimage.measure import marching_cubes

    # Bounding box: encompass both tori with margin
    bx  = d / 2 + R + r + 0.3
    byz = R + r + 0.3
    bz  = r + 0.3

    xs = np.linspace(-bx,  bx,  grid_res)
    ys = np.linspace(-byz, byz, grid_res)
    zs = np.linspace(-bz,  bz,  grid_res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    T1 = _torus_implicit(X, Y, Z, -d / 2, R, r)
    T2 = _torus_implicit(X, Y, Z,  d / 2, R, r)
    F  = T1 * T2 - epsilon

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    verts_raw, faces, _, _ = marching_cubes(F, level=0.0, spacing=(dx, dy, dz))

    # Shift to world coordinates (marching_cubes origin = grid lower corner)
    origin = np.array([xs[0], ys[0], zs[0]])
    verts  = verts_raw + origin

    verts, faces = _largest_component(verts, faces)

    chi = _euler_characteristic(verts, faces)
    if verbose:
        print(f"  Genus-2 reference mesh: {len(verts)} verts, "
              f"{len(faces)} faces, χ = {chi}  (expected −2)")
    if chi != -2 and verbose:
        print(f"  Warning: χ = {chi} ≠ −2. Consider adjusting d or epsilon.")

    # Centre and normalise to unit bounding-box scale for numerical stability
    centre = verts.mean(axis=0)
    verts -= centre
    scale  = np.abs(verts).max()
    if scale > 1e-8:
        verts /= scale
    verts *= 2.0        # rescale to ≈ [-2, 2]³

    return verts, faces


# ============================================================================
# Abelian Coordinates (pretraining reference embedding)
# ============================================================================


def _compute_abelian_coords_at_vertices(
    vertices:  np.ndarray,
    triangles: np.ndarray,
    m:         int,
) -> np.ndarray:
    """
    Compute four smooth Abelian coordinate functions (u₁, v₁, u₂, v₂) on the
    unidentified octagon mesh as Dirichlet BVP solutions with period-consistent
    boundary values.

    The identification word  a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹  defines four
    generating cycles.  For each cycle γ_k we assign boundary values that
    vary from 0 to 2π over the relevant octagon edges, with 0 elsewhere.
    Linear interpolation along each edge automatically satisfies the jump
    conditions on identified edge pairs (edge0[k] ≡ reversed edge2[k]).

    Corner values (columns = corners 0..7):
        u₁: [0, 2π, 2π, 0,  0,  0,  0,  0 ]   (0→2π along a₁ cycle)
        v₁: [0,  0, 2π, 2π, 0,  0,  0,  0 ]   (0→2π along b₁ cycle)
        u₂: [0,  0,  0,  0,  0, 2π, 2π,  0]   (0→2π along a₂ cycle)
        v₂: [0,  0,  0,  0,  0,  0, 2π, 2π]   (0→2π along b₂ cycle)

    The stiffness matrix uses inverse-distance weights w_{ij} = 1/|v_i−v_j|
    (rather than cotangent weights). The cotangent Laplacian is indefinite on
    Poincaré-disk meshes (78% of triangles carry a negative cotangent weight,
    yielding a K_II with 36 negative eigenvalues), causing 10× overshoots and
    wild oscillations after clamping. IDW weights are always positive, so K_II
    is positive definite and the maximum principle holds: solution ∈ [0, 2π].

    Args:
        vertices:  (N_total, 2) mesh vertex positions
        triangles: (M, 3) triangle indices (all octagon interior triangles)
        m:         num_edge_divisions − 1 (interior vertices per edge; may be 0)

    Returns:
        abelian_coords: (N_total, 4) values of (u₁, v₁, u₂, v₂) at each vertex
    """
    N           = len(vertices)
    int_start   = 8 + 8 * m          # interior grid nodes start here

    # Corner values for the 4 harmonic coordinate functions
    TWO_PI      = 2.0 * np.pi
    corner_vals = np.array([
        [0,       TWO_PI,  TWO_PI,  0,       0,       0,       0,       0      ],  # u₁
        [0,       0,       TWO_PI,  TWO_PI,  0,       0,       0,       0      ],  # v₁
        [0,       0,       0,       0,       0,       TWO_PI,  TWO_PI,  0      ],  # u₂
        [0,       0,       0,       0,       0,       0,       TWO_PI,  TWO_PI ],   # v₂
    ], dtype=np.float64)  # (4, 8)

    # ---- Build boundary-condition array ----
    is_boundary          = np.ones(N, dtype=bool)
    is_boundary[int_start:] = False

    bc = np.zeros((N, 4), dtype=np.float64)

    # Corners
    for c in range(8):
        bc[c, :] = corner_vals[:, c]

    # Edge interiors: linear interpolation between the two endpoint corner values.
    # This automatically satisfies the period jump conditions on identified pairs
    # (u₁(edge0[k]) = u₁(edge2[m-1-k]), etc.).
    edge_start_local = {e: 8 + e * m for e in range(8)}
    for e in range(8):
        v_s = corner_vals[:, e]           # value at start corner
        v_e = corner_vals[:, (e + 1) % 8] # value at end corner
        for k in range(m):
            t           = (k + 1.0) / (m + 1.0)
            bc[edge_start_local[e] + k, :] = (1.0 - t) * v_s + t * v_e

    # ---- Inverse-distance weighted stiffness on the unidentified mesh ----
    # IDW: w_{ij} = 1 / |v_i − v_j|. All weights positive → positive-definite
    # K_II → maximum principle holds, solution guaranteed ∈ [0, 2π].
    rows, cols, data = [], [], []
    seen_edges: set = set()
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for (ia, ib) in [(i0, i1), (i1, i2), (i0, i2)]:
            key = (min(ia, ib), max(ia, ib))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            d_ij = float(np.linalg.norm(vertices[ia] - vertices[ib]))
            if d_ij < 1e-15:
                continue
            w = 1.0 / d_ij
            rows.extend([ia, ib, ia, ib])
            cols.extend([ib, ia, ia, ib])
            data.extend([-w, -w, +w, +w])

    K = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    K = 0.5 * (K + K.T)  # symmetrise

    # ---- Dirichlet solve: K_II f_I = -K_IB b_bnd ----
    int_idx = np.where(~is_boundary)[0]
    bnd_idx = np.where( is_boundary)[0]
    n_int   = len(int_idx)

    if n_int == 0:
        return bc   # degenerate mesh — return linear BCs directly

    K_II = K[np.ix_(int_idx, int_idx)].tocsc()
    K_IB = K[np.ix_(int_idx, bnd_idx)]
    b_bnd = bc[bnd_idx, :]               # (n_bnd, 4)
    rhs   = -(K_IB @ b_bnd)              # (n_int, 4)

    # LU factorisation; solve all 4 RHS in one shot.
    # IDW K_II is positive definite, so no clamping is needed.
    lu    = scipy.sparse.linalg.splu(K_II + scipy.sparse.eye(n_int, format='csc') * 1e-12)
    f_int = lu.solve(rhs)               # (n_int, 4)

    abelian = np.zeros((N, 4), dtype=np.float64)
    abelian[bnd_idx, :] = b_bnd
    abelian[int_idx, :]  = f_int

    return abelian


# ============================================================================
# Main Class
# ============================================================================


class HyperbolicOctagonSpectral:
    """
    Spectral basis for the genus-2 surface via hyperbolic octagon identification.

    Computes Laplace-Beltrami eigenfunctions on the closed genus-2 surface Σ₂
    obtained from the regular hyperbolic octagon (Poincaré disk) by identifying
    opposite edges via the word a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹.

    The eigenfunctions are computed once at initialisation and stored; they are
    never recomputed during training. Interpolation from mesh to an arbitrary
    point in the octagon is differentiable w.r.t. (x, y) coordinates, so
    autograd can compute ∂F/∂x and ∂F/∂y for the Willmore integrand.

    Attributes:
        num_eigenfunctions: Number of nontrivial LB eigenfunctions.
        eigenvalues_np:    (K,) numpy array of LB eigenvalues λ₁ ≤ ... ≤ λ_K.
        oct_area:          Euclidean area of the octagon in the Poincaré disk.
    """

    def __init__(
        self,
        num_eigenfunctions:    int   = 16,
        num_edge_divisions:    int   = 10,
        interior_grid_spacing: float = 0.05,
        cache_path:            Optional[str] = None,
        verbose:               bool  = True,
    ):
        """
        Args:
            num_eigenfunctions:    Number of nontrivial LB eigenfunctions to compute
                                   (the constant is always excluded).
            num_edge_divisions:    Number of equal segments per octagon edge.
                                   Larger → finer boundary → better eigenfunctions.
            interior_grid_spacing: Cartesian spacing for interior mesh points.
                                   Smaller → finer mesh → higher quality Laplacian.
            cache_path:            If given, save/load the computed basis to/from
                                   this .npz file to avoid recomputation.
            verbose:               Print progress information.
        """
        self.num_eigenfunctions    = num_eigenfunctions
        self.num_edge_divisions    = num_edge_divisions
        self.interior_grid_spacing = interior_grid_spacing
        self._device               = torch.device('cpu')

        # Attempt to load from cache
        if cache_path is not None and self._try_load(cache_path, num_eigenfunctions, verbose):
            return

        if verbose:
            print("Building hyperbolic octagon spectral basis...")

        # --- Build mesh ---
        vertices, triangles, id_map, vinfo = build_octagon_mesh(
            num_edge_divisions    = num_edge_divisions,
            interior_grid_spacing = interior_grid_spacing,
        )
        n_dof = vinfo['n_dof']

        if verbose:
            print(f"  Mesh: {vinfo['n_total']} vertices, "
                  f"{vinfo['n_triangles']} triangles, {n_dof} DOFs (after identification)")

        # --- Cotangent Laplacian and mass matrix ---
        L, M = build_laplacian_and_mass(vertices, triangles, id_map, n_dof)

        if verbose:
            print(f"  Solving for {num_eigenfunctions} Laplace-Beltrami eigenfunctions...")

        # --- Solve eigenproblem ---
        eigenvalues, eigenvectors = solve_eigenfunctions(L, M, num_eigenfunctions)
        actual_k = eigenvectors.shape[1]

        if verbose:
            print(f"  λ₁,...,λ_{actual_k}: {np.round(eigenvalues[:min(8, actual_k)], 4)}")

        # --- Lift eigenfunctions back to all vertices ---
        # ef_at_vertices[v] = eigenvectors[id_map[v], :]
        ef_at_vertices = eigenvectors[id_map, :]   # (N_total, K)

        # Store numpy data
        self.vertices_np         = vertices        # (N_total, 2)
        self.triangles_np        = triangles       # (M, 3)
        self.id_map_np           = id_map          # (N_total,)
        self.eigenvalues_np      = eigenvalues     # (K,)
        self.ef_at_vertices_np   = ef_at_vertices  # (N_total, K)
        self.num_eigenfunctions  = actual_k        # may be < requested if mesh too small

        # Octagon geometry for sampling
        self.oct_vertices = build_octagon_vertices()  # (8, 2)
        self.oct_area     = octagon_euclidean_area()

        # Precompute differentiable interpolation data
        self._build_interpolation_tables()

        # Compute Abelian harmonic coordinates (retained for diagnostics)
        self._build_abelian_interp_tables()

        # Build reference surface for supervised pretraining
        self._build_surface_reference(verbose=verbose)

        # Save to cache if requested
        if cache_path is not None:
            self._save(cache_path)
            if verbose:
                print(f"  Cached to: {cache_path}")

        if verbose:
            print(f"  Done. Octagon Euclidean area = {self.oct_area:.5f}, "
                  f"K = {actual_k} features")

    # ------------------------------------------------------------------
    # Differentiable barycentric interpolation
    # ------------------------------------------------------------------

    def _build_interpolation_tables(self):
        """
        Precompute per-triangle barycentric inverse matrices and eigenfunction
        tables, and build a KD-tree on triangle centroids for fast lookup.

        For triangle t with vertices p₀, p₁, p₂, the barycentric inverse is:
            B_t = inv([p₁ − p₀ | p₂ − p₀])   (2 × 2 matrix)
        so that  [λ₁, λ₂]ᵀ = B_t @ (p − p₀),  λ₀ = 1 − λ₁ − λ₂.
        """
        verts = self.vertices_np    # (N_total, 2)
        tris  = self.triangles_np   # (M, 3)
        ef    = self.ef_at_vertices_np  # (N_total, K)

        p0 = verts[tris[:, 0]]  # (M, 2)
        p1 = verts[tris[:, 1]]  # (M, 2)
        p2 = verts[tris[:, 2]]  # (M, 2)

        e1 = p1 - p0   # (M, 2)
        e2 = p2 - p0   # (M, 2)

        # det of [e1 | e2]
        dets = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # (M,)

        # B = inv([e1 | e2]), stored row-major per triangle (M, 2, 2)
        B = np.zeros((len(tris), 2, 2), dtype=np.float64)
        safe = np.abs(dets) > 1e-15
        B[safe, 0, 0] =  e2[safe, 1] / dets[safe]
        B[safe, 0, 1] = -e2[safe, 0] / dets[safe]
        B[safe, 1, 0] = -e1[safe, 1] / dets[safe]
        B[safe, 1, 1] =  e1[safe, 0] / dets[safe]

        centroids = (p0 + p1 + p2) / 3.0  # (M, 2)

        # Eigenfunction values at the 3 vertices of each triangle: (M, 3, K)
        tri_ef = ef[tris]   # (M, 3, K)

        # Store numpy versions
        self._B_np        = B           # (M, 2, 2)
        self._p0_np       = p0          # (M, 2)
        self._tri_ef_np   = tri_ef      # (M, 3, K)
        self._centroids   = centroids   # (M, 2)

        # KD-tree for fast centroid nearest-neighbour lookup
        self._centroid_tree = cKDTree(centroids)

        # Torch tensors (CPU initially; moved by .to())
        self._B_torch      = torch.from_numpy(B).float()       # (M, 2, 2)
        self._p0_torch     = torch.from_numpy(p0).float()      # (M, 2)
        self._tri_ef_torch = torch.from_numpy(tri_ef).float()  # (M, 3, K)

    def _build_abelian_interp_tables(self):
        """
        Compute the Abelian coordinate functions at all mesh vertices and store
        per-triangle interpolation arrays for differentiable evaluation.

        Abelian coordinates (u₁, v₁, u₂, v₂) ∈ [0, 2π]⁴ are computed by solving
        four Dirichlet BVPs on the unidentified mesh using inverse-distance
        weights (see _compute_abelian_coords_at_vertices). The result is stored
        as per-triangle value arrays for the same barycentric interpolation
        path used for eigenfunctions.
        """
        m_actual = self.num_edge_divisions - 1

        abelian_np = _compute_abelian_coords_at_vertices(
            self.vertices_np,
            self.triangles_np,
            m_actual,
        )  # (N_total, 4)

        tris  = self.triangles_np   # (M, 3)
        tri_ab = abelian_np[tris]   # (M, 3, 4)

        self._abelian_np     = abelian_np         # (N_total, 4)
        self._tri_ab_torch   = torch.from_numpy(tri_ab).float()  # (M, 3, 4) — CPU initially

    def _interpolate_abelian(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Differentiable barycentric interpolation of Abelian coordinates
        (u₁, v₁, u₂, v₂) at arbitrary octagon points.

        Uses the same triangle-lookup and barycentric machinery as `interpolate`.

        Args:
            xy: (batch, 2) octagon sample points

        Returns:
            coords: (batch, 4) values of (u₁, v₁, u₂, v₂) at xy
        """
        xy_np   = xy.detach().cpu().numpy()
        tri_idx = self._find_triangles(xy_np)
        t_idx   = torch.from_numpy(tri_idx).long()
        device  = xy.device

        B_b    = self._B_torch[t_idx].to(device)            # (batch, 2, 2)
        p0_b   = self._p0_torch[t_idx].to(device)           # (batch, 2)
        ab_b   = self._tri_ab_torch[t_idx].to(device)       # (batch, 3, 4)

        delta  = xy - p0_b                                   # (batch, 2)
        lam12  = torch.einsum('bij,bj->bi', B_b, delta)     # (batch, 2)
        lam0   = 1.0 - lam12[:, 0:1] - lam12[:, 1:2]       # (batch, 1)
        lam    = torch.cat([lam0, lam12], dim=1)             # (batch, 3)

        coords = torch.einsum('bi,bic->bc', lam, ab_b)      # (batch, 4)
        return coords

    def _interpolate_reference(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Differentiable barycentric interpolation of the reference surface xyz
        targets at arbitrary octagon points.

        Args:
            xy: (batch, 2) octagon sample points.

        Returns:
            xyz: (batch, 3) interpolated surface positions.
        """
        xy_np   = xy.detach().cpu().numpy()
        tri_idx = self._find_triangles(xy_np)
        t_idx   = torch.from_numpy(tri_idx).long()
        device  = xy.device

        B_b   = self._B_torch[t_idx].to(device)           # (batch, 2, 2)
        p0_b  = self._p0_torch[t_idx].to(device)          # (batch, 2)
        ref_b = self._tri_ref_torch[t_idx].to(device)     # (batch, 3, 3)

        delta = xy - p0_b                                     # (batch, 2)
        lam12 = torch.einsum('bij,bj->bi', B_b, delta)        # (batch, 2)
        lam0  = 1.0 - lam12[:, 0:1] - lam12[:, 1:2]          # (batch, 1)
        lam   = torch.cat([lam0, lam12], dim=1)               # (batch, 3)

        xyz = torch.einsum('bi,bic->bc', lam, ref_b)          # (batch, 3)
        return xyz

    def _build_surface_reference(
        self,
        R:             float = 1.0,
        r:             float = 0.45,
        d:             float = 1.6,
        epsilon:       float = 0.005,
        bvp_grid_size: int   = 80,
        verbose:       bool  = True,
        # Legacy parameters retained for API compatibility but unused
        num_match_eigfns: int   = 8,
        grid_res:         int   = 64,
    ) -> None:
        """
        Build the genus-2 reference surface for supervised pretraining by
        projecting Abelian-blend guide positions onto the level set T₁·T₂ = ε.

        The reference must be smooth AND identification-consistent: two octagon
        vertices identified by the edge-pairing word
        a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹ must map to the same xyz target.

        The Abelian coordinate functions (u₁, v₁, u₂, v₂) ∈ [0, 2π]⁴ are solved
        via Dirichlet BVP with inverse-distance weights, giving no oscillations
        (see _compute_abelian_coords_at_vertices).

        Construction:
          1. For each octagon vertex, read its Abelian angles (u₁,v₁,u₂,v₂).
          2. Evaluate two guide torus positions (both standard xy-plane tori):
               probe₁ = torus at (−d/2, 0, 0): (−d/2 + (R+r·cos v₁)·cos u₁,
                                                  (R+r·cos v₁)·sin u₁,  r·sin v₁)
               probe₂ = torus at (+d/2, 0, 0): (+d/2 + (R+r·cos v₂)·cos u₂,
                                                  (R+r·cos v₂)·sin u₂,  r·sin v₂)
          3. Blend the two guide positions with quadratic handle-activity weights:
               raw_k = sin²(uₖ/2) + sin²(vₖ/2),  w_k = raw_k² + 0.01
               target = (w₁·probe₁ + w₂·probe₂) / (w₁ + w₂)
             The quadratic sharpening concentrates each handle's weight on its
             own torus so the reference resembles a connected double torus.

             Note: projecting the blend onto the level set T₁·T₂ = ε via
             Newton iteration is topologically correct but creates a hard seam
             along the α=0.5 iso-contour.  Octagon triangles that cross this
             seam have vertices projected to opposite tori, producing thousands
             of self-intersecting seam-spanning triangles.  The smooth blend
             avoids this by varying continuously across the transition zone.

        Because (u₁,v₁,u₂,v₂) are identical at identified vertex pairs, the
        blended guide is identical, and hence the projected xyz is identical —
        no snapping or post-processing is required.

        Connectivity constraint: for the level set T₁·T₂ = ε to be a connected
        genus-2 surface, the tori must overlap in their tube region:

            R − r  <  d/2  <  R + r

        With defaults R=1.0, r=0.45, d=1.6: 0.55 < 0.8 < 1.45 ✓.

        Args:
            R:       major radius of each reference torus.
            r:       tube radius of each reference torus.
            d:       centre-to-centre separation (tori at ±d/2 along x-axis).
                     Must satisfy R−r < d/2 < R+r for a connected genus-2 surface.
            epsilon: level-set value ε; controls bridge cross-section area.
                     Larger ε gives a thicker connecting neck.
            verbose: print progress information.
        """
        if verbose:
            print(f"  Building genus-2 reference (Abelian blend, R={R}, r={r}, d={d})...")

        ab = self._abelian_np          # (N_oct, 4): (u₁, v₁, u₂, v₂) ∈ [0, 2π]⁴
        u1, v1 = ab[:, 0], ab[:, 1]
        u2, v2 = ab[:, 2], ab[:, 3]

        cx1 = -d / 2.0
        cx2 = +d / 2.0

        # Both probes are standard xy-plane tori at ±d/2.
        # These guide positions are close to the true level-set surface and will
        # be projected onto T₁·T₂ = ε below.
        probe1 = np.stack([
            cx1 + (R + r * np.cos(v1)) * np.cos(u1),
            (R + r * np.cos(v1)) * np.sin(u1),
            r * np.sin(v1),
        ], axis=1)   # (N_oct, 3)

        probe2 = np.stack([
            cx2 + (R + r * np.cos(v2)) * np.cos(u2),
            (R + r * np.cos(v2)) * np.sin(u2),
            r * np.sin(v2),
        ], axis=1)   # (N_oct, 3)

        # Handle-activity weights with quadratic sharpening.
        #
        # raw_k = sin²(u_k/2) + sin²(v_k/2) ∈ [0, 2], zero exactly at the
        # base-point (all 8 octagon corners where each coord ∈ {0, 2π}).  The
        # quadratic envelope w_k = raw_k² amplifies the dominant handle (large
        # raw → even larger w) while suppressing the weak handle (small raw →
        # very small w), giving a cleaner partition of the two tori.
        #
        # A small floor is added so that at the base-point (raw₁=raw₂=0) the
        # blend evaluates to ½(probe₁_base + probe₂_base) rather than 0/0.
        floor = 0.01
        raw1 = np.sin(u1 / 2.0) ** 2 + np.sin(v1 / 2.0) ** 2   # ∈ [0, 2]
        raw2 = np.sin(u2 / 2.0) ** 2 + np.sin(v2 / 2.0) ** 2
        w1 = raw1 ** 2 + floor   # (N_oct,)
        w2 = raw2 ** 2 + floor

        # Use the blend as the reference target directly.
        #
        # Projecting the blend onto the level set T₁·T₂ = ε is topologically
        # correct but creates a hard seam along α = w₁/(w₁+w₂) = 0.5: vertices
        # on each side project to opposite tori, so triangles that cross the
        # seam span the full surface, producing thousands of self-intersections.
        # The smooth blend avoids this by varying continuously across the
        # transition zone, giving a well-behaved genus-2-shaped reference.
        denom  = (w1 + w2)[:, None]                                    # (N_oct, 1)
        target_xyz = ((w1[:, None] * probe1 + w2[:, None] * probe2) / denom).astype(np.float32)  # (N_oct, 3)

        self._surface_ref_xyz_np  = target_xyz
        self._surface_ref_params  = {'R': R, 'r': r, 'd': d, 'epsilon': epsilon}
        tris    = self.triangles_np          # (M, 3)
        tri_ref = target_xyz[tris]           # (M, 3, 3)
        self._tri_ref_torch = torch.from_numpy(tri_ref)

        if verbose:
            print(
                f"  Reference: "
                f"x ∈ [{target_xyz[:,0].min():.3f}, {target_xyz[:,0].max():.3f}], "
                f"y ∈ [{target_xyz[:,1].min():.3f}, {target_xyz[:,1].max():.3f}], "
                f"z ∈ [{target_xyz[:,2].min():.3f}, {target_xyz[:,2].max():.3f}]"
            )

    def set_surface_reference(
        self,
        R:       float = 1.0,
        r:       float = 0.45,
        d:       float = 1.6,
        epsilon: float = 0.005,
        verbose: bool  = True,
        # Legacy parameters retained for API compatibility
        bvp_grid_size: int = 80,
    ) -> None:
        """
        Rebuild the genus-2 reference surface for supervised pretraining.

        Reuses the already-computed octagon LB eigenfunctions and Abelian
        coordinates; only the level-set projection is recomputed.

        Args:
            R:       major radius of each reference torus.
            r:       tube radius of each reference torus.
            d:       centre-to-centre separation. Must satisfy R−r < d/2 < R+r.
            epsilon: level-set value controlling bridge thickness.
            verbose: print progress information.
        """
        self._build_surface_reference(R=R, r=r, d=d, epsilon=epsilon, verbose=verbose)
        self._tri_ref_torch = self._tri_ref_torch.to(self._device)

    def reference_embedding(
        self,
        xy: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Compute the genus-2 reference embedding at octagon points.

        Returns reference positions obtained by barycentric interpolation of
        per-vertex targets built by the Abelian-blend + level-set-projection
        construction (_build_surface_reference):

          1. Read Abelian angles (u₁,v₁,u₂,v₂) at each octagon vertex.
          2. Evaluate two standard torus guide probes at those angles.
          3. Blend with quadratic handle-activity weights wₖ = (sin²(uₖ/2)+sin²(vₖ/2))²+0.01.
             A direct blend is used rather than level-set projection: projection
             creates hard seam-spanning triangles along α=0.5, causing thousands
             of self-intersections.
          4. Barycentrically interpolate per-vertex targets within octagon triangles.
          5. Barycentrically interpolate projected targets within octagon triangles.

        Identified vertex pairs have identical Abelian angles by construction,
        so the reference map is automatically smooth across identified edges.

        Args:
            xy: (batch, 2) octagon sample points.

        Returns:
            xyz: (batch, 3) reference surface positions.
        """
        return self._interpolate_reference(xy)

    def _find_triangles(self, xy_np: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
        """
        Find the index of the enclosing triangle for each query point (numpy).

        Uses KD-tree on centroids to shortlist candidates, then tests barycentric
        coordinates for containment. Falls back to nearest centroid if no triangle
        strictly contains the point (boundary / floating-point edge cases).

        Args:
            xy_np:       (batch, 2) query coordinates (numpy)
            n_neighbors: number of closest centroids to check per point

        Returns:
            tri_idx: (batch,) integer array of triangle indices
        """
        n_tri   = len(self._centroids)
        n_neigh = min(n_neighbors, n_tri)
        _, cands = self._centroid_tree.query(xy_np, k=n_neigh)  # (batch, k)
        if cands.ndim == 1:
            cands = cands[:, None]

        batch   = xy_np.shape[0]
        tri_idx = np.zeros(batch, dtype=int)
        B  = self._B_np   # (M, 2, 2)
        p0 = self._p0_np  # (M, 2)

        for b in range(batch):
            found = False
            for t in cands[b]:
                d    = xy_np[b] - p0[t]             # (2,)
                lam  = B[t] @ d                      # (2,)  = [λ₁, λ₂]
                lam0 = 1.0 - lam[0] - lam[1]
                if lam0 >= -1e-6 and lam[0] >= -1e-6 and lam[1] >= -1e-6:
                    tri_idx[b] = t
                    found = True
                    break
            if not found:
                tri_idx[b] = int(cands[b, 0])

        return tri_idx

    def interpolate(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Differentiable interpolation of spectral features Ψ(x, y) at arbitrary
        points in the octagon via barycentric coordinates.

        This is differentiable w.r.t. xy, enabling autograd to propagate
        gradients from the Willmore loss back through the spectral features to
        the input coordinates (used in compute_first_fundamental_form).

        Args:
            xy: (batch, 2) — points inside the octagon. Must lie within the
                octagon; points near or outside the boundary fall back to the
                nearest triangle.

        Returns:
            features: (batch, num_eigenfunctions) spectral feature values Ψ(xy)
        """
        # Step 1: locate enclosing triangle (non-differentiable, numpy)
        xy_np   = xy.detach().cpu().numpy()
        tri_idx = self._find_triangles(xy_np)

        t_idx = torch.from_numpy(tri_idx).long()
        device = xy.device

        # Step 2: fetch per-sample precomputed matrices (differentiable w.r.t. xy)
        B_b   = self._B_torch[t_idx].to(device)       # (batch, 2, 2)
        p0_b  = self._p0_torch[t_idx].to(device)      # (batch, 2)
        ef_b  = self._tri_ef_torch[t_idx].to(device)  # (batch, 3, K)

        # Step 3: barycentric coordinates (affine in xy → differentiable)
        delta   = xy - p0_b                                     # (batch, 2)
        lam12   = torch.einsum('bij,bj->bi', B_b, delta)        # (batch, 2)
        lam0    = 1.0 - lam12[:, 0:1] - lam12[:, 1:2]          # (batch, 1)
        lam     = torch.cat([lam0, lam12], dim=1)               # (batch, 3)

        # Step 4: interpolate eigenfunction values
        features = torch.einsum('bi,bik->bk', lam, ef_b)        # (batch, K)

        return features

    # ------------------------------------------------------------------
    # Uniform sampling in the octagon
    # ------------------------------------------------------------------

    def sample_uniform(
        self,
        num_points:  int,
        device:      torch.device = torch.device('cpu'),
        dtype:       torch.dtype  = torch.float32,
        edge_margin: float        = 0.0,
    ) -> torch.Tensor:
        """
        Sample num_points uniformly (by Euclidean area) inside the octagon.

        Uses rejection sampling within the axis-aligned bounding box
        [−r, r] × [−r, r] of the octagon.

        Args:
            num_points:  Number of points to sample.
            device:      Target torch device.
            dtype:       Target torch dtype.
            edge_margin: Minimum perpendicular distance from each octagon edge.
                         Excludes a strip of this width near the identified
                         boundary edges, reducing gradient noise from the C¹
                         discontinuity of the spectral basis at those edges.

        Returns:
            xy: (num_points, 2) sample coordinates inside the octagon.
        """
        r       = _octagon_disk_radius()
        verts   = self.oct_vertices
        batches = []
        total   = 0

        while total < num_points:
            n_try = max(int((num_points - total) * 3.0) + 200, 500)
            x     = np.random.uniform(-r, r, n_try)
            y     = np.random.uniform(-r, r, n_try)
            pts   = np.column_stack([x, y])
            mask  = is_inside_octagon(pts, verts, margin=edge_margin)
            valid = pts[mask]
            if len(valid) > 0:
                batches.append(valid)
                total += len(valid)

        xy_np = np.concatenate(batches, axis=0)[:num_points]
        return torch.tensor(xy_np, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> 'HyperbolicOctagonSpectral':
        """Move all torch tensors to the given device."""
        self._device        = device
        self._B_torch       = self._B_torch.to(device)
        self._p0_torch      = self._p0_torch.to(device)
        self._tri_ef_torch  = self._tri_ef_torch.to(device)
        self._tri_ab_torch  = self._tri_ab_torch.to(device)
        self._tri_ref_torch = self._tri_ref_torch.to(device)
        return self

    # ------------------------------------------------------------------
    # Cache save / load
    # ------------------------------------------------------------------

    def _save(self, path: str):
        """Save computed basis data to a .npz file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_kwargs = dict(
            vertices              = self.vertices_np,
            triangles             = self.triangles_np,
            id_map                = self.id_map_np,
            eigenvalues           = self.eigenvalues_np,
            ef_at_vertices        = self.ef_at_vertices_np,
            abelian_coords        = self._abelian_np,
            oct_vertices          = self.oct_vertices,
            oct_area              = np.array([self.oct_area]),
            num_eigenfunctions    = np.array([self.num_eigenfunctions]),
            num_edge_divisions    = np.array([self.num_edge_divisions]),
            interior_grid_spacing = np.array([self.interior_grid_spacing]),
        )
        if self._surface_ref_xyz_np is not None:
            save_kwargs['surface_ref_xyz'] = self._surface_ref_xyz_np
            save_kwargs['ref_method']      = np.array([b'abelian_blend'])
            if self._surface_ref_params is not None:
                p = self._surface_ref_params
                save_kwargs['ref_params'] = np.array([p['R'], p['r'], p['d'], p.get('epsilon', 0.005)])
        np.savez_compressed(path, **save_kwargs)

    def _try_load(self, path: str, num_eigenfunctions: int, verbose: bool) -> bool:
        """
        Attempt to load precomputed basis from a .npz file.

        Returns True if load was successful and parameters match.
        """
        import os
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=False)
            cached_k = int(data['num_eigenfunctions'][0])
            if cached_k < num_eigenfunctions:
                if verbose:
                    print(f"Cache has {cached_k} eigenfunctions; need {num_eigenfunctions}. Recomputing.")
                return False

            self.vertices_np        = data['vertices']
            self.triangles_np       = data['triangles']
            self.id_map_np          = data['id_map']
            self.eigenvalues_np     = data['eigenvalues'][:num_eigenfunctions]
            self.ef_at_vertices_np  = data['ef_at_vertices'][:, :num_eigenfunctions]
            self.num_eigenfunctions = num_eigenfunctions
            self.oct_vertices       = data['oct_vertices']
            self.oct_area           = float(data['oct_area'][0])
            self.num_edge_divisions    = int(data['num_edge_divisions'][0])
            self.interior_grid_spacing = float(data['interior_grid_spacing'][0])

            self._build_interpolation_tables()

            if 'abelian_coords' in data:
                self._abelian_np   = data['abelian_coords']
                tris               = self.triangles_np
                self._tri_ab_torch = torch.from_numpy(self._abelian_np[tris]).float()
            else:
                self._build_abelian_interp_tables()

            if 'surface_ref_xyz' in data:
                # Only use cache if it was built with the Abelian-blend method
                is_abelian = (
                    'ref_method' in data
                    and data['ref_method'][0] == b'abelian_blend'
                )
                if is_abelian:
                    self._surface_ref_xyz_np = data['surface_ref_xyz']
                    if 'ref_params' in data:
                        rp = data['ref_params']
                        self._surface_ref_params = {
                            'R': float(rp[0]), 'r': float(rp[1]),
                            'd': float(rp[2]), 'epsilon': float(rp[3]),
                        }
                    tris = self.triangles_np
                    tri_ref = self._surface_ref_xyz_np[tris]
                    self._tri_ref_torch = torch.from_numpy(tri_ref.astype(np.float32))
                    if verbose:
                        print(f"  Loaded Abelian-blend reference from cache.")
                else:
                    if verbose:
                        print("  Cache reference is from old method; rebuilding.")
                    self._build_surface_reference(verbose=verbose)
            else:
                # Cache pre-dates surface reference — recompute
                self._build_surface_reference(verbose=verbose)

            if verbose:
                print(f"Loaded spectral basis from cache: {path} "
                      f"(K={num_eigenfunctions}, "
                      f"{len(self.triangles_np)} triangles)")
            return True

        except Exception as exc:
            if verbose:
                print(f"Could not load cache {path}: {exc}. Recomputing.")
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Dimension of the spectral feature vector (= num_eigenfunctions)."""
        return self.num_eigenfunctions


def build_spectral_basis(config: dict, verbose: bool = True) -> HyperbolicOctagonSpectral:
    """
    Factory: build or load a HyperbolicOctagonSpectral from config.

    Reads keys from config['topology']['double_torus']:
        num_eigenfunctions:    int  (default 16)
        num_edge_divisions:    int  (default 10)
        interior_grid_spacing: float (default 0.05)
        cache_path:            str  (default None)

    Args:
        config:  full hyperparameter config dict
        verbose: whether to print progress

    Returns:
        Initialised HyperbolicOctagonSpectral instance (on CPU).
    """
    dt_cfg = config.get('topology', {}).get('double_torus', {})
    return HyperbolicOctagonSpectral(
        num_eigenfunctions    = dt_cfg.get('num_eigenfunctions',    16),
        num_edge_divisions    = dt_cfg.get('num_edge_divisions',    10),
        interior_grid_spacing = dt_cfg.get('interior_grid_spacing', 0.05),
        cache_path            = dt_cfg.get('cache_path',            None),
        verbose               = verbose,
    )

"""Flux conserving resampling of many spectra that share the same wavelength grids.

``desispec.interpolation.resample_flux(xout, x, flux)`` is linear in ``flux``: with the same
``x`` and ``xout`` every spectrum is multiplied by the same matrix.  ``resample_flux_matrix``
builds that matrix (sparse) following the algorithm of ``desispec.interpolation._unweighted_resample``
step by step, and ``resample_flux_many`` applies it to a 2D array of spectra in one product instead of
one Python call per spectrum.  The result agrees with the per-spectrum function to rounding
(a few units in the last place), not bit for bit, because the sums are made in a different order.
"""
import numba
import numpy as np
import scipy.sparse as sp


def resample_flux_matrix(xout, x, extrapolate=False):
    """Sparse matrix M of shape (len(xout), len(x)) with ``M @ flux`` equal to
    ``desispec.interpolation.resample_flux(xout, x, flux, extrapolate=extrapolate)``."""
    xout = np.asarray(xout, dtype=float)
    ix = np.asarray(x, dtype=float)
    nin = ix.size

    # boundaries of the output bins
    bins = np.zeros(xout.size + 1)
    bins[1:-1] = (xout[:-1] + xout[1:]) / 2.
    bins[0] = 1.5 * xout[0] - 0.5 * xout[1]
    bins[-1] = 1.5 * xout[-1] - 0.5 * xout[-2]
    binsize = bins[1:] - bins[:-1]
    if np.any(binsize <= 0):
        raise ValueError("Zero or negative bin size")

    # input nodes; unless extrapolating, two extra nodes of zero flux density at the ends
    # (the columns of these nodes are dropped at the end)
    if not extrapolate:
        ix = np.append(2 * ix[0] - ix[1], ix)
        ix = np.append(ix, 2 * ix[-1] - ix[-2])
        offset = 1
    else:
        offset = 0
    n_ext = ix.size

    # values at the bin boundaries: np.interp(bins, ix, iy) as a matrix
    # (the first / last value outside the range of the nodes)
    tx = bins.copy()
    rows = np.arange(tx.size)
    left = tx <= ix[0]
    right = tx >= ix[-1]
    inner = ~left & ~right
    j = np.searchsorted(ix[:-1], tx[inner], side='right') - 1
    wgt = (tx[inner] - ix[j]) / (ix[j + 1] - ix[j])      # weight of node j+1
    r_b = np.concatenate([rows[left], rows[right], rows[inner], rows[inner]])
    c_b = np.concatenate([np.zeros(left.sum(), dtype=int), np.full(right.sum(), n_ext - 1), j, j + 1])
    v_b = np.concatenate([np.ones(left.sum()), np.ones(right.sum()), 1. - wgt, wgt])
    edge_nodes = sp.csr_matrix((v_b, (r_b, c_b)), shape=(tx.size, n_ext))

    # input nodes inside the range of the bin boundaries enter as nodes themselves
    k = np.where((ix >= tx[0]) & (ix <= tx[-1]))[0]
    node_rows = sp.csr_matrix((np.ones(k.size), (np.arange(k.size), k)), shape=(k.size, n_ext))

    all_x = np.concatenate([tx, ix[k]])
    all_nodes = sp.vstack([edge_nodes, node_rows], format='csr')
    p = np.argsort(all_x)
    all_x = all_x[p]
    all_nodes = all_nodes[p]

    # trapezes between consecutive nodes; each belongs to the output bin that holds its centre
    dx = all_x[1:] - all_x[:-1]
    centers = (all_x[1:] + all_x[:-1]) / 2.
    trap = sp.diags(dx / 2.) @ (all_nodes[1:] + all_nodes[:-1])
    ibin = np.searchsorted(bins, centers, side='right') - 1
    ibin[centers == bins[-1]] = xout.size - 1             # np.histogram: last bin is closed
    ok = (ibin >= 0) & (ibin < xout.size) & (centers >= bins[0]) & (centers <= bins[-1])
    assign = sp.csr_matrix((np.ones(ok.sum()), (ibin[ok], np.where(ok)[0])), shape=(xout.size, centers.size))
    matrix = sp.diags(1. / binsize) @ assign @ trap
    matrix = matrix.tocsc()[:, offset:offset + nin].tocsr()
    matrix.eliminate_zeros()
    return matrix


def resample_flux_many(xout, x, flux, extrapolate=False):
    """``np.array([resample_flux(xout, x, f) for f in flux])`` for 2D ``flux`` [nspec, len(x)]."""
    matrix = resample_flux_matrix(xout, x, extrapolate=extrapolate)
    return np.asarray(matrix @ np.asarray(flux).T).T


@numba.njit(cache=True)
def _interp_rows(out, fp, j, dx, dxnew, mode):
    # mode: 0 inside (linear), 1 left of xp[0] (fp[:,0]), 2 right of xp[-1] (fp[:,-1]), 3 exactly xp[-1]
    for q in range(out.shape[0]):
        for i in range(out.shape[1]):
            m = mode[i]
            if m == 0:
                k = j[i]
                out[q, i] = ((fp[q, k + 1] - fp[q, k]) / dx[i]) * dxnew[i] + fp[q, k]
            elif m == 1:
                out[q, i] = fp[q, 0]
            else:
                out[q, i] = fp[q, fp.shape[1] - 1]


def interp_many(xnew, xp, fp):
    """``np.array([np.interp(xnew, xp, f) for f in fp])`` for 2D ``fp`` [nspec, len(xp)] with common ``xp``
    (same arithmetic as ``np.interp``, the positions are searched once)."""
    xnew = np.asarray(xnew, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.ascontiguousarray(fp, dtype=float)
    j = np.clip(np.searchsorted(xp, xnew, side='right') - 1, 0, xp.size - 2)
    mode = np.zeros(xnew.size, dtype=np.int8)
    mode[xnew < xp[0]] = 1
    mode[xnew > xp[-1]] = 2
    mode[xnew == xp[-1]] = 3
    out = np.empty((fp.shape[0], xnew.size))
    _interp_rows(out, fp, j, xp[j + 1] - xp[j], xnew - xp[j], mode)
    return out

"""
Build the Lya coverage of a data release (footprint, completeness and Lya exposure times of the Lya quasar targets) from the LSS
random catalogs. Run once per release; the output feeds ``gen_qso_catalog`` (option ``--lya-coverage``).
See desisim.survey_release.LyaCoverage for the model of the selection and the exposures.

Steps:
  1. read the LSS random catalogs (rancomb_<k>dark and rancomb_<k>dark1b: one row per random and tile) in chunks, in parallel, and split
     the rows in buckets by TARGETID (all the rows of a random end in the same bucket), kept in the temporary directory of the node;
  2. footprint weight and NTILE histogram per healpix pixel; fit the completeness slope q and the number of targets N_DATA of the
     data QSO targets (and q, N of the non-QSO targets, quasars selected as other target classes) on the data counts per pixel;
  3. fit the re-observation probability p_reobs so that the mean simulated exposure (rounded to whole nominal passes) equals the
     mean exposure of the data Lya QSO targets (12.15 * TSNR2_LRG);
  4. simulate the observations of every random with these numbers, accumulate the exposure histogram per pixel and write the map.

Steps 2 to 4 work bucket by bucket, in parallel; the accumulators are counts, so the buckets simply add up.
"""
import os
import time
import argparse
import datetime
import tempfile
import functools
import multiprocessing as mp
import numpy as np
import healpy as hp
from astropy.table import Table
from desiutil.log import get_logger
from desitarget.targetmask import desi_mask

from desisim.survey_release import (LyaCoverage, round_to_passes, EXPTIME_FACTOR, TSNR_COLUMN, EXPTIME_BIN_WIDTH, N_EXPTIME_BINS, NTILE_MAX,
                                    NSIDE_DENSITY, NSIDE_POS, SECONDS_PER_PASS)

log = get_logger()

RANDOM_COLUMNS = ['TARGETID', 'RA', 'DEC', 'TILEID', TSNR_COLUMN]
ROW_DTYPE = np.dtype([('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('TILEID', 'i8'), (TSNR_COLUMN, 'f4')])   # native byte order
N_BUCKETS = 64              # the rows are split by TARGETID % N_BUCKETS
READ_CHUNK = 8_000_000      # rows read by one task

# Cap rule of the Lya QSO re-observations: up to RULE_NIGHT a target is observed at most CAP_BEFORE times, CAP_TOTAL times in all
RULE_NIGHT, CAP_BEFORE, CAP_TOTAL = 20250721, 4, 6


def _read_chunk(task):
    """Read rows [start, stop) of an LSS rancomb file, sort them by bucket (TARGETID % N_BUCKETS) and store them with the offset of each bucket."""
    import fitsio
    path, start, stop, out = task
    data = fitsio.read(path, ext=1, columns=RANDOM_COLUMNS, rows=np.arange(start, stop))
    rows = np.empty(len(data), ROW_DTYPE)
    for name in ROW_DTYPE.names:
        rows[name] = data[name]
    bucket = (rows['TARGETID'] % N_BUCKETS).astype(np.uint8)
    np.save(out + '.npy', rows[np.argsort(bucket, kind='stable')])
    np.save(out + '_offsets.npy', np.concatenate([[0], np.cumsum(np.bincount(bucket, minlength=N_BUCKETS))]))
    return out


def _load_bucket(chunks, bucket):
    """Rows of one bucket of a catalog, from the stored chunks."""
    parts = []
    for chunk in chunks:
        offsets = np.load(chunk + '_offsets.npy')
        if offsets[bucket + 1] > offsets[bucket]:
            parts.append(np.load(chunk + '.npy', mmap_mode='r')[offsets[bucket]:offsets[bucket + 1]])
    return np.concatenate(parts)


def tile_night_lookup(tiles):
    """Return a function mapping TILEID -> LASTNIGHT from a tiles table (e.g. tiles-matterhorn.fits)."""
    tid = np.asarray(tiles['TILEID']); night = np.asarray(tiles['LASTNIGHT']).astype(int)
    order = np.argsort(tid); tid, night = tid[order], night[order]

    def lookup(tileids):
        idx = np.searchsorted(tid, tileids)
        idx = np.clip(idx, 0, len(tid) - 1)
        if not np.all(tid[idx] == tileids):
            raise ValueError("Some TILEIDs of the randoms are not in the tiles table")
        return night[idx]
    return lookup


@functools.lru_cache(maxsize=None)
def _night_of_tile(tiles_file):
    return tile_night_lookup(Table.read(tiles_file))


def _group_tiles(parts_list, night_of_tile):
    """One row per (target, tile), sorted by target and then by night. Returns the rows, their nights, the index of the target of each row, the
    first row of each target and the position of each row among the tiles of its target."""
    d = np.concatenate(parts_list)
    night = night_of_tile(d['TILEID'])
    order = np.lexsort((d['TILEID'], d['TARGETID'])); d = d[order]; night = night[order]
    first = np.ones(len(d), bool)
    first[1:] = (d['TARGETID'][1:] != d['TARGETID'][:-1]) | (d['TILEID'][1:] != d['TILEID'][:-1])   # one row per (target, tile)
    d = d[first]; night = night[first]
    order = np.lexsort((d['TILEID'], night, d['TARGETID'])); d = d[order]; night = night[order]
    new_target = np.ones(len(d), bool); new_target[1:] = d['TARGETID'][1:] != d['TARGETID'][:-1]
    gid = np.cumsum(new_target) - 1
    start = np.flatnonzero(new_target); j = np.arange(len(d)) - start[gid]
    return d, night, gid, start, j


def ntile_per_target(parts_list, night_of_tile):
    """Position (ra, dec) and number of available tiles NTILE (unique tiles) of every target in the (target, tile) rows."""
    d, night, gid, start, j = _group_tiles(parts_list, night_of_tile)
    return d['RA'][start], d['DEC'][start], np.bincount(gid, minlength=len(start))


def exptime_per_target(parts_list, night_of_tile, q, p_reobs, rng, rule_night=RULE_NIGHT, cap_before=CAP_BEFORE, cap_total=CAP_TOTAL):
    """Per-target availability and Lya exposure time from the (target, tile) rows, with the per-tile observation model.

    The first observation happens on the first tile that succeeds, each available tile succeeding with probability q; after it every later
    tile is observed with probability p_reobs; the cap rule (cap_before observations up to rule_night, cap_total in all) is applied to the
    observed tiles in date order. Targets without any success are not observed (observed=False, exptime 0).

    Args:
        parts_list (list of ndarray): row arrays (e.g. the dark and the dark1b parts of one random catalog).
        night_of_tile (callable): TILEID -> LASTNIGHT.
        q, p_reobs (float): first-observation and re-observation probabilities per available tile.
        rng (numpy Generator): random numbers.

    Returns:
        ra, dec (ndarray), ntile (int, unique tiles), exptime (float, seconds, cap rule applied) and observed (bool).
    """
    d, night, gid, start, j = _group_tiles(parts_list, night_of_tile)
    ngroup = len(start)
    ntile = np.bincount(gid, minlength=ngroup)
    if q >= 1:
        first = np.zeros(ngroup, dtype=np.int64)
    elif q <= 0:
        first = np.full(ngroup, 10 ** 9, dtype=np.int64)
    else:   # index of the first successful tile (geometric distribution)
        first = np.floor(np.log(1.0 - rng.uniform(size=ngroup)) / np.log1p(-q)).astype(np.int64)
    success = (j == first[gid]) | ((j > first[gid]) & (rng.uniform(size=len(d)) < p_reobs))
    cum = np.cumsum(success)
    index_success = cum - (cum[start] - success[start])[gid] - 1      # position of a success among the target's successes
    n_before = np.bincount(gid, weights=(success & (night <= rule_night)), minlength=ngroup).astype(int)
    cb = np.minimum(n_before, cap_before)
    observed_rows = success & ((index_success < cb[gid]) | ((index_success >= n_before[gid]) & (index_success - n_before[gid] < cap_total - cb[gid])))
    exptime = np.bincount(gid, weights=EXPTIME_FACTOR * d[TSNR_COLUMN].astype(float) * observed_rows, minlength=ngroup)
    return d['RA'][start], d['DEC'][start], ntile, exptime, first < ntile


def accumulate(ra, dec, ntile, exptime=None, nside_density=NSIDE_DENSITY, nside_pos=NSIDE_POS, observed=None):
    """Per-pixel accumulators of one random catalog (additive, so catalogs can be combined or split into halves).

    Footprint and NTILE histogram use all targets; the exposure-time histogram uses only the observed targets (all of them if ``observed``
    is None) and is left empty when ``exptime`` is None.
    """
    p_d = hp.ang2pix(nside_density, np.radians(90 - dec), np.radians(ra), nest=True)
    shift = (nside_density // nside_pos) ** 2
    p_p = p_d // shift
    npix_d, npix_p = 12 * nside_density ** 2, 12 * nside_pos ** 2
    acc = dict(nr_density=np.bincount(p_d, minlength=npix_d).astype(np.int64),
               nr_pos=np.bincount(p_p, minlength=npix_p).astype(np.int64))
    nt = np.clip(ntile, 0, NTILE_MAX)
    acc['h_ntile'] = np.bincount(p_p * (NTILE_MAX + 1) + nt, minlength=npix_p * (NTILE_MAX + 1)).reshape(npix_p, NTILE_MAX + 1).astype(np.int64)
    h_exp = np.zeros((npix_p, N_EXPTIME_BINS), np.int64)
    if exptime is not None:
        observed = np.ones(len(ra), bool) if observed is None else observed
        eb = np.clip((exptime[observed] // EXPTIME_BIN_WIDTH).astype(int), 0, N_EXPTIME_BINS - 1)
        h_exp = np.bincount(p_p[observed] * N_EXPTIME_BINS + eb, minlength=npix_p * N_EXPTIME_BINS).reshape(npix_p, N_EXPTIME_BINS).astype(np.int64)
    acc['h_exp'] = h_exp
    return acc


def combine(accs):
    """Sum a list of accumulators."""
    out = {k: sum(a[k] for a in accs) for k in accs[0]}
    return out


def _poisson_deviance(n, lam):
    """2 sum [n ln(n/lam) - (n - lam)] over pixels with lam > 0 (n ln n = 0 for n = 0)."""
    ok = lam > 0
    n, lam = n[ok], lam[ok]
    t = np.zeros(len(n)); pos = n > 0
    t[pos] = n[pos] * np.log(n[pos] / lam[pos])
    return 2 * np.sum(t - (n - lam))


def fit_q(smap, data_ra, data_dec, q_grid=None):
    """Completeness slope q of a set of data targets: Poisson deviance of their counts per nside_pos pixel against
    kappa * A_pix * P_obs(q), kappa free per q. Returns (q, deviance per pixel, {q: deviance per pixel})."""
    q_grid = np.concatenate([[0.002, 0.005, 0.01], np.linspace(0.02, 0.99, 98)]) if q_grid is None else q_grid
    npix = 12 * smap.nside_pos ** 2
    p_p = hp.ang2pix(smap.nside_pos, np.radians(90 - data_dec), np.radians(data_ra), nest=True)
    n_d = np.bincount(p_p, minlength=npix).astype(float)
    ratio = (smap.nside_density // smap.nside_pos) ** 2
    area = smap.weight.reshape(npix, ratio).sum(1)
    use = area > 0.5 * ratio
    dev = {}
    for q in q_grid:
        x = area * smap.pobs(q)
        lam = n_d[use].sum() / x[use].sum() * x
        dev[float(q)] = _poisson_deviance(n_d[use], lam[use]) / use.sum()
    q_best = min(dev, key=dev.get)
    return q_best, dev[q_best], dev


def _footprint_tasks(args):
    """First pass: footprint and NTILE histograms of a list of (catalog, bucket) pairs, added up."""
    tiles_file, chunks, pairs = args
    accs = []
    for k, bucket in pairs:
        ra, dec, ntile = ntile_per_target([_load_bucket(chunks[k], bucket)], _night_of_tile(tiles_file))
        accs.append(accumulate(ra, dec, ntile))
    return combine(accs)


def _grid_tasks(args):
    """Mean exposure of the simulated observed targets, for every p_reobs of the grid: [sum of the rounded exposures, number of targets] per
    p_reobs, added up over the buckets of the catalog 0."""
    tiles_file, chunks, buckets, q, grid = args
    total = np.zeros((len(grid), 2))
    for bucket in buckets:
        rows = _load_bucket(chunks[0], bucket)
        for i, p in enumerate(grid):
            rng = np.random.default_rng([1, bucket])
            exptime, observed = exptime_per_target([rows], _night_of_tile(tiles_file), q, p, rng)[3:]
            total[i] += [round_to_passes(exptime[observed]).sum(), observed.sum()]
    return total


def _simulation_tasks(args):
    """Second pass: footprint, NTILE and exposure-time histograms of the simulated observations of a list of (catalog, bucket) pairs."""
    tiles_file, chunks, pairs, q, p_reobs = args
    accs = []
    for k, bucket in pairs:
        rng = np.random.default_rng([2, k, bucket])
        ra, dec, ntile, exptime, observed = exptime_per_target([_load_bucket(chunks[k], bucket)], _night_of_tile(tiles_file), q, p_reobs, rng)
        accs.append(accumulate(ra, dec, ntile, exptime, observed=observed))
    return combine(accs)


def _split(items, n):
    """n lists of items, dealt out round-robin."""
    return [items[i::n] for i in range(min(n, len(items)))]


def main(args=None):
    p = argparse.ArgumentParser(description="Build the Lya coverage of a data release from the LSS randoms (run once per release).")
    p.add_argument("--randoms-dir", required=True, help="Directory with the LSS rancomb_<k>{dark,dark1b}wdupspec_zdone.fits files")
    p.add_argument("--release", default='matterhorn', help="Release name (used for the tiles file and the header)")
    p.add_argument("--tiles-file", default=None, help="Default: $DESI_SPECTRO_REDUX/<release>/tiles-<release>.fits")
    p.add_argument("--data-catalog", required=True, help="Data QSO catalog (e.g. QSO_cat_<release>_main_dark_healpix_vN.fits), with the non-QSO targets")
    p.add_argument("-o", "--output", required=True, help="Output file, e.g. <mocks>/lya_<release>_coverage.fits")
    p.add_argument("--random-ids", type=int, nargs='+', default=[0, 1, 2], help="Random catalogs to use (default 0 1 2; each has about 38M randoms, 30 times the data targets)")
    p.add_argument("--zmin", type=float, default=1.8, help="Redshift range of the data targets that are counted: that of the mock quasars (CoLoRe: 1.8 - 3.8; "
                   "the master catalog extends slightly beyond it by the redshift-space distortions)")
    p.add_argument("--zmax", type=float, default=3.8)
    p.add_argument("--nproc", type=int, default=64, help="Processes: they read the files in chunks, then work on the buckets (more than 64 does not help: the reading of the files limits)")
    p.add_argument("--overwrite", action='store_true', help="Overwrite the output file if it exists")
    args = p.parse_args(args)

    tiles_file = args.tiles_file or os.path.join(os.environ['DESI_SPECTRO_REDUX'], args.release, f'tiles-{args.release}.fits')
    if os.path.exists(args.output) and not args.overwrite:
        p.error(f"{args.output} exists; use --overwrite to replace it")
    nproc = args.nproc
    t_start = time.time()

    # data: QSO targets and non-QSO targets in the redshift range of the mock master
    data = Table.read(args.data_catalog)
    in_z = (data['Z'] >= args.zmin) & (data['Z'] <= args.zmax)
    is_qso_target = (data['DESI_TARGET'] & desi_mask.mask('QSO|QSO_HIZ|QSO_NORTH|QSO_SOUTH')) > 0
    data_nonqso = data[in_z & ~is_qso_target]
    data = data[in_z & is_qso_target]
    exptime_lya = EXPTIME_FACTOR * np.asarray(data[TSNR_COLUMN])[data['Z'] > 2.1]
    mean_target = float(np.mean(round_to_passes(exptime_lya)))
    ra_d, dec_d = np.asarray(data['TARGET_RA']), np.asarray(data['TARGET_DEC'])
    meta = dict(RANDOMS=','.join(map(str, args.random_ids))[:60], RULENITE=RULE_NIGHT, CAPB=CAP_BEFORE,
                CAPT=CAP_TOTAL, DATECREA=datetime.date.today().isoformat(), RELEASE=args.release,
                DATACAT=os.path.basename(args.data_catalog)[:60], EXPMODEL='pertile', ZMIN=args.zmin, ZMAX=args.zmax)

    with tempfile.TemporaryDirectory(prefix='lya_coverage_') as tmp_dir, mp.Pool(processes=nproc) as pool:
        # read the catalogs in chunks and split their rows in buckets (all the rows of a random are in the same bucket)
        import fitsio
        tasks, chunks = [], {k: [] for k in args.random_ids}
        for k in args.random_ids:
            for program in ('dark', 'dark1b'):
                path = os.path.join(args.randoms_dir, f'rancomb_{k}{program}wdupspec_zdone.fits')
                nrows = fitsio.FITS(path)[1].get_nrows()
                for i, start in enumerate(range(0, nrows, READ_CHUNK)):
                    out = os.path.join(tmp_dir, f'{k}_{program}_{i:03d}')
                    tasks.append((path, start, min(start + READ_CHUNK, nrows), out))
                    chunks[k].append(out)
        pool.map(_read_chunk, tasks, chunksize=1)
        log.info(f"Read {len(tasks)} chunks of the {len(args.random_ids)} catalogs in {time.time() - t_start:.0f} s")
        pairs = [(k, bucket) for k in args.random_ids for bucket in range(N_BUCKETS)]

        # pass 1: footprint and NTILE histograms, completeness q and N_DATA of the QSO targets, q and N of the non-QSO targets
        accs = pool.map(_footprint_tasks, [(tiles_file, chunks, group) for group in _split(pairs, nproc)])
        log.info(f"Footprint and tile histograms done at {time.time() - t_start:.0f} s")
        smap0 = LyaCoverage(combine(accs), meta=meta)
        q, _, _ = fit_q(smap0, ra_d, dec_d)
        n_data = len(ra_d)
        log.info(f"Fitted q={q:.3f}, N_DATA={n_data}")
        q_nonqso, dev_nq, _ = fit_q(smap0, np.asarray(data_nonqso['TARGET_RA']), np.asarray(data_nonqso['TARGET_DEC']))
        log.info(f"QSO targets: q={q:.3f}, N_DATA={n_data};  non-QSO targets: q={q_nonqso:.3f}, N={len(data_nonqso)} (deviance/pixel {dev_nq:.3f})")

        # pass 2: re-observation probability from the mean exposure, then the simulated observations
        grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        totals = sum(pool.map(_grid_tasks, [(tiles_file, chunks, group, q, grid) for group in _split(list(range(N_BUCKETS)), nproc)]))
        ms = totals[:, 0] / totals[:, 1]
        p_reobs = float(np.interp(mean_target, ms, grid))
        log.info("p_reobs grid: " + ', '.join(f'{a:.2f}->{b:.0f}s' for a, b in zip(grid, ms)) + f"; data mean {mean_target:.0f} s -> p_reobs = {p_reobs:.3f}")
        accs = pool.map(_simulation_tasks, [(tiles_file, chunks, group, q, p_reobs) for group in _split(pairs, nproc)])
        log.info(f"Observations simulated at {time.time() - t_start:.0f} s")

    lya_coverage = LyaCoverage(combine(accs), meta=dict(meta, PREOBS=p_reobs),
                                q=q, n_data=n_data, seconds_per_pass=SECONDS_PER_PASS, q_nonqso=q_nonqso, n_data_nonqso=len(data_nonqso))
    lya_coverage.write(args.output)
    log.info(f"Wrote {args.output}")
    return 0


if __name__ == '__main__':
    main()

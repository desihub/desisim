# quickquasars: faster and with less memory (branch `gpu-port`)

The branch is called `gpu-port` for historical reasons; nothing here uses a GPU. It speeds up `quickquasars`
and `quickspectra.sim_spectra` on CPU nodes and lowers their memory. All the new behaviour that changes how a run
is organised is opt-in; the spectra do not change (details below).

## What is new

### Options of `quickquasars`

| Option | What it does |
|---|---|
| `--fiber-chunk N` | Simulates the quasars of a healpix in chunks of N fibres. The specsim simulator holds about 20 float64 arrays of (fibres x 31,000 wavelengths), so its memory is proportional to the number of fibres. With a chunk of 200 the peak of a worker for a healpix of 1500 quasars falls from about 11 GiB to 4-5 GiB. The output is identical to the output without the option. Default: off. |
| `--mem-per-worker GIB` | Lowers `--nproc` to what fits in the memory of the job (`SLURM_MEM_PER_NODE`, else the free memory of the node), at GIB per worker. Default: off. |

### Behaviour changes that are not options

- **A worker that dies ends the run at once.** The pool is a `ProcessPoolExecutor` (one task per healpix) instead
  of `multiprocessing.Pool.map`, which never returns when a worker is killed (for example by the out-of-memory
  killer) and left jobs idle until their walltime. The run now stops within a second and logs the healpixes that
  did not finish. `bin/quickquasars` has the `if __name__ == "__main__":` guard that Python 3.14 (start method
  `forkserver`) needs.
- **Memory of a worker is returned after each healpix** (`gc.collect()` and `malloc_trim(0)`). glibc kept the memory
  freed by one healpix in the heap of the worker, so a worker that ran many healpixes grew to 12-20 GB (a pool of 2
  workers over 16 healpixes, also with the unmodified code).
- **The R and Z wavelength arrays of the output spectra keep full float64 precision.** `Spectra.update()` rounded the
  appended R and Z grids to float32 (at most 4e-4 Angstrom). Nothing else in the files changes.

### Speed-ups with unchanged output

- `quickspectra.sim_spectra` builds one multi-band `Spectra` instead of merging cameras with `Spectra.update()`,
  whose EXPID/FIBER matching is a quadratic pure-Python loop (24% of the serial time).
- The metals absorption and the resampling work on all the quasars of a healpix at once (`desisim/resample.py`;
  the metals step no longer builds 16 arrays of [quasars, pixels] at once, 10.6 to 3.1 GiB).
- Compiled versions of `render_gaussians`, the broken power law and the flux normalisation in the simqso templates
  (`desisim/simqso_fast.py`, `templates.py`).
- The camera output resolution matrix is computed once per process and reused when the inputs are identical.

## Why `--fiber-chunk` gives the same noise

With `use_poisson=False` (as in `quickquasars`) specsim draws the noise of each camera with one
`normal(scale=...)` call on the array (wavelengths, fibres), then the sky error of the fibres from the same random
state. Standard normals of the same shapes drawn in the same order, multiplied by the scale of each chunk, give the
same numbers, so they are drawn for all the fibres first. Two more things depend on the chunking and are handled
(`simexp.simulate_spectra` got the arguments `acceptance_only` and `fiber_acceptance_fraction`):
specsim draws a random focal-plane offset for each fibre from a generator of the simulator, which changes the loss
of light in the fibre, so the fibre acceptance of all the fibres is computed once and each chunk gets its slice;
and the exposure time is a per-quasar array in the observing conditions, which is sliced per chunk. The chunking
applies only with `use_poisson=False`, without `fullsim` and for the DESI (not eBOSS) configuration; otherwise the
option is ignored.

## What was checked

- `fiber-chunk`: outputs identical (spectra, truth, zbest, all columns except the date stamp
  `FIBERMAP.NIGHT`) for chunks of 64, 173, 200 and 500 on one healpix of 1506 quasars and of 100 on another, and
  for all 882 files of a share of mock-15 (302k quasars) between chunked and unchunked runs.
- Earlier changes: spectra, truth and zbest agree with the original code to 1e-6 relative or better (the metals and
  the last step are identical bit for bit), and the pool change gives the same files apart from the date stamp
  `FIBERMAP.NIGHT`.
- The existing tests of the package pass (66 passed with the integration script excluded, DESI 26.9).
- A whole-mock test on one CPU node (1151 healpixes, 1.21M quasars): the stock script and this branch produce
  spectra that agree to 1.5e-4 of the noise in the flux and 3e-6 relative in `IVAR`; the difference includes the
  Python and library versions of the two environments (DESI 26.3 and 26.9), which I did not separate.

## Timings and memory

One share of mock-15 (302k quasars, `--nproc 48`, production arguments, a CPU node with 128 cores): 5 min 10 s
without `--fiber-chunk`, 3 min 56 s with `--fiber-chunk 200`; peak node memory 367 GB against 209 GB; largest
worker 11.1 against 5.3 GB. The whole mock (1.21M quasars) on one CPU node: 3934 s with the stock lyatools script
(48 workers with `srun -c 48`), 752 s with the lyatools changes (see the lyatools branch) and this branch; most of
that gain comes from giving the job all the CPUs of the node (on Perlmutter `-c` counts logical CPUs, so 48
workers had 24 physical cores). Single runs.

## Commits

`9ba3fe3` multi-band Spectra; `66e9fc4` ProcessPoolExecutor; `2149636` metals and resampling for all quasars at
once; `7584ee4`, `33fb6ab` compiled simqso pieces; `b0d8862` resolution matrix cache; `72da6bf`
`--mem-per-worker`; `06f0d6d` `--fiber-chunk`; `3a12df8` memory returned after each healpix.

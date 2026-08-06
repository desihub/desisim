# Dev environment setup — desisim-fsps fork

Reproducible steps for the sandbox environment used to develop this fork.
Written after first bootstrapping the environment on 2026-08-06.

## 1. Python environment

```
python3 -m venv venv
venv/bin/python3 -m pip install --upgrade pip "setuptools<66" wheel
```

`setuptools<66` is required: `desiutil`'s legacy `setup.py` still imports
`pkg_resources._namespace_packages`, which modern setuptools removed.

## 2. python-fsps / FSPS Fortran backend

```
venv/bin/python3 -m pip install fsps          # ships a prebuilt compiled extension (PyPI wheel) - no gfortran needed
git clone --depth=1 https://github.com/cconroy20/fsps.git fsps_data   # ~3.4 GB: isochrones + spectral libraries
export SPS_HOME=/absolute/path/to/fsps_data
```

Note: the PyPI `fsps` wheel (0.5.0 as of this writing) bundles the *compiled*
Fortran extension, so no local Fortran compiler is required. `SPS_HOME` still
needs to point at the `cconroy20/fsps` data tree (isochrones, spectral
libraries) at runtime — this is ~3.4 GB and was NOT compileable in the
original sandbox used for this session (no gfortran, no root/apt access,
no conda). If a future environment also lacks a Fortran compiler, the
PyPI-wheel route above is the workaround; if a compiler is available, follow
python-fsps's standard "build from source" instructions instead for closer
version parity between the extension and the data (this session did not
verify exact version-matching beyond the fact that `StellarPopulation()` and
`get_spectrum()` ran and returned a physically sane, positive-flux array).

## 3. desisim's declared dependencies (`requirements.txt`)

```
venv/bin/python3 -m pip install --no-build-isolation "git+https://github.com/desihub/desiutil.git@main#egg=desiutil"
venv/bin/python3 -m pip install --no-build-isolation -r requirements.txt
venv/bin/python3 -m pip install "scipy<1.14"
```

Two deviations from what's literally written in `requirements.txt`:

- **desiutil**: the file's comment pins `@3.1.0`, but the current
  `desispec@main` (also pulled in by requirements.txt) imports
  `desiutil.healpix`, which does not exist in 3.1.0. Installed `@main`
  instead of `@3.1.0`.
- **scipy**: requirements.txt installs whatever `astropy`/`desispec`/etc.
  pull in, which resolved to scipy 1.15.3. `scipy.integrate.simps` (used by
  simqso 1.2.4, see below) was removed in scipy 1.14. Pinned `scipy<1.14`
  after the fact.

## 4. simqso — NOT installed, by design

desisim's `requirements.txt` itself comments out the simqso line. This
session confirmed why beyond "the install script needs numpy": simqso
v1.2.4 (`imcgreer/simqso`, the version desisim's own requirements.txt is
pinned to) imports `astropy.analytic_functions.blackbody_lambda`, an API
removed from astropy years ago. Installing a compatible ancient astropy
would break `desispec`/`desitarget`/`desimodel@main`, which need modern
astropy. There is no version of astropy that satisfies both simultaneously.

This is not a blocker for the GALAXY/EMSpectrum work this fork targets:
every `import simqso` in `desisim` is a lazy, function-local import inside
the QSO-only code path (`templates.py`'s `SIMQSO`/`QSO` classes,
`_make_simqso_templates`), already wrapped in a try/except with a
user-facing "please install simqso" message. `EMSpectrum` and the
GALAXY-type template classes never import it. Confirmed by direct source
read (`grep -rn simqso py/`) and by importing `EMSpectrum` and calling
`.spectrum()` successfully with simqso absent.

**Consequence**: QSO-flavor tests (`test_templates.py`'s use of the `QSO`/
`SIMQSO` classes, `test_quickquasars.py`, `lya_simqso_model.py`) cannot be
exercised in this environment. This is pre-existing/expected, not a
regression, and out of scope per this project's own instructions (simqso
is a separate future session).

## 5. External DESI data products — NOT available in this sandbox

Two environment variables gate large chunks of the existing test suite via
`unittest.skipUnless`, and are not present here:

- `$DESI_BASIS_TEMPLATES` — the pre-baked continuum FITS files this whole
  project (see §1.1 of the handoff) is partly about supplementing/replacing.
  No public download location was found from inside desisim's own docs; the
  paths referenced in `doc/nb/*.ipynb` are individual collaborators' local
  NERSC paths, implying DESI-collaboration-internal distribution.
- `$DESIMODEL` — desimodel's data package (throughput curves, PSFs, survey
  geometry, `targets.yaml`, etc.), fetched via `desimodel.install.install()`,
  which shells out to `svn export https://desi.lbl.gov/svn/code/desimodel`.
  This sandbox has neither an `svn` binary nor apt/root access to install
  one, and it's unconfirmed whether that SVN URL is anonymously readable.

**If you (the PI) have NERSC/DESI collaboration credentials or a public
mirror URL for either data product, provide them and this step can be
redone with fuller test coverage.**

## 6. Baseline test result (unmodified fork, before any of this project's changes)

Run: `venv/bin/python3 -m pytest py/desisim/test/ -v`, with `SPS_HOME` and
`PYTHONPATH=py` set, `DESI_BASIS_TEMPLATES`/`DESIMODEL` unset.

**14 passed, 22 skipped (correctly gated by `skipUnless` on the missing env
vars above), 26 failed + 2 errors.**

All 26 failures + 2 errors trace to the same two missing external data
products (§5) — e.g. `KeyError: 'DESI_BASIS_TEMPLATES'`,
`FileNotFoundError: .../desimodel/data/targets/targets.dat`. A few of these
(e.g. `test_templates.py::test_input_wave`,
`test_targets.py::test_sample_objtype`) are cases where upstream simply
didn't wrap the test in `skipUnless` even though it needs the data — a
pre-existing gap in the unmodified test suite, not something introduced by
this fork. None of the failures are attributable to code we've touched, since
no source files have been modified yet at the point this baseline was taken.

**Directly relevant to this project's actual first coding target**:
`EMSpectrum` (the class §1.2/§1.4 of the handoff modify) has zero dependency
on `$DESI_BASIS_TEMPLATES` or `$DESIMODEL` — it only reads bundled package
data (`py/desisim/data/*.ecsv`, `forbidden_mog.fits`). Verified directly:
`EMSpectrum()` instantiates and `.spectrum(...)` runs and returns a
correctly-shaped flux array with this environment as-is. So §1.2/§1.4 work
(and their unit tests) can proceed without resolving §5's data-access gap;
§1.1 (FSPS continuum) and full `GALAXY`/`LRG`/`ELG`/`BGS` integration tests
will need it.

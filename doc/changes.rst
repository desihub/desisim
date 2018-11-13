==================
desisim change log
==================

0.31.1 (unreleased)
-------------------

* No changes yet

0.31.0 (2018-11-08)
-------------------

* Update to new fibermap format for consistency with targeting and
  fiber assignment; requires desispec >= 0.26.0 (`PR #446`_).
* Update `desisim.templates.BGS` to use latest selection cuts (`PR #439`_).
* Fix quickquasar to work with Saclay mocks (`PR #435`_).
* Add support for >v3.0 stellar templates, with notebook to boot (`PR #434`_).
* Update notebook describing the construction of the LRG templates (`PR
  #433`_). 
* Fix quicksurvey (`PR #431`_).
* Update quickcat model (`PR #430`_, `PR #427`_).
* Fix archetype computation for redrock (`PR #429`_).
* Change ``electron`` to ``count`` for FITS compliance (`PR #428`_).
* Do not include Mg II emission by default (`PR #426`_).
* Add and adjust the nebular emission line spectra added to galaxy templates
  (`PR #424`_).
* quickquasar options for random z, ignoring transmission, random seeds,
  desisim.templates.SIMQSO vs. QSO
  (`PR #419`_, `PR #408`_, `PR #406`_, `PR #401`_).
* Read and write `select_mock_targets` style `simspec` file (`PR #416`_).
* Restore `quickquasars` to a functioning state, after being broken in `PR #409`_ (`PR #413`_).
* Add optional `nside` and `overwrite` arguments to `wrap-newexp` and
  `obs.new_exposure`, respectively (`PR #412`_).
* Major (and backwards-incompatible) refactor of how the template/simulated
  metadata are returned by desisim.templates (`PR #409`_).
* Adding reading metals from LyA transmission files (`PR #407`_).

.. _`PR #401`: https://github.com/desihub/desisim/pull/401
.. _`PR #406`: https://github.com/desihub/desisim/pull/406
.. _`PR #407`: https://github.com/desihub/desisim/pull/407
.. _`PR #408`: https://github.com/desihub/desisim/pull/408
.. _`PR #409`: https://github.com/desihub/desisim/pull/409
.. _`PR #412`: https://github.com/desihub/desisim/pull/412
.. _`PR #413`: https://github.com/desihub/desisim/pull/413
.. _`PR #416`: https://github.com/desihub/desisim/pull/416
.. _`PR #419`: https://github.com/desihub/desisim/pull/419
.. _`PR #424`: https://github.com/desihub/desisim/pull/424
.. _`PR #426`: https://github.com/desihub/desisim/pull/426
.. _`PR #427`: https://github.com/desihub/desisim/pull/427
.. _`PR #428`: https://github.com/desihub/desisim/pull/428
.. _`PR #429`: https://github.com/desihub/desisim/pull/429
.. _`PR #430`: https://github.com/desihub/desisim/pull/430
.. _`PR #431`: https://github.com/desihub/desisim/pull/431
.. _`PR #433`: https://github.com/desihub/desisim/pull/433
.. _`PR #434`: https://github.com/desihub/desisim/pull/434
.. _`PR #435`: https://github.com/desihub/desisim/pull/435
.. _`PR #439`: https://github.com/desihub/desisim/pull/439
.. _`PR #446`: https://github.com/desihub/desisim/pull/446

0.30.0 (2018-08-09)
-------------------

* Update templates to DR7+ standard-star designation (FSTD-->STD) (`PR #400`_).
* Update standard star bit name again STD -> STD_FAINT;
  requires desitarget 0.23.0 (`PR #402`_).

.. _`PR #400`: https://github.com/desihub/desisim/pull/400
.. _`PR #402`: https://github.com/desihub/desisim/pull/402

0.29.0 (2018-07-26)
-------------------

* Option in quickspectra to write the full sim table (`PR #392`_).
* Option to use Gaussian instead of Poisson for QSO DLA.
  Requires specsim >= v0.12 (`PR #393`_).
* Use `overwrite` instead of `clobber` for `astropy.io.fits` (`PR #395`_).

.. _`PR #392`: https://github.com/desihub/desisim/pull/392
.. _`PR #393`: https://github.com/desihub/desisim/pull/393
.. _`PR #395`: https://github.com/desihub/desisim/pull/395

0.28.0 (2018-07-18)
-------------------

* Add BALs to templates.QSO class (`PR #321`_).
* Enable redshift QA using input summary catalogs of truth and redshifts
  (`PR #349`_).
* Add zstats-like good/fail/miss/list QA method from desitest mini
  notebook and refactor previous code to enable it (`PR #351`_).
* quickquasar mags and random seed (`PR #350`_ and `PR #352`_).
* New pixsim and pixsim_nights (`PR #353`_, `PR #354`_, and `PR #358`_).
* Generate confusion matrix related to Spectype (`PR #359`_).
* Update QA to use qaprod_dir
* Fix newexp-mock wrapper when first expid != 0 (`PR #361`_).
* newexp-mock options for production running (`PR #363`_).
* Add BALs to QSO spectra outside of desisim.templates (`PR #370`_).
* Add rest-frame option to templates.SIMQSO (`PR #377`_).
* Optionally change output wave vector in templates.SIMQSO when noresample=True
  or restframe=True (`PR #383`_).
* Fix ``newexp-mock`` and ``wrap-fastframe`` file parsing for ``NIGHT/EXPID/*.*``
  vs. ``NIGHT/*.*``.
* Speed up emission line simulation when using ``MKL >= 2018.0.2`` (`PR #390`_).

.. _`PR #321`: https://github.com/desihub/desisim/pull/321
.. _`PR #349`: https://github.com/desihub/desisim/pull/349
.. _`PR #350`: https://github.com/desihub/desisim/pull/350
.. _`PR #351`: https://github.com/desihub/desisim/pull/351
.. _`PR #352`: https://github.com/desihub/desisim/pull/352
.. _`PR #353`: https://github.com/desihub/desisim/pull/353
.. _`PR #354`: https://github.com/desihub/desisim/pull/354
.. _`PR #358`: https://github.com/desihub/desisim/pull/358
.. _`PR #359`: https://github.com/desihub/desisim/pull/359
.. _`PR #361`: https://github.com/desihub/desisim/pull/361
.. _`PR #363`: https://github.com/desihub/desisim/pull/363
.. _`PR #370`: https://github.com/desihub/desisim/pull/370
.. _`PR #377`: https://github.com/desihub/desisim/pull/377
.. _`PR #383`: https://github.com/desihub/desisim/pull/383
.. _`PR #390`: https://github.com/desihub/desisim/pull/390

0.27.0 (2018-03-29)
-------------------

* Fix pixsim_mpi; make it faster with scatter/gather
  (`PR #329`_, `PR #332`_, and `PR #344`_).
* Fix PSF convolution for newexp-mock (`PR #331`_).
* BGS redshift bug fix (`PR #333`_).
* Astropy 2 compatibility (`PR #334`_).
* Fix newexp-mock --nspec option (`PR #340`_).
* Fix fibermap EXTNAME (`PR #340`_).
* Fix PSF convolution for newexp_mock (`PR #331`_).
* Match desispec renaming and relocating of of pix -> preproc
  (`PR #337`_ and `PR #339`_).
* More robust handling of unassigned fiber inputs (`PR #341`_).

.. _`PR #329`: https://github.com/desihub/desisim/pull/329
.. _`PR #331`: https://github.com/desihub/desisim/pull/331
.. _`PR #332`: https://github.com/desihub/desisim/pull/332
.. _`PR #333`: https://github.com/desihub/desisim/pull/333
.. _`PR #334`: https://github.com/desihub/desisim/pull/334
.. _`PR #337`: https://github.com/desihub/desisim/pull/337
.. _`PR #339`: https://github.com/desihub/desisim/pull/339
.. _`PR #340`: https://github.com/desihub/desisim/pull/340
.. _`PR #341`: https://github.com/desihub/desisim/pull/341
.. _`PR #344`: https://github.com/desihub/desisim/pull/344

0.26.0 (2018-02-27)
-------------------

Requires desitarget >= 0.19.0

* Update BGS fiber acceptance vs. z (`PR #326`_)
* Update desitarget imports for desitarget/0.19.0 (`PR #328`_)

.. _`PR #326`: https://github.com/desihub/desisim/pull/326
.. _`PR #328`: https://github.com/desihub/desisim/pull/328

0.25.1 (2018-02-23)
-------------------

Requires desitarget < 0.19.0

* Fix set_xscale(...) nonposy -> nonposx for qa_zfind

0.25.0 (2018-02-23)
-------------------

* Fix double PSF convolution in pixsims (`PR #320`_).
* Additional edits to QA scripts and doc to run with mini Notebook (`PR #322`_).
* Optional specsim config for simulating spectra (`PR #325`_)

.. _`PR #320`: https://github.com/desihub/desisim/pull/320
.. _`PR #322`: https://github.com/desihub/desisim/pull/322
.. _`PR #325`: https://github.com/desihub/desisim/pull/325

0.24.0 (2018-01-30)
-------------------

* Support new LRG templates (v2.0). (`PR #302`_).
* Bug fixes and additional features added to SIMQSO template maker. (`PR
  #303`_).
* Fixes quickspectra (broken by desispec change) (`PR #306`_).
* Fixes quickspectra random seed (never worked?) (`PR #306`_).
* Improves pixsim_mpi performance (`PR #312`_).
* Optionally do not wavelength resample simqso templates (`PR #310`_).
* Default to basis templates v2.4 instead of 2.3
* Minor edits to QA scripts and doc (`PR #311`_).
* Adds quickspectra --skyerr option (`PR #313`_).
* Correct fastframe output BUNIT (`PR #317`_).

.. _`PR #302`: https://github.com/desihub/desisim/pull/302
.. _`PR #303`: https://github.com/desihub/desisim/pull/303
.. _`PR #306`: https://github.com/desihub/desisim/pull/306
.. _`PR #312`: https://github.com/desihub/desisim/pull/312
.. _`PR #310`: https://github.com/desihub/desisim/pull/310
.. _`PR #311`: https://github.com/desihub/desisim/pull/311
.. _`PR #313`: https://github.com/desihub/desisim/pull/313
.. _`PR #317`: https://github.com/desihub/desisim/pull/317

0.23.0 (2017-12-20)
-------------------

* Fixed crash in newexp-mock success print message.
* Refactor DLA code into its own module (`PR #294`_).
* Adds reader for LyA skewer v2.x format (`PR #297`_).
* Removed deprecated brick output from quickgen.
* Preliminary support for simqso based QSO templates (`PR #293`_).
* fastframe can directly output cframes (`PR #287`_).
* adds BGS efficiency notebooks (`PR #285`_ and `PR #286`_).

.. _`PR #285`: https://github.com/desihub/desisim/pull/285
.. _`PR #286`: https://github.com/desihub/desisim/pull/286
.. _`PR #287`: https://github.com/desihub/desisim/pull/287
.. _`PR #294`: https://github.com/desihub/desisim/pull/294
.. _`PR #293`: https://github.com/desihub/desisim/pull/293
.. _`PR #297`: https://github.com/desihub/desisim/pull/297

0.22.0 (2017-11-10)
-------------------

* Scaling updates to wrap-fastframe and wrap-newexp (`PR #274`_).
* Fix a minor units scaling bug in lya_spectra (`PR #264`_).
* newexp takes exposures list with EXPID and arcs/flats (`PR #275`_).
* lyman alpha QSOs with optional DLAs (`PR #275`_).
* Update arc lamp line list (`PR #272`_).
* Fix MPI pixsim wrappers (`PR #265`_ and `PR #262`_).
* quicksurvey updats for latest surveysim outputs (`PR #270`_).
* Adds fastfiber method of fiber input loss calculations (`PR #261`_).
* Fix quickgen moon input parameters (`PR #263`_).
* Adds quickspectra script (`PR #259`_).

.. _`PR #264`: https://github.com/desihub/desisim/pull/264
.. _`PR #274`: https://github.com/desihub/desisim/pull/274
.. _`PR #275`: https://github.com/desihub/desisim/pull/275
.. _`PR #272`: https://github.com/desihub/desisim/pull/272
.. _`PR #265`: https://github.com/desihub/desisim/pull/265
.. _`PR #270`: https://github.com/desihub/desisim/pull/270
.. _`PR #261`: https://github.com/desihub/desisim/pull/261
.. _`PR #262`: https://github.com/desihub/desisim/pull/262
.. _`PR #263`: https://github.com/desihub/desisim/pull/263
.. _`PR #259`: https://github.com/desihub/desisim/pull/259

0.21.0 (2017-09-29)
-------------------

* Major refactor of newexp to add connection to upstream mocks, surveysims,
  and fiber assignment (`PR #250`_).
* Support latest (>DR4) data model in the templates metadata table and also
  scale simulated templates by 1e17 erg/s/cm2/Angstrom (`PR #252`_).
* Add desi_qa_s2n script (`PR #254`_)
* Refactor desi_qa_zfind script (`PR #254`_)
* Refactor redshift QA for new data model (`PR #254`_)
* Refactor shared QA methods to desisim.spec_qa.utils (`PR #254`_)
* New plots for S/N of spectra for various objects (ELG, LRG, QSO) (`PR #254`_)
* Add BGS, MWS to z_find QA
* Miscellaneous polishing in QA (velocity, clip before RMS, extend [OII] flux, S/N per Ang)
* Bug fix: correctly select both "bright" and "faint" BGS templates by default
  (`PR #257`_).
* Updates for newexp/fastframe wrappers for end-to-end sims (`PR #258`_).

.. _`PR #250`: https://github.com/desihub/desisim/pull/250
.. _`PR #252`: https://github.com/desihub/desisim/pull/252
.. _`PR #254`: https://github.com/desihub/desisim/pull/254
.. _`PR #257`: https://github.com/desihub/desisim/pull/257
.. _`PR #258`: https://github.com/desihub/desisim/pull/258

0.20.0 (2017-07-12)
-------------------

* Adds tutorial on simulating spectra (`PR #244`_).
* Fixes QSO template wavelength extrapolation (`PR #247`_);
  requires desispec > 0.15.1.
* Uses ``desitarget.cuts.isLRG_colors``; requires desitarget >= 0.14.0
  (`PR #246`_).
* Uses ``desiutil.log`` instead of ``desispec.log``.

.. _`PR #244`: https://github.com/desihub/desisim/pull/244
.. _`PR #246`: https://github.com/desihub/desisim/pull/246
.. _`PR #247`: https://github.com/desihub/desisim/pull/247

0.19.0 (2017-06-15)
-------------------

* "FLAVOR" keyword is arc/flat/science but not dark/bright/bgs/mws/etc to match
  desispec usage (`PR #243`_).
* Add ``nocolorcuts`` option for LyA spectra (`PR #242`_).
* Fixes for ``targets.dat`` to ``targets.yaml`` change (`PR #240`_).
* Changed refs to ``desispec.brick`` to its new location at :mod:`desiutil.brick` (`PR #241`_).
* Remove LyA absorption below the LyA limit (`PR #236`_).
* Refactor and speed-up of QSO templates; add Lya forest on-the-fly (`PR #234`_).

.. _`PR #234`: https://github.com/desihub/desisim/pull/234
.. _`PR #236`: https://github.com/desihub/desisim/pull/236
.. _`PR #240`: https://github.com/desihub/desisim/pull/240
.. _`PR #241`: https://github.com/desihub/desisim/pull/241
.. _`PR #242`: https://github.com/desihub/desisim/pull/242
.. _`PR #243`: https://github.com/desihub/desisim/pull/243

0.18.3 (2017-04-13)
-------------------

* Add DLAs to lya spectra (`PR #220`_)
* Fix quickgen for specsim v0.8 (`PR #226`_).
* Add verbose output to templates code (`PR #230`_).
* Much faster quickcat (`PR #233`_).

.. _`PR #226`: https://github.com/desihub/desisim/pull/226
.. _`PR #230`: https://github.com/desihub/desisim/pull/230
.. _`PR #233`: https://github.com/desihub/desisim/pull/233
.. _`PR #220`: https://github.com/desihub/desisim/pull/220

0.18.2 (2017-03-27)
-------------------

* Fixed a number of documentation errors (`PR #224`_).
* Removed unneeded Travis scripts in ``etc/``.
* Fixed N^2 scaling of :meth:`desisim.templates.QSO.make_templates`.
* Speed up :class:`desisim.templates.GALAXY` by factor of
  8-12 by caching velocity dispersions (`PR #229`_)

.. _`PR #224`: https://github.com/desihub/desisim/pull/224
.. _`PR #229`: https://github.com/desihub/desisim/pull/229

0.18.1 (2016-03-05)
-------------------

* Update ``desisim.module`` to use :envvar:`DESI_BASIS_TEMPLATES` v2.3.

0.18.0 (2016-03-04)
-------------------

* pixsims add new required keywords DOSVER, FEEVER, DETECTOR.
* Small bug fixes in quickcat; drop unused truth,targets columns to save memory
  in quicksurvey loop (PRs #198, #199).
* quickgen update to support white dwarf templates (PR #204)
* several enhancements of the templates code

  * optionally output rest-frame templates (PR #208)
  * rewrite of lya_spectra to achieve factor of 10 speedup; use COSMO
    (astropy.cosmology setup) as a new optional keyword for qso_desi_templates;
    updated API (PRs #210, #212)
  * various small changes to desisim.templates (PR #211)
  * support for DA and DB white dwarf subtypes (PR #213)

* update test dependencies (PR #214)

0.17.1 (2016-12-05)
-------------------

* Fix bug when obsconditions contain tiles that don't overlap catalog
* Add ``surveysim --start_epoch`` option

0.17.0 (2016-12-02)
-------------------

* fixes tests for use with latest desitarget master
* Refactor quickgen and quickbrick to reduce duplicated code (PR #184)
* Makes BGS compatible with desitarget master after
  isBGS -> isBGS_faint vs. isBGS_bright
* Refactor quickcat to include dependency on observing conditions
* Update quicksurvey to use observing conditions from surveysim
* Fixes use of previous zcatalog when updating catalog with new observations

0.16.0 (2016-11-10)
-------------------

* Requires specsim >= v0.6
* Add integration test for quickgen (PR #179)
* Cache specsim Simulator for faster testing (PR #178)
* Add lya_spectra.get_spectra (PR #156)
* Add quickgen and quickbrick unit tests and bug fixes (PR #176, #177)

0.15.0 (2016-10-14)
-------------------

* Fix some ``build_sphinx`` errors.
* Run coverage tests under Python 2.7 for now.
* Update template Module file to new DESI+Anaconda infrastructure.
* quickbrick unit tests and bug fixes (#166)
* new quickgen features (PR #173 and #175)

  * fix exptime and airmass for specsim v0.5
  * new --frameonly option
  * moon phase, angle, and zenith options
  * misc cleanup and unit tests

0.14.0 (2016-09-14)
-------------------

* updates for python 3.5

0.13.1 (2016-08-18)
-------------------

* fix batch.pixsim seeds vs. seed typo

0.13.0 (2016-08-18)
-------------------

* desi_qa_zfind: fixed --reduxdir option; improved plots
* PR#132: major refactor of template generation, including ability to give
  input redshifts, magnitudes, or random seeds from metadata table.
* desisim.batch.pixsim functions propagate random seeds for reproducibility

0.12.0 (2016-07-14)
-------------------

* desi_qa_zfind options to override raw and processed data directories
* PRODNAME -> SPECPROD and TYPE -> SPECTYPE to match latest desispec
* remove unused get_simstds.py
* fix #142 so that pixsim only optionally runs preprocessing
* fix #141 to avoid repeated TARGETIDs when simulating both
  bright and dark tiles together
* add io.load_simspec_summary() convenience function to load and merge
  truth information from fibermap and simspec files.
* adjusts which magnitudes were plotted for each target class

0.11.0 (2016-07-12)
-------------------

Pixsim updates:

* simulate fully raw data, then call preprocessing
* bug fix for simulating tiles in parallel
* fix pixsim loading of non-default PSFs

0.10.0 and prior
----------------

* No changes.rst yet

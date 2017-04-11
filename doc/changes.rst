==================
desisim change log
==================

0.18.3 (unreleased)
-------------------

* Add DLAs to lya spectra (PR #220)
* Fix quickgen for specsim v0.8 (`PR #226`_).
* Add verbose output to templates code (`PR #230`_).

.. _`PR #226`: https://github.com/desihub/desisim/pull/226
.. _`PR #230`: https://github.com/desihub/desisim/pull/230

0.18.2 (2017-03-27)
-------------------

* Fixed a number of documentation errors (`PR #224`_).
* Removed unneeded Travis scripts in ``etc/``.
* Fixed N^2 scaling of ``QSO.make_templates``
* Speed up desisim.templates.GALAXY by factor of 8-12 by caching velocity
  dispersions (PR #229)

.. _`PR #224`: https://github.com/desihub/desisim/pull/224

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

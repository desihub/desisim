==================
desisim change log
==================

0.15.1 (unreleased)
-------------------

* add integration test for quickgen

0.15.0 (2016-10-20)
-------------------

* add moon phase, moon angle, and zenith angle to quickgen
* add a unit test for each moon property

0.14.0 (2016-10-13)
-------------------

* change $PRODNAME to $SPECPROD in quickgen
* change print statement to log.info() in quickgen
* change os.path.join to desispec.io.findfile in quickgen

0.14.0 (2016-10-12)
-------------------

* add keyword frameonly to quickgen
* allow for only uncalibrated frame files to be output by quickgen

0.14.0 (2016-10-12)
-------------------

* add airmass unit test for quickgen
* add exposure time unit test for quickgen
* update how exposure time is set in quickgen with specsim v0.5

0.14.0 (2016-09-28)
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

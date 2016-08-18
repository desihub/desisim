==================
desisim change log
==================

0.13.0 (unreleased)
-------------------

* desi_qa_zfind: fixed --reduxdir option; improved plots
* PR#132: major refactor of template generation, including ability to give
  input redshifts, magnitudes, or random seeds from metadata table.

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

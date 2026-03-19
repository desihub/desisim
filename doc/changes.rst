==================
desisim change log
==================

0.39.1 (unreleased)
-------------------

* No changes yet

0.39.0 (2026-03-18)
-------------------

Features:

* Support for more data releases on SurveyRelease/Quickquasars (PR `#590`_).

Test infrastructure:

* Remove DesiTest (PR `#582`_).
* Add numpy/2.x and scipy/1.16.x support. General cleanup. (PR `#589`_).
* Add astropy/7.x test support on GitHub (PR `#591`_).
* Minor docs cleanup; add support for parallel tests (PR `#594`_).

.. _`#582`: https://github.com/desihub/desisim/pull/582
.. _`#589`: https://github.com/desihub/desisim/pull/589
.. _`#590`: https://github.com/desihub/desisim/pull/590
.. _`#591`: https://github.com/desihub/desisim/pull/591
.. _`#594`: https://github.com/desihub/desisim/pull/594

0.38.2 (2024-12-17)
-------------------

* Update to generate DESI-Y3 Lya mocks (PR `#581`_).

.. _`#581`: https://github.com/desihub/desisim/pull/581

0.38.1 (2024-04-30)
-------------------

* Followup requested changes for DESI-Y1 LyA mocks (PR `#580`_).

.. _`#580`: https://github.com/desihub/desisim/pull/580

0.38.0 (2024-03-05)
-------------------

* Fixed bug on quickquasars quasar sampling method to reproduce z distribution
  from SV causing a wrong shape in cross-correlation (PR `#577`_).
* Updates for DESI-Y1 LyA mocks with quickquasars (PR `#578`_).

.. _`#577`: https://github.com/desihub/desisim/pull/577
.. _`#578`: https://github.com/desihub/desisim/pull/578

0.37.1 (2023-01-13)
-------------------

* Added sort to io.find_basis_template() (PR `#576`_).

.. _`#576`: https://github.com/desihub/desisim/pull/576

0.37.0 (2023-01-12)
-------------------

* Smoothing source contribution to noise in quickquasars (PR `#566`_).
* Quickquasars updates to reproduce actual z and mag distribution
  as obtained in SV for DESIY5 mocks (PR `#569`_).
* Make multiprocessing Pool safe in quickquasars (PR `#570`_).
* Renamed master -> main (PR `#571`_).
* Updated emission line and continuum models (PR `#572`_).
* Fix pixsim (PR `#573`_).

.. _`#566`: https://github.com/desihub/desisim/pull/566
.. _`#569`: https://github.com/desihub/desisim/pull/569
.. _`#570`: https://github.com/desihub/desisim/pull/570
.. _`#571`: https://github.com/desihub/desisim/pull/571
.. _`#572`: https://github.com/desihub/desisim/pull/572
.. _`#573`: https://github.com/desihub/desisim/pull/573

0.36.0 (2022-01-20)
-------------------

* Major fixes to desisim unit tests, particularly for templates (see PR thread
  for details about algorithmic changes; PR `#559`_).
* Restore template-generating code to a working state (PR `#556`_).
* Flux bug fix in quicktransients simulator (PR `#541`_).

.. _`#541`: https://github.com/desihub/desisim/pull/541
.. _`#556`: https://github.com/desihub/desisim/pull/556
.. _`#559`: https://github.com/desihub/desisim/pull/559


0.35.6 (2021-03-31)
-------------------

* lighterweight quickquasars (PR `#552`_).

.. _`#552`: https://github.com/desihub/desisim/pull/552

0.35.5 (2021-02-15)
-------------------

* Migrated unit tests to GitHub Actions (`PR #546`_).
* Fix negative arc line ringing (`PR #548`_).
* Fix sim header keywords for TSNR calc (`PR #550`_, `PR #551`_).
* Add option to save true continuum (`PR #544`_).

.. _`PR #544`: https://github.com/desihub/desisim/pull/544
.. _`PR #546`: https://github.com/desihub/desisim/pull/546
.. _`PR #548`: https://github.com/desihub/desisim/pull/548
.. _`PR #550`: https://github.com/desihub/desisim/pull/550
.. _`PR #551`: https://github.com/desihub/desisim/pull/551

0.35.4 (2020-11-12)
-------------------

* QSO BAL bug fix (`PR #545`_).

.. _`PR #545`: https://github.com/desihub/desisim/pull/545

0.35.3 (2020-08-04)
-------------------

* desisim.spec_qa.redshifts.zstats support for astropy 4 (hotfix to master).

0.35.2 (2020-08-04)
-------------------

* Support astropy/4.x and fitsio/1.x (`PR #539`_ and `PR #542`_).
* New keyword in quickcat to indicate which HDU to read in fba files (`PR #538`_)
* Fix sky level Travis test failure (#534) and "low QSO flux" template unit test
  failure (#507) (`PR #536`_).
* Add freeze_iers to more functions in simexp (direct to master).
* Add the option to run quickquasars in eBOSS mode (`PR #481`_)
* Add dwave out to quickquasars (`PR #533`_).

.. _`PR #481`: https://github.com/desihub/desisim/pull/481
.. _`PR #533`: https://github.com/desihub/desisim/pull/533
.. _`PR #536`: https://github.com/desihub/desisim/pull/536
.. _`PR #538`: https://github.com/desihub/desisim/pull/538
.. _`PR #539`: https://github.com/desihub/desisim/pull/539
.. _`PR #542`: https://github.com/desihub/desisim/pull/542

0.35.1 (2020-04-15)
-------------------

* Add freeze_iers to quickgen (direct fix to master).

0.35.0 (2020-04-13)
-------------------

* Use desiutil.iers.freeze_iers instead of desisurvey (requires
  desiutil >= 2.0.3) (`PR #530`_).
* Update BAL_META columns (`PR #527`_).

.. _`PR #527`: https://github.com/desihub/desisim/pull/527
.. _`PR #530`: https://github.com/desihub/desisim/pull/530

0.34.3 (2020-04-07)
-------------------

* Add functionality to simulate transients into galaxy spectra; includes several
  example notebooks (`PR #525`_).
* Added a new table with a development emission line table  (`PR #523`_).

.. _`PR #525`: https://github.com/desihub/desisim/pull/525
.. _`PR #523`: https://github.com/desihub/desisim/pull/523

0.34.2 (2019-12-27)
-------------------

* Add ``desisurvey.utils.freeze_iers()`` to any code that uses
  ``astropy.time`` (`PR #520`_).

.. _`PR #520`: https://github.com/desihub/desisim/pull/520

0.34.1 (2019-12-20)
-------------------

* New fiberassign file names (`PR #519`_).
* Minor change to DLA metadata (`PR #517`_).
* Minor fixes (`PR #515`_).
* Use `desitarget.io.find_target_files` to find (mock) target catalogs (`PR #514`_).
* Update `desisim.module` to use latest `v3.2` basis templates (`PR #513`_).

.. _`PR #513`: https://github.com/desihub/desisim/pull/513
.. _`PR #514`: https://github.com/desihub/desisim/pull/514
.. _`PR #515`: https://github.com/desihub/desisim/pull/515
.. _`PR #517`: https://github.com/desihub/desisim/pull/517
.. _`PR #519`: https://github.com/desihub/desisim/pull/519


0.34.0 (2019-10-17)
-------------------

Requires desispec/0.30.0 or later.

* Support mocks in bright/dark subdirs (`PR #508`_).
* Support FIBERASSIGN_X/Y instead of DESIGN_X/Y from fiberassign (`PR #512`_).

.. _`PR #508`: https://github.com/desihub/desisim/pull/508
.. _`PR #512`: https://github.com/desihub/desisim/pull/512

0.33.1 (2019-10-01)
-------------------

* *No code or API changes in this tag.*
* Pinned Numpy version to fix broken tests (`PR #505`_).
* Minor changes to documentation configuration and docstrings.

.. _`PR #505`: https://github.com/desihub/desisim/pull/505

0.33.0 (2019-09-30)
-------------------

* Running quickquasar on recent London mocks (`PR #495`_, `PR #497`_).
* Update eBOSS to DR16 (`PR #498`_).
* Updates to pixsim for SV (`PR #502`_).
* Fix bug in counting repeat observations (`PR #503`_).

.. _`PR #495`: https://github.com/desihub/desisim/pull/495
.. _`PR #497`: https://github.com/desihub/desisim/pull/497
.. _`PR #498`: https://github.com/desihub/desisim/pull/498
.. _`PR #502`: https://github.com/desihub/desisim/pull/502
.. _`PR #503`: https://github.com/desihub/desisim/pull/503


0.32.0 (2019-05-30)
-------------------

* Remove remaining matplotlib (`PR #470`_).
* Modify simqsotemplate (`PR #474`_).
* LyA/QSO sims updates (`PR #471`_, `PR #472`_, `PR #473`_, `PR #475`_,
  `PR #478`_, `PR #483`_, `PR #485`_, `PR #488`_
* Misc cleanup (`PR #480`, `PR #479`_,
* Support new tile naming in quicksurvey (`PR #486`_).
* Fix crashing bug in mock spectra OBJMETA tracking (`PR #490`_).
* Support SV1_DESI_TARGET (`PR #494`_).

.. _`PR #470`: https://github.com/desihub/desisim/pull/470
.. _`PR #471`: https://github.com/desihub/desisim/pull/471
.. _`PR #472`: https://github.com/desihub/desisim/pull/472
.. _`PR #473`: https://github.com/desihub/desisim/pull/473
.. _`PR #474`: https://github.com/desihub/desisim/pull/474
.. _`PR #475`: https://github.com/desihub/desisim/pull/475
.. _`PR #478`: https://github.com/desihub/desisim/pull/478
.. _`PR #479`: https://github.com/desihub/desisim/pull/479
.. _`PR #480`: https://github.com/desihub/desisim/pull/480
.. _`PR #483`: https://github.com/desihub/desisim/pull/483
.. _`PR #485`: https://github.com/desihub/desisim/pull/485
.. _`PR #486`: https://github.com/desihub/desisim/pull/486
.. _`PR #488`: https://github.com/desihub/desisim/pull/488
.. _`PR #490`: https://github.com/desihub/desisim/pull/490
.. _`PR #494`: https://github.com/desihub/desisim/pull/494

0.31.2 (2019-02-28)
-------------------

* Update quickquasars default redshift error (`PR #466`_).
* Support for London mocks v5.0 + DLAs (`PR #467`_).

.. _`PR #466`: https://github.com/desihub/desisim/pull/466
.. _`PR #467`: https://github.com/desihub/desisim/pull/467

0.31.1 (2018-12-14)
-------------------

* quickquasars updates:

  * support eBOSS (`PR #450`_).
  * mimic redshift fitter uncertainties (`PR #452`_).
  * adding shift to redshift (`PR #454`_).
  * fix error in size of Z_noFOG (`PR #455`_).
  * Fix quickquasars targetid truth (`PR #457`_).

* Precompute colors for star and galaxy templates. (`PR #453`_).
* Refactor S/N qa to load cframes only once (also updates OII for new TRUTH table) (`PR #459`_, `PR #465`_).
* Use basis_templates v3.1 and matching desisim-testdata 0.6.1 (`PR #464`_).

.. _`PR #450`: https://github.com/desihub/desisim/pull/450
.. _`PR #452`: https://github.com/desihub/desisim/pull/452
.. _`PR #453`: https://github.com/desihub/desisim/pull/453
.. _`PR #454`: https://github.com/desihub/desisim/pull/454
.. _`PR #455`: https://github.com/desihub/desisim/pull/455
.. _`PR #457`: https://github.com/desihub/desisim/pull/457
.. _`PR #459`: https://github.com/desihub/desisim/pull/459
.. _`PR #464`: https://github.com/desihub/desisim/pull/464
.. _`PR #465`: https://github.com/desihub/desisim/pull/465

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

* bring changes from master (`PR #396`_).
* Update to use fiberassign script in place of fiberassign_exec from c++ code (`PR #397`_).
* below wanted_min_wave, corresponding to z=1.7, we set F=1 (`PR #399`_).
* Update templates to DR7+ standard-star designation (FSTD-->STD) (`PR #400`_).
* Update standard star bit name again STD -> STD_FAINT;
  requires desitarget 0.23.0 (`PR #402`_).

.. _`PR #396`: https://github.com/desihub/desisim/pull/396
.. _`PR #397`: https://github.com/desihub/desisim/pull/397
.. _`PR #399`: https://github.com/desihub/desisim/pull/399
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
* Quick lya (`PR #343`_).
* Healpix in spectra header (`PR #345`_).
* Enable redshift QA using input summary catalogs of truth and redshifts
  (`PR #349`_).
* quickquasar mags and random seed (`PR #350`_ and `PR #352`_).
* Add zstats-like good/fail/miss/list QA method from desitest mini
  notebook and refactor previous code to enable it (`PR #351`_).
* New pixsim and pixsim_nights (`PR #353`_, `PR #354`_, and `PR #358`_).
* Generate confusion matrix related to Spectype (`PR #359`_).
* Update QA to use qaprod_dir
* Fix newexp-mock wrapper when first expid != 0 (`PR #361`_).
* newexp-mock options for production running (`PR #363`_).
* Add dla (`PR #366`_).
* read healpix nside and scheme from header (`PR #367`_).
* add transmission of metals from LYA transmission (`PR #368`_).
* Add BALs to QSO spectra outside of desisim.templates (`PR #370`_).
* Qqso dla bal (`PR #371`_).
* Revert "Qqso dla bal" (`PR #372`_).
* add parameter for metal lines (`PR #373`_).
* Qqsuasar dlas (`PR #374`_).
* Add rest-frame option to templates.SIMQSO (`PR #377`_).
* update extension name of fiber-assignment output file (`PR #380`_).
* correct computation of chi2 and amplitude between archetypes (`PR #382`_).
* Optionally change output wave vector in templates.SIMQSO when noresample=True
  or restframe=True (`PR #383`_).
* raw data now under NIGHT/EXPID/ not just NIGHT/ (`PR #384`_).
* Fix dlabugs (`PR #385`_).
* bringing things from master (`PR #387`_).
* fix newexp-mock and wrap-fastframe NIGHT/EXPID/ vs. NIGHT/ parsing (`PR #389`_).
* Fix ``newexp-mock`` and ``wrap-fastframe`` file parsing for ``NIGHT/EXPID/*.*``
  vs. ``NIGHT/*.*``.
* Speed up emission line simulation when using ``MKL >= 2018.0.2`` (`PR #390`_).

.. _`PR #321`: https://github.com/desihub/desisim/pull/321
.. _`PR #343`: https://github.com/desihub/desisim/pull/343
.. _`PR #345`: https://github.com/desihub/desisim/pull/345
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
.. _`PR #366`: https://github.com/desihub/desisim/pull/366
.. _`PR #367`: https://github.com/desihub/desisim/pull/367
.. _`PR #368`: https://github.com/desihub/desisim/pull/368
.. _`PR #370`: https://github.com/desihub/desisim/pull/370
.. _`PR #371`: https://github.com/desihub/desisim/pull/371
.. _`PR #372`: https://github.com/desihub/desisim/pull/372
.. _`PR #373`: https://github.com/desihub/desisim/pull/373
.. _`PR #374`: https://github.com/desihub/desisim/pull/374
.. _`PR #377`: https://github.com/desihub/desisim/pull/377
.. _`PR #380`: https://github.com/desihub/desisim/pull/380
.. _`PR #382`: https://github.com/desihub/desisim/pull/382
.. _`PR #383`: https://github.com/desihub/desisim/pull/383
.. _`PR #384`: https://github.com/desihub/desisim/pull/384
.. _`PR #385`: https://github.com/desihub/desisim/pull/385
.. _`PR #387`: https://github.com/desihub/desisim/pull/387
.. _`PR #389`: https://github.com/desihub/desisim/pull/389
.. _`PR #390`: https://github.com/desihub/desisim/pull/390

0.27.0 (2018-03-29)
-------------------

* Fix pixsim_mpi; make it faster with scatter/gather
  (`PR #329`_, `PR #332`_, and `PR #344`_).
* Fix PSF convolution for newexp-mock (`PR #331`_).
* BGS redshift bug fix (`PR #333`_).
* Astropy 2 compatibility (`PR #334`_).
* Match desispec renaming and relocating of of pix -> preproc
  (`PR #337`_ and `PR #339`_).
* Fix newexp-mock --nspec option (`PR #340`_).
* Fix fibermap EXTNAME (`PR #340`_).
* Fix PSF convolution for newexp_mock (`PR #331`_).
* More robust handling of unassigned fiber inputs (`PR #341`_).
* Fix minor problems in doc/changes.rst (`PR #342`_).

.. _`PR #329`: https://github.com/desihub/desisim/pull/329
.. _`PR #331`: https://github.com/desihub/desisim/pull/331
.. _`PR #332`: https://github.com/desihub/desisim/pull/332
.. _`PR #333`: https://github.com/desihub/desisim/pull/333
.. _`PR #334`: https://github.com/desihub/desisim/pull/334
.. _`PR #337`: https://github.com/desihub/desisim/pull/337
.. _`PR #339`: https://github.com/desihub/desisim/pull/339
.. _`PR #340`: https://github.com/desihub/desisim/pull/340
.. _`PR #341`: https://github.com/desihub/desisim/pull/341
.. _`PR #342`: https://github.com/desihub/desisim/pull/342
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
* fix bug in pixsim when simulating more than one spectro (`PR #304`_).
* Fixes quickspectra (broken by desispec change) (`PR #306`_).
* Fixes quickspectra random seed (never worked?) (`PR #306`_).
* Optionally do not wavelength resample simqso templates (`PR #310`_).
* Minor edits to QA scripts and doc (`PR #311`_).
* Improves pixsim_mpi performance (`PR #312`_).
* Adds quickspectra --skyerr option (`PR #313`_).
* Correct fastframe output BUNIT (`PR #317`_).
* Default to basis templates v2.4 instead of 2.3

.. _`PR #302`: https://github.com/desihub/desisim/pull/302
.. _`PR #303`: https://github.com/desihub/desisim/pull/303
.. _`PR #304`: https://github.com/desihub/desisim/pull/304
.. _`PR #306`: https://github.com/desihub/desisim/pull/306
.. _`PR #310`: https://github.com/desihub/desisim/pull/310
.. _`PR #311`: https://github.com/desihub/desisim/pull/311
.. _`PR #312`: https://github.com/desihub/desisim/pull/312
.. _`PR #313`: https://github.com/desihub/desisim/pull/313
.. _`PR #317`: https://github.com/desihub/desisim/pull/317

0.23.0 (2017-12-20)
-------------------

* Reintroduce nspec option (`PR #279`_).
* Add objtypes in qa_zfind (`PR #280`_).
* handles new date formats (`PR #281`_).
* QA: docs and qa_s2n improvement (`PR #282`_).
* code and notebook for generating archetypes for BGS (`PR #283`_).
* adds BGS efficiency notebooks (`PR #285`_ and `PR #286`_).
* fastframe can directly output cframes (`PR #287`_).
* do not add random lya forest when also reading lya forest from file (`PR #292`_).
* Preliminary support for simqso based QSO templates (`PR #293`_).
* Refactor DLA code into its own module (`PR #294`_).
* Adds reader for LyA skewer v2.x format (`PR #297`_).
* fix bright/dark test (`PR #299`_).
* Fixed crash in newexp-mock success print message.
* Removed deprecated brick output from quickgen.

.. _`PR #279`: https://github.com/desihub/desisim/pull/279
.. _`PR #280`: https://github.com/desihub/desisim/pull/280
.. _`PR #281`: https://github.com/desihub/desisim/pull/281
.. _`PR #282`: https://github.com/desihub/desisim/pull/282
.. _`PR #283`: https://github.com/desihub/desisim/pull/283
.. _`PR #285`: https://github.com/desihub/desisim/pull/285
.. _`PR #286`: https://github.com/desihub/desisim/pull/286
.. _`PR #287`: https://github.com/desihub/desisim/pull/287
.. _`PR #292`: https://github.com/desihub/desisim/pull/292
.. _`PR #293`: https://github.com/desihub/desisim/pull/293
.. _`PR #294`: https://github.com/desihub/desisim/pull/294
.. _`PR #297`: https://github.com/desihub/desisim/pull/297
.. _`PR #299`: https://github.com/desihub/desisim/pull/299

0.22.0 (2017-11-10)
-------------------

* Adds quickspectra script (`PR #259`_).
* Adds fastfiber method of fiber input loss calculations (`PR #261`_).
* Fix MPI pixsim wrappers (`PR #265`_ and `PR #262`_).
* Fix quickgen moon input parameters (`PR #263`_).
* Fix a minor units scaling bug in lya_spectra (`PR #264`_).
* Quickspectra (`PR #266`_).
* fix newexp-random --outdir option (`PR #267`_).
* quicksurvey updats for latest surveysim outputs (`PR #270`_).
* drop columns from tables used in quicksurvey (`PR #271`_).
* Update arc lamp line list (`PR #272`_).
* Scaling updates to wrap-fastframe and wrap-newexp (`PR #274`_).
* newexp takes exposures list with EXPID and arcs/flats (`PR #275`_).
* lyman alpha QSOs with optional DLAs (`PR #275`_).
* Fix coverage error (`PR #277`_).
* qa_zfind even if specter wasn't used (`PR #278`_).

.. _`PR #259`: https://github.com/desihub/desisim/pull/259
.. _`PR #261`: https://github.com/desihub/desisim/pull/261
.. _`PR #262`: https://github.com/desihub/desisim/pull/262
.. _`PR #263`: https://github.com/desihub/desisim/pull/263
.. _`PR #264`: https://github.com/desihub/desisim/pull/264
.. _`PR #265`: https://github.com/desihub/desisim/pull/265
.. _`PR #266`: https://github.com/desihub/desisim/pull/266
.. _`PR #267`: https://github.com/desihub/desisim/pull/267
.. _`PR #270`: https://github.com/desihub/desisim/pull/270
.. _`PR #271`: https://github.com/desihub/desisim/pull/271
.. _`PR #272`: https://github.com/desihub/desisim/pull/272
.. _`PR #274`: https://github.com/desihub/desisim/pull/274
.. _`PR #275`: https://github.com/desihub/desisim/pull/275
.. _`PR #277`: https://github.com/desihub/desisim/pull/277
.. _`PR #278`: https://github.com/desihub/desisim/pull/278

0.21.0 (2017-09-29)
-------------------

* quicksurvey on surveysim outputs (`PR #249`_).
* Major refactor of newexp to add connection to upstream mocks, surveysims,
  and fiber assignment (`PR #250`_).
* Support latest (>DR4) data model in the templates metadata table and also
  scale simulated templates by 1e17 erg/s/cm2/Angstrom (`PR #252`_).
* BGS Simulation parameter propagation (`PR #253`_).
* Add desi_qa_s2n script (`PR #254`_)
* Refactor desi_qa_zfind script (`PR #254`_)
* Refactor redshift QA for new data model (`PR #254`_)
* Refactor shared QA methods to desisim.spec_qa.utils (`PR #254`_)
* New plots for S/N of spectra for various objects (ELG, LRG, QSO) (`PR #254`_)
* Add BGS, MWS to z_find QA
* QA Polishing (`PR #255`_).
* Miscellaneous polishing in QA (velocity, clip before RMS, extend [OII] flux, S/N per Ang)
* Bug fix: correctly select both "bright" and "faint" BGS templates by default
  (`PR #257`_).
* Updates for newexp/fastframe wrappers for end-to-end sims (`PR #258`_).

.. _`PR #249`: https://github.com/desihub/desisim/pull/249
.. _`PR #250`: https://github.com/desihub/desisim/pull/250
.. _`PR #252`: https://github.com/desihub/desisim/pull/252
.. _`PR #253`: https://github.com/desihub/desisim/pull/253
.. _`PR #254`: https://github.com/desihub/desisim/pull/254
.. _`PR #255`: https://github.com/desihub/desisim/pull/255
.. _`PR #257`: https://github.com/desihub/desisim/pull/257
.. _`PR #258`: https://github.com/desihub/desisim/pull/258

0.20.0 (2017-07-12)
-------------------

* Adds tutorial on simulating spectra (`PR #244`_).
* Uses ``desitarget.cuts.isLRG_colors``; requires desitarget >= 0.14.0
  (`PR #246`_).
* Fixes QSO template wavelength extrapolation (`PR #247`_);
  requires desispec > 0.15.1.
* converts to using desiutil.log instead of desispec.log (`PR #248`_).
* Uses ``desiutil.log`` instead of ``desispec.log``.

.. _`PR #244`: https://github.com/desihub/desisim/pull/244
.. _`PR #246`: https://github.com/desihub/desisim/pull/246
.. _`PR #247`: https://github.com/desihub/desisim/pull/247
.. _`PR #248`: https://github.com/desihub/desisim/pull/248

0.19.0 (2017-06-15)
-------------------

* Refactor and speed-up of QSO templates; add Lya forest on-the-fly (`PR #234`_).
* Remove LyA absorption below the LyA limit (`PR #236`_).
* updated changes.rst for #234 (`PR #237`_).
* add notebook comparing various interpolation methods (`PR #239`_).
* Fixes for ``targets.dat`` to ``targets.yaml`` change (`PR #240`_).
* Changed refs to ``desispec.brick`` to its new location at :mod:`desiutil.brick` (`PR #241`_).
* Add ``nocolorcuts`` option for LyA spectra (`PR #242`_).
* "FLAVOR" keyword is arc/flat/science but not dark/bright/bgs/mws/etc to match
  desispec usage (`PR #243`_).

.. _`PR #234`: https://github.com/desihub/desisim/pull/234
.. _`PR #236`: https://github.com/desihub/desisim/pull/236
.. _`PR #237`: https://github.com/desihub/desisim/pull/237
.. _`PR #239`: https://github.com/desihub/desisim/pull/239
.. _`PR #240`: https://github.com/desihub/desisim/pull/240
.. _`PR #241`: https://github.com/desihub/desisim/pull/241
.. _`PR #242`: https://github.com/desihub/desisim/pull/242
.. _`PR #243`: https://github.com/desihub/desisim/pull/243

0.18.3 (2017-04-13)
-------------------

* Add DLAs to lya spectra (`PR #220`_)
* Fix quickgen for specsim v0.8 (`PR #226`_).
* set z_wind=0 in lya_spectra (`PR #228`_).
* Add verbose output to templates code (`PR #230`_).
* do not cache blurmatrix when novdisp=True (`PR #231`_).
* Much faster quickcat (`PR #233`_).

.. _`PR #220`: https://github.com/desihub/desisim/pull/220
.. _`PR #226`: https://github.com/desihub/desisim/pull/226
.. _`PR #228`: https://github.com/desihub/desisim/pull/228
.. _`PR #230`: https://github.com/desihub/desisim/pull/230
.. _`PR #231`: https://github.com/desihub/desisim/pull/231
.. _`PR #233`: https://github.com/desihub/desisim/pull/233

0.18.2 (2017-03-27)
-------------------

* Fixed a number of documentation errors (`PR #224`_).
* fix N^2 scaling of QSO.make_templates (`PR #225`_).
* Removed unneeded Travis scripts in ``etc/``.
* Fixed N^2 scaling of :meth:`desisim.templates.QSO.make_templates`.
* Speed up :class:`desisim.templates.GALAXY` by factor of
  8-12 by caching velocity dispersions (`PR #229`_)

.. _`PR #224`: https://github.com/desihub/desisim/pull/224
.. _`PR #225`: https://github.com/desihub/desisim/pull/225
.. _`PR #229`: https://github.com/desihub/desisim/pull/229

0.18.1 (2016-03-05)
-------------------

* change z-band magnitude prior for LRGs from 19<z<20.5 to 19<z<20.4 (`PR #215`_).
* Update ``desisim.module`` to use :envvar:`DESI_BASIS_TEMPLATES` v2.3.

.. _`PR #215`: https://github.com/desihub/desisim/pull/215

0.18.0 (2016-03-04)
-------------------

* pixsims add new required keywords DOSVER, FEEVER, DETECTOR.
* Debug quickcat (`PR #198`_).
* quicker quickcat maybe (`PR #199`_).
* Small bug fixes in quickcat; drop unused truth,targets columns to save memory
  in quicksurvey loop (PRs #198, #199).
* notebook documenting how to connect BGS/MXXL mock to spectral templates (`PR #202`_).
* added wd templates (`PR #204`_).
* quickgen update to support white dwarf templates (PR #204)
* add option for restframe templates (`PR #208`_).
* several enhancements of the templates code

  * optionally output rest-frame templates (PR #208)
  * rewrite of lya_spectra to achieve factor of 10 speedup; use COSMO
    (astropy.cosmology setup) as a new optional keyword for qso_desi_templates;
    updated API (PRs #210, #212)
  * various small changes to desisim.templates (PR #211)
  * support for DA and DB white dwarf subtypes (PR #213)

* Lyaspeedup (`PR #210`_).
* miscellaneous tweaks, mostly to templates.py (`PR #211`_).
* update the API of lya_spectra to work with mock-->spectra connection (`PR #212`_).
* support DA and DB white dwarf subtypes (`PR #213`_).
* Fix test failures (`PR #214`_).
* update test dependencies (PR #214)

.. _`PR #198`: https://github.com/desihub/desisim/pull/198
.. _`PR #199`: https://github.com/desihub/desisim/pull/199
.. _`PR #202`: https://github.com/desihub/desisim/pull/202
.. _`PR #204`: https://github.com/desihub/desisim/pull/204
.. _`PR #208`: https://github.com/desihub/desisim/pull/208
.. _`PR #210`: https://github.com/desihub/desisim/pull/210
.. _`PR #211`: https://github.com/desihub/desisim/pull/211
.. _`PR #212`: https://github.com/desihub/desisim/pull/212
.. _`PR #213`: https://github.com/desihub/desisim/pull/213
.. _`PR #214`: https://github.com/desihub/desisim/pull/214

0.17.1 (2016-12-05)
-------------------

* fix tiles vs. obsconditions bug; add start_epoch option (`PR #197`_).
* Fix bug when obsconditions contain tiles that don't overlap catalog
* Add ``surveysim --start_epoch`` option

.. _`PR #197`: https://github.com/desihub/desisim/pull/197

0.17.0 (2016-12-02)
-------------------

* Quickgen refactor (`PR #184`_).
* New quickcat interface (`PR #186`_).
* fix tests to work with latest desitarget master (`PR #189`_).
* Add observing conditions to quicksurvey (`PR #190`_).
* Model quickcat for ELGs (`PR #191`_).
* Fix bgs (`PR #192`_).
* fixes zeff (`PR #194`_).
* fixes tests for use with latest desitarget master
* Refactor quickgen and quickbrick to reduce duplicated code (PR #184)
* Makes BGS compatible with desitarget master after
  isBGS -> isBGS_faint vs. isBGS_bright
* Refactor quickcat to include dependency on observing conditions
* Update quicksurvey to use observing conditions from surveysim
* Fixes use of previous zcatalog when updating catalog with new observations

.. _`PR #184`: https://github.com/desihub/desisim/pull/184
.. _`PR #186`: https://github.com/desihub/desisim/pull/186
.. _`PR #189`: https://github.com/desihub/desisim/pull/189
.. _`PR #190`: https://github.com/desihub/desisim/pull/190
.. _`PR #191`: https://github.com/desihub/desisim/pull/191
.. _`PR #192`: https://github.com/desihub/desisim/pull/192
.. _`PR #194`: https://github.com/desihub/desisim/pull/194

0.16.0 (2016-11-10)
-------------------

* Lya sims (`PR #156`_).
* Quickgen moon (`PR #176`_).
* update moon keywords (`PR #177`_).
* Cache specsim (`PR #178`_).
* add integration test for quickgen (`PR #179`_).
* Requires specsim >= v0.6
* Add integration test for quickgen (PR #179)
* Cache specsim Simulator for faster testing (PR #178)
* Add lya_spectra.get_spectra (PR #156)
* Add quickgen and quickbrick unit tests and bug fixes (PR #176, #177)

.. _`PR #156`: https://github.com/desihub/desisim/pull/156
.. _`PR #176`: https://github.com/desihub/desisim/pull/176
.. _`PR #177`: https://github.com/desihub/desisim/pull/177
.. _`PR #178`: https://github.com/desihub/desisim/pull/178
.. _`PR #179`: https://github.com/desihub/desisim/pull/179

0.15.0 (2016-10-14)
-------------------

* Sphinx and coverage fixes (`PR #164`_).
* Get desisim working on ReadTheDocs (`PR #165`_).
* quickbrick tests and bug fixes (`PR #166`_).
* Tiny one-line fix so that function from desisim.pixsim can be called. (`PR #168`_).
* added unit tests to quickgen and updated exposure time (`PR #169`_).
* update module file (`PR #170`_).
* Quickgen (`PR #173`_).
* Quickgen update (`PR #175`_).
* Fix some ``build_sphinx`` errors.
* Run coverage tests under Python 2.7 for now.
* Update template Module file to new DESI+Anaconda infrastructure.
* quickbrick unit tests and bug fixes (#166)
* new quickgen features (PR #173 and #175)

  * fix exptime and airmass for specsim v0.5
  * new --frameonly option
  * moon phase, angle, and zenith options
  * misc cleanup and unit tests

.. _`PR #164`: https://github.com/desihub/desisim/pull/164
.. _`PR #165`: https://github.com/desihub/desisim/pull/165
.. _`PR #166`: https://github.com/desihub/desisim/pull/166
.. _`PR #168`: https://github.com/desihub/desisim/pull/168
.. _`PR #169`: https://github.com/desihub/desisim/pull/169
.. _`PR #170`: https://github.com/desihub/desisim/pull/170
.. _`PR #173`: https://github.com/desihub/desisim/pull/173
.. _`PR #175`: https://github.com/desihub/desisim/pull/175

0.14.0 (2016-09-14)
-------------------

* Contaminants (`PR #150`_).
* fix template fibermap mag; fix quickgen outputs (`PR #151`_).
* 2to3 (`PR #154`_).
* interpolate stellar templates on an input physical grid for the MWS mocks (`PR #155`_).
* change default cdelt in desisim.templates from 2 A/pix to 0.2 A/pix (`PR #157`_).
* Py3 (`PR #159`_).
* updates for python 3.5

.. _`PR #150`: https://github.com/desihub/desisim/pull/150
.. _`PR #151`: https://github.com/desihub/desisim/pull/151
.. _`PR #154`: https://github.com/desihub/desisim/pull/154
.. _`PR #155`: https://github.com/desihub/desisim/pull/155
.. _`PR #157`: https://github.com/desihub/desisim/pull/157
.. _`PR #159`: https://github.com/desihub/desisim/pull/159

0.13.1 (2016-08-18)
-------------------

* fix batch.pixsim seeds vs. seed typo

0.13.0 (2016-08-18)
-------------------

* generate templates at given/input redshifts (`PR #132`_).
* Quickgen update (`PR #145`_).
* QA zfind plot updates (`PR #146`_).
* fix obs.get_next_tileid() for UPPERCASE program in desi-tiles.fits (`PR #147`_).
* add rand seeds for batch pixsim (`PR #148`_).
* desi_qa_zfind: fixed --reduxdir option; improved plots
* PR#132: major refactor of template generation, including ability to give
  input redshifts, magnitudes, or random seeds from metadata table.
* desisim.batch.pixsim functions propagate random seeds for reproducibility

.. _`PR #132`: https://github.com/desihub/desisim/pull/132
.. _`PR #145`: https://github.com/desihub/desisim/pull/145
.. _`PR #146`: https://github.com/desihub/desisim/pull/146
.. _`PR #147`: https://github.com/desihub/desisim/pull/147
.. _`PR #148`: https://github.com/desihub/desisim/pull/148

0.12.0 (2016-07-14)
-------------------

* newexp + pixsim + QA fixes (`PR #144`_).
* desi_qa_zfind options to override raw and processed data directories
* PRODNAME -> SPECPROD and TYPE -> SPECTYPE to match latest desispec
* remove unused get_simstds.py
* fix #142 so that pixsim only optionally runs preprocessing
* fix #141 to avoid repeated TARGETIDs when simulating both
  bright and dark tiles together
* add io.load_simspec_summary() convenience function to load and merge
  truth information from fibermap and simspec files.
* adjusts which magnitudes were plotted for each target class

.. _`PR #144`: https://github.com/desihub/desisim/pull/144

0.11.0 (2016-07-12)
-------------------

* wrapper to construct bright-time bricks (BGS, MWS, or a mixture) (`PR #123`_).
* Rawpixsim (`PR #131`_).
* newexp different tileids in parallel (`PR #133`_).
* fix quickgen mask dtype (`PR #137`_).
* Resource filename use (`PR #139`_).
* use desisim throughput for varying arc lines in cameras (`PR #140`_).

Pixsim updates:

* simulate fully raw data, then call preprocessing
* bug fix for simulating tiles in parallel
* fix pixsim loading of non-default PSFs

.. _`PR #123`: https://github.com/desihub/desisim/pull/123
.. _`PR #131`: https://github.com/desihub/desisim/pull/131
.. _`PR #133`: https://github.com/desihub/desisim/pull/133
.. _`PR #137`: https://github.com/desihub/desisim/pull/137
.. _`PR #139`: https://github.com/desihub/desisim/pull/139
.. _`PR #140`: https://github.com/desihub/desisim/pull/140

0.10.0 (2016-05-19)
-------------------

* adds the option of including a Type Ia supernova spectrum in the BGS, ELG, and LRG spectra (`PR #102`_).
* Quicksurvey (`PR #111`_).
* Redshift truth (`PR #112`_).
* clean up of quickbrick to support bright-time survey (`PR #114`_).
* Wave in vacuum (`PR #116`_).
* Merging test stand simulation options (`PR #117`_).
* Make sure data directory really is part of the package. (`PR #120`_).
* Fix cosmicsims (`PR #122`_).
* Integration with ztruth (`PR #124`_).
* QA edits for cron job (`PR #125`_).
* QA backend for cronjobs (`PR #126`_).
* update template set to v2.2 (`PR #127`_).

.. _`PR #102`: https://github.com/desihub/desisim/pull/102
.. _`PR #111`: https://github.com/desihub/desisim/pull/111
.. _`PR #112`: https://github.com/desihub/desisim/pull/112
.. _`PR #114`: https://github.com/desihub/desisim/pull/114
.. _`PR #116`: https://github.com/desihub/desisim/pull/116
.. _`PR #117`: https://github.com/desihub/desisim/pull/117
.. _`PR #120`: https://github.com/desihub/desisim/pull/120
.. _`PR #122`: https://github.com/desihub/desisim/pull/122
.. _`PR #124`: https://github.com/desihub/desisim/pull/124
.. _`PR #125`: https://github.com/desihub/desisim/pull/125
.. _`PR #126`: https://github.com/desihub/desisim/pull/126
.. _`PR #127`: https://github.com/desihub/desisim/pull/127


0.9.3 (2016-04-08)
------------------

* fix random seed usage with parallelism (`PR #108`_).

.. _`PR #108`: https://github.com/desihub/desisim/pull/108


0.9.1 (2016-03-10)
------------------

* add cosmic ray masking to pixsim (`PR #91`_).
* fix BRIGHT vs bright flavor bug (`PR #92`_).
* Streamline use of specsim by quickgen and quickbrick (`PR #93`_).
* refactor stellar templates, issue #64 (`PR #94`_).
* desisim fix to MWS_STAR colors (`PR #95`_).
* Cleanup (`PR #98`_).
* Batch pixsim (`PR #99`_).

.. _`PR #91`: https://github.com/desihub/desisim/pull/91
.. _`PR #92`: https://github.com/desihub/desisim/pull/92
.. _`PR #93`: https://github.com/desihub/desisim/pull/93
.. _`PR #94`: https://github.com/desihub/desisim/pull/94
.. _`PR #95`: https://github.com/desihub/desisim/pull/95
.. _`PR #98`: https://github.com/desihub/desisim/pull/98
.. _`PR #99`: https://github.com/desihub/desisim/pull/99


0.9 (2016-03-03)
----------------

* In quickbrick: replace actual srcflux in b brick by pre-downsampling … (`PR #63`_).
* Newexp desi flavor (`PR #65`_).
* Refactor QSO_templates (`PR #67`_).
* Bright pixsim (`PR #68`_).
* use bright sky for flavor=bright,bgs,mws (`PR #69`_).
* Testlite (`PR #79`_).
* add BGS templates and speclite & desitarget dependencies (`PR #81`_).
* Quickbrick zrange (`PR #83`_).
* bump numpy/scipy/astropy versions (`PR #84`_).
* quickcat2 (`PR #85`_).
* Update quickgen and quickbrick to new specsim API (`PR #86`_).
* fix mag range of simulated stdstars (`PR #88`_).
* update fibermap format for target masks (`PR #90`_).

.. _`PR #63`: https://github.com/desihub/desisim/pull/63
.. _`PR #65`: https://github.com/desihub/desisim/pull/65
.. _`PR #67`: https://github.com/desihub/desisim/pull/67
.. _`PR #68`: https://github.com/desihub/desisim/pull/68
.. _`PR #69`: https://github.com/desihub/desisim/pull/69
.. _`PR #79`: https://github.com/desihub/desisim/pull/79
.. _`PR #81`: https://github.com/desihub/desisim/pull/81
.. _`PR #83`: https://github.com/desihub/desisim/pull/83
.. _`PR #84`: https://github.com/desihub/desisim/pull/84
.. _`PR #85`: https://github.com/desihub/desisim/pull/85
.. _`PR #86`: https://github.com/desihub/desisim/pull/86
.. _`PR #88`: https://github.com/desihub/desisim/pull/88
.. _`PR #90`: https://github.com/desihub/desisim/pull/90


0.8.2 (2016-01-13)
------------------

* Pip install (`PR #55`_).
* implement specsim into quickgen (`PR #56`_).
* Quickgen specsim (`PR #57`_).
* Modification of quickbrick output in channel b brick (`PR #60`_).
* fixes (more!) random seed issues for ELGs identified in #59 (`PR #61`_).

.. _`PR #55`: https://github.com/desihub/desisim/pull/55
.. _`PR #56`: https://github.com/desihub/desisim/pull/56
.. _`PR #57`: https://github.com/desihub/desisim/pull/57
.. _`PR #60`: https://github.com/desihub/desisim/pull/60
.. _`PR #61`: https://github.com/desihub/desisim/pull/61


0.8 (2015-12-17)
----------------

* Quickbrick2 (`PR #50`_).
* Fix issue/bug #52 and add fiducial velocity broadening to the LRG templates (`PR #53`_).

.. _`PR #50`: https://github.com/desihub/desisim/pull/50
.. _`PR #53`: https://github.com/desihub/desisim/pull/53


0.7 (2015-12-14)
----------------

* fix template I/O unit test (`PR #36`_).
* fix units of object flux (`PR #42`_).
* fix FLUX vs. SKYFLUX units (`PR #45`_).
* fix emission-line strengths in templates and ensure reproducibility (`PR #49`_).

.. _`PR #36`: https://github.com/desihub/desisim/pull/36
.. _`PR #42`: https://github.com/desihub/desisim/pull/42
.. _`PR #45`: https://github.com/desihub/desisim/pull/45
.. _`PR #49`: https://github.com/desihub/desisim/pull/49


0.6 (2015-10-31)
----------------

* newexp-desi integrated with templates generated on-the-fly (`PR #34`_).

.. _`PR #34`: https://github.com/desihub/desisim/pull/34


0.5 (2015-10-30)
----------------

* Cleanup, including pixsim-desi interface change. (`PR #9`_).
* Resolution data added to frame output (`PR #10`_).
* resampling and resolution defined inside (`PR #11`_).
* Cleanup (`PR #14`_).
* fix --nspec usage when nspec>500 (`PR #15`_).
* options updates: redefine --nspec, add --arms and --spectrographs opt… (`PR #16`_).
* Format flow (`PR #17`_).
* Highlevel qa (`PR #18`_).
* quickgen wrapper for quicksim outputs to spectro pipeline formats (`PR #21`_).
* updated to rawdata_root (`PR #22`_).
* First-pass desisim templates (`PR #23`_).
* Cosmics (`PR #24`_).
* Add support for simulating F-type standard-star templates (`PR #25`_).
* Header keywords (`PR #27`_).
* Fix the spectrograph higher than 0 and nspec beyond 500 (`PR #28`_).
* Corrects the number of fiberflats simulated (`PR #30`_).
* add support for generating white dwarf and QSO templates (`PR #31`_).

.. _`PR #9`: https://github.com/desihub/desisim/pull/9
.. _`PR #10`: https://github.com/desihub/desisim/pull/10
.. _`PR #11`: https://github.com/desihub/desisim/pull/11
.. _`PR #14`: https://github.com/desihub/desisim/pull/14
.. _`PR #15`: https://github.com/desihub/desisim/pull/15
.. _`PR #16`: https://github.com/desihub/desisim/pull/16
.. _`PR #17`: https://github.com/desihub/desisim/pull/17
.. _`PR #18`: https://github.com/desihub/desisim/pull/18
.. _`PR #21`: https://github.com/desihub/desisim/pull/21
.. _`PR #22`: https://github.com/desihub/desisim/pull/22
.. _`PR #23`: https://github.com/desihub/desisim/pull/23
.. _`PR #24`: https://github.com/desihub/desisim/pull/24
.. _`PR #25`: https://github.com/desihub/desisim/pull/25
.. _`PR #27`: https://github.com/desihub/desisim/pull/27
.. _`PR #28`: https://github.com/desihub/desisim/pull/28
.. _`PR #30`: https://github.com/desihub/desisim/pull/30
.. _`PR #31`: https://github.com/desihub/desisim/pull/31


0.4.1 (2015-02-13)
------------------

* Qso template (`PR #3`_).

.. _`PR #3`: https://github.com/desihub/desisim/pull/3


0.4 (2015-01-30)
----------------

* Added Directory tree, modified coding style, indentation, dictionary etc (`PR #6`_).

.. _`PR #6`: https://github.com/desihub/desisim/pull/6


0.3.2 (2015-01-16)
------------------

* Desi install (`PR #4`_).

.. _`PR #4`: https://github.com/desihub/desisim/pull/4


0.2.1 (2015-01-02)
------------------

* Sjb/arcflat (`PR #2`_).

.. _`PR #2`: https://github.com/desihub/desisim/pull/2


0.2 (2015-01-02)
----------------

* tie randseed to tileid; scale sky with airmass (`PR #1`_).

.. _`PR #1`: https://github.com/desihub/desisim/pull/1

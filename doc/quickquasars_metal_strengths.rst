==================================
Metal strengths for quickquasars
==================================

``quickquasars --metals`` adds the absorption of metals and of the higher-order Lyman lines,
computed from the Lyα optical depth of the skewers: each absorber gets an optical depth equal
to its strength times the Lyα optical depth, at its own rest-frame wavelength. The strengths are
given with ``--metal-strengths``, one value per entry of ``--metals`` and in the same order.
Without ``--metal-strengths``, the default ``COEF`` of ``absorber_IGM`` in
``desisim.lya_spectra`` is used; SiII and SiIII have no default and always need
``--metal-strengths``.

The strengths depend on the mock suite and on the data they are tuned to. The tunings used for
DESI mocks so far are listed below, so that they can be reproduced without other tools. Mock
productions with lyatools keep the same values in their configurations.

All tunings use the same metals, in this order::

    --metals LYB LY3 LY4 LY5 SiII(1260) SiIII(1207) SiII(1193) SiII(1190)

Tunings
=======

=============================  ======  ======  ======  ======  ==========  ===========  ==========  ==========
Tuning                         LYB     LY3     LY4     LY5     SiII(1260)  SiIII(1207)  SiII(1193)  SiII(1190)
=============================  ======  ======  ======  ======  ==========  ===========  ==========  ==========
LyaCoLoRe and Saclay, DR14     0.1901  0.0697  0.0335  0.0187  4.2504e-4   9.4595e-4    6.3540e-4   4.4960e-4
LyaCoLoRe 2LPT, DR2            0.1901  0.0697  0.0335  0.0187  1.3e-3      3.5e-3       0.7e-3      1.4e-3
Saclay, DR2                    0.1901  0.0697  0.0335  0.0187  5.7e-4      1.6e-3       5.3e-4      6.8e-4
LyaCoLoRe 2LPT, DR2, HCD fix   0.1901  0.0697  0.0335  0.0187  1.80e-3     3.66e-3      1.83e-3     1.89e-3
=============================  ======  ======  ======  ======  ==========  ===========  ==========  ==========

- **LyaCoLoRe and Saclay, DR14**: SiII and SiIII tuned to the DR14 data; the same values were
  used for both mock suites (formerly selected with ``--raw-mock lyacolore`` or
  ``--raw-mock saclay``).
- **LyaCoLoRe 2LPT, DR2** and **Saclay, DR2**: tuned to the DR2 data, used for the DR2 mocks.
- **LyaCoLoRe 2LPT, DR2, HCD fix**: LyaCoLoRe 2LPT tuning redone on the DR2 data biases after the
  HCD absorption was decoupled from the metals (``--dla`` no longer feeds the metal absorption).
  Use it for mocks made with ``--dla`` and this or a later version of desisim.
- LYB, LY3, LY4 and LY5 are the same in every tuning and equal the ``absorber_IGM`` defaults.

For example, the LyaCoLoRe 2LPT tuning after the HCD fix::

    quickquasars ... --dla file \
        --metals LYB LY3 LY4 LY5 'SiII(1260)' 'SiIII(1207)' 'SiII(1193)' 'SiII(1190)' \
        --metal-strengths 0.1901 0.0697 0.0335 0.0187 1.80e-3 3.66e-3 1.83e-3 1.89e-3

Please add a row here when a new tuning is used for a mock production.

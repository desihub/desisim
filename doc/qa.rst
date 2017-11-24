.. _qa:

*****************
Quality Assurance
*****************

Overview
========

The desisim pacakge includes a few scripts for running
QA on the outputs.


Scripts
=======

desi_qa_s2n
+++++++++++

Generate S/N QA for all of the nights in a production.

usage
-----

Here is the usage::

    usage: desi_qa_s2n [-h] [--qafig_root QAFIG_ROOT]

    Generate S/N QA for a production [v0.2.0]

    optional arguments:
      -h, --help            show this help message and exit
      --qafig_root QAFIG_ROOT
                            Root name (and path) of QA figure files (default:
                            None)


examples
--------

Generate the figures::

    desi_qa_s2n

A series of PNG files are created for the various
cameras and object types.


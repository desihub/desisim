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

desi_qa_zfind
+++++++++++++

Generate redshift accuracy QA based on the truth
vs. RedRock outputs.

usage
-----

Here is the usage::

    usage: desi_qa_zfind [-h] [--verbose] [--load_simz_table LOAD_SIMZ_TABLE]
                         [--reduxdir PATH] [--rawdir PATH] [--yaml_file YAML_FILE]
                         [--qafig_path QAFIG_PATH]
                         [--write_simz_table WRITE_SIMZ_TABLE]

    Generate QA on redshift for a production [v0.2.1]

    optional arguments:
      -h, --help            show this help message and exit
      --verbose             Provide verbose reporting of progress. (default:
                            False)
      --load_simz_table LOAD_SIMZ_TABLE
                            Load an existing simz Table to remake figures
                            (default: None)
      --reduxdir PATH       Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                            to processed data. (default: None)
      --rawdir PATH         Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                            to processed data. (default: None)
      --yaml_file YAML_FILE
                            YAML file for debugging (primarily). (default: None)
      --qafig_path QAFIG_PATH
                            Path to where QA figure files are generated. Default
                            is specprod_dir+/QA (default: None)
      --write_simz_table WRITE_SIMZ_TABLE
                            Write simz to this filename (default: None)

example
-------

Here is the typical execution::

    desi_qa_zfind


desi_qa_s2n
+++++++++++

Generate S/N QA for all object types for
all of the nights in a production.
The object types and redshifts are taken from *truth*.

usage
-----

Here is the usage::

    usage: desi_qa_s2n [-h] [--reduxdir PATH] [--qafig_path QAFIG_PATH]

    Generate S/N QA for a production [v0.2.1]

    optional arguments:
      -h, --help            show this help message and exit
      --reduxdir PATH       Override default path ($DESI_SPECTRO_REDUX) to
                            processed data. (default: None)
      --qafig_path QAFIG_PATH
                            Path to where QA figure files are generated. Default
                            is specprod_dir+/QA (default: None)



examples
--------

Generate the figures::

    desi_qa_s2n

A series of PNG files are created for the various
cameras and object types.


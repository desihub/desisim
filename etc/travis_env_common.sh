#!/bin/bash -x

# CONDA
conda create --yes -n test -c astropy-ci-extras python=$PYTHON_VERSION pip
source activate test

# EGG_INFO
# if [[ $SETUP_CMD == egg_info ]]
# then
#   exit  # no more dependencies needed
# fi

# PEP8
if [[ $MAIN_CMD == pep8* ]]
then
  $PIP_INSTALL pep8
  # exit  # no more dependencies needed
fi

# CORE DEPENDENCIES
conda install --yes pytest Cython jinja2 psutil pyyaml requests

# NUMPY scipy
conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION matplotlib

# ASTROPY
if [[ $ASTROPY_VERSION == development ]]
then
  $PIP_INSTALL git+http://github.com/astropy/astropy.git#egg=astropy
  export CONDA_INSTALL="conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"
else
  conda install --yes astropy=$ASTROPY_VERSION
  export CONDA_INSTALL="conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION astropy=$ASTROPY_VERSION"
fi

# Now set up shortcut to conda install command to make sure the Python and Numpy
# versions are always explicitly specified.

# OPTIONAL DEPENDENCIES
if $OPTIONAL_DEPS
then
  $CONDA_INSTALL h5py scikit-image pandas
  $PIP_INSTALL beautifulsoup4
fi

# REQUIRED BY DESI DEPENDENCIES
# $PIP_INSTALL fitsio
# DESI DEPENDENCIES
$PIP_INSTALL git+https://github.com/desihub/desiutil.git@${DESIUTIL_VERSION}#egg=desiutil
$PIP_INSTALL git+https://github.com/desihub/specter.git@${SPECTER_VERSION}#egg=specter
$PIP_INSTALL git+https://github.com/desihub/desispec.git@${DESISPEC_VERSION}#egg=desispec
$PIP_INSTALL git+https://github.com/desihub/desitarget.git@${DESITARGET_VERSION}#egg=desitarget
$PIP_INSTALL speclite

# DESI_ROOT and DESI_BASIS_TEMPLATES with test data
export DESISIM=$PWD
testdata_version=0.3.2
wget https://github.com/desihub/desisim-testdata/archive/$testdata_version.zip
unzip $testdata_version.zip
source desisim-testdata-$testdata_version/setup-testdata.sh

# DESIMODEL
# This gets cloned into desisim/desimodel - is that a problem?
git clone https://github.com/desihub/desimodel
cd desimodel
svn export https://desi.lbl.gov/svn/code/desimodel/branches/test-0.4/data
export DESIMODEL=$PWD
echo DESIMODEL=$DESIMODEL
python setup.py install
cd ..

# DOCUMENTATION DEPENDENCIES
# build_sphinx needs sphinx and matplotlib (for plot_directive). Note that
# this matplotlib will *not* work with py 3.x, but our sphinx build is
# currently 2.7, so that's fine
if [[ $SETUP_CMD == build_sphinx* ]]
then
  $CONDA_INSTALL Sphinx=$SPHINX_VERSION Pygments matplotlib
fi

# COVERAGE DEPENDENCIES
# cpp-coveralls must be installed first.  It installs two identical
# scripts: 'cpp-coveralls' and 'coveralls'.  The latter will overwrite
# the script installed by 'coveralls', unless it's installed first.
if [[ $SETUP_CMD == 'test --coverage' ]]
then
  $PIP_INSTALL cpp-coveralls;
  $PIP_INSTALL coverage coveralls;
fi

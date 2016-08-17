#!/bin/bash -x
set -e
#
# If this script is being run by Travis, the Travis install script will
# already be in the correct directory.
# If this script is being run by desiInstall, then we need to make sure
# we are running this in ${WORKING_DIR}.
#
[[ -n "${WORKING_DIR}" ]] && cd ${WORKING_DIR}
#
# Make sure DESIMODEL_VERSION is set.
#
if [[ -z "${DESIMODEL_DATA}" ]]; then
    echo "DESIMODEL_DATA is not set!"
    exit 1
fi
export DESIMODEL=${HOME}/desimodel/${DESIMODEL_VERSION}
mkdir -p ${DESIMODEL}
svn export https://desi.lbl.gov/svn/code/desimodel/${DESIMODEL_DATA}/data ${DESIMODEL}/data
# echo DESIMODEL=${DESIMODEL}
if [[ -z "${DESISIM_TESTDATA_VERSION}" ]]; then
    echo "DESISIM_TESTDATA_VERSION is not set!"
    exit 1
fi
export DESISIM=$PWD
wget https://github.com/desihub/desisim-testdata/archive/${DESISIM_TESTDATA_VERSION}.zip
unzip ${DESISIM_TESTDATA_VERSION}.zip
source desisim-testdata-${DESISIM_TESTDATA_VERSION}/setup-testdata.sh

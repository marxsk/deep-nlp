#!/bin/bash

TIMESTAMP=`date '+%s'`
OUTPUT="output-${TIMESTAMP}"

REFERENCE=`ls -td -- output-* | head -n1`

if [ -z "$1" ]; then
    echo "You have to enter file with documents to parse"
    exit 1
fi

mkdir -p ${OUTPUT}

python3 pipeline.py $1 ${OUTPUT}
cd ${OUTPUT}
find *pretty -type f -exec md5sum {} + | sort -k 2 > tree.md5
cd ..

diff -c1 ${OUTPUT}/tree.md5 ${REFERENCE}/tree.md5

if [ "$?" -eq "0" ]; then
    # results are same, so there is no need to preserve them
    rm -rf ${OUTPUT}
fi
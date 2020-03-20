#!/bin/bash

## runner FILE PRODUCTION(yes|null)

TIMESTAMP=`date '+%s'`
OUTPUT="output-${TIMESTAMP}"

REFERENCE=`ls -td -- output-* | grep -v output-tmp | head -n1`

if [ -z "$1" ]; then
    echo "You have to enter file with documents to parse"
    exit 1
fi

if [ -z "$2" ]; then
    # we are not in production mode
    OUTPUT="output-tmp"
    # remove old content so we do not compare output of previous builds 
    rm ${OUTPUT}/*
fi

mkdir -p ${OUTPUT}

python3 pipeline.py $1 ${OUTPUT}
cd ${OUTPUT}
find *pretty -type f -exec md5sum {} + | sed 's/-[[:digit:]][[:digit:]]\././' | sort -k 2 > tree.md5
cd ..

wc -l ${OUTPUT}/tree.md5 $1 | head -n2
diff -c1 ${REFERENCE}/tree.md5 ${OUTPUT}/tree.md5

if [ "$?" -eq "0" ]; then
    # results are same, so there is no need to preserve them
    if [ ! -z "$2" ]; then
        rm -rf ${OUTPUT}
    fi
fi

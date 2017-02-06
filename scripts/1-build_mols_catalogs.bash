#!/bin/bash
pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

srcDir="${myDir}/../src"

echo "Welcome to TDT1 Malaria Bootstrapping Script."
echo "We will download and build catalogs for the competition molecules."
echo "Go take a nap, (this took one hour in our machine)."

PYTHONPATH="${srcDir}:${PYTHONPATH}" python -u ${srcDir}/ccl_malaria/molscatalog.py init

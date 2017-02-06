#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "We will organize and make easily available the Morgan fingerprints computed in step 3a."
echo "This is a single-threaded program, really hard in memory!!!"
echo "Go to sleep, (this took around 3 hours in 1 core, with a peak of 55 GB of RAM)."

PYTHONPATH="${myDir}/src:${PYTHONPATH}" python2 -u ${myDir}/src/malaria/features.py munge-ecfps

#!/bin/bash

echo "Welcome to TDT1 Malaria Bootstrapping Script."
echo "We will download and build catalogs for the competition molecules."
echo "Go take a nap, (this took one hour in our machine)."

PYTHONUNBUFFERED=1 ccl-malaria catalog init

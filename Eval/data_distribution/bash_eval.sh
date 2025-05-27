#!/bin/bash

distributions=("cluster" "explosion" "mixed" "rotation")

for dist in "${distributions[@]}"; do
  python Eval/data_distribution/CMIP_distribution.py --distribution="$dist"
done


 
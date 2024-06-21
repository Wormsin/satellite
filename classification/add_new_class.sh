#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_name>"
  exit 1
fi

mkdir "datasets/multi/test/$1"
mkdir "datasets/multi/train/$1"

threshold=" 0.75"
file='classification/metrics.txt'
old_metrics=$(sed -n '2p' "$file")
new_metrics="$old_metrics$threshold"

if [ $? -eq 0 ]; then
  echo "Directory '$1' created successfully."
  sed -i "2s/.*/$new_metrics/" "$file"
else
  echo "Failed to create directory '$1'."
fi




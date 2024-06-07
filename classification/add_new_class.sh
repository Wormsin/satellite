#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_name>"
  exit 1
fi

mkdir "datasets/multi/test/$1"
mkdir "datasets/multi/train/$1"

if [ $? -eq 0 ]; then
  echo "Directory '$1' created successfully."
else
  echo "Failed to create directory '$1'."
fi


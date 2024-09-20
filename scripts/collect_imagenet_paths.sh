#!/bin/bash

original_dir="$PWD"

TARGET_DIR="${1:-$HOME/eagle/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi

current_path=$(pwd)
echo "Current Path: $current_path"

# start timer
SECONDS=0

files=$(find "$TARGET_DIR" -type f)

# date for csv
date=$(date +'%Y-%m-%d')
csv_file="imagenet_paths_${date}.csv"

# switch to where paths will be stored
output_dir="$original_dir/assets"
mkdir -p "$output_dir"

# create csv
echo "path" > "$output_dir/$csv_file"

while IFS= read -r file; do
    echo "\"$file\"" >> "$output_dir/$csv_file"
done <<< "$image_files"

elapsed_time=$SECONDS

echo "CSV file created at: $output_dir/$csv_file"
echo "Time taken: $elapsed_time seconds"

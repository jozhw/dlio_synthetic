#!/bin/bash

original_dir="$PWD"

TARGET_DIR="${1:-/lus/eagle/projects/datasets/ImageNet/ILSVRC/Data}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi

# Convert TARGET_DIR to absolute path
TARGET_DIR=$(readlink -f "$TARGET_DIR")

current_path=$(pwd)
echo "Current Path: $current_path"

# start timer
SECONDS=0

# Change to the target directory
cd "$TARGET_DIR"

files=$(find "$TARGET_DIR" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.JPEG' -o -name '*.JPG' \))

# date for csv
date=$(date +'%Y-%m-%d')
csv_file="imagenet_ilsvrc_paths.csv"

# switch to where paths will be stored
output_dir="$original_dir/assets"
mkdir -p "$output_dir"

# create csv
echo "path" > "$output_dir/$csv_file"

while IFS= read -r file; do
    echo "\"$file\"" >> "$output_dir/$csv_file"
done <<< "$files"

# Change back to the original directory
cd "$original_dir"

elapsed_time=$SECONDS
echo "CSV file created at: $output_dir/$csv_file"
echo "Time taken: $elapsed_time seconds"

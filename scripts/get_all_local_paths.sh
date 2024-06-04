#!/bin/bash


read -p "Which directory in assets would you like to use? " dir_used

# Start timer
start_time=$(date +%s.%N)

# store current working dir
original_dir="$PWD"

current_path=$(pwd)
echo "Current Path: $current_path"


# find image paths given NUM_IMG_PATHS 
jpg_files=$(find ./assets/$dir_used  -type f -name '*.jpg' )

# create an array to store the paths
paths=()

# make dated file
datetime=$(date +'%Y%m%dT%H%M%S')

# name of json file for storage
json_file="$datetime-local-$dir_used-all_imgs.json"


# get image path and store to paths array
for file in $jpg_files; do
    # Get the absolute path of the file
    absolute_path=$(realpath "$file")
    # Add the absolute path to the array
    paths+=("$absolute_path")
done

num_img_paths=${#paths[@]}

# Create directory if it doesn't exist
output_dir="results/image_paths"
mkdir -p "$output_dir"

# Store paths in a temporary file
tmp_file=$(mktemp)
for file in $jpg_files; do
    realpath "$file"
done > "$tmp_file"

# create json file containing all of the paths
jq -Rs '{paths: split("\n") | map(select(. != ""))}' "$tmp_file" > "$output_dir/$json_file"

# Remove the temporary file
rm "$tmp_file"

# End timer
end_time=$(date +%s.%N)

# Calculate elapsed time
execution_time=$(echo "$end_time - $start_time" | bc)

# Print execution time
echo "Execution time: $execution_time seconds for $num_img_paths image paths."


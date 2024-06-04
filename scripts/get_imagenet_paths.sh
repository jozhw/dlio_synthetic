#!/bin/bash

# Start timer
start_time=$(date +%s.%N)

# define a constant for number of image paths to get
num_img_paths=1000

# input number of images to get
read -p "How many image paths would you like to fetch? " num_img_paths

if [[ $num_img_paths =~ ^[0-9]+$ ]]; then
   echo "You entered ${num_img_paths}."
else
   echo "${num_img_paths} is not a number. Please enter a number."
   exit 1
fi

# imagenet path
IMAGENET_PATH="eagle/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"

# Change directory directly instead of using multiple 'cd' commands
IMAGENET_DIR="$HOME/../../$IMAGENET_PATH"

# Check if the directory exists
if [ ! -d "$IMAGENET_DIR" ]; then
    echo "Could not find $IMAGENET_PATH"
    exit 1
fi

# Find image paths given num_img_paths 
jpg_files=$(find "$IMAGENET_DIR" -type f -name '*.JPEG' | shuf -n "$num_img_paths")

datetime=$(date +'%Y%m%dT%H%M%S')

# Generate JSON file name
json_file="$datetime-polaris-imagenet-rand-$num_img_paths.json"

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

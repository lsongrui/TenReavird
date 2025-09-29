#!/bin/bash

# ==============================================================================
# SCRIPT: create_pseudo_dir.sh
#
# AUTHOR: Songrui Li, lsongrui2024@gmail.com
#
# DESCRIPTION:
#   This script creates a "pseudo" directory containing empty files that have
#   the same names as the regular files in a specified original directory.
#   It will not replicate subdirectories, only files.
#
# USAGE:
#   ./create_pseudo_dir.sh <original_directory> <pseudo_directory>
#
# EXAMPLE:
#   ./create_pseudo_dir.sh ./my_data ./my_data_pseudo
#
# ==============================================================================

# --- 1. Argument Validation ---
# Check if exactly two arguments were provided.
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments." >&2
    echo "Usage: $0 <original_directory> <pseudo_directory>" >&2
    exit 1
fi

# --- 2. Assign Arguments to Descriptive Variables ---
# Using quotes to handle potential spaces in directory names.
ORIG_DIR="$1"
PSEUDO_DIR="$2"

# --- 3. Check if the Original Directory Exists ---
if [ ! -d "$ORIG_DIR" ]; then
    echo "Error: The source directory '$ORIG_DIR' does not exist." >&2
    exit 1
fi

# --- 4. Create the Pseudo Directory ---
# The '-p' flag creates parent directories if needed and does not
# return an error if the directory already exists.
echo "Creating destination directory: '$PSEUDO_DIR'"
mkdir -p "$PSEUDO_DIR"

# --- 5. Main Logic: Loop and Create Empty Files ---
echo "Processing files from '$ORIG_DIR'..."

# Initialize a counter for the created files.
file_count=0

# Loop through all items in the original directory.
# Using '/*' will list all files and directories at the top level.
for filepath in "$ORIG_DIR"/*; do
    # Check if the item is a regular file (and not a directory or other special file).
    if [ -f "$filepath" ]; then
        # Extract just the filename from the full path (e.g., "dir/file.txt" -> "file.txt").
        filename=$(basename "$filepath")

        # Create an empty file with the same name in the pseudo directory.
        # The 'touch' command is perfect for this.
        touch "$PSEUDO_DIR/$filename"

        # Increment the counter.
        ((file_count++))
    fi
done

# --- 6. Final Report ---
echo "Done. Created $file_count empty pseudo-files in '$PSEUDO_DIR'."
exit 0
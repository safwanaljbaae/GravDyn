#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="Data"

KEEP_NAMES=(
  "Pot_Expansion"
  "modified_v.dat"
  "modified_f.dat"
  "shape_f.dat"
  "shape_v.dat"
  "shape_verification.log"
)

for folder in "$DATA_DIR"/*/; do
    echo "Cleaning: $folder"

    find_args=()

    for name in "${KEEP_NAMES[@]}"; do
        find_args+=( ! -name "$name" )
    done

    find "$folder" -mindepth 1 -maxdepth 1 "${find_args[@]}" -exec rm -rf {} +
done

echo "Done."

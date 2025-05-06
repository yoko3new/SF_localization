#!/bin/bash

# run_all_batches.sh
# This script runs multiple batches of AIA image downloads in parallel using the JSOC client.
# It splits a flare event list into multiple index ranges and downloads each range concurrently.

# -------------------- Define batch ranges -------------------- #
# Each string represents a start and end index for a subset of flare events.
batches=(
  "0 100"
  "100 200"
  "200 300"
  "300 400"
  "400 500"
  "500 561"
)

# -------------------- Launch parallel download tasks -------------------- #
# Iterate over each batch and run the download script in the background.
for batch in "${batches[@]}"; do
    read start end <<< "$batch"
    echo "▶️ Running batch: $start to $end"
    python scripts/download_aia_images.py --start "$start" --end "$end" &
done

# -------------------- Wait for all background tasks -------------------- #
wait
echo "All batches finished."

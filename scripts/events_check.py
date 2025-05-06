import os

# ---------- Path Configuration ---------- #
DIFF_OUTPUT_DIR = "/data/kyang30/flare_localization/diff_images"
HEATMAP_OUTPUT_DIR = "/data/kyang30/flare_localization/hek_heatmap"
OUTPUT_LIST_PATH = "available_events.txt"

# ---------- Initialize List of Valid Events ---------- #
valid_events = []

# ---------- Traverse All Event Directories ---------- #
# We only retain events that contain both a diff.npy and a heatmap.npy
for event_id in sorted(os.listdir(DIFF_OUTPUT_DIR)):
    diff_path = os.path.join(DIFF_OUTPUT_DIR, event_id, "diff.npy")
    heatmap_path = os.path.join(HEATMAP_OUTPUT_DIR, event_id, "heatmap.npy")

    if os.path.exists(diff_path) and os.path.exists(heatmap_path):
        valid_events.append(event_id)

print(f"Found {len(valid_events)} events with both diff.npy and heatmap.npy")

# ---------- Save Valid Event IDs ---------- #
with open(OUTPUT_LIST_PATH, "w") as f:
    for event_id in valid_events:
        f.write(f"{event_id}\n")

print(f"Saved valid event list to: {OUTPUT_LIST_PATH}")

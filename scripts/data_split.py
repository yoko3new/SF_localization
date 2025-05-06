import os
import random

# ---------- Configuration ---------- #
available_events_path = "available_events.txt"  # Input: list of valid event IDs, one per line
split_dir = "splits"
os.makedirs(split_dir, exist_ok=True)

# ---------- Load Event IDs ---------- #
with open(available_events_path, "r") as f:
    event_ids = [line.strip() for line in f.readlines() if line.strip()]

# Ensure reproducibility of the split
random.seed(42)
random.shuffle(event_ids)

total = len(event_ids)

# ---------- Dataset Split Ratios ---------- #
# - 20% for train_labeled.txt
# - 10% for val_labeled.txt
# - 50% for pseudo_unlabeled.txt
# - 20% for test_labeled.txt
# Note: final group is calculated to ensure all samples are used (no rounding issues)

n_train = int(total * 0.2)
n_val = int(total * 0.1)
n_pseudo = int(total * 0.5)
n_test = total - n_train - n_val - n_pseudo

train_ids = event_ids[:n_train]
val_ids = event_ids[n_train:n_train + n_val]
pseudo_ids = event_ids[n_train + n_val:n_train + n_val + n_pseudo]
test_ids = event_ids[n_train + n_val + n_pseudo:]

# ---------- Save Splits to Files ---------- #
with open(os.path.join(split_dir, "train_labeled.txt"), "w") as f:
    f.writelines([eid + "\n" for eid in train_ids])

with open(os.path.join(split_dir, "val_labeled.txt"), "w") as f:
    f.writelines([eid + "\n" for eid in val_ids])

with open(os.path.join(split_dir, "pseudo_unlabeled.txt"), "w") as f:
    f.writelines([eid + "\n" for eid in pseudo_ids])

with open(os.path.join(split_dir, "test_labeled.txt"), "w") as f:
    f.writelines([eid + "\n" for eid in test_ids])

# ---------- Summary ---------- #
print("Dataset split complete.")
print(f"Train labeled: {len(train_ids)}")
print(f"Val labeled: {len(val_ids)}")
print(f"Pseudo unlabeled: {len(pseudo_ids)}")
print(f"Test labeled: {len(test_ids)}")

# scripts/concat_joint_split.py
with open("splits/train_labeled.txt", "r") as f1, open("splits/pseudo_selected.txt", "r") as f2:
    lines = f1.readlines() + f2.readlines()

with open("splits/joint_train.txt", "w") as f:
    f.writelines(lines)

print(f" joint_train.txt saved with {len(lines)} entries.")
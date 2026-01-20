import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("MFCC_Features/dataset.npz", allow_pickle=True)
X = data["X"]                 # (N, time, n_mfcc)
y = data["y"]                 # (N,)
class_names = data["class_names"]

np.random.seed(0)  # reproducible randomness

num_classes = len(class_names)
samples_per_class = 3

plt.figure(figsize=(samples_per_class * 4, num_classes * 2.5))

plot_idx = 1

for class_id, class_name in enumerate(class_names):
    indices = np.where(y == class_id)[0]

    if len(indices) == 0:
        continue

    # Pick up to 3 random samples from this class
    chosen = np.random.choice(
        indices,
        size=min(samples_per_class, len(indices)),
        replace=False
    )

    for j, idx in enumerate(chosen):
        mfcc = X[idx]

        # Normalize for visualization only
        mfcc_vis = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-6)

        plt.subplot(num_classes, samples_per_class, plot_idx)
        plt.imshow(mfcc_vis.T, aspect="auto", origin="lower")
        plt.title(f"{class_name} #{j+1}", fontsize=9)
        plt.xlabel("Time")
        if j == 0:
            plt.ylabel("MFCC")
        else:
            plt.yticks([])

        plot_idx += 1

plt.suptitle("MFCC Feature Maps – 3 Random Samples per Class", fontsize=14)
plt.tight_layout()
plt.show()

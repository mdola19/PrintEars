import numpy as np
from pathlib import Path
import librosa

# -------- CONFIG --------
RAW_DIR = Path("Raw PCM")
OUT_DIR = Path("MFCC_Features")

CLASSES = ["motion_error", "z_slip", "normal_extrusion", "normal", "clogged_extrusion"]

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
TARGET_FRAMES = 100
# ------------------------


def infer_label(filename: str) -> str:
    name = filename.lower()
    for c in CLASSES:
        if c in name:
            return c
    return ""


def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "per_clip").mkdir(exist_ok=True)

    X, y = [], []
    class_to_id = {c: i for i, c in enumerate(CLASSES)}

    for f in sorted(RAW_DIR.glob("*.csv")):
        label = infer_label(f.name)
        if not label:
            print(f"SKIP (no label): {f.name}")
            continue

        # Load: row 0 = timesteps, row 1 = PCM values
        arr = np.loadtxt(f, delimiter=",", dtype=np.float32)
        audio = arr[1]  # <- THIS is the only audio we want

        # Convert int16 PCM to float [-1, 1] (safe if already scaled)
        audio = audio / 32768.0

        # Normalize
        m = np.max(np.abs(audio))
        if m > 0:
            audio = audio / m

        # MFCC (time, n_mfcc)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).T

        # Pad/crop time frames to fixed length for CNN batching
        T = mfcc.shape[0]
        if T > TARGET_FRAMES:
            mfcc = mfcc[:TARGET_FRAMES]
        elif T < TARGET_FRAMES:
            mfcc = np.vstack([mfcc, np.zeros((TARGET_FRAMES - T, N_MFCC), dtype=np.float32)])

        mfcc = mfcc.astype(np.float32)

        # Save per-clip MFCC
        np.save(OUT_DIR / "per_clip" / f"{f.stem}.npy", mfcc)

        # Build dataset arrays
        X.append(mfcc)
        y.append(class_to_id[label])

        print(f"OK: {f.name} -> {label}, mfcc={mfcc.shape}")

    # Save one dataset file for training
    X = np.stack(X, axis=0)   # (N, frames, n_mfcc)
    y = np.array(y, dtype=np.int64)

    np.savez_compressed(OUT_DIR / "dataset.npz", X=X, y=y, class_names=np.array(CLASSES))
    print("\nSaved MFCC_Features/dataset.npz")
    print("Done.")


if __name__ == "__main__":
    main()

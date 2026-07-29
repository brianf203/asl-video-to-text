import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def main(npy_path):
    data = np.load(npy_path)
    print(f"Loaded {npy_path}, shape: {data.shape}")

    fig, ax = plt.subplots(figsize=(6, 8))

    def draw_frame(frame_idx):
        ax.clear()
        frame = data[frame_idx]

        pose = frame[0:33]
        right_hand = frame[33:54]
        left_hand = frame[54:75]

        pose_mask = ~np.all(pose == 0, axis=1)
        ax.scatter(pose[pose_mask, 0], -pose[pose_mask, 1], c='blue', s=20)
        for a, b in POSE_CONNECTIONS:
            if pose_mask[a] and pose_mask[b]:
                ax.plot([pose[a, 0], pose[b, 0]], [-pose[a, 1], -pose[b, 1]], 'b-')

        rh_mask = ~np.all(right_hand == 0, axis=1)
        if rh_mask.any():
            ax.scatter(right_hand[rh_mask, 0], -right_hand[rh_mask, 1], c='red', s=15)
            for a, b in HAND_CONNECTIONS:
                if rh_mask[a] and rh_mask[b]:
                    ax.plot([right_hand[a, 0], right_hand[b, 0]], [-right_hand[a, 1], -right_hand[b, 1]], 'r-')

        lh_mask = ~np.all(left_hand == 0, axis=1)
        if lh_mask.any():
            ax.scatter(left_hand[lh_mask, 0], -left_hand[lh_mask, 1], c='green', s=15)
            for a, b in HAND_CONNECTIONS:
                if lh_mask[a] and lh_mask[b]:
                    ax.plot([left_hand[a, 0], left_hand[b, 0]], [-left_hand[a, 1], -left_hand[b, 1]], 'g-')

        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 0)
        ax.set_title(f"Frame {frame_idx+1}/{data.shape[0]}")

    ani = animation.FuncAnimation(fig, draw_frame, frames=data.shape[0], interval=100, repeat=True)
    plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "live_sample.npy"
    main(path)

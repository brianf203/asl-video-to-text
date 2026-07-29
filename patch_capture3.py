with open('capture_live_onnx.py', 'r') as f:
    lines = f.readlines()

lines[93] = "def extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector, prev_landmarks=None):\n"
lines[94] = "    if prev_landmarks is not None:\n        frame_landmarks = prev_landmarks.copy()\n    else:\n        frame_landmarks = np.zeros((543, 2), dtype=np.float32)\n"

insert_idx = 147
lines.insert(insert_idx, "    prev_landmarks = None\n")

for i, line in enumerate(lines):
    if "landmarks = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector)" in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = f"{indent}landmarks = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector, prev_landmarks)\n{indent}prev_landmarks = landmarks\n"
        break

with open('capture_live_onnx.py', 'w') as f:
    f.writelines(lines)

print("Patched successfully")

with open('capture_live_onnx.py', 'r') as f:
    lines = f.readlines()

func_start = None
func_end = None
for i, line in enumerate(lines):
    if line.startswith("def extract_frame_landmarks"):
        func_start = i
    if func_start is not None and "return frame_landmarks" in line:
        func_end = i
        break

new_func = '''def extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector, prev_landmarks=None, miss_counts=None, max_misses=5):
    if prev_landmarks is not None:
        frame_landmarks = prev_landmarks.copy()
    else:
        frame_landmarks = np.zeros((543, 2), dtype=np.float32)

    if miss_counts is None:
        miss_counts = {'right': 0, 'left': 0}

    h, w, _ = frame.shape
    persons = person_detector.infer(frame)
    if len(persons) > 0:
        pose_result = pose_detector.infer(frame, persons[0])
        if pose_result is not None:
            pose_landmarks = pose_result[1]
            for i in range(33):
                frame_landmarks[i][0] = pose_landmarks[i][0] / w
                frame_landmarks[i][1] = pose_landmarks[i][1] / h

    right_seen = False
    left_seen = False

    palms = palm_detector.infer(frame)
    for palm in palms:
        hand_result = hand_detector.infer(frame, palm)
        if hand_result is not None:
            handedness = hand_result[130]
            landmarks_2d = np.array([[hand_result[4 + i*3] / w, hand_result[4 + i*3 + 1] / h] for i in range(21)])
            if handedness > 0.5:
                for i in range(21):
                    frame_landmarks[33 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[33 + i][1] = landmarks_2d[i][1]
                right_seen = True
                miss_counts['right'] = 0
            else:
                for i in range(21):
                    frame_landmarks[54 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[54 + i][1] = landmarks_2d[i][1]
                left_seen = True
                miss_counts['left'] = 0

    if not right_seen:
        miss_counts['right'] += 1
        if miss_counts['right'] > max_misses:
            frame_landmarks[33:54] = 0
    if not left_seen:
        miss_counts['left'] += 1
        if miss_counts['left'] > max_misses:
            frame_landmarks[54:75] = 0

    return frame_landmarks, miss_counts
'''

lines[func_start:func_end+1] = [new_func]

with open('capture_live_onnx.py', 'w') as f:
    f.writelines(lines)

print(f"Replaced function spanning original lines {func_start+1}-{func_end+1}")

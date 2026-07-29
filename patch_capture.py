with open('capture_live_onnx.py', 'r') as f:
    content = f.read()

old_func = '''def extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector):
    frame_landmarks = np.zeros((543, 2), dtype=np.float32)
    h, w, _ = frame.shape
    persons = person_detector.infer(frame)
    if len(persons) > 0:
        pose_result = pose_detector.infer(frame, persons[0])
        if pose_result is not None:
            pose_landmarks = pose_result[1]
            for i in range(33):
                frame_landmarks[i][0] = pose_landmarks[i][0] / w
                frame_landmarks[i][1] = pose_landmarks[i][1] / h
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
            else:
                for i in range(21):
                    frame_landmarks[54 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[54 + i][1] = landmarks_2d[i][1]
    return frame_landmarks'''

new_func = '''def extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector, prev_landmarks=None):
    if prev_landmarks is not None:
        frame_landmarks = prev_landmarks.copy()
    else:
        frame_landmarks = np.zeros((543, 2), dtype=np.float32)
    h, w, _ = frame.shape
    persons = person_detector.infer(frame)
    if len(persons) > 0:
        pose_result = pose_detector.infer(frame, persons[0])
        if pose_result is not None:
            pose_landmarks = pose_result[1]
            for i in range(33):
                frame_landmarks[i][0] = pose_landmarks[i][0] / w
                frame_landmarks[i][1] = pose_landmarks[i][1] / h
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
            else:
                for i in range(21):
                    frame_landmarks[54 + i][0] = landmarks_2d[i][0]
                    frame_landmarks[54 + i][1] = landmarks_2d[i][1]
    return frame_landmarks'''

if old_func not in content:
    print("ERROR: old function text not found exactly, aborting to avoid corrupting file")
else:
    content = content.replace(old_func, new_func)

    old_loop_init = '''    is_recording = False
    recorded_frames = []'''
    new_loop_init = '''    is_recording = False
    recorded_frames = []
    prev_landmarks = None'''
    content = content.replace(old_loop_init, new_loop_init, 1)

    old_call = '''        landmarks = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector)'''
    new_call = '''        landmarks = extract_frame_landmarks(frame, palm_detector, hand_detector, person_detector, pose_detector, prev_landmarks)
        prev_landmarks = landmarks'''
    content = content.replace(old_call, new_call, 1)

    with open('capture_live_onnx.py', 'w') as f:
        f.write(content)
    print("Patched successfully")

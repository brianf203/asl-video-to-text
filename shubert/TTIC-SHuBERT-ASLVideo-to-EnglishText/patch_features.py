with open('features.py', 'r') as f:
    content = f.read()

old_import = "import decord\n"
new_import = "import cv2\n"
content = content.replace(old_import, new_import, 1)

old_block = '''        signer_video = decord.VideoReader(video_path)
        
        signer_video_fps = signer_video.get_avg_fps()
        # target_fps = 12
        # stride = max(1, int(round(signer_video_fps / target_fps)))
        stride = 1
        index_list = list(range(0, len(signer_video), stride))
        signer_video = signer_video.get_batch(index_list)
        signer_video = signer_video.asnumpy()'''

new_block = '''        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        signer_video = np.array(frames)'''

if old_block not in content:
    print("ERROR: exact block not found, need to inspect file more closely")
else:
    content = content.replace(old_block, new_block)
    with open('features.py', 'w') as f:
        f.write(content)
    print("Patched successfully - decord replaced with cv2")

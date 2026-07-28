import cv2
import numpy as np

# 1. Load the landmark data
landmarks = np.load('live_sample.npy')
print(f"Playing skeletal data with shape: {landmarks.shape}")

# Canvas settings (Width, Height)
canvas_width = 800
canvas_height = 800

# 2. Loop through each of the 88 frames
for frame_idx, frame_landmarks in enumerate(landmarks):
    # Create a solid black canvas for this frame
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 3. Loop through and draw all 543 coordinates
    for pt in frame_landmarks:
        x, y = pt[0], pt[1]
        
        # Check if coordinates are normalized (between 0 and 1)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            # Scale normalized coordinates to fit the canvas size
            pixel_x = int(x * canvas_width)
            pixel_y = int(y * canvas_height)
        else:
            # Coordinates are already in absolute pixel values
            pixel_x, pixel_y = int(x), int(y)
            
        # Ensure points fall inside the window boundaries before drawing
        if 0 <= pixel_x < canvas_width and 0 <= pixel_y < canvas_height:
            # Draw a small green dot for each tracking point
            cv2.circle(canvas, (pixel_x, pixel_y), 3, (0, 255, 0), -1)
            
    # Add a frame counter overlay
    cv2.putText(canvas, f"Frame: {frame_idx+1}/88", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the animated frame
    cv2.imshow('MediaPipe Landmark Playback', canvas)
    
    # Slow down or speed up playback by changing the waitKey value (ms)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

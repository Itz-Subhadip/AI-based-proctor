'''import cv2
import numpy as np

# Initialize the camera
camera = cv2.VideoCapture(0)

# Function to detect head movement
def detect_head_movement(prev_frame, curr_frame, threshold):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of optical flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate average magnitude as a measure of movement
    avg_magnitude = np.mean(magnitude)
    
    return avg_magnitude

# Main loop
threshold = 5  # Define the threshold for head movement detection
previous_frame = None  # Initialize previous frame
while True:
    # Capture current frame
    ret, frame = camera.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Detect head movement
    if previous_frame is not None:
        head_movement = detect_head_movement(previous_frame, frame, threshold)
        if head_movement > threshold:
            print("Head movement detected!")
    
    # Update previous frame
    previous_frame = frame.copy()
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
'''


import cv2
import numpy as np
import pyaudio
import audioop

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize audio variables
audio_chunk_size = 1024
audio_format = pyaudio.paInt16
audio_channels = 1
audio_rate = 44100

# Initialize audio stream
audio = pyaudio.PyAudio()
audio_stream = audio.open(format=audio_format, channels=audio_channels,
                          rate=audio_rate, input=True,
                          frames_per_buffer=audio_chunk_size)

# Function to detect head movement
def detect_head_movement(prev_frame, curr_frame, threshold):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of optical flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate average magnitude as a measure of movement
    avg_magnitude = np.mean(magnitude)
    
    return avg_magnitude

# Function to detect background noise level
def detect_background_noise():
    audio_data = audio_stream.read(audio_chunk_size)
    rms = audioop.rms(audio_data, 2)  # Calculate Root Mean Square (RMS) value
    return rms

# Main loop
threshold = 5  # Define the threshold for head movement detection
previous_frame = None  # Initialize previous frame
while True:
    # Capture current frame
    ret, frame = camera.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Detect head movement
    if previous_frame is not None:
        head_movement = detect_head_movement(previous_frame, frame, threshold)
        if head_movement > threshold:
            print("Head movement detected!")
    
    # Detect background noise
    noise_level = detect_background_noise()
    print("Background noise level:", noise_level)
    # Update previous frame
    previous_frame = frame.copy()
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close audio stream and terminate PyAudio
audio_stream.stop_stream()
audio_stream.close()
audio.terminate()

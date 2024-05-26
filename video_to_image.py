import cv2
import os

# Define the path to the input video file
video_file = './videos/7.mp4'

# Create a directory to save the extracted frames
output_dir = './tmp/frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Start frame counter
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    # Display the frame number
    print(f"Extracted frame {frame_count}")

    # Increment frame counter
    frame_count += 1

# Release the video capture object
cap.release()

print("Frames extraction completed.")


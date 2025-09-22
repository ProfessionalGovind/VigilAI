"""
VigilAI Foundation Script
Basic video processing with YOLOv8 model loading and frame display
This is the foundational code before adding advanced features
"""

# ==========================================
# IMPORTS - All necessary libraries
# ==========================================

import cv2                    # OpenCV for video processing and display
import os                     # For file path operations
from ultralytics import YOLO  # YOLOv8 model from ultralytics
import time                   # For timing and performance measurement

# ==========================================
# CONFIGURATION - Basic settings
# ==========================================

# Video file path - change this to your video file
VIDEO_PATH = "data/videos/test_video_720p_1.mp4"  # Update this path

# Model path - YOLOv8 model file
MODEL_PATH = "models/yolov8n.pt"  # nano model (fastest, smaller)

# Display window settings
WINDOW_NAME = "VigilAI - Video Processing"
WINDOW_WIDTH = 800   # Resize window width for display
WINDOW_HEIGHT = 600  # Resize window height for display

# ==========================================
# STEP 1: Load YOLOv8 Model
# ==========================================

print("Loading YOLOv8 model...")

# Check if model file exists, if not YOLO will download it automatically
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    print("YOLOv8 will download the model automatically...")
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load the YOLO model
# This will download yolov8n.pt automatically if not found
model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model

print("✓ YOLOv8 model loaded successfully!")

# Display model information
print(f"Model classes: {len(model.names)} classes")
print(f"Class names: {list(model.names.values())[:10]}...")  # Show first 10 classes

# ==========================================
# STEP 2: Open Video File
# ==========================================

print(f"\nOpening video file: {VIDEO_PATH}")

# Check if video file exists
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: Video file not found at {VIDEO_PATH}")
    print("Please:")
    print("1. Create the 'data/videos/' directory")
    print("2. Add your video file to that directory") 
    print("3. Update VIDEO_PATH in the script")
    exit()

# Open the video file using OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"❌ Error: Could not open video file {VIDEO_PATH}")
    print("Make sure the video file is not corrupted and is in a supported format")
    exit()

print("✓ Video file opened successfully!")

# ==========================================
# STEP 3: Get Video Properties
# ==========================================

# Get video properties for information and processing
fps = int(cap.get(cv2.CAP_PROP_FPS))                    # Frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # Total number of frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Frame width in pixels
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height in pixels
duration = frame_count / fps                            # Video duration in seconds

# Display video information
print(f"\n📹 Video Information:")
print(f"   Resolution: {frame_width}x{frame_height}")
print(f"   FPS: {fps}")
print(f"   Total Frames: {frame_count}")
print(f"   Duration: {duration:.2f} seconds")

# ==========================================
# STEP 4: Create Display Window
# ==========================================

# Create a named window for displaying the video
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Resize the window to fit the screen better
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

# ==========================================
# STEP 5: Main Video Processing Loop
# ==========================================

print(f"\n🚀 Starting video processing...")
print("Controls:")
print("   SPACE: Pause/Resume")
print("   Q: Quit")
print("   S: Save current frame")
print("   ESC: Exit")

# Initialize variables for loop
current_frame = 0      # Counter for current frame number
paused = False         # Flag to track if video is paused
start_time = time.time()  # Start time for performance measurement

# Main loop - iterate through each frame
while True:
    
    # ==========================================
    # STEP 5A: Read Next Frame
    # ==========================================
    
    if not paused:
        # Read the next frame from video
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print("\n✓ End of video reached or error reading frame")
            break
        
        current_frame += 1
    
    # ==========================================
    # STEP 5B: Process Frame with YOLO (Basic)
    # ==========================================
    
    # Run YOLO inference on the current frame
    # This detects objects in the frame
    results = model(frame, verbose=False)  # verbose=False to reduce console output
    
    # Get the annotated frame with detection boxes drawn
    # results[0].plot() draws bounding boxes and labels on the frame
    annotated_frame = results[0].plot()
    
    # ==========================================
    # STEP 5C: Add Information Overlay
    # ==========================================
    
    # Add frame information as text overlay on the video
    info_text = f"Frame: {current_frame}/{frame_count}"
    progress_text = f"Progress: {(current_frame/frame_count)*100:.1f}%"
    
    # Add text to the frame
    # cv2.putText(image, text, position, font, scale, color, thickness)
    cv2.putText(annotated_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, progress_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add pause indicator if video is paused
    if paused:
        cv2.putText(annotated_frame, "PAUSED - Press SPACE to resume", 
                   (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
    
    # ==========================================
    # STEP 5D: Display Frame
    # ==========================================
    
    # Show the annotated frame in the window
    cv2.imshow(WINDOW_NAME, annotated_frame)
    
    # ==========================================
    # STEP 5E: Handle User Input
    # ==========================================
    
    # Wait for key press (1ms timeout for smooth video playback)
    key = cv2.waitKey(1) & 0xFF
    
    # Handle different key presses
    if key == ord('q') or key == 27:  # 'q' or ESC key
        print("\n🛑 Stopping video processing...")
        break
        
    elif key == ord(' '):  # SPACE key - pause/resume
        paused = not paused
        if paused:
            print("⏸️  Video paused - Press SPACE to resume")
        else:
            print("▶️  Video resumed")
            
    elif key == ord('s'):  # 's' key - save current frame
        # Create output directory if it doesn't exist
        os.makedirs("data/output", exist_ok=True)
        
        # Save current frame as image file
        filename = f"data/output/frame_{current_frame:06d}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"💾 Saved frame to: {filename}")
    
    # ==========================================
    # STEP 5F: Performance Information (Optional)
    # ==========================================
    
    # Show progress every 100 frames
    if current_frame % 100 == 0:
        elapsed_time = time.time() - start_time
        processing_fps = current_frame / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Processing... Frame {current_frame}/{frame_count} "
              f"({processing_fps:.1f} FPS)")

# ==========================================
# STEP 6: Cleanup and Exit
# ==========================================

print("\n🔄 Cleaning up...")

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Calculate and display final statistics
end_time = time.time()
total_time = end_time - start_time
avg_fps = current_frame / total_time if total_time > 0 else 0

print(f"\n📊 Processing Complete!")
print(f"   Processed Frames: {current_frame}")
print(f"   Total Time: {total_time:.2f} seconds") 
print(f"   Average FPS: {avg_fps:.2f}")
print(f"   Video FPS: {fps}")

print("\n✅ VigilAI Foundation Script completed successfully!")

# ==========================================
# NEXT STEPS (Comments for future development)
# ==========================================

"""
🚀 NEXT DEVELOPMENT STEPS:

1. ADD PERSON-SPECIFIC DETECTION:
   - Filter YOLO results to only show 'person' class (class ID = 0)
   - Count number of persons in each frame
   - Add person counter overlay

2. ADD TRACKING:
   - Implement person tracking across frames
   - Assign unique IDs to each person
   - Track person movement paths

3. ADD ANALYTICS:
   - Calculate crowd density
   - Detect crowding hotspots  
   - Generate statistical reports

4. ADD ZONE ANALYSIS:
   - Define regions of interest
   - Count people in specific zones
   - Alert on zone capacity limits

5. ADD EXPORT FEATURES:
   - Save processed video with annotations
   - Export detection data to CSV/JSON
   - Generate analytical reports

6. OPTIMIZE PERFORMANCE:
   - Add GPU acceleration
   - Implement frame skipping for faster processing
   - Add multi-threading support
"""
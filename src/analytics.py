import numpy as np
import cv2

class Heatmap:
    """
    My class for generating and updating a crowd density heatmap.
    It takes the bounding box locations and builds a visual map.
    """
    def __init__(self, frame_size, decay_rate=0.98):
        # I'm creating a blank heatmap layer that will have the same dimensions as my video.
        self.heatmap = np.zeros(frame_size, dtype=np.float32)
        self.decay_rate = decay_rate
        self.frame_size = frame_size

    def update(self, detections):
        # This is where the magic happens. I'll update the heatmap with new detections.
        # I'm applying a decay so that older detections fade over time.
        self.heatmap *= self.decay_rate
        
        for detection in detections:
            # Getting the center point of each bounding box.
            x1, y1, x2, y2 = detection['box']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Adding a "hotspot" to the heatmap at the center of the detection.
            # The stronger the value, the brighter the spot will be.
            cv2.circle(self.heatmap, (center_x, center_y), 30, 1.0, -1, lineType=cv2.LINE_AA)

    def draw(self, frame):
        # Now I'll create a colored version of my heatmap and overlay it on the video frame.
        # I'll normalize the heatmap so the values are between 0 and 255 for the color map.
        normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized_heatmap = np.uint8(normalized_heatmap)
        
        # Applying a color map to the heatmap.
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # I'll blend the heatmap with the original frame for a cool semi-transparent effect.
        output_frame = cv2.addWeighted(frame, 0.7, colored_heatmap, 0.3, 0)
        return output_frame
    
class FlowAnalyzer:
    """
    My class for a simple flow analysis. I'll track people as they move and
    count them as they cross a specific line.
    """
    def __init__(self, line_y_coord, line_direction='up'):
        # I'm setting up a line to count people as they cross it.
        self.line_y_coord = line_y_coord
        self.line_direction = line_direction
        self.tracked_objects = {}  # A dictionary to hold the last known position of each person.
        self.person_count = 0  # My counter.
    
    def update(self, detections):
        # This is where I'll match new detections to the people I'm already tracking.
        # This is a very basic method, but it's enough for a simple demo.
        
        # A list to hold the center points of the new detections.
        current_centroids = {i: self._get_centroid(d) for i, d in enumerate(detections)}
        
        for person_id, old_centroid in list(self.tracked_objects.items()):
            # Find the new detection closest to the old centroid.
            new_centroid_id = self._find_closest_centroid(old_centroid, current_centroids)
            
            if new_centroid_id is not None:
                new_centroid = current_centroids[new_centroid_id]
                self._check_and_count_person(old_centroid, new_centroid)
                self.tracked_objects[person_id] = new_centroid
                del current_centroids[new_centroid_id]  # Remove the matched centroid
            else:
                # If a person isn't found in the new frame, I'll stop tracking them.
                del self.tracked_objects[person_id]

        # Add any remaining new centroids as new tracked objects.
        for new_centroid_id, new_centroid in current_centroids.items():
            # A simple way to give new people a unique ID.
            new_id = len(self.tracked_objects) + 1
            self.tracked_objects[new_id] = new_centroid

    def _get_centroid(self, detection):
        # Just a helper function to find the center point of a bounding box.
        box = detection['box']
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        return np.array([center_x, center_y])

    def _find_closest_centroid(self, old_centroid, new_centroids):
        # This finds the new person who is closest to a person I was already tracking.
        closest_id = None
        min_distance = float('inf')
        
        for new_id, new_centroid in new_centroids.items():
            distance = np.linalg.norm(old_centroid - new_centroid)
            if distance < min_distance:
                min_distance = distance
                closest_id = new_id
        
        return closest_id if min_distance < 100 else None # 100 is my max distance to match

    def _check_and_count_person(self, old_centroid, new_centroid):
        # This is where I'll check if a person crossed my predefined line.
        if self.line_direction == 'up':
            # Check if the person crossed the line from below.
            if old_centroid[1] > self.line_y_coord and new_centroid[1] <= self.line_y_coord:
                self.person_count += 1
        elif self.line_direction == 'down':
            # Check if the person crossed the line from above.
            if old_centroid[1] < self.line_y_coord and new_centroid[1] >= self.line_y_coord:
                self.person_count += 1
import cv2
import numpy as np
import os

def compute_center(corners):
    """Calculate center of ArUco marker from corners"""
    center = np.mean(corners[0], axis=0)
    return tuple(center.astype(int))

class Node:
    def __init__(self, id, center):
        self.id = id
        self.center = center

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))

def distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def list_video_devices():
    """List all video devices in /dev/"""
    return [f for f in os.listdir('/dev/') if f.startswith('video')]

def test_cameras():
    """Test available cameras and return working indices"""
    available_cameras = []
    video_devices = list_video_devices()
    print(f"Found video devices: {video_devices}")
    
    for i in range(len(video_devices)):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera {i} is working")
                    available_cameras.append(i)
            cap.release()
        except Exception as e:
            print(f"Error testing camera {i}: {e}")
    return available_cameras

# Get available cameras
cameras = test_cameras()
if not cameras:
    print("No working cameras found!")
    exit(1)

print("Available cameras:", cameras)

# Let user select camera with validation
while True:
    try:
        camera_idx = int(input(f"Select camera index {cameras}: "))
        if camera_idx in cameras:
            break
        print("Invalid camera index. Please try again.")
    except ValueError:
        print("Please enter a valid number.")

cap = cv2.VideoCapture(camera_idx)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ArUco detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(frame)
    graph = Graph()  # Reset graph each frame

    if ids is not None:
        # Process detected markers
        for i in range(len(ids)):
            center = compute_center(corners[i])
            node = Node(id=ids[i][0], center=center)
            graph.add_node(node)
            
            # Draw circle around marker
            radius = 50
            cv2.circle(frame, center, radius, (0, 0, 255), 2)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw edges between markers
        distance_threshold = 300
        for i in range(len(graph.nodes)):
            for j in range(i + 1, len(graph.nodes)):
                # Calculate both distances
                pixels_per_meter = 100  # Calibration factor - adjust as needed
                dist_pixels = distance(graph.nodes[i].center, graph.nodes[j].center)
                dist_meters = dist_pixels / pixels_per_meter
                
                # Draw line between markers
                cv2.line(frame, 
                        graph.nodes[i].center, 
                        graph.nodes[j].center, 
                        (0, 255, 0), 3)
                
                # Position text for both measurements
                text_pos = (
                    (graph.nodes[i].center[0] + graph.nodes[j].center[0]) // 2,
                    (graph.nodes[i].center[1] + graph.nodes[j].center[1]) // 2
                )
                
                # Draw both distances with outline
                text = f"{dist_pixels:.1f}px ({dist_meters:.2f}m)"
                cv2.putText(frame, text, text_pos, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # outline
                cv2.putText(frame, text, text_pos, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # text

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
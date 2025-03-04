import cv2
import depthai as dai
import numpy as np

# ------------------------------
# Función para calcular el centro de un marcador
# ------------------------------
def compute_center(corners):
    """
    Calcula el centro de un marcador ArUco a partir de sus esquinas.
    """
    center = np.mean(corners[0], axis=0)
    return tuple(center.astype(int))

# ------------------------------
# Clase para gestionar la red de nodos
# ------------------------------
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
    """
    Calcula la distancia euclidiana entre dos puntos en la imagen.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Configuración de la cámara Oak-D (DepthAI)
# ------------------------------
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# ------------------------------
# Procesamiento en tiempo real con Oak-D
# ------------------------------
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        # ------------------------------
        # Detección de Marcadores ArUco
        # ------------------------------
        # Create ArUco dictionary and detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        graph = Graph()  # Reset graph each frame

        if ids is not None:
            # Draw markers first
            for i in range(len(ids)):
                center = compute_center(corners[i])
                node = Node(id=ids[i][0], center=center)
                graph.add_node(node)
                
                # Draw circle around marker
                radius = 50
                cv2.circle(frame, center, radius, (0, 0, 255), 2)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw edges between markers
            for i in range(len(graph.nodes)):
                for j in range(i + 1, len(graph.nodes)):
                    dist = distance(graph.nodes[i].center, graph.nodes[j].center)
                    print(f"Drawing edge between markers {graph.nodes[i].id} and {graph.nodes[j].id}")
                    
                    # Draw thicker green line for better visibility
                    cv2.line(frame, 
                            graph.nodes[i].center, 
                            graph.nodes[j].center, 
                            (0, 255, 0),  # Green color
                            3)            # Increased thickness
                    
                    # Add distance label
                    text_pos = (
                        (graph.nodes[i].center[0] + graph.nodes[j].center[0]) // 2,
                        (graph.nodes[i].center[1] + graph.nodes[j].center[1]) // 2
                    )
                    text = f"{dist:.1f}px"
                    cv2.putText(frame, text, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    cv2.putText(frame, text, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ------------------------------
            # Conectar nodos según distancia
            # ------------------------------
            distance_threshold = 300  # Increased threshold for testing
            for i in range(len(graph.nodes)):
                for j in range(i + 1, len(graph.nodes)):
                    dist = distance(graph.nodes[i].center, graph.nodes[j].center)
                    print(f"Distance between markers {graph.nodes[i].id} and {graph.nodes[j].id}: {dist:.1f}px")
                    
                    if dist < distance_threshold:
                        graph.add_edge(graph.nodes[i], graph.nodes[j])
                        # Draw line between nodes
                        cv2.line(frame, graph.nodes[i].center, graph.nodes[j].center, (0, 255, 0), 2)
                        
                        # Calculate text position (midpoint of line)
                        text_pos = (
                            (graph.nodes[i].center[0] + graph.nodes[j].center[0]) // 2,
                            (graph.nodes[i].center[1] + graph.nodes[j].center[1]) // 2
                        )
                        # Add distance text with background for better visibility
                        text = f"{dist:.1f}px"
                        cv2.putText(frame, text, text_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # thick black outline
                        cv2.putText(frame, text, text_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # white text

        # ------------------------------
        # Visualización
        # ------------------------------
        cv2.imshow("Mesa Interactiva", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

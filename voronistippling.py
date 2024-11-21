import cv2
import numpy as np
from scipy.spatial import Voronoi

def enhance_live_feed(frame):
    """
    Enhance the live feed by improving brightness, contrast, and sharpness.
    """
    # Convert to LAB and apply CLAHE for adaptive brightness/contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_frame = cv2.merge((l, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

    # Apply sharpening to make the image crisper
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_frame = cv2.filter2D(enhanced_frame, -1, sharpening_kernel)

    # Optional: Denoise the image
    denoised_frame = cv2.GaussianBlur(sharp_frame, (5, 5), 0)
    return denoised_frame

def compute_centroids(points, shape):
    """
    Compute centroids for Voronoi regions, excluding invalid regions.
    """
    vor = Voronoi(points)
    centroids = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not region or -1 in region:  # Ignore regions with infinite vertices
            continue
        polygon = vor.vertices[region]
        if np.all((polygon >= 0) & (polygon <= [shape[1], shape[0]])):  # Within bounds
            centroids.append(np.mean(polygon, axis=0))
    return np.array(centroids, dtype=np.float32)

def voronoi_stippling(frame, num_points=300, iterations=5):
    """
    Perform Voronoi stippling using Lloyd's algorithm.
    """
    # Enhance the frame for better feature representation
    frame = enhance_live_feed(frame)
    
    # Convert to grayscale and normalize pixel intensities
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_normalized = gray / 255.0

    # Initialize random points
    points = np.array([
        [np.random.randint(0, frame.shape[1]), np.random.randint(0, frame.shape[0])]
        for _ in range(num_points)
    ], dtype=np.float32)

    # Perform Lloyd's iterations
    for _ in range(iterations):
        centroids = compute_centroids(points, frame.shape)
        if len(centroids) > 0:  # Ensure valid centroids
            points = centroids

    # Draw the stippling effect
    stippled = np.zeros_like(frame)
    for point in points:
        intensity = gray_normalized[int(point[1]), int(point[0])]  # Use pixel intensity for size
        radius = max(1, int(4 * (1.0 - intensity)))  # Larger dots for darker regions
        cv2.circle(stippled, (int(point[0]), int(point[1])), radius, (255, 255, 255), -1)
    return stippled

# Main loop to capture and process video
def live_voronoi_stippling():
    cap = cv2.VideoCapture(0)  # Open laptop camera
    if not cap.isOpened():
        print("Camera not found!")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break
        
        frame = cv2.resize(frame, (640, 480))  # Resize for consistent processing
        stippled_frame = voronoi_stippling(frame, num_points=400, iterations=6)  # Generate stippling effect

        # Display enhanced live feed alongside stippling
        enhanced_frame = enhance_live_feed(frame)
        combined_display = np.hstack((enhanced_frame, stippled_frame))
        cv2.imshow("Live Feed (Left) & Voronoi Stippling (Right)", combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_voronoi_stippling()

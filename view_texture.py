import cv2
import numpy as np

class SaliencyDetectors:
    def __init__(self):
        # Static saliency detectors
        self.static_fine = cv2.saliency.StaticSaliencyFineGrained_create()
        self.static_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
        
        # Objectness detector
        self.objectness = cv2.saliency.ObjectnessBING_create()

    def compute_static_fine(self, frame):
        """Fine-grained static saliency"""
        success, saliency_map = self.static_fine.computeSaliency(frame)
        if success:
            return (saliency_map * 255).astype(np.uint8)
        return None

    def compute_static_spectral(self, frame):
        """Spectral residual static saliency"""
        success, saliency_map = self.static_spectral.computeSaliency(frame)
        if success:
            return (saliency_map * 255).astype(np.uint8)
        return None

    def compute_objectness(self, frame):
        """BING Objectness saliency"""
        success, saliency_map = self.objectness.computeSaliency(frame)
        if success and saliency_map is not None:
            return (saliency_map * 255).astype(np.uint8)
        return None

def process_video():
    cap = cv2.VideoCapture(0)
    detectors = SaliencyDetectors()

    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Static Fine-Grained', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Static Spectral', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

    # Create trackbar for visualization
    def nothing(x):
        pass
    
    cv2.createTrackbar('Alpha', 'Controls', 50, 100, nothing)  # Blend factor
    cv2.createTrackbar('Threshold', 'Controls', 128, 255, nothing)  # Binary threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get trackbar values
        alpha = cv2.getTrackbarPos('Alpha', 'Controls') / 100
        threshold = cv2.getTrackbarPos('Threshold', 'Controls')

        # Original frame
        cv2.imshow('Original', frame)

        # Process with each detector
        results = {
            'Static Fine-Grained': detectors.compute_static_fine(frame),
            'Static Spectral': detectors.compute_static_spectral(frame)
        }

        # Display results
        for name, saliency_map in results.items():
            if saliency_map is not None:
                # Create binary mask
                _, binary_mask = cv2.threshold(
                    saliency_map, threshold, 255, cv2.THRESH_BINARY
                )


                # Create normalized grayscale visualization (0-255)
                gray_map = ((1.0-saliency_map) * 255).astype(np.uint8)

                # Apply colormap for visualization
                colored_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

                # Blend with original
                blended = cv2.addWeighted(
                    frame, 1-alpha, colored_map, alpha, 0
                )

                # Stack results horizontally
                display = np.hstack([
                    cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR),  # Raw normalized
                    colored_map,
                    cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR),
                    blended
                ])

                cv2.imshow(name, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
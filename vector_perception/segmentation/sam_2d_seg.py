import cv2
import numpy as np
import time
from ultralytics import FastSAM
from utils import extract_masks_bboxes_probs_names, filter_segmentation_results, plot_results, SimpleTracker


class Sam2DSegmenter:
    def __init__(self, model_path="models/FastSAM-s.engine", device="cuda"):
        self.device = device
        self.model = FastSAM(model_path)
        self.tracker = SimpleTracker(history_size=100, min_count=10, count_window=20)  # Increased history size

    def process_image(self, image):
        """
        Process an image and return segmentation results.
        :param image: RGB image as a NumPy array
        :return: masks, bboxes, track_ids, probs, names, total_counts
        """
        results = self.model.track(
            source=image,
            device=self.device,
            retina_masks=True,
            conf=0.6,
            iou=0.9,
            persist=True,
            verbose=False,
            tracker="vector_perception/segmentation/config/custom_tracker.yaml"
        )

        if len(results) > 0:
            masks, bboxes, track_ids, probs, names, areas = extract_masks_bboxes_probs_names(results[0])
            filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, _ = \
                filter_segmentation_results(image, self.tracker, masks, bboxes, track_ids, probs, names, areas)
            total_counts = self.tracker.get_total_counts()
            return filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, total_counts
        return [], [], [], [], [], {}

    def visualize_results(self, image, masks, bboxes, track_ids, probs, names):
        """
        Generate an overlay visualization with segmentation results.
        :param image: RGB image as a NumPy array
        :return: Overlayed image
        """
        overlay = plot_results(image.copy(), masks, bboxes, track_ids, probs, names)
        return overlay


def main():
    cap = cv2.VideoCapture(-1)
    segmenter = Sam2DSegmenter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        masks, bboxes, track_ids, probs, names, total_counts = segmenter.process_image(frame)
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time * 1000:.1f}ms")

        overlay = segmenter.visualize_results(frame, masks, bboxes, track_ids, probs, names)

        cv2.imshow("Segmentation with Tracking", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

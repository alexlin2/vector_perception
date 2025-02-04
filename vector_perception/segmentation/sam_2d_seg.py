import cv2
import numpy as np
import time
from ultralytics import FastSAM
from utils import extract_masks_bboxes_probs_names, \
                  filter_segmentation_results, \
                  plot_results, \
                  crop_images_from_bboxes
from vector_perception.common.detection2d_tracker import target2dTracker, get_tracked_results
import os


class Sam2DSegmenter:
    def __init__(self, model_path="models/FastSAM-s.engine", device="cuda"):
        self.device = device
        self.model = FastSAM(model_path)

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
            filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, filtered_texture_values = \
                filter_segmentation_results(image, masks, bboxes, track_ids, probs, names, areas)
            return filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, filtered_texture_values
        return [], [], [], [], [], []

    def visualize_results(self, image, masks, bboxes, track_ids, probs, names):
        """
        Generate an overlay visualization with segmentation results.
        :param image: RGB image as a NumPy array
        :return: Overlayed image
        """
        overlay = plot_results(image.copy(), masks, bboxes, track_ids, probs, names)
        return overlay


def main():
    cap = cv2.VideoCapture(0)
    segmenter = Sam2DSegmenter()

    tracker = target2dTracker(
        history_size=100,
        score_threshold_start=0.7,
        score_threshold_stop=0.2,
        min_frame_count=10,
        max_missed_frames=20,
        min_area_ratio=0.02,
        max_area_ratio=0.4,
        texture_range=(0.0, 0.35),
        border_safe_distance=100,  # pixels
        weights={"prob": 1.0, "temporal": 3.0, "texture": 2.0, "border": 3.0, "size": 1.0}
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        masks, bboxes, track_ids, probs, names, texture_values = segmenter.process_image(frame)

        tracked_targets = tracker.update(
            frame,
            masks,
            bboxes,
            track_ids,
            probs,
            names,
            texture_values,
        )

        tracked_masks, tracked_bboxes, tracked_track_ids, tracked_probs, tracked_names = get_tracked_results(tracked_targets)

        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time * 1000:.1f}ms")

        overlay = segmenter.visualize_results(frame, tracked_masks, tracked_bboxes, tracked_track_ids, tracked_probs, tracked_names)

        cv2.imshow("Segmentation with Tracking", overlay)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_path = "cropped_images/"
    os.makedirs(save_path, exist_ok=True)
    main()

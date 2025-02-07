import cv2
import numpy as np
import time
from ultralytics import FastSAM
from utils import extract_masks_bboxes_probs_names, \
                  filter_segmentation_results, \
                  plot_results, \
                  crop_images_from_bboxes
from vector_perception.common.detection2d_tracker import target2dTracker, get_tracked_results
from image_analyzer import ImageAnalyzer
from collections import deque
import os
from concurrent.futures import ThreadPoolExecutor, Future


class Sam2DSegmenter:
    def __init__(self, model_path="models/FastSAM-s.engine", device="cuda"):
        # Core components
        self.device = device
        self.model = FastSAM(model_path)
        self.image_analyzer = ImageAnalyzer()
        
        # Tracker initialization
        self.tracker = target2dTracker(
            history_size=100,
            score_threshold_start=0.7,
            score_threshold_stop=0.2,
            min_frame_count=10,
            max_missed_frames=20,
            min_area_ratio=0.02,
            max_area_ratio=0.4,
            texture_range=(0.0, 0.35),
            border_safe_distance=100,
            weights={"prob": 1.0, "temporal": 3.0, "texture": 2.0, "border": 3.0, "size": 1.0}
        )
        
        # Analysis components
        self.to_be_analyzed = deque()  # Queue of objects waiting for analysis
        self.object_names = {}  # Dictionary to store track_id -> name mappings
        self.analysis_executor = ThreadPoolExecutor(max_workers=1)
        self.current_future = None
        self.current_analysis_id = None

    def process_image(self, image):
        """Process an image and return segmentation results."""
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
            # Get initial segmentation results
            masks, bboxes, track_ids, probs, names, areas = extract_masks_bboxes_probs_names(results[0])
            
            # Filter results
            filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, filtered_texture_values = \
                filter_segmentation_results(image, masks, bboxes, track_ids, probs, names, areas)
            
            # Update tracker with filtered results
            tracked_targets = self.tracker.update(
                image,
                filtered_masks,
                filtered_bboxes,
                filtered_track_ids,
                filtered_probs,
                filtered_names,
                filtered_texture_values,
            )
            
            # Get tracked results
            tracked_masks, tracked_bboxes, tracked_track_ids, tracked_probs, tracked_names = get_tracked_results(tracked_targets)
            
            # Update analysis queue with tracked IDs
            tracked_id_set = set(tracked_track_ids)
            
            # Remove untracked objects from queue and results
            self.to_be_analyzed = deque([track_id for track_id in self.to_be_analyzed 
                                       if track_id in tracked_id_set])
            self.object_names = {k: v for k, v in self.object_names.items() 
                               if k in tracked_id_set}
            
            # If current analysis object is no longer tracked, cancel it
            if self.current_analysis_id and self.current_analysis_id not in tracked_id_set:
                if self.current_future and not self.current_future.done():
                    self.current_future.cancel()
                self.current_analysis_id = None
                self.current_future = None
            
            # Add new track_ids to analysis queue
            for track_id in tracked_track_ids:
                if track_id not in self.object_names and track_id not in self.to_be_analyzed:
                    self.to_be_analyzed.append(track_id)
            
            return tracked_masks, tracked_bboxes, tracked_track_ids, tracked_probs, tracked_names
        return [], [], [], [], []

    def check_analysis_status(self, tracked_track_ids):
        """Check if analysis is complete and start new analysis if needed."""
        # Check if current analysis is complete
        if self.current_future and self.current_future.done():
            try:
                result = self.current_future.result()
                if result is not None:
                    self.object_names[self.current_analysis_id] = result
            except Exception as e:
                print(f"Analysis failed: {e}")
            self.current_future = None
            self.current_analysis_id = None

        # Start new analysis if none is running
        if not self.current_future and self.to_be_analyzed:
            track_id = self.to_be_analyzed[0]
            if track_id in tracked_track_ids:
                bbox_idx = tracked_track_ids.index(track_id)
                self.current_analysis_id = track_id
                self.to_be_analyzed.popleft()
                return bbox_idx
        return None

    def run_analysis(self, frame, tracked_bboxes, tracked_track_ids):
        """Run image analysis in background."""
        bbox_idx = self.check_analysis_status(tracked_track_ids)
        if bbox_idx is not None:
            bbox = tracked_bboxes[bbox_idx]
            cropped_images = crop_images_from_bboxes(frame, [bbox])
            if cropped_images:
                self.current_future = self.analysis_executor.submit(
                    self.image_analyzer.analyze_images, cropped_images
                )

    def get_object_names(self, track_ids, tracked_names):
        """Get object names for the given track IDs, falling back to tracked names."""
        return [self.object_names.get(track_id, tracked_name) 
                for track_id, tracked_name in zip(track_ids, tracked_names)]

    def visualize_results(self, image, masks, bboxes, track_ids, probs, names):
        """Generate an overlay visualization with segmentation results and object names."""
        return plot_results(image, masks, bboxes, track_ids, probs, names)

    def cleanup(self):
        """Cleanup resources."""
        self.analysis_executor.shutdown()


def main():
    cap = cv2.VideoCapture(0)
    segmenter = Sam2DSegmenter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            
            # Process image and get tracked results
            tracked_masks, tracked_bboxes, tracked_track_ids, tracked_probs, tracked_names = segmenter.process_image(frame)
            
            # Run analysis in background
            segmenter.run_analysis(frame, tracked_bboxes, tracked_track_ids)
            
            # Get current object names
            updated_names = segmenter.get_object_names(tracked_track_ids, tracked_names)

            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time * 1000:.1f}ms")

            overlay = segmenter.visualize_results(
                frame, 
                tracked_masks, 
                tracked_bboxes, 
                tracked_track_ids, 
                tracked_probs, 
                updated_names
            )

            cv2.imshow("Segmentation with Tracking", overlay)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    finally:
        segmenter.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
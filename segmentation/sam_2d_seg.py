from ultralytics import FastSAM
import cv2
import numpy as np
import time
from common.utils import extract_masks_bboxes_probs_names, \
                         filter_segmentation_results, \
                         plot_results
from common.utils import SimpleTracker

# use cuda
device = 'cuda'

# Load FastSAM model
model = FastSAM("models/FastSAM-s.engine")

# Open video capture
cap = cv2.VideoCapture(0)

tracker = SimpleTracker(history_size=10, min_count=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run FastSAM inference with tracking
    # Note: using track() instead of predict()
    results = model.track(
        source=frame,
        device=device,
        retina_masks=True,
        conf=0.6,
        iou=0.9,
        persist=True,
        verbose=False,
        tracker="segmentation/config/custom_tracker.yaml"
    )

    if len(results) > 0:
        # Extract masks, boxes, and other info - tracking IDs will be included automatically
        # Time the extraction
        start_time = time.time()
        masks, bboxes, track_ids, probs, names, areas = extract_masks_bboxes_probs_names(results[0])
        filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, texture_map = \
            filter_segmentation_results(frame, tracker, masks, bboxes, track_ids, probs, names, areas)
        texture_map = texture_map.cpu().numpy()
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time*1000:.1f}ms")

        # Plot results with tracking information
        overlay = plot_results(frame.copy(), filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names)
    else:
        overlay = frame
        texture_map = np.zeros_like(frame)

    cv2.imshow("Segmentation with Tracking", overlay)
    cv2.imshow("texture Map", texture_map)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
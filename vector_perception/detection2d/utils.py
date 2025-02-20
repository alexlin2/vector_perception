import numpy as np
import cv2


def extract_detection_results(result):
    """
    Extract detection information from a YOLO result object.
    
    Args:
        result: Ultralytics result object
        
    Returns:
        tuple: (bboxes, track_ids, class_ids, confidences, names)
            - bboxes: list of [x1, y1, x2, y2] coordinates
            - track_ids: list of tracking IDs
            - class_ids: list of class indices
            - confidences: list of detection confidences
            - names: list of class names
    """
    bboxes = []
    track_ids = []
    class_ids = []
    confidences = []
    names = []

    if result.boxes is None:
        return bboxes, track_ids, class_ids, confidences, names

    for box in result.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bboxes.append([x1, y1, x2, y2])
        
        # Extract tracking ID if available
        track_id = -1
        if hasattr(box, 'id') and box.id is not None:
            track_id = int(box.id[0].item())
        track_ids.append(track_id)
        
        # Extract class information
        cls_idx = int(box.cls[0])
        class_ids.append(cls_idx)
        names.append(result.names[cls_idx])
        
        # Extract confidence
        confidences.append(float(box.conf[0]))

    return bboxes, track_ids, class_ids, confidences, names


def plot_results(image, bboxes, track_ids, class_ids, confidences, names, alpha=0.5):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Original input image
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        track_ids: List of tracking IDs
        class_ids: List of class indices
        confidences: List of detection confidences
        names: List of class names
        alpha: Transparency of the overlay
        
    Returns:
        Image with visualized detections
    """
    vis_img = image.copy()

    for bbox, track_id, conf, name in zip(bboxes, track_ids, confidences, names):
        # Generate consistent color based on track_id or class name
        if track_id != -1:
            np.random.seed(track_id)
        else:
            np.random.seed(hash(name) % 100000)
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        np.random.seed(None)
            
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color.tolist(), 2)

        # Prepare label text
        if track_id != -1:
            label = f"ID:{track_id} {name} {conf:.2f}"
        else:
            label = f"{name} {conf:.2f}"

        # Calculate text size for background rectangle
        (text_w, text_h), _ = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            1
        )

        # Draw background rectangle for text
        cv2.rectangle(
            vis_img, 
            (x1, y1-text_h-8), 
            (x1+text_w+4, y1), 
            color.tolist(), 
            -1
        )

        # Draw text with white color for better visibility
        cv2.putText(
            vis_img,
            label,
            (x1+2, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return vis_img
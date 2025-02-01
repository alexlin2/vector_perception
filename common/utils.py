import numpy as np
import cv2
import torch
from collections import deque, Counter


class SimpleTracker:
    def __init__(self, history_size=10, min_count=5):
        """
        Simple temporal tracker that counts appearances in a fixed window.
        
        Parameters:
        history_size: Number of past frames to remember
        min_count: Minimum number of appearances required
        """
        self.history = deque(maxlen=history_size)
        self.min_count = min_count
        
    def update(self, track_ids):
        # Add new frame's track IDs to history
        self.history.append(track_ids)
        
        # Count appearances of each ID in history
        all_tracks = [id for frame_ids in self.history for id in frame_ids]
        counts = Counter(all_tracks)
        
        # Return IDs that appear often enough
        return [id for id, count in counts.items() if count >= self.min_count]


def extract_masks_bboxes_probs_names(result, max_size=0.7):
    """
    Extracts masks, bounding boxes, probabilities, and class names from one Ultralytics result object.
    
    Parameters:
    result: Ultralytics result object
    max_size: float, maximum allowed size of object relative to image (0-1)
    
    Returns:
    tuple: (masks, bboxes, track_ids, probs, names, areas)
    """
    masks = []
    bboxes = []
    track_ids = []
    probs = []
    names = []
    areas = []

    if result.masks is None:
        return masks, bboxes, track_ids, probs, names, areas
    
    total_area = result.masks.orig_shape[0] * result.masks.orig_shape[1]

    for box, mask_data in zip(result.boxes, result.masks.data):
        mask_numpy = mask_data

        # Extract bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Extract track_id if available
        track_id = -1  # default if no tracking
        if hasattr(box, 'id') and box.id is not None:
            track_id = int(box.id[0].item())
        
        # Extract probability and class index
        conf = float(box.conf[0])
        cls_idx = int(box.cls[0])
        area = (x2 - x1) * (y2 - y1)

        if area / total_area > max_size:
            continue

        masks.append(mask_numpy)
        bboxes.append([x1, y1, x2, y2])
        track_ids.append(track_id)
        probs.append(conf)
        names.append(result.names[cls_idx])
        areas.append(area)

    return masks, bboxes, track_ids, probs, names, areas

def compute_texture_map(frame, blur_size=3):
    """
    Compute texture map using gradient statistics.
    Returns high values for textured regions and low values for smooth regions.
    
    Parameters:
    frame: BGR image
    blur_size: Size of Gaussian blur kernel for pre-processing
    
    Returns:
    numpy array: Texture map with values normalized to [0,1]
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Pre-process with slight blur to reduce noise
    if blur_size > 0:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute local standard deviation of gradient magnitude
    texture_map = cv2.GaussianBlur(magnitude, (15, 15), 0)
    
    # Normalize to [0,1]
    texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    return texture_map


def filter_segmentation_results(frame, tracker, masks, bboxes, track_ids, probs, names, areas, texture_threshold=0.1, size_filter=600):
    """
    Filters segmentation results using both overlap and saliency detection.
    Uses mask_sum tensor for efficient overlap detection.
    
    Parameters:
    masks: list of torch.Tensor containing mask data
    bboxes: list of bounding boxes [x1, y1, x2, y2]
    track_ids: list of tracking IDs
    probs: list of confidence scores
    names: list of class names
    areas: list of object areas
    frame: BGR image for computing saliency
    texture_threshold: Average texture value required for mask to be kept
    
    Returns:
    tuple: (filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names)
    """
    if len(masks) <= 1:
        return masks, bboxes, track_ids, probs, names, torch.zeros(frame.shape[:2], dtype=torch.float32)
    
    # Get stable track IDs
    stable_ids = tracker.update(track_ids)
    
    # Keep only temporally stable detections
    keep_temporal = [i for i, track_id in enumerate(track_ids) if track_id in stable_ids]
    if not keep_temporal:
        return [], [], [], [], [], torch.zeros(frame.shape[:2], dtype=torch.float32)

    masks = [masks[i] for i in keep_temporal]
    bboxes = [bboxes[i] for i in keep_temporal]
    track_ids = [track_ids[i] for i in keep_temporal]
    probs = [probs[i] for i in keep_temporal]
    names = [names[i] for i in keep_temporal]
    areas = [areas[i] for i in keep_temporal]
        
    # Compute texture map once and convert to tensor
    texture_map = compute_texture_map(frame)
    
    # Sort by area (largest to smallest)
    sorted_indices = torch.tensor(areas).argsort(descending=False)

    device = masks[0].device  # Get the device of the first mask
    
    # Create mask_sum tensor where each pixel stores the index of the mask that claims it
    mask_sum = torch.zeros_like(masks[0], dtype=torch.int32)
    
    texture_map = torch.from_numpy(texture_map).to(device)  # Convert texture_map to tensor and move to device
    
    for i, idx in enumerate(sorted_indices):
        mask = masks[idx]
        # Compute average texture value within mask
        texture_value = torch.mean(texture_map[mask > 0]) if torch.any(mask > 0) else 0
        
        # Only claim pixels if mask passes texture threshold
        if texture_value >= texture_threshold:
            mask_sum[mask > 0] = i
    
    # Get indices that appear in mask_sum (these are the masks we want to keep)
    keep_indices, counts = torch.unique(mask_sum[mask_sum > 0], return_counts=True)
    size_indices = counts > size_filter
    keep_indices = keep_indices[size_indices]

    sorted_indices = sorted_indices.cpu()
    keep_indices = keep_indices.cpu()
    
    # Map back to original indices and filter
    final_indices = sorted_indices[keep_indices].tolist()
    
    filtered_masks = [masks[i] for i in final_indices]
    filtered_bboxes = [bboxes[i] for i in final_indices]
    filtered_track_ids = [track_ids[i] for i in final_indices]
    filtered_probs = [probs[i] for i in final_indices]
    filtered_names = [names[i] for i in final_indices]

    return filtered_masks, filtered_bboxes, filtered_track_ids, filtered_probs, filtered_names, texture_map


def plot_results(image, masks, bboxes, track_ids, probs, names, alpha=0.5):
    """
    Draws bounding boxes, masks, and labels on the given image.
    """
    h, w = image.shape[:2]
    overlay = image.copy()

    for mask, bbox, track_id, prob, name in zip(masks, bboxes, track_ids, probs, names):
        # Convert mask tensor to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if track_id != -1:
            # Use track_id to generate consistent color
            np.random.seed(track_id)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            np.random.seed(None)  # Reset seed
        else:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay[mask_resized > 0.5] = color

        # Bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color.tolist(), 2)

        # Label
        label = f"{track_id} {prob:.2f}"
        cv2.putText(
            overlay,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            1
        )

    # Blend
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image
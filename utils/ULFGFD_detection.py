import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import warnings
import logging

# Suppress ONNX warnings
warnings.filterwarnings('ignore')
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

def _load_model(model_path: str):
    """Load ONNX model and return session with input/output info."""
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape
    
    return session, input_name, output_names, input_shape[2], input_shape[3]

def _preprocess_image(image: np.ndarray, model_width: int, model_height: int) -> Tuple[np.ndarray, float, float]:
    """Preprocess image for model inference."""
    original_height, original_width = image.shape[:2]
    
    # Resize to model input size
    resized_image = cv2.resize(image, (model_width, model_height))
    
    # Convert BGR to RGB and normalize
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0
    
    # Transpose to CHW format and add batch dimension
    chw_image = np.transpose(normalized_image, (2, 0, 1))
    input_tensor = np.expand_dims(chw_image, axis=0)
    
    # Calculate scale factors
    scale_x = original_width / model_width
    scale_y = original_height / model_height
    
    return input_tensor, scale_x, scale_y

def _postprocess_detections(outputs: List[np.ndarray], scale_x: float, scale_y: float,
                           model_width: int, model_height: int, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """Extract face detections from model outputs."""
    detections = []
    
    if len(outputs) == 2:
        scores = outputs[0][0]  # [N, 2]
        boxes = outputs[1][0]   # [N, 4]
        
        face_scores = scores[:, 1]
        valid_indices = face_scores > confidence_threshold
        valid_boxes = boxes[valid_indices]
        valid_scores = face_scores[valid_indices]
        
        for box, score in zip(valid_boxes, valid_scores):
            x1, y1, x2, y2 = box
            
            # Convert normalized coordinates to pixel coordinates
            x1_pixel = int(x1 * model_width * scale_x)
            y1_pixel = int(y1 * model_height * scale_y)
            x2_pixel = int(x2 * model_width * scale_x)
            y2_pixel = int(y2 * model_height * scale_y)
            
            # Ensure coordinates are valid
            x1_pixel = max(0, x1_pixel)
            y1_pixel = max(0, y1_pixel)
            x2_pixel = min(int(model_width * scale_x), x2_pixel)
            y2_pixel = min(int(model_height * scale_y), y2_pixel)
            
            # Convert to (x, y, w, h)
            x, y = x1_pixel, y1_pixel
            w, h = max(1, x2_pixel - x1_pixel), max(1, y2_pixel - y1_pixel)
            
            detections.append((x, y, w, h, float(score)))
    
    return detections

def _filter_detections(detections: List[Tuple[int, int, int, int, float]], 
                      image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
    """Filter and apply NMS to detections."""
    if not detections:
        return []
    
    img_height, img_width = image_shape
    filtered = []
    
    for x, y, w, h, confidence in detections:
        # Basic filtering
        min_size = min(img_width, img_height) * 0.03
        max_size = min(img_width, img_height) * 0.8
        face_size = max(w, h)
        
        if (face_size < min_size or face_size > max_size or
            x < 0 or y < 0 or x + w > img_width or y + h > img_height or
            w / h < 0.5 or w / h > 2.0):
            continue
            
        filtered.append((x, y, w, h, confidence))
    
    # Sort by confidence
    filtered.sort(key=lambda x: x[4], reverse=True)
    
    # Apply NMS
    if len(filtered) > 1:
        boxes = [[x, y, w, h] for x, y, w, h, _ in filtered]
        confidences = [conf for _, _, _, _, conf in filtered]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        
        if len(indices) > 0:
            filtered = [filtered[i] for i in indices.flatten()] # type: ignore
    
    return filtered

def _expand_to_head_bbox(face_bbox: Tuple[int, int, int, int], 
                        image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Expand face bbox to include full head (hair to neck)."""
    x, y, w, h = face_bbox
    img_height, img_width = image_shape
    
    # Expansion factors for head coverage
    top_expansion = 0.6    # Hair
    bottom_expansion = 0.3  # Neck
    side_expansion = 0.35    # Ears
    
    new_x = max(0, x - int(w * side_expansion))
    new_y = max(0, y - int(h * top_expansion))
    new_w = min(img_width - new_x, w + int(w * 2 * side_expansion))
    new_h = min(img_height - new_y, h + int(h * (top_expansion + bottom_expansion)))
    
    return (new_x, new_y, new_w, new_h)

def getBoundingBoxsULFGFD(image: np.ndarray, model_path: str = "weights/version-RFB-640.onnx", 
                   confidence_threshold: float = 0.5) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect heads in image and return bounding boxes in format ((x1, y1), (x2, y2)).
    
    Args:
        image: Input image (BGR format from cv2.imread)
        model_path: Path to ONNX model file
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of ((x1, y1), (x2, y2)) tuples for detected heads
    """
    try:
        # Load model
        session, input_name, output_names, model_height, model_width = _load_model(model_path)
        
        # Preprocess
        input_tensor, scale_x, scale_y = _preprocess_image(image, model_width, model_height)
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # {len(outputs)} outputs
        
        # Post-process
        raw_detections = _postprocess_detections(outputs, scale_x, scale_y, model_width, model_height, confidence_threshold) # type: ignore
        
        # {len(raw_detections)} raw detections
        
        # Filter
        filtered_detections = _filter_detections(raw_detections, image.shape[:2])
        
        # {len(filtered_detections)} detections after filtering
        
        # Convert to head bboxes and required format
        head_boxes = []
        for x, y, w, h, conf in filtered_detections:
            head_x, head_y, head_w, head_h = _expand_to_head_bbox((x, y, w, h), image.shape[:2])
            x1, y1 = head_x, head_y
            x2, y2 = head_x + head_w, head_y + head_h
            head_boxes.append(((x1, y1), (x2, y2)))
        
        # {len(head_boxes)}
        return head_boxes
        
    except Exception as e:
        print(f"Error during head detection using Ultra-Light-Fast-Generic-Face-Detector-1MB: {str(e)}")
        return []


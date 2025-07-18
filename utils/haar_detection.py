from .logger import Logger
logger = Logger()

def getBoundingBoxsHaar(gray, face_cascade):
    """Get bounding boxes for faces using Haar Cascade detection"""
    try:
        if gray is None or gray.size == 0:
            logger.error("Invalid gray image provided to Haar detection")
            return []
        
        if face_cascade is None or face_cascade.empty():
            logger.error("Invalid or empty Haar cascade classifier")
            return []
        
        logger.info("Running Haar cascade face detection")
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            logger.warning("No faces detected by Haar cascade")
            return []
        
        logger.info(f"Haar cascade detected {len(faces)} potential face(s)")
        
        faces_boxes = []
        img_height, img_width = gray.shape[:2]
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Expand bounding box to include more head area
            x1, y1 = int(x/1.2), int(y/5)
            x2, y2 = x1 + int(w*1.5), y1 + int(h*2)

            # Clamp coordinates to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Validate bounding box
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box for face {idx + 1}: ({x1},{y1}) to ({x2},{y2})")
                continue
            
            # Check if bounding box is reasonable size
            box_width = x2 - x1
            box_height = y2 - y1
            min_size = min(img_width, img_height) * 0.05  # At least 5% of image
            
            if box_width < min_size or box_height < min_size:
                logger.warning(f"Bounding box {idx + 1} too small: {box_width}x{box_height}")
                continue
            
            face = ((x1, y1), (x2, y2))
            faces_boxes.append(face)
            logger.info(f"Valid face {len(faces_boxes)}: ({x1},{y1}) to ({x2},{y2})")

        logger.info(f"Haar cascade validated {len(faces_boxes)} face(s)")
        return faces_boxes
    
    except Exception as e:
        logger.error(f"Error during Haar cascade detection: {str(e)}")
        return []
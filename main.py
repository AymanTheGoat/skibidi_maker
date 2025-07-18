import os
import sys
import cv2 as cv
import onnxruntime as ort
from utils.remove_bg import remBg
from utils.haar_detection import getBoundingBoxsHaar
from utils.image_utils import getOverlayImage, alpha_blend
from utils.ULFGFD_detection import getBoundingBoxsULFGFD
from utils.file_utils import check_file_exists, getSmallestAvailableNumber
from utils.logger import Logger

# Initialize logger
logger = Logger()

def main(image, method):
    try:
        logger.info("Starting face detection and processing")
        
        if method == 1:
            logger.info("Using Haar Cascade detection method")
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            boundingboxs = getBoundingBoxsHaar(image_gray, face_cascade)
        else:
            logger.info("Using ULFGFD detection method")
            boundingboxs = getBoundingBoxsULFGFD(image, ulfgfd_path)
        
        if not boundingboxs:
            logger.warning("No faces detected in the image")
            return
        
        logger.info(f"Found {len(boundingboxs)} face(s) in the image")
        
        logger.info("Removing background from image")
        noBg = remBg(image, u2net_session)
        
        cropped_images = []
        
        # Crop the faces from the image
        for idx, (pt1, pt2) in enumerate(boundingboxs):
            logger.info(f"Processing face {idx + 1}/{len(boundingboxs)}")
            
            x1, y1 = pt1
            x2, y2 = pt2
            
            # Validate bounding box
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid bounding box for face {idx + 1}: ({x1},{y1}) to ({x2},{y2})")
                continue
            
            cropped_image = noBg[y1:y2, x1:x2] # 
            
            # Check if cropped image is valid
            if cropped_image.size == 0:
                logger.warning(f"Empty cropped image for face {idx + 1}")
                continue
                
            cropped_images.append(cropped_image)
        
        if not cropped_images:
            logger.error("No valid faces could be cropped from the image")
            return
        
        logger.info(f"Successfully cropped {len(cropped_images)} face(s)")
        
        i = getSmallestAvailableNumber()
        
        for idx, cropped_image in enumerate(cropped_images):
            logger.info(f"Creating final output for face {idx + 1}")
            
            # Resize the cropped image and overlay it on a blank canvas
            overlay_image = getOverlayImage(cropped_image)
            toilet_image = toiletImage
            toilet_overlay_image = toiletOverlayImage
            
            stack1 = alpha_blend(overlay_image, toilet_image)
            stack2 = alpha_blend(toilet_overlay_image, stack1)
            
            # Save the final output
            output_filename = f'output/output_{i}.png'
            success = cv.imwrite(output_filename, stack2)
            
            if success:
                logger.info(f"Successfully saved output as {output_filename}")
            else:
                logger.error(f"Failed to save output as {output_filename}")
            
            i += 1
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during main processing: {str(e)}")
        sys.exit(1)

def load_image(path):
    """Load and validate image file"""
    try:
        if not path.lower().endswith('.png'):
            logger.error(f"Only PNG files are supported. Got: {path}")
            return None
        
        image = cv.imread(path)
        if image is None:
            logger.error(f"Could not load image from {path}. File may be corrupted or not a valid PNG.")
            return None
        
        logger.info(f"Successfully loaded image: {path} (size: {image.shape[1]}x{image.shape[0]})")
        return image
    
    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        return None

def load_model_weights():
    """Load all required model weights with error handling"""
    global face_cascade, u2net_session
    
    try:
        # Load Haar Cascade
        logger.info("Loading Haar Cascade classifier")
        face_cascade = cv.CascadeClassifier(haarcascade_path)
        if face_cascade.empty():
            logger.error("Failed to load Haar Cascade classifier")
            return False
        logger.info("Haar Cascade loaded successfully")
        
        # Load U2NET model
        logger.info("Loading U2NET model for background removal")
        u2net_session = ort.InferenceSession(u2net_path, providers=['CPUExecutionProvider'])
        logger.info("U2NET model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model weights: {str(e)}")
        return False


if __name__ == "__main__":
    # File paths
    input_path = 'input/image3.png'
    output_path = 'assets/output.png'
    toilet_path = 'assets/toilet.png'
    toilet_overlay_path = 'assets/toilet_overlay.png'
    u2net_path = 'weights/u2net.onnx'
    ulfgfd_path = 'weights/version-RFB-640.onnx'
    haarcascade_path = 'weights/haarcascade.xml'
    method = 2  # 1 for Haar Cascade, 2 for ULFGFD

    logger.info("Starting Skibidi Face Generator")
    
    # Check if all required files exist
    logger.info("Checking required files")
    required_files = [input_path, toilet_path, toilet_overlay_path, u2net_path, ulfgfd_path, haarcascade_path]
    
    for path in required_files:
        if not check_file_exists(path):
            logger.error(f"Required file missing: {path}")
            sys.exit(1)
    
    logger.info("All required files found")
    
    # Load the input image
    logger.info("Loading input image")
    image = load_image(input_path)
    if image is None:
        sys.exit(1)
    
    # Load toilet assets
    logger.info("Loading toilet assets")
    toiletImage = cv.imread(toilet_path, cv.IMREAD_UNCHANGED)
    toiletOverlayImage = cv.imread(toilet_overlay_path, cv.IMREAD_UNCHANGED)
    
    if toiletImage is None:
        logger.error(f"Could not load toilet image: {toilet_path}")
        sys.exit(1)
    
    if toiletOverlayImage is None:
        logger.error(f"Could not load toilet overlay image: {toilet_overlay_path}")
        sys.exit(1)
    
    logger.info("Toilet assets loaded successfully")
    
    # Load model weights
    if not load_model_weights():
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    logger.info("Output directory ready")
    
    # Run main processing
    main(image, method)
import onnxruntime as ort
import numpy as np
import cv2 as cv
from .logger import Logger
logger = Logger()


def _preprocess(image):
    """Preprocess image for U2NET model"""
    try:
        if image is None or image.size == 0:
            logger.error("Invalid image provided for preprocessing")
            return None, None, None
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        logger.info(f"Preprocessing image of size {original_size}")

        # Resize to 320x320
        image_resized = cv.resize(image, (320, 320), interpolation=cv.INTER_LINEAR)

        # Convert to float32 and normalize
        img_np = image_resized.astype(np.float32) / 255.0
        img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)

        return img_np, original_size, image
    
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        return None, None, None

def _postprocess(mask, original_size):
    """Postprocess mask from U2NET model"""
    try:
        if mask is None or mask.size == 0:
            logger.error("Invalid mask provided for postprocessing")
            return None
        
        # Normalize mask
        mask_min, mask_max = mask.min(), mask.max()
        if mask_max - mask_min == 0:
            logger.warning("Mask has no variation, using default mask")
            mask = np.ones_like(mask)
        else:
            mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
        
        # Resize back to original size
        mask = cv.resize(mask, original_size, interpolation=cv.INTER_LINEAR)
        mask = (mask * 255).astype(np.uint8)
        
        return mask
    
    except Exception as e:
        logger.error(f"Error during mask postprocessing: {str(e)}")
        return None

def _apply_mask_with_transparency(original_image, mask):
    """Apply mask to create transparent background"""
    try:
        if original_image is None or mask is None:
            logger.error("Invalid image or mask provided")
            return None
        
        if original_image.shape[:2] != mask.shape[:2]:
            logger.error(f"Image size {original_image.shape[:2]} doesn't match mask size {mask.shape[:2]}")
            return None
        
        # Ensure mask is single channel uint8
        alpha = mask

        # Convert original image to BGRA
        bgr = original_image
        bgra = cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)

        # Set alpha channel to the mask
        bgra[:, :, 3] = alpha
        
        return bgra
    
    except Exception as e:
        logger.error(f"Error applying mask with transparency: {str(e)}")
        return None

def _remove_background_onnx(image, model, output_path="output.png"):
    """Remove background using U2NET ONNX model"""
    try:
        logger.info("Starting background removal process")
        
        # Preprocess
        input_tensor, original_size, original_image = _preprocess(image)
        if input_tensor is None:
            return None

        # Run ONNX inference
        session: ort.InferenceSession = model
        input_name = session.get_inputs()[0].name
        
        logger.info("Running U2NET inference for background removal")
        output = session.run(None, {input_name: input_tensor})

        if not output or len(output) == 0:
            logger.error("No output from U2NET model")
            return None

        # Get mask from output
        d1 = output[0][0][0]  # type: ignore
        mask = _postprocess(d1, original_size)
        
        if mask is None:
            return None

        # Create transparent result
        result = _apply_mask_with_transparency(original_image, mask)
        
        if result is not None:
            logger.info("Background removal completed successfully")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during background removal: {str(e)}")
        return None


def remBg(image: cv.typing.MatLike, model: ort.InferenceSession):
    """Remove background from image using U2NET model"""
    try:
        if image is None:
            logger.error("No image provided for background removal")
            return None
        
        result = _remove_background_onnx(image, model)
        
        if result is None:
            logger.error("Background removal failed")
            # Return original image with alpha channel as fallback
            if len(image.shape) == 3:
                fallback = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
                logger.warning("Using original image as fallback")
                return fallback
            return image
        
        return result
    
    except Exception as e:
        logger.error(f"Error in remBg function: {str(e)}")
        return None
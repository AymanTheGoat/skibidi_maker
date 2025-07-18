import cv2 as cv
import numpy as np
from .logger import Logger
logger = Logger()


# Constants for toilet positioning
SIZE = 1080
WIDTH = 383
HEIGHT = 513
X = 105
Y = 215

def getOverlayImage(image):
    """Resize and position face image for toilet overlay"""
    try:
        if image is None or image.size == 0:
            logger.error("Invalid image provided for overlay")
            return None
        
        logger.info(f"Processing overlay image of size {image.shape}")
        
        # Check if image is RGBA, if not convert to RGBA
        if len(image.shape) == 4 and image.shape[2] == 4:
            resized_image = image
        else:
            resized_image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        
        # Resize image based on current size
        current_height, current_width = resized_image.shape[:2]
        
        if current_height > WIDTH or current_width > WIDTH:
            logger.info(f"Resizing large image from {current_width}x{current_height} to {WIDTH}x{HEIGHT}")
            resized_image = cv.resize(resized_image, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)
        elif current_height < WIDTH or current_width < WIDTH:
            logger.info(f"Upscaling small image from {current_width}x{current_height} to {WIDTH}x{HEIGHT}")
            resized_image = cv.resize(resized_image, (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR)
        else:
            logger.info("Image already at target size")
            if resized_image.shape[:2] != (HEIGHT, WIDTH):
                resized_image = cv.resize(resized_image, (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR)
        
        # Create blank canvas and place image
        blank = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
        
        # Validate positioning
        if Y + HEIGHT > SIZE or X + WIDTH > SIZE:
            logger.error(f"Image positioning ({X}, {Y}) with size ({WIDTH}, {HEIGHT}) exceeds canvas size ({SIZE})")
            return None
        
        blank[Y:Y + HEIGHT, X:X + WIDTH] = resized_image
        
        logger.info(f"Successfully created overlay image positioned at ({X}, {Y})")
        return blank
    
    except Exception as e:
        logger.error(f"Error creating overlay image: {str(e)}")
        return None

def alpha_blend(foreground, background):
    """Alpha blend two RGBA images"""
    try:
        if foreground is None or background is None:
            logger.error("Invalid images provided for alpha blending")
            return None
        
        if foreground.shape != background.shape:
            logger.error(f"Image size mismatch: foreground {foreground.shape} vs background {background.shape}")
            return None
        
        if len(foreground.shape) != 3 or foreground.shape[2] != 4:
            logger.error(f"Expected RGBA images, got foreground shape: {foreground.shape}")
            return None
        
        if len(background.shape) != 3 or background.shape[2] != 4:
            logger.error(f"Expected RGBA images, got background shape: {background.shape}")
            return None
        
        logger.info("Performing alpha blending")
        
        # Convert to float for calculations
        fg = foreground.astype(float) / 255.0
        bg = background.astype(float) / 255.0

        # Separate RGB and alpha channels
        fg_rgb, fg_alpha = fg[..., :3], fg[..., 3:]
        bg_rgb, bg_alpha = bg[..., :3], bg[..., 3:]

        # Calculate output alpha
        out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

        # Avoid division by zero
        out_alpha_safe = np.clip(out_alpha, 1e-6, 1)

        # Calculate output RGB
        out_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)) / out_alpha_safe

        # Combine RGB and alpha
        out_image = np.dstack((out_rgb, out_alpha))
        
        # Convert back to uint8
        result = (out_image * 255).astype(np.uint8)
        
        logger.info("Alpha blending completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during alpha blending: {str(e)}")
        return None
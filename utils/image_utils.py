import cv2 as cv
import numpy as np
# 105 215 , 488 215
# 105 728 , 488 728

# dimesions = 383 × 513 
# position = (0.097, 0.199)
size = 1080

width = 383
height = 513

x = 105
y = 215

def getOverlayImage(image):
    # check if image is rgba
    if image.shape[2] == 4:
        resized_image = image
    else:
        resized_image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        
    if image.shape[0] > width:
        resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    elif image.shape[0] < width: 
        resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)

    else:
        resized_image = image

    blank = np.zeros((size, size, 4), dtype=np.uint8)
    blank[y:y + height, x:x + width] = resized_image
    return blank




def alpha_blend(foreground, background):
    fg = foreground.astype(float) / 255.0
    bg = background.astype(float) / 255.0

    fg_rgb, fg_alpha = fg[..., :3], fg[..., 3:]
    bg_rgb, bg_alpha = bg[..., :3], bg[..., 3:]

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

    out_alpha_safe = np.clip(out_alpha, 1e-6, 1)

    out_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)) / out_alpha_safe

    out_image = np.dstack((out_rgb, out_alpha))
    return (out_image * 255).astype(np.uint8)
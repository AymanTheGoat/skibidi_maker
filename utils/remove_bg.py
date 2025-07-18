import onnxruntime as ort
import numpy as np
import cv2 as cv

def _preprocess(image):
    original_size = (image.shape[1], image.shape[0])  # (width, height)

    # Resize to 320x320
    image_resized = cv.resize(image, (320, 320), interpolation=cv.INTER_LINEAR)

    # Convert to float32 and normalize
    img_np = image_resized.astype(np.float32) / 255.0
    img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)

    return img_np, original_size, image

def _postprocess(mask, original_size):
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = cv.resize(mask, original_size, interpolation=cv.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    return mask

def _apply_mask_with_transparency(original_image, mask):
    # Ensure mask is single channel uint8
    alpha = mask

    # Convert original image to BGRA
    bgr = original_image
    bgra = cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)

    # Set alpha channel to the mask
    bgra[:, :, 3] = alpha
    return bgra

def _remove_background_onnx(image, model, output_path="output.png"):
    input_tensor, original_size, original_image = _preprocess(image)

    # Run ONNX inference
    session: ort.InferenceSession = model
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})

    # Get mask from output
    d1 = output[0][0][0]  # type: ignore
    mask = _postprocess(d1, original_size)

    # Create transparent result
    result = _apply_mask_with_transparency(original_image, mask)

    return result

def remBg(image:cv.typing.MatLike, model:ort.InferenceSession):
    return _remove_background_onnx(image, model)



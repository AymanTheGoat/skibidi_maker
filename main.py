import os
import cv2 as cv
import onnxruntime as ort
from utils.remove_bg import remBg
from utils.haar_detection import getBoundingBoxsHaar
from utils.image_utils import getOverlayImage, alpha_blend
from utils.ULFGFD_detection import getBoundingBoxsULFGFD
from utils.file_utils import check_file_exists, getSmallestAvailableNumber


def main(image, method):
    if method == 1:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        boundingboxs = getBoundingBoxsHaar(image_gray, face_cascade)
    else :
        boundingboxs = getBoundingBoxsULFGFD(image, ulfgfd_path)
    cropped_images = []

    noBg = remBg(image, u2net_session)

    # Crop the faces from the image
    for pt1, pt2 in boundingboxs:
        # Uncomment to visualize bounding boxes
        # cv.rectangle(image, pt1, pt2, (255, 0, 0), 2) 
        
        x1, y1 = pt1
        x2, y2 = pt2

        cropped_image = noBg[y1:y2, x1:x2]
        cropped_images.append(cropped_image)

    i = getSmallestAvailableNumber()

    for cropped_image in cropped_images:
        # Resize the cropped image and overlay it on a blank canvas
        overlay_image = getOverlayImage(cropped_image)
        toilet_image = toiletImage
        toilet_overlay_image = toiletOverlayImage

        stack1 = alpha_blend(overlay_image, toilet_image)
        stack2 = alpha_blend(toilet_overlay_image, stack1)

        # Save the final output
        cv.imwrite(f'output/output_{i}.png', stack2)
        i += 1



if __name__ == "__main__":

    input_path = 'input/image5.png'
    output_path = 'assets/output.png'
    toilet_path = 'assets/toilet.png'
    toilet_overlay_path = 'assets/toilet_overlay.png'
    u2net_path = 'weights/u2net.onnx'
    ulfgfd_path = 'weights/version-RFB-640.onnx'
    haarcascade_path = 'weights/haarcascade.xml'
    method = 1  # 1 for Haar Cascade, 2 for ULFGFD


    # Load the image
    image = cv.imread(input_path)
    toiletImage = cv.imread(toilet_path, cv.IMREAD_UNCHANGED)
    toiletOverlayImage = cv.imread(toilet_overlay_path, cv.IMREAD_UNCHANGED)


    # Check if files exist
    for path in [input_path, toilet_path, toilet_overlay_path, u2net_path, ulfgfd_path, haarcascade_path]:
        check_file_exists(path)


    # Load the necessary weights
    face_cascade = cv.CascadeClassifier(haarcascade_path)
    u2net_session = ort.InferenceSession(u2net_path)


    main(image, method)

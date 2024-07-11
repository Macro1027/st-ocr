import cv2
import numpy as np
from deskew import determine_skew

def grayscale(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def remove_noise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced

def deskew(image):
    # Deskewing (assuming text is horizontal)
    angle = determine_skew(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the actual rotation and return the image
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def binarize(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def preprocess_image(image):
    """Preprocesses the image by enhancing contrast, removing noise, and deskewing."""
    gray = grayscale(image)
    denoised = remove_noise(gray)
    enhanced = enhance_contrast(denoised)
    # deskewed = deskew(enhanced)
    # binary = binarize(deskewed)
    binary = binarize(enhanced)
    return binary

def test_preprocess_image():
    # Create a sample image for testing
    test_image = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(test_image, 'skibidi!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Preprocess the image
    preprocessed = preprocess_image(test_image)

    # Save the preprocessed image
    cv2.imwrite('images/test_image.png', test_image)
    cv2.imwrite('images/preprocessed_test_image.png', preprocessed)
    print('Preprocessed image saved as preprocessed_test_image.png')

# Run the test function
if __name__ == "__main__":
    test_preprocess_image()

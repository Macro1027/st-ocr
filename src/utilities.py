import cv2
import numpy as np
from deskew import determine_skew
from spellchecker import SpellChecker
import streamlit as st

from src.perplexity_api import chat_completion
from src.st_context import with_streamlit_context

class Utilities:
    def __init__(self):
        # Initialize the spell checker
        self.spell = SpellChecker()

    def fetch_likely_text(self):
        """Fetches likely text based on latest OCR values."""
        # Use the chat_completion function to fetch the latest OCR values from the session state
        return chat_completion(f"latest_ocr_values = {st.session_state['latest']}")

    @with_streamlit_context
    def detect_annotations(self, frame, text_queue, conf_thresh):
        """Detects annotations for a single video frame."""
        # If the text queue is empty, return an empty list
        if text_queue.empty():
            return []

        # Get detections from the text queue
        detections = text_queue.get()
        annotations = []
        for (box, text, confidence) in detections:
            # Only consider detections with confidence above the threshold
            if confidence > conf_thresh / 100.0:
                # Correct the spelling of the detected text
                corrected_text = self.correct_spelling(text)
                # Append the bounding box and corrected text to annotations
                annotations.append((box, corrected_text))
        return annotations

    @with_streamlit_context
    def draw_annotations(self, frame, annotations):
        """Draws annotations on the frame."""
        for (box, text) in annotations:
            try:
                # Calculate the size of the text box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size

                # Calculate the position for the rectangle and text
                p1 = (int(box[0][0]), int(box[0][1]))
                p2 = (p1[0] + text_width, p1[1] - text_height - baseline)

                # Draw a filled rectangle with transparency
                overlay = frame.copy()
                cv2.rectangle(overlay, p1, p2, (0, 255, 0), -1)
                alpha = 0.4  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Put the text on top of the rectangle
                cv2.putText(frame, text, (p1[0], p1[1] - baseline), font, font_scale, (0, 0, 0), thickness)
            except Exception as e:
                # Log an error message if annotation fails
                st.error(f"Failed to annotate frame: {e}")
        return frame

    def _grayscale(self, image):
        """Converts the image to grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray

    def _remove_noise(self, image):
        """Removes noise from the image using Non-Local Means Denoising."""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def _enhance_contrast(self, image):
        """Enhances the contrast of the image using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        return enhanced

    def _deskew(self, image):
        """Deskews the image assuming the text is horizontal."""
        angle = determine_skew(image)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform the actual rotation and return the image
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def _binarize(self, image):
        """Converts the image to a binary image using Otsu's binarization."""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def preprocess_image(self, image):
        """Preprocesses the image by enhancing contrast, removing noise, and deskewing."""
        gray = self._grayscale(image)
        denoised = self._remove_noise(gray)
        enhanced = self._enhance_contrast(denoised)
        deskewed = self._deskew(enhanced)
        binary = self._binarize(deskewed)
        return binary

    def correct_spelling(self, text):
        """Corrects the spelling of the given text."""
        corrected_text = []
        for word in text.split():
            corrected_word = self.spell.correction(word)
            if corrected_word:
                corrected_text.append(corrected_word)      

        return ' '.join(corrected_text)
    
    def overlay_annotations(self, frame, annotated_frame):
        """Overlay annotations from the annotated frame onto the current frame."""
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(annotated_frame, alpha, frame, 1 - alpha, 0, frame)
        return frame
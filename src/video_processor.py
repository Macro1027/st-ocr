import cv2
from collections import deque
from streamlit_webrtc import VideoProcessorBase
from src.utilities import Utilities
from src.st_context import with_streamlit_context

class VideoProcessor(VideoProcessorBase):
    def __init__(self, queues, conf_thresh, n, k):
        self.queues = queues
        self.conf_thresh = conf_thresh
        self.n = n
        self.k = k
        self.frame_counter = 0
        self.utilities = Utilities()
        self.annotation_counter = 0  

    @with_streamlit_context
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        if self.frame_counter % self.n == 0:
            # Preprocess the frame
            preprocessed_frame = self.utilities.preprocess_image(img)

            # Add frame to queue if it is empty
            if self.queues['frame_queue'].empty():
                self.queues['frame_queue'].put(preprocessed_frame)
            
            annotations = self.utilities.detect_annotations(preprocessed_frame, self.queues['text_queue'], self.conf_thresh)
            if annotations:
                self.queues['annotation_queue'].put((annotations, self.k))  # Store annotations with counter
                self.annotation_counter = self.k  # Reset the counter

        # Draw annotations from the queue
        if not self.queues['annotation_queue'].empty() and self.annotation_counter > 0:
            annotations, _ = self.queues['annotation_queue'].get()
            img = self.utilities.draw_annotations(img, annotations)
            self.annotation_counter -= 1

        return img
import cv2
import queue
import numpy as np
import threading
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.ocr import ocr_thread
from src.chatbot import run_chatbot
from src.perplexity_api import chat_completion
from src.spelling import correct_spelling
from src.preprocessing import preprocess_image  
from src.st_decorator import with_streamlit_context

# Globals
thread_lock = threading.Lock()

def initialize_system():
    """Initializes queues and starts OCR thread."""
    queues = {
        'frame_queue': queue.Queue(maxsize=1),
        'text_queue': queue.Queue(maxsize=1),
        'prompt_queue': queue.Queue(maxsize=1),
        'ppx_queue': queue.Queue(maxsize=1)
    }
    ocr_thread_with_ctx = threading.Thread(target=ocr_thread, args=(queues['frame_queue'], queues['text_queue']))
    add_script_run_ctx(ocr_thread_with_ctx)
    ocr_thread_with_ctx.start()
    return queues

def fetch_likely_text():
    """Fetches likely text based on latest OCR values."""
    return chat_completion(f"latest_ocr_values = {st.session_state['latest']}")

@with_streamlit_context
def process_frame(frame, text_queue, conf_thresh):
    """Processes a single video frame for OCR results."""
    cv2.imwrite("images/img.png", frame)
    if text_queue.empty():
        return None, ""

    detections = text_queue.get()
    annotated_frame = frame.copy()
    detected_texts = []
    for (box, text, confidence) in detections:
        if confidence > conf_thresh / 100.0:
            print(detected_texts)
            try:
                # Adjust the bounding box coordinates according to the resized frame
                p1 = (int(box[0][0]), int(box[0][1]))
                p2 = (int(box[2][0]), int(box[2][1]))
                cv2.rectangle(annotated_frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(annotated_frame, correct_spelling(text), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Failed to annotate frame: {e}")
            detected_texts.append(correct_spelling(text))
    return annotated_frame, ' '.join(detected_texts)

class VideoProcessor(VideoProcessorBase):
    def __init__(self, queues, conf_thresh, n):
        self.queues = queues
        self.conf_thresh = conf_thresh
        self.n = n
        self.frame_counter = 0

    @with_streamlit_context
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        if self.frame_counter % self.n == 0:
            # Preprocess the frame
            preprocessed_frame = preprocess_image(img)

            # Add frame to queue if it is empty
            if self.queues['frame_queue'].empty():
                self.queues['frame_queue'].put(preprocessed_frame)
            
            processed_frame, detected_text = process_frame(img, self.queues['text_queue'], self.conf_thresh)
            if detected_text:
                st.session_state.latest.append(detected_text)
            return processed_frame if processed_frame is not None else img
        else:
            return img

def main():
    st.title('OCR and Chatbot Application')
    if "camera_frozen" not in st.session_state:
        st.session_state.update({"camera_frozen": False, "latest": [], "likely_text": ""})

    queues = initialize_system()
    conf_thresh = st.slider('Confidence Threshold', 0, 100, 50)
    n = st.slider('Process every n frames', 1, 30, 5)  # Add a slider to select n frames

    if st.button("Freeze" if not st.session_state.camera_frozen else "Resume"):
        st.session_state.camera_frozen = not st.session_state.camera_frozen
        if st.session_state.camera_frozen:
            st.session_state.likely_text = fetch_likely_text()

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: VideoProcessor(queues, conf_thresh, n),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.session_state.camera_frozen and st.session_state.likely_text:
        st.write(st.session_state.likely_text)
    else:
        st.write("No text found")

    run_chatbot()

if __name__ == '__main__':
    main()
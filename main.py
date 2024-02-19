import cv2
import queue
import numpy as np
import threading
import streamlit as st

from src.ocr import ocr_thread
from src.chatbot import run_chatbot
from src.perplexity_api import chat_completion
from src.spelling import correct_spelling

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
    threading.Thread(target=ocr_thread, args=(queues['frame_queue'], queues['text_queue'])).start()
    return queues

def setup_video_capture():
    """Configures and returns the video capture object."""
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return capture

def fetch_likely_text():
    """Fetches likely text based on latest OCR values."""
    return chat_completion(f"latest_ocr_values = {st.session_state["latest"]}")

def process_frame(frame, text_queue, conf_thresh):
    """Processes a single video frame for OCR results."""
    if text_queue.empty():
        return None, ""

    detections = text_queue.get()
    annotated_frame = frame.copy()
    detected_texts = []
    for (box, text, confidence) in detections:
        if confidence > conf_thresh / 100.0:
            try:
                cv2.rectangle(annotated_frame, tuple(box[0]), tuple(box[2]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, correct_spelling(text), box[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print("Failed to add rectangle")
            detected_texts.append(correct_spelling(text))
    return annotated_frame, ' '.join(detected_texts)

def main():
    st.title('OCR and Chatbot Application')
    if "camera_frozen" not in st.session_state:
        st.session_state.update({"camera_frozen": False, "latest": [], "likely_text": ""})

    queues = initialize_system()
    cap = setup_video_capture()

    FRAME_WINDOW = st.image([])
    run = st.checkbox('Run')
    conf_thresh = st.slider('Confidence Threshold', 0, 100, 50)

    if st.button("Freeze" if not st.session_state.camera_frozen else "Resume"):
        st.session_state.camera_frozen = not st.session_state.camera_frozen
        if st.session_state.camera_frozen:
            st.session_state.likely_text = fetch_likely_text()

    while run and not st.session_state.camera_frozen:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (840, 480))
        processed_frame, detected_text = process_frame(frame, queues['text_queue'], conf_thresh)
        print(detected_text)
        if detected_text:
            st.session_state.latest.append(detected_text)
        FRAME_WINDOW.image(processed_frame if processed_frame is not None else frame, channels="BGR")

        # Add frame to queue if it is empty
        if queues['frame_queue'].empty():
            queues['frame_queue'].put(frame)

    if st.session_state.camera_frozen and st.session_state.likely_text:
        st.write(st.session_state.likely_text)
        print(st.session_state.likely_text)
    else:
        print("no text found")

    run_chatbot()
    cap.release()

if __name__ == '__main__':
    main()

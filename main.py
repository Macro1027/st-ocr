import cv2
import time
import queue
import threading
import streamlit as st

from scripts.ocr import ocr_thread
from scripts.chatbot import run_chatbot
from scripts.spelling import correct_spelling
from scripts.perplexity_api import chat_completion

def initialize_queues():
    # Create and return all necessary queues for the application.
    return {
        'frame_queue': queue.Queue(maxsize=1),
        'text_queue': queue.Queue(maxsize=1),
        'prompt_queue': queue.Queue(maxsize=1),
        'ppx_queue': queue.Queue(maxsize=1)
    }

def start_threads(queues):
    # Start OCR and chat completion threads.
    threading.Thread(target=ocr_thread, args=(queues['frame_queue'], queues['text_queue'])).start()

def capture_video():
    # Set up and return video capture object.
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return video_capture

def freeze_frame():
    likely_text = chat_completion(f"latest_ocr_values = {st.session_state.latest}")
    print(likely_text)


def display_ocr_results(frame, text_queue, conf_thresh=50, text_placeholder=None):
    # Display OCR results on the frame if available.
    if not text_queue.empty():

        d = text_queue.get()

        n_boxes = len(d)
        to_append = []
        # Loop through each text box
        for i in range(n_boxes):
            # Check if confidence level is above 50%
            if d[i][2] > 0.5:
                edge1 = tuple([int(i) for i in d[i][0][0]])
                edge2 = tuple([int(i) for i in d[i][0][2]])
                detected_text = correct_spelling(d[i][1])

                to_append.append(detected_text)
                # Display bounding box and text
                cv2.rectangle(frame, edge1, edge2, (0, 255, 0), 2)
                cv2.putText(frame, detected_text, edge1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if len(to_append) > 0:
            text_placeholder.markdown(' '.join(to_append))
            return to_append
        
        return None

def main():
    # Global variables
    if "latest" not in st.session_state:
        st.session_state.latest = []

    if st.button('hello'):
        freeze_frame()

    # Initialise queues and start threads
    queues = initialize_queues()
    start_threads(queues)

    # Create streamlit widgets
    st.title('Webcam Live Feed')
    FRAME_WINDOW = st.image([])
    run = st.checkbox('Run')
    text_placeholder = st.empty()
    conf_thresh = st.slider('Confidence Threshold', 0, 100, 50)
    run_chatbot()

    # Start video capture
    cap = capture_video()
    while run:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (840, 480))

        # Display OCR results on the frame if available, whilst retrieving latest text.
        output = display_ocr_results(frame, queues['text_queue'], conf_thresh, text_placeholder)
        
        # Add latest text to list
        if output:
            st.session_state["latest"].append(output)
            if len(st.session_state.latest) > 3:
                st.session_state.latest = st.session_state.latest[1:]

        # Add frame to queue if it is empty
        if queues['frame_queue'].empty():
            queues['frame_queue'].put(frame)


        FRAME_WINDOW.image(frame, channels="BGR")
    else:
        st.write('Stopped')
        
    # Release reesources
    cap.release()
    cv2.destroyAllWindows()


# Plan - 
#    1. Make the text stay longer
#    2. Allow questions to be asked to ppx

if __name__ == '__main__':
    main()

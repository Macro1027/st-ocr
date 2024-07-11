import queue
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from src.ocr import ocr_thread
from src.chatbot import run_chatbot

class SystemInitializer:
    def initialize_system(self):
        """Initializes queues and starts OCR thread."""
        queues = {
            'frame_queue': queue.Queue(maxsize=1),
            'text_queue': queue.Queue(maxsize=1),
            'annotation_queue': queue.Queue(maxsize=1),
            'prompt_queue': queue.Queue(maxsize=1),
            'ppx_queue': queue.Queue(maxsize=1)
        }
        ocr_thread_with_ctx = threading.Thread(target=ocr_thread, args=(queues['frame_queue'], queues['text_queue']))
        add_script_run_ctx(ocr_thread_with_ctx)
        ocr_thread_with_ctx.start()
        return queues

    def run_chatbot(self):
        run_chatbot()
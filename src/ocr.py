# Perform OCR in a separate thread
def ocr_thread(frame_queue, text_queue):
    from easyocr import Reader
    reader = Reader(lang_list=['en'])
    
    while True:
        frame = frame_queue.get()

        # If queue is empty, exit the loop
        if frame is None:
            break
 
        texts = reader.readtext(frame)
        text_queue.put(texts)
        print(texts)

if __name__ == "__main__":
    import sys
    print(sys.path)

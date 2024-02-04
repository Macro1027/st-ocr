import requests

APY_KEY = "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"

url = "https://api.perplexity.ai/chat/completions"

ppx_prompt = '''Given a Python list named latest_ocr_values, such as ['I am a Python', 'I a pthon', 
'I python', 'I am a python', 'a'], directly provide the text that is most consistently detected by the OCR. 
If multiple answers are possible, choose the most likely one only. If there is no clear answer, state 'None'. 
The output must follow the format: 'your_answer'. Do not explain yourself afterwards, do not include
multiple valid outputs. Do not include any other information.'''

headers = {
    "Authorization": "Bearer " + APY_KEY,
    "accept": "application/json",
    "content-type": "application/json"
}

payload = {
    "model": "mistral-7b-instruct",
    "messages": [
        {
            "role": "system",
            "content": ppx_prompt
        },
        {
            "role": "user",
            "content": ""
        }
    ]
}
    
# Perform a chat completion in a separate thread
def chat_completion(prompt):
    while True:
        # If queue is empty, exit the loop
        if not prompt:
            break
        
        # Copy payload and insert prompt
        pl = payload.copy()
        pl["messages"][1]["content"] = prompt

        # Perform chat completion and add results to queue
        response = requests.post(url, json=pl, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            output = response_data.get("choices")[0].get("message").get("content")
            return output
        else:
            return "Error"

    
if __name__ == "__main__":
    print(chat_completion("latest_ocr_values = ['I am a cat', 'I a cat', 'I cat', 'I am cat', 'a']"))
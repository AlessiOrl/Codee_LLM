import json
import os
from threading import Thread
import time
import torch
from transformers import pipeline
from PIL import Image
import requests
import torch
from app.Model import Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, TextStreamer

def history_isvalid(file_path) -> bool:
    # Get the file status using os.stat() function
    filestat = os.stat(file_path)
    # Get the last modified date and time in local time
    date = time.localtime((filestat.st_mtime))

    # Get the current date and time in local time
    current_date = time.localtime()

    # Get the time difference in seconds
    time_diff = time.mktime(current_date) - time.mktime(date)

    # Get the time difference in minutes
    time_diff = time_diff / 60
    # If the file was modified in the last 180 minutes
    if time_diff < 180:
        return  True
    print("History do not respect the conditions to remain valid, it will be overwritten.")
    return False

class Mistral(Model):
    def __init__(self):
        super().__init__(model_id = "mistralai/Mistral-7B-Instruct-v0.3",
                         device = "cuda" if torch.cuda.is_available() else "cpu", 
                         torch_dtype = "auto")
        self.model_running = False

    def load_prompt(self, file_path):
        # load prompt from file
        with open(file_path, "r", encoding="utf8" ) as f:
            prompt = f.read()
        return prompt


    def add_message(self, user_id, user, message):
        messages = self.load_message_history(f"./data/{user_id}/message_history.json")
        if messages is None:
            messages = []
        messages.append({"role": user, "content": message})
        self.save_message_history(messages, f"./data/{user_id}/message_history.json")
        return messages

    def save_message_history(self,messages,file_path):
        with open(file_path, "w", encoding="utf8" ) as f:
            json.dump(messages, f)

    def load_message_history(self,file_path):
        messages = []
        try:
            if not history_isvalid(file_path):
                
                return messages
            with open(file_path, "r", encoding="utf8" ) as f:
                old_messages = json.load(f)
                # get the last 10 messages
                if len(old_messages) > 10:
                    old_messages = old_messages[-10:]
                for message in old_messages:
                    messages.append(message)
        except:
            pass
        return messages

    def currently_running(self):
        return self.model_running


    def load_model(self):
        if self.model is not None:
            return True
        # create quantization config with bitsandbytes, not with FineGrainedFP8Config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_activation_dtype=torch.float16),
            use_flash_attention_2=False
        )
        return True


    def get_prediction(self, string: str, user_id="default"):
        try:
            self.model_running = True
            self.load_model()
            messages = self.add_message(user_id, "user", string)
            prompt = [{"role": "system", "content": self.load_prompt("./data/codee_assistant_prompt.txt")}]
            full_prompt = prompt + messages
            
            full_prompt = self.tokenizer.apply_chat_template(full_prompt, add_generation_prompt=True, tokenize=False)
            full_prompt = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

            generation_kwargs = dict(full_prompt, max_new_tokens=500, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, streamer=streamer)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
        except Exception as e:
            print("Error: ", e)
            streamer = ["I'm sorry, I'm having an internal error right now. Please contact the mantainer of the OrlandosNas Sever."]
        return streamer
 
    def save_model(self):
        self.model.save_pretrained("data/models/llama")
        pass
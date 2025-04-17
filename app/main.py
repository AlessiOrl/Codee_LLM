from time import sleep
import datetime, os
from pydantic import BaseModel
from app.WhisperModel import WhisperModel
from app.Llama import Llama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

def load_queue():
    queue = {}
    for user_id in os.listdir("data"):
        # check if the user_id is a folder
        if not os.path.isdir(f"data/{user_id}"):
            continue
        queue[user_id] = set()
        for audio_file in os.listdir(f"data/{user_id}"):
            # check if the file is an audio file in wav format or mp3 format or ogg format
            if audio_file.endswith(".wav") or audio_file.endswith(".mp3") or audio_file.endswith(".ogg"):
                queue[user_id].add(audio_file)
    return queue

app = FastAPI(title="Whisper Large", description="This is a Whisper Large model for speech recognition")
queue = load_queue()
model_whisper = WhisperModel()
model_llm = Llama()
print("Loading model... ", end="")
model_llm.load_model()
print("Model loaded.")

class Message(BaseModel):
  message: str

@app.get("/")
def health_check():
    return {"health_check": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/help")
def help():
    return {"GET /": "Health check", 
            "GET /help": "Returns this help message", 
            "GET /info": "Returns information about the model and the device it is running on.",
            "GET /predict_whisper/{user_id}": "Predicts the text from the audio files in the queue of the user_id. Optionally removes the audio files after prediction.", 
            "GET /queue/{user_id}": "Returns the list of audio files in the queue of the user_id. Optionally removes the audio files after prediction with the remove_after query parameter.", 
            "POST /queue/{user_id}": "Adds audio file to the queue of the user_id. The audio file should be sent in the request body binary format. Returns the list of audio files in the queue of the user_id.",
            "GET /clear_queue/{user_id}": "Clears the queue of the user_id. Optionally removes the audio files after prediction with the remove_after query parameter.",
            "POST /predict_llm/{user_id}": "Chatbot call to LLM model. Returns the response from the model."
            }

@app.get("/info")
def get_info():
    return {"whisper model": f"{model_whisper.model_id} on {model_whisper.device}: {'is' if model_whisper.currently_running() else 'is not'} currently running.",
            "gemma model": f"{model_llm.model_id} on {model_llm.device}: {'is' if model_llm.currently_running() else 'is not'} currently running."}

@app.post("/queue/{user_id}")
def add_queue(user_id: str, audio_file: UploadFile):
    dir_path = os.path.join("data", f"{user_id}")
    os.makedirs(dir_path, exist_ok=True)
    audio_file_path = os.path.join(dir_path, audio_file.filename)

    print(f"Saving audio file to {audio_file_path}")
    with open(audio_file_path, "wb") as f: 
        f.write(audio_file.file.read())

    if user_id not in queue:
        queue[user_id] = set()
    
    queue[user_id].add(audio_file.filename)

    return get_queue(user_id)

@app.get("/queue/{user_id}")
def get_queue(user_id: str):
    return queue.get(user_id, [])

@app.get("/clear_queue/{user_id}")
def clear_queue(user_id: str):
    if user_id in queue:
        queue[user_id] = set()
    # delete all files in the data folder of the user
    # check if the folder exists
    if os.path.exists(f"data/{user_id}"):
        for filename in os.listdir(f"data/{user_id}"):
            os.remove(os.path.join("data",f"{user_id}", filename))
    return get_queue(user_id)

@app.get("/predict_whisper/{user_id}")
def predict(user_id: str, remove: bool = True):
    if len(queue.get(user_id, [])) == 0:
        raise HTTPException(status_code=401, detail= "No audio files in the queue.")
    if model_whisper.currently_running():
        raise HTTPException(status_code=401, detail="Model is currently running, please try again later.")
    
    result = {}
    for audio_file in queue.get(user_id, []):
        audio_file_path = os.path.join("data", f"{user_id}", audio_file)

        text = model_whisper.get_prediction(audio_file_path=audio_file_path)
        result[audio_file] = text

    if remove:
        print("Removing audio files...")
        while len(queue.get(user_id, [])) > 0:
            audio_file_path = os.path.join("data", f"{user_id}", queue[user_id].pop())
            os.remove(audio_file_path)
    return result

@app.get("/restart_model")
def restart_model():
    model_whisper.unload_model()

    return {"status": "Model restarted."}


@app.get("/predict_llm/{user_id}")
def predict_llm(user_id: str, message: Message):
    message = message.message
    while model_llm.currently_running():
        raise HTTPException(status_code=401, detail="Model is currently running, please try again later.")
    dir_path = os.path.join("data", f"{user_id}")
    os.makedirs(dir_path, exist_ok=True)
    
    streamer = model_llm.get_prediction(message, user_id=user_id)

    return StreamingResponse(streamer, media_type='text/event-stream')


@app.post("/predicted_text/{user_id}")
def get_predicted_text(user_id: str, message: Message=None):
    if message is not None:
        message = message.message
        model_llm.add_message(user_id, "assistant", message)
    model_llm.model_running = False
    return {"status": "Done"}
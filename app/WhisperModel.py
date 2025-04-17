from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch, gc
from app.Model import Model


class WhisperModel(Model):
    def __init__(self):
        super().__init__(model_id = "openai/whisper-large-v3-turbo", 
                         device = "cpu", 
                         torch_dtype = torch.float16)

    def currently_running(self):
        return self.model is not None

    def load_model(self):
        if self.model is not None:
            return True
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        return True
    
    def get_prediction(self, audio_file_path):
        self.load_model()

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=20,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        generate_kwargs = {
        "num_beams": 5,
        "condition_on_prev_tokens": True,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        "temperature": (0.1, 0.3, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "return_timestamps": True,
        }

        result = pipe(audio_file_path, generate_kwargs=generate_kwargs)

        self.unload_model()

        return result["text"]


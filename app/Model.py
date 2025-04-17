from abc import abstractmethod
import torch, gc

class Model():
    def __init__(self, model_id: str, device: str = "cpu", torch_dtype = torch.float32):
        self.model = None
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

    @abstractmethod
    def get_prediction(self, audio_file_path: str) -> str:
        pass

    @abstractmethod
    def currently_running(self) -> bool:
        pass

    def unload_model(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return True
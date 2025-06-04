import torch
import fairseq2
import pandas as pd

from sonar.models.mutox import get_mutox_model_hub
from sonar.models.mutox.model import MutoxClassifier
from sonar.models.sonar_speech import get_sonar_speech_encoder_hub
from sonar.models.sonar_speech.model import SonarSpeechEncoderModel



fairseq2.setup_fairseq2()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


def load_mutox_model(model_name: str, device=None, dtype=None) -> MutoxClassifier:
    model_hub = get_mutox_model_hub()
    model = model_hub.load(model_name).to(device=device, dtype=dtype)
    return model

def load_sonar_model(model_name: str, device=None, dtype=None):
    model_hub = get_sonar_speech_encoder_hub()
    model = model_hub.load(model_name).to(device=device, dtype=dtype)
    return model

class MuToxClassifier:
    def __init__(self, lang, device='cuda:0'):
        self.encoder = load_sonar_model(f'sonar_speech_encoder_{lang}', device=device)
        self.model = load_mutox_model(model_name='sonar_mutox', device=device)
    def __call__(self, x):
        return self.model(self.encoder(x))

encoder = load_sonar_model('sonar_speech_encoder_arb', device='cuda:0')
print(sum(p.numel() for p in model.parameters()))
print(encoder)
# print(sum(p.numel() for p in model.parameters()))


# load phonemizer
import torch
import numpy as np
import random
import nltk
import time
import random
import yaml
import phonemizer
import torch.nn.functional as F
from StyleTTS2.models import *
from StyleTTS2.text_utils import TextCleaner
from utils import length_to_mask
from StyleTTS2.Utils.PLBERT.util import load_plbert
from StyleTTS2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
textcleaner = TextCleaner()
# --- Text to speech ---
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(0)
np.random.seed(0)

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
# load packages
from munch import Munch

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch({k: recursive_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_munch(i) for i in d]
    else:
        return d 
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

textcleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
config = yaml.safe_load(open("./StyleTTS2/Models/LJSpeech/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# # load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from StyleTTS2.Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_tts = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model_tts[key].eval() for key in model_tts]
_ = [model_tts[key].to(device) for key in model_tts]

params_whole = torch.load("./StyleTTS2/Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
params = params_whole['net']

for key in model_tts:
    if key in params:
        print('%s loaded' % key)
        try:
            model_tts[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model_tts[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model_tts[key].eval() for key in model_tts]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sampler = DiffusionSampler(
    model_tts.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
  text = text.strip()
  text = text.replace('"', '')
  ps = global_phonemizer.phonemize([text])
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)

  tokens = textcleaner(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
      text_mask = length_to_mask(input_lengths).to(tokens.device)

      t_en = model_tts.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model_tts.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model_tts.bert_encoder(bert_dur).transpose(-1, -2)

      s_pred = sampler(noise,
            embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
            embedding_scale=embedding_scale).squeeze(0)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = alpha * s_prev + (1 - alpha) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      d = model_tts.predictor.text_encoder(d_en, s, input_lengths, text_mask)

      x, _ = model_tts.predictor.lstm(d)
      duration = model_tts.predictor.duration_proj(x)
      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)

      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      F0_pred, N_pred = model_tts.predictor.F0Ntrain(en, s)
      out = model_tts.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))

  return out.squeeze().cpu().numpy(), s_pred
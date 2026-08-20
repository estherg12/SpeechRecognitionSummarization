from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
from transformers import pipeline

FRAME_RATE = 16000 # Cualidad del audio
CHANNELS = 1 # Mono

model = Model(model_name="vosk-model-es-0.42") # Cargar el modelo de reconocimiento de voz

rec = KaldiRecognizer(model, FRAME_RATE) # Crear un reconocedor de voz con el modelo y la calidad del audio
rec.SetWords(True) 

# Cargar el archivo de audio mp3 y convertirlo a mono y a la calidad de audio deseada
mp3 = AudioSegment.from_file("NoticiaMuyCorta.mp3", format="mp3") 
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

# Transcribir el audio a texto
rec.AcceptWaveform(mp3.raw_data)
result = rec.Result() 
text = json.loads(result).get("text", "")
print(text)

# Añadir puntuación al texto transcrito
pipe = pipeline("token-classification", model="HiTZ/cap-punct-es")
predictions = pipe(text)
print(predictions)
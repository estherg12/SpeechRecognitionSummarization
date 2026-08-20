from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FRAME_RATE = 16000 # Cualidad del audio
CHANNELS = 1 # Mono

def voice_recognition(filename):
    model = Model(model_name="vosk-model-es-0.42") # Cargar el modelo de reconocimiento de voz

    rec = KaldiRecognizer(model, FRAME_RATE) # Crear un reconocedor de voz con el modelo y la calidad del audio
    rec.SetWords(True) 

    # Cargar el archivo de audio mp3 y convertirlo a mono y a la calidad de audio deseada
    mp3 = AudioSegment.from_file(filename, format="mp3") 
    mp3 = mp3.set_channels(CHANNELS)
    mp3 = mp3.set_frame_rate(FRAME_RATE)

    step = 40000 # Tamaño del fragmento de audio a procesar en milisegundos
    transcript = ""
    # Procesar el audio en fragmentos y transcribirlo a texto
    for i in range(0, len(mp3), step):
        print(f"Progress: {i}/{len(mp3)}")
        segment = mp3[i:i+step]
        # Transcribir el audio a texto
        rec.AcceptWaveform(segment.raw_data)
        result = rec.Result() 
        text = json.loads(result).get("text", "")
        transcript += text + " "

    # Añadir puntuación al texto transcrito
    tokenizer = AutoTokenizer.from_pretrained("HiTZ/cap-punct-es")
    modelPunctuation = AutoModelForSeq2SeqLM.from_pretrained("HiTZ/cap-punct-es")

    inputs = tokenizer(transcript, return_tensors="pt")
    outputs = modelPunctuation.generate(**inputs)
    punctuated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(punctuated_text)

voice_recognition("NoticiaMuyCorta.mp3")
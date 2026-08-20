from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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

    raw_words = transcript.split(" ")
    chunk_size = 300
    punctuated_chunks = []

    for i in range(0, len(raw_words), chunk_size):
        chunk = " ".join(raw_words[i:i+chunk_size])
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
        outputs = modelPunctuation.generate(**inputs)
        decoded_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        punctuated_chunks.append(decoded_chunk)

    punctuated_text = " ".join(punctuated_chunks)
    print(punctuated_text)

    # Resumir el texto transcrito
    sum_tokenizer = AutoTokenizer.from_pretrained("LeoCordoba/mt5-small-mlsum")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("LeoCordoba/mt5-small-mlsum")

    split_tokens = punctuated_text.split(" ")
    docs = []

    for i in range(0, len(split_tokens), 512):
        selection = split_tokens[i:i+512]
        docs.append(" ".join(selection))

    summaries = []

    for doc in docs:
        inputs = sum_tokenizer(doc, return_tensors="pt", max_length=512, truncation=True)
        outputs = sum_model.generate(**inputs, max_new_tokens=150, min_length=40, num_beams=4, early_stopping=True)
        summary_text = sum_tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary_text)
    final_summary = "\n\n".join(summaries)
    print(final_summary)


voice_recognition("HistoriaCorto.mp3")


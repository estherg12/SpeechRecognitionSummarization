# Speech Recognition and Summarization System
This project is a Python-based pipeline that takes a Spanish audio file (such as a downloaded YouTube MP3), transcribes it into text, restores the missing punctuation and capitalization, and finally generates a concise summary of the content.

## Features
* **Speech-to-Text:** Utilizes offline speech recognition via Vosk.
* **Audio Processing:** Automatically converts incoming audio to the correct format (16kHz, mono) using `pydub` and `ffmpeg`.
* **Punctuation Restoration:** Uses a Hugging Face translation model (`HiTZ/cap-punct-es`) to intelligently add periods, commas, and capitalization to the raw transcript.
* **Summarization:** Employs a multilingual T5 model (`LeoCordoba/mt5-small-mlsum`) to break down long transcriptions into readable summaries.
* **Smart Chunking:** Bypasses token limits (512 tokens) by processing both punctuation and summarization in manageable text chunks.

## Requirements
We will be using the following libraries:
- Vosk: from which you can choose your own model (we chose the big model for Spanish) https://alphacephei.com/vosk/models
  In order to install this librarie execute the following code in your bash: `pip install vosk`
- Pydub: https://github.com/jiaaro/pydub
  In order to install this librarie execute the following code in your bash: `pip install pydub`
  Since pydub can only read `WAV` files natively, it is also required to have `ffmpeg` installed, so it can decode and open compressed formats like MP3s. Execute `winget install ffmpeg` in your bash.
- Hugging Face Models for punctuation (HiTZ/cap-punct-es) and summarization (https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner): `pip install sentencepiece tiktoken protobuf`
  Also, for text tokenization execute `pip install sacremoses`.
- Transformers: `pip install transformers`
- Torch: `pip install torch -f https://download.pytorch.org/whl/torch_stable.html`

## Usage
1. Install all the libraries and the recognition.py file from the repository.
2. Place your MP3 file in the project directory.
3. Update the voice_recognition("YourFile.mp3") function call at the bottom of the script with your file's name.
4. Run the script.


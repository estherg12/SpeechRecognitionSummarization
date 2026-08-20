# Speech Recognition and Summarization System
This project was selected and is based on a tutorial from @Dataquest.

## Requirements
We will be using the following libraries:
- Vosk: from which you can choose your own model (we chose the big model for Spanish) https://alphacephei.com/vosk/models
  In order to install this librarie execute the following code in your bash: `pip install vosk`
- Pydub: https://github.com/jiaaro/pydub
  In order to install this librarie execute the following code in your bash: `pip install pydub`
  Since pydub can only read `WAV` files natively, it is also required to have `ffmpeg` installed, so it can decode and open compressed formats like MP3s. Execute `winget install ffmpeg` in your bash.
- Hugging Face Models for punctuation (HiTZ/cap-punct-es): `pip install sentencepiece`
  Also, for text tokenization execute `pip install sacremoses`.
- Transformers: `pip install transformers`
- Torch: `pip install torch -f https://download.pytorch.org/whl/torch_stable.html`
- pyaudio: `pip install pyaudio`
- ipywidgets: `pip install ipywidgets`
- json


from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pyaudio
import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from avatar import Avatar

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Pyaudio
p = pyaudio.PyAudio()

# Initialize NeMo ASR and TTS models
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
tts_model = nemo_tts.models.Tacotron2Model.from_pretrained(model_name="Tacotron2-22050Hz")

# Initialize 3D avatar
avatar = Avatar()

# Initialize AI model and tokenizer
model_name = "NousResearch/Nous-Hermes-13b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    model_name = "NousResearch/Nous-Hermes-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_predictor = LLMPredictor(llm=model, temperature=0.7, max_tokens=num_outputs)
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return index

def tts(text):
    # Convert text to audio using NeMo TTS
    parsed = nemo_tts.models.parse_tts_args(None)
    parsed['text'] = text
    audio = tts_model(**parsed)
    return audio.data
    
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio')
def handle_audio(data):
    # Convert audio to text using NeMo ASR
    text = asr_model.transcribe(paths2audio_files=data)

    # Process text with chatbot
    response_text = chatbot(text, model, tokenizer)

    # Convert response text to audio using NeMo TTS
    response_audio = tts(response_text)

    # Emit response audio
    emit('audio', response_audio)

    # Update avatar based on response
    avatar.update(response_text)

def chatbot(input_text):
    # Process input text with chatbot
    model_name = "NousResearch/Nous-Hermes-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_predictor = LLMPredictor(llm=model, temperature=0.7, max_tokens=512)
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, llm_predictor=llm_predictor, response_mode="compact")
    return response.response

if __name__ == '__main__':
    # Construct the chatbot index
    construct_index('docs', model, tokenizer)
    # Run the Flask app
    socketio.run(app)

import streamlit as st
from moviepy.editor import *
import tempfile
import os
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import librosa
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

# Load the h5 model
main_model = load_model('D:/S8/ui/multimodal_model.h5')
# Load the h5 model
# Load the pre-trained model and processor
output_dir = "D:/S8/ui/wav2text"
model = Speech2TextForConditionalGeneration.from_pretrained(output_dir + "/model")
processor = Speech2TextProcessor.from_pretrained(output_dir + "/processor")

def extract_features(data, sample_rate):
    # Extracting features
    features = {
        "zcr": librosa.feature.zero_crossing_rate(y=data).mean(),
        "chroma_stft": librosa.feature.chroma_stft(y=data, sr=sample_rate).mean(axis=1),
        "mfcc": librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).mean(axis=1),
        "rms": librosa.feature.rms(y=data).mean(),
        "mel": librosa.feature.melspectrogram(y=data, sr=sample_rate).mean(axis=1)
    }
    
    return np.concatenate([v if isinstance(v, np.ndarray) else [v] for v in features.values()])

def get_features(data,sampling_rate):
    # Load audio file
    # Extract features for the original audio
    features_original = extract_features(data, sampling_rate)
    # Augment data and extract features    
    return np.vstack([features_original])

# Function to preprocess text
def add_text_embeddings(text):
    # Load the sentence transformer model
    model = SentenceTransformer("D:/S8/New/embedding")

    # Encode sentences one by one and handle potential errors gracefully
    embeddings = []
    embedding = model.encode(text)
    return embedding

# Define emotions dictionary
emotions_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Surprise",
    6: "sad"
}

# Title of the app
st.title("Emotion detection In Conversations")

# File uploader widget to allow users to upload a video file
st.sidebar.header("Upload Video")
video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Check if a video file is uploaded
if video_file is not None:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Save the uploaded video to a temporary file
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())

        # Load the video
        st.header("Input Video:")
        st.video(video_path)

        # Load the video and extract audio
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        
        # Convert audio to WAV format
        wav_file_path = os.path.join(temp_dir, "output_audio.wav")
        audio_clip.write_audiofile(wav_file_path)

        # Display the WAV audio
        st.header("Extracted WAV Audio:")
        st.audio(wav_file_path)

        # Transcribe audio
        audio_array, sampling_rate = librosa.load(wav_file_path, sr=16000)  # Resample to 16000 Hz
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Display the transcription
        st.header("Transcription:")
        st.write(transcription)

        # Preprocess text
        embeddings = add_text_embeddings([transcription])
        # st.write("Preprocessed Text Embeddings:")
        # st.write(embeddings)

        # Extract audio features
        audio_array, sampling_rate = librosa.load(wav_file_path)  # Resample to 16000 Hz

        audio_features = get_features(audio_array, sampling_rate)
        # st.write("Audio Features:")
        # st.write(audio_features)
        # st.write("Audio Features Shape:")
        # st.write(audio_features.shape)

        # Make prediction using multimodal model
        prediction = main_model.predict([embeddings,audio_features])

        # Get the predicted emotion
        predicted_emotion = emotions_dict[np.argmax(prediction)]
        
        # Display the predicted emotion
        st.header("Predicted Emotion:")
        st.text(predicted_emotion)
        
    finally:
        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()

        # Clean up: Delete the temporary directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

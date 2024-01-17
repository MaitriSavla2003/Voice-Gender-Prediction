# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import librosa
from keras.models import load_model
import pygame
import soundfile as sf  # Added import for soundfile

# Loading the Model
model = load_model('results/model.h5')

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Voice Gender Detector')
top.configure(background='#CDCDCD')

# Initializing the Labels (1 for gender)
label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

# Defining Detect function which detects the gender from the voice
def Detect(file_path):
    global label_packed
    try:
        # Load audio using soundfile
        X, sample_rate = sf.read(file_path)
        
        features = extract_feature(X, sample_rate, mel=True).reshape(1, -1)
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "Male" if male_prob > female_prob else "Female"

        # Update the label to display the result
        label.configure(foreground="#011638", text="Predicted Gender: " + gender)
    except Exception as e:
        print(e)

# Defining Detect function which detects the gender from the voice
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Voice Gender", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Defining Play Audio Function
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Defining Upload Audio Function
def upload_audio():
    try:
        file_path = filedialog.askopenfilename()
        label.configure(text='')
        show_Detect_button(file_path)
        play_audio_button = Button(top, text="Play Audio", command=lambda: play_audio(file_path), padx=10, pady=5)
        play_audio_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        play_audio_button.place(relx=0.79, rely=0.55)
    except Exception as e:
        print(e)

def extract_feature(X, sample_rate, **kwargs):
    """
    Extract feature from audio data `X` with sample rate `sample_rate`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(X, sample_rate, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

upload = Button(top, text="Upload an Audio File", command=upload_audio, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
label.pack(side="bottom", expand=True)
heading = Label(top, text="Voice Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()

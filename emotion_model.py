import torch
import io
from torchaudio.utils import download_asset
import torchaudio
import numpy as np
import soundfile as sf
import struct
import time
import scipy
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

#loads model
model = torch.jit.load("final_emotion_model_scripted.pt", map_location="cpu")
model.eval()
model.double()

emotion = "Neutral"
gender = "Male"
speaking_rate, pitch_mean, pitch_std, rms, relative_db = 0, 0, 0, 0, 0

#GUI title, project members, explanation
st.title("Embedded Real-Time Speech Classifier")
st.text("Norman Smith and Vignesh Saravanan")
st.text("Our project performs real-time speech classification from 5-second clips recorded by the MAX9814 microphone module. A DMA channel running on an STM32L476RG microcontroller continually "
        "samples data from the microphone module using the ADC at a sampling frequency of 16 kHz. This data fills up a buffer that is sent to a PC via the UART peripheral. A Python program then "
        "uses the PySerial module to accumulate and process these buffers to create a "
        "5-second audio clip, which, after going through a digital bandpass filter (to remove low-frequency and high-frequency noise), is then processed by a speech classification model trained on PyTorch in order to produce data such as emotion, gender, speaking rate, etc. This data, along with a spectrogram of the 5-second"
        "audio clip is present below and continually updated every 2 seconds.")
dispArr = np.array([["Emotion", "Gender", "Speaking Rate", "Pitch Mean", "Pitch Standard Deviation", "RMS", "Relative Decibels"], [emotion, gender, speaking_rate, pitch_mean, pitch_std, rms, relative_db]])
ser = pd.DataFrame(dispArr.transpose(), columns=["Category", "Value"])
displayed = st.dataframe(ser)

sr_value, x_value = scipy.io.wavfile.read("output.wav")
f, t, Sxx = scipy.signal.spectrogram(x_value, sr_value)

#plots spectrogram
fig, ax = plt.subplots(figsize=(10, 4))
mesh = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
ax.set_ylabel("Frequency (Hz")
ax.set_xlabel("Time (s)")
ax.set_title("Spectrogram of 5-second Audio File")
fig.colorbar(mesh, ax=ax, label="Power (dB)")
temp = st.pyplot(fig)


while True:
    #unpacks wav file, converting into a tensor
    data, sr = sf.read("output.wav")

    if len(data.shape) == 1:
        wav = torch.tensor(data).unsqueeze(0)
        print(wav.shape)
    else:
        wav = torch.tensor(data).T

    #processes tensor data, converting into human-readable form
    wav = wav.type('torch.DoubleTensor')
    output = model(wav)
    emotion_logits = output[0][0]
    m = torch.nn.Softmax(dim=0)
    emotion_norm = m(emotion_logits)
    binary_predictions = torch.argmax(emotion_norm, dim=0)
    emotions = ['Neutral', 'Happy', 'Sad', 'Angry']
    emotion = emotions[int(binary_predictions)]
    print(f"Emotions: {emotion}")
    gender = output[1][0][0]
    gender = np.exp(int(gender))/(np.exp(int(gender))+1)
    if gender >= 0.5:
        gender = "Male"
    else:
        gender = "Female"
    print(f"Gender: {gender}")
    #print(output)

    acoustic = output[2][0]
    mean_vector = torch.tensor([ 9.0108e+00,  1.7562e+02,  4.1693e+01,  1.9548e-02, -1.5083e+01])
    std_vector = torch.tensor([4.1678e+00, 6.3619e+01, 3.0374e+01, 2.9597e-02, 4.2030e+00])
    unnorm_acoustic = mean_vector + acoustic * std_vector
    [speaking_rate, pitch_mean, pitch_std, rms, relative_db] = unnorm_acoustic
    print(f"Speaking Rate: {speaking_rate}")
    print(f"Pitch Mean: {pitch_mean}")
    print(f"Pitch Std: {pitch_std}")
    print(f"RMS: {rms}")
    print(f"Relative DB: {relative_db}")

    ser.iloc[0,1] = emotion
    ser.iloc[1,1] = gender
    ser.iloc[2,1] = speaking_rate
    ser.iloc[3,1] = pitch_mean
    ser.iloc[4,1] = pitch_std
    ser.iloc[5,1] = rms
    ser.iloc[6,1] = relative_db

    displayed.empty()
    time.sleep(0.1)
    displayed = st.dataframe(ser)

    sr_value, x_value = scipy.io.wavfile.read("output.wav")
    f, t, Sxx = scipy.signal.spectrogram(x_value, sr_value)

    #new spectrogram
    temp.empty()
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    ax.set_ylabel("Frequency (Hz")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram of 5-second Audio File")
    fig.colorbar(mesh, ax = ax, label = "Power (dB)")
    temp = st.pyplot(fig)
    time.sleep(2)


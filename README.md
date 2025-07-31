# Embedded Speech Classification
Vignesh Saravanan & Norman Smith

Our project performs **real-time speech classification** from 5-second clips recorded by the MAX9814 microphone module. A **DMA** channel running on an STM32L476RG microcontroller continually samples data from the microphone module using the **ADC** at a sampling frequency of 16 kHz. This data fills up a buffer that is sent to a PC via the UART peripheral. A Python program then uses the **PySerial** module to accumulate and process these buffers to create a 5-second audio clip, which, after going through a **digital bandpass filter** (to remove low-frequency and high-frequency noise), is then processed by a speech classification model trained on **PyTorch** in order to produce data such as emotion, gender, speaking rate, etc. This data, along with a spectrogram of the 5-second audio clip, is continually updated every 2 seconds and displayed on a web app deployed using Streamlit.

The following link contains a demo video of our project: https://www.youtube.com/watch?v=ADdkXfiX4Y0

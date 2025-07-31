import serial
import wave
import numpy as np
from scipy import signal
import torch

def bandpass(sig):
    order = 4
    low_cutoff = 60
    high_cutoff = 600
    th  = signal.butter(4, low_cutoff,  "highpass", False, "ba", fs=16000)
    filtered_sig = signal.filtfilt(th[0], th[1], sig)
    filtered_sig = [int(val) for val in filtered_sig]
    filtered_sig = np.array(filtered_sig)
    filtered_sig = filtered_sig/np.max(np.abs(filtered_sig)) * np.max(np.abs(sig))
    for i, val in enumerate(filtered_sig):
        if (val > 32767):
            val = 32767
        elif (val < -32768):
            val = -32768
    filtered_sig = np.array([np.int16(val) for val in filtered_sig], dtype=np.int16)
    return filtered_sig

def wav():
    intTotal = []
    oldInt = 0
    counter = 0
    for element in total:
        newInt = (int(element) - 1551) * 16
        if newInt > 32767:
            newInt = 32767
        elif newInt < -32768:
            newInt = -32768
        intTotal.append(newInt)
    intTotal = bandpass(np.array(intTotal))
    for i in intTotal:
        if i > 32767 or i < -32768:
            i = 0
    data = np.array(intTotal, dtype =np.int16)
    with wave.open("output.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.setnframes(80000)
        wav_file.writeframes(data.tobytes())


ser = serial.Serial('COM4', baudrate=1000000)

print("Connected to", ser.port)
a = 0
counter = 0
current = []
buf = []
total = []
try:
    line = 0
    while True:
        if ser.in_waiting:  # Check if data is available
           counter = 0
           while counter < 16000:
               pastLine = line
               line = ser.read(4).decode("ASCII")
               buf.append(line)
               counter += 1
           print(buf[0:10])
           a += 1
           a = a%5
           total += buf
           if a == 0:
               wav()
               total = []
           buf = []


except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
    print("Serial port closed.")




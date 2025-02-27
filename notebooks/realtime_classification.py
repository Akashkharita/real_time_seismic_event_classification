import json
import time
from obspy import UTCDateTime

from scipy import stats,signal
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../src')
from neural_network_architectures import QuakeXNet_2d

import torch.nn.functional as F
from datetime import datetime
import json
import time


# Initialize IRIS FDSN client
client = Client("IRIS")


## setting up some important parameters (not to be changed)
num_channels = 3
dropout = 0.9
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_quakexnet_2d = QuakeXNet_2d(num_classes=4, num_channels=num_channels,dropout_rate=dropout).to(device)  # Use 'cuda' if you have a GPU available

# Load the saved model state dict (weights)
saved_model_quakexnet_2d = torch.load('../deep_learning_models/best_model_QuakeXNet_2d.pth', map_location=torch.device('cpu'))  # No 'weights_only' argument
model_quakexnet_2d.load_state_dict(saved_model_quakexnet_2d)
model_quakexnet_2d.to(device)
model_quakexnet_2d.eval()


model = model_quakexnet_2d
model.to('cpu')



lowcut = 1
highcut = 20
nyquist = 0.5 * 50  # Nyquist frequency (original sampling rate is 100 Hz)
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(4, [low, high], btype='band')  # Bandpass filter coefficients
taper_alpha = 0.1
fs = 50




def extract_spectrograms(waveforms = [], fs = 50, nperseg=256, overlap=0.5):
    noverlap = int(nperseg * overlap)  # Calculate overlap

    # Example of how to get the shape of one spectrogram
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Initialize an array of zeros with the shape: (number of waveforms, channels, frequencies, time_segments)
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))

    for i in range(waveforms.shape[0]):  # For each waveform
        for j in range(waveforms.shape[1]):  # For each channel
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx  # Fill the pre-initialized array

    return spectrograms



"""
## Next steps

- Run the app.py on the cascadia server. This creates the endpoint from where external apps can acess the results. 
- Then run this code on the terminal of our local machine - ssh -L 5000:127.0.0.1:5000 -p7777 ak287@cascadia.ess.washington.edu  This allows to connect local machine to the remote server where the flask is running. 
- And access the results using  http://127.0.0.1:5000/classification on the browser.

"""

# Define stations
stations = ["UW.STAR", "UW.RCM"]  # Replace with actual station codes

location = '*'
channel = '*H*'

while True:
    try:
        # Fetch new waveform data for all stations (last 60 seconds)
        start_time = UTCDateTime() - 200
        end_time = UTCDateTime() - 50

        station_results = {}

        for station in stations:
            network, station = station.split('.')
            st = client.get_waveforms(network, station, location, channel, start_time, end_time)
            st.resample(50)

            # Process data
            data = np.array([st[i].data[0:5000] for i in range(3)]).reshape([1,3,5000])
            tapered_data = np.array([np.multiply(signal.windows.tukey(data.shape[-1], alpha=taper_alpha), row) for row in data])
            filtered_data = np.array([signal.filtfilt(b, a, row) for row in tapered_data])
            norm = np.std(abs(filtered_data), axis=2)
            normalized_data = (filtered_data / norm[:, :, None])

            specs = extract_spectrograms(normalized_data)

            # Run classification
            with torch.no_grad():
                output = model(torch.Tensor(specs))
                softmax_probs = F.softmax(output, dim=1).cpu().numpy()

            # Get class labels
            class_labels = ["earthquake", "explosion", "noise", "surface event"]
            max_class_index = np.argmax(softmax_probs[0])  # Index of max probability class
            max_class_label = class_labels[max_class_index]

            # Store results
            station_results[station] = {
                "timestamp": start_time.isoformat(),
                "probabilities": {
                    "earthquake": float(softmax_probs[0][0]),
                    "explosion": float(softmax_probs[0][1]),
                    "noise": float(softmax_probs[0][2]),
                    "surface event": float(softmax_probs[0][3])
                },
                "max_label": max_class_label
            }

        # Check if UW.STAR and UW.RCM are both NOT classified as "noise"
        if (station_results["STAR"]["max_label"] != "noise" and 
            station_results["RCM"]["max_label"] != "noise"):

            # **Append new result directly to file in JSONL format**
            with open("realtime_classification.json", "a") as f:
                f.write(json.dumps(station_results) + "\n")  # Write new entry as a single line

            #print("Updated classification:", station_results)

    except Exception as e:
        print("Error:", e)

    time.sleep(20)  # Wait 2 seconds before fetching new data

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
import sys

print("===================================")
print("   EEG Recognition Detection App   ")
print("===================================")

def process_subject(file_name, threshold):

    try:
        data = sio.loadmat(file_name)
    except:
        print("File not found ❌")
        return

    eeg = data['test_data']
    print("EEG Shape:", eeg.shape)

    trials, samples, channels = eeg.shape

    sfreq = 240
    ch_names = [f"EEG{i+1}" for i in range(channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    eeg = np.transpose(eeg, (0, 2, 1))
    epochs = mne.EpochsArray(eeg, info)
    evoked = epochs.average()

    avg_signal = evoked.data[0]

    p300_index = int(0.3 * sfreq)
    window = 10

    peak = np.max(np.abs(avg_signal[p300_index-window:p300_index+window]))
    peak_time = np.argmax(np.abs(avg_signal[p300_index-window:p300_index+window]))
    reaction_time = (p300_index-window + peak_time) / sfreq * 1000

    print("\nP300 Peak Value:", round(float(peak), 2))
    print("Reaction Time:", round(float(reaction_time), 2), "ms")

    if peak > threshold:
        result = "Recognition Detected ✅"
    else:
        result = "Not Recognition ❌"

    print("Final Result:", result)

    times = np.arange(samples) / sfreq * 1000

    plt.figure()
    plt.plot(times, avg_signal)
    plt.axvline(x=300)
    plt.title(f"{file_name} - ERP Waveform")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.show()

while True:

    print("\nMenu:")
    print("1. Analyze EEG File")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        file_name = input("Enter EEG file name (Example: S1.mat): ")
        threshold = float(input("Enter Detection Threshold (Example: 2.0): "))
        process_subject(file_name, threshold)

    elif choice == "2":
        print("Exiting Program... 👋")
        sys.exit()

    else:
        print("Invalid choice ❌ Try again.") 
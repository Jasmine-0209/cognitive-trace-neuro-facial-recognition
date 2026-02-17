import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
import sys

print("===================================")
print("   EEG Recognition Detection App   ")
print("===================================")

def process_subject(file_name, threshold=None):
    try:
        data = sio.loadmat(file_name)
    except:
        print(f"File '{file_name}' not found ❌")
        return

    eeg = data['test_data']
    print("EEG Shape:", eeg.shape)

    trials, samples, channels = eeg.shape
    sfreq = 240
    ch_names = [f"EEG{i+1}" for i in range(channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Reshape EEG to (trials, channels, samples)
    eeg = np.transpose(eeg, (0, 2, 1))
    epochs = mne.EpochsArray(eeg, info)
    evoked = epochs.average()

    avg_signal = evoked.data[0]  # First channel

    # P300 expected around 300ms
    p300_index = int(0.3 * sfreq)
    window = 10  # +/- samples around 300ms

    peak = np.max(np.abs(avg_signal[p300_index - window : p300_index + window]))
    peak_time = np.argmax(np.abs(avg_signal[p300_index - window : p300_index + window]))
    reaction_time = (p300_index - window + peak_time) / sfreq * 1000  # in ms

    print(f"\nP300 Peak Value: {round(float(peak),2)}")
    print(f"Reaction Time: {round(float(reaction_time),2)} ms")

    # If threshold not given, ask user
    if threshold is None:
        user_input = input(f"Enter detection threshold (leave blank for auto 50% of max {round(np.max(np.abs(avg_signal)),2)}): ")
        if user_input.strip() == "":
            threshold = 0.5 * np.max(np.abs(avg_signal))
        else:
            try:
                threshold = float(user_input)
            except:
                print("Invalid input. Using automatic threshold.")
                threshold = 0.5 * np.max(np.abs(avg_signal))
    else:
        print(f"Using batch threshold: {threshold}")

    # Detection logic
    if peak > threshold:
        detected = True
        reason = f"peak amplitude ({round(peak,2)}) exceeded threshold ({round(threshold,2)})"
    else:
        detected = False
        reason = f"peak amplitude ({round(peak,2)}) did not exceed threshold ({round(threshold,2)})"

    # Reaction time explanation
    if reaction_time < 250:
        latency_note = "P300 occurred earlier than typical (early)."
    elif reaction_time > 500:
        latency_note = "P300 occurred later than typical (late)."
    else:
        latency_note = "P300 occurred within typical latency (on time)."

    # Final statement
    if detected:
        result = f"P300 Detected ✅ because {reason}. {latency_note}"
    else:
        result = f"P300 Not Detected ❌ because {reason}. {latency_note}"

    print("Final Result:", result)

    # Plot ERP waveform
    times = np.arange(samples) / sfreq * 1000
    plt.figure()
    plt.plot(times, avg_signal)
    plt.axvline(x=300, color='r', linestyle='--', label="Expected P300")
    plt.title(f"{file_name} - ERP Waveform")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


# ===== Main Menu =====
while True:
    print("\nMenu:")
    print("1. Analyze Single EEG File")
    print("2. Analyze Multiple EEG Files (Batch)")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        file_name = input("Enter EEG file name (Example: S1.mat): ")
        process_subject(file_name)

    elif choice == "2":
        # Predefine the list of files here
        eeg_files = ["S1.mat", "S2.mat"]  # <-- Add your EEG files
        # Ask user once for a batch threshold
        user_input = input("Enter threshold for all files (leave blank for auto 50% of each file): ")
        if user_input.strip() == "":
            batch_threshold = None
        else:
            try:
                batch_threshold = float(user_input)
            except:
                print("Invalid input. Using automatic thresholds for each file.")
                batch_threshold = None

        for file_name in eeg_files:
            print("\n===================================")
            print(f"Processing {file_name}...")
            process_subject(file_name, threshold=batch_threshold)

    elif choice == "3":
        print("Exiting Program... 👋")
        sys.exit()

    else:
        print("Invalid choice ❌ Try again.")

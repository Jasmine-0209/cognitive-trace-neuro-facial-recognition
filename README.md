🧠 Cognitive Trace: Neuro-Facial Recognition Analysis

An ethical AI system that estimates the probability of recognition or concealed information using EEG (P300 signals) and facial expression analysis.

⚠️ This system does NOT detect lies. It only provides probabilistic recognition analysis.

🚀 Overview

Traditional lie detection methods rely on behavioral cues, which can be unreliable.

This project uses a multimodal AI approach:

🧠 EEG (P300 signal detection) – detects involuntary recognition

🙂 Facial Expression Analysis – detects stress indicators

🔗 Fusion Model – combines both for final probability output

⚙️ How It Works

Present stimulus (neutral / known / test items)

Detect P300 brain response (~300ms peak)

Analyze facial stress patterns

Combine outputs using weighted/Bayesian fusion

Generate final probability score

Example Output:

“72% probability of concealed information under stress.”

🛠 Tech Stack

Python

MNE

OpenCV / MediaPipe

Scikit-learn

NumPy / Pandas

Streamlit (Demo UI)

📊 Datasets

BCI Competition (P300 EEG)

Kaggle FER-2013

CK+ Facial Dataset

📌 Key Features

✔️ Non-invasive & ethical
✔️ Multimodal fusion approach
✔️ Probabilistic output (not lie detection)
✔️ Explainable results (ERP plots & facial indicators)

👩‍💻 Team Members & Roles
🔹 Jasmine Shafi – Team Lead & System Integration

Designed overall system architecture

Integrated EEG & facial modules

Implemented multimodal fusion

Coordinated testing and final demo

🔹 Ahinaya – EEG Module

EEG preprocessing (filtering, artifact removal, epoching)

P300 feature extraction

Trained and evaluated EEG model

🔹 Facial Module Developer

Facial dataset preprocessing

Landmark & stress feature extraction

Trained facial expression classifier

🔹 Ruby – Fusion Analysis & Documentation

Multimodal performance comparison

Explainability (ERP plots, heatmaps)

Documentation & presentation
🎓 Project Summary

A multimodal AI system that combines brain and facial signals to estimate recognition probability in an ethical, research-focused framework.

import tkinter as tk
from tkinter import messagebox
import numpy as np
import librosa
import sounddevice as sd
import wavio
import pickle

SR = 8000
DURATION = 4  # seconds
N_MFCC = 40
MODEL_FILE = "cough_classifier.pkl"
TARGET_FEATURES = 1280  # Match training features

# Load model
with open(MODEL_FILE, "rb") as f:
    model, le = pickle.load(f)

class CoughPredictGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 Cough Predictor")
        self.root.geometry("400x300")
        self.root.config(bg="#2b2b2b")

        header = tk.Label(self.root, text="Cough Predictor", font=("Arial", 18, "bold"), bg="#2b2b2b", fg="white")
        header.pack(pady=20)

        self.record_button = tk.Button(self.root, text="📹 Record Cough", command=self.record_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10, borderwidth=0)
        self.record_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Ready to record", font=("Arial", 12), bg="#2b2b2b", fg="white")
        self.status_label.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#FFD700")
        self.result_label.pack(pady=20)

    def record_audio(self, filename="temp.wav"):
        sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        wavio.write(filename, sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32"), SR, sampwidth=2)
        return filename

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path, sr=SR)
        max_samples = SR * DURATION
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_flat = mfcc.flatten()
        if len(mfcc_flat) < TARGET_FEATURES:
            mfcc_flat = np.pad(mfcc_flat, (0, TARGET_FEATURES - len(mfcc_flat)))
        elif len(mfcc_flat) > TARGET_FEATURES:
            mfcc_flat = mfcc_flat[:TARGET_FEATURES]
        return mfcc_flat.reshape(1, -1)

    def predict_cough(self, file_path):
        features = self.extract_features(file_path)
        pred_idx = model.predict(features)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        return pred_label

    def record_and_predict(self):
        self.status_label.config(text="Recording...")
        self.root.update()
        file_path = self.record_audio()
        self.status_label.config(text="Analyzing...")
        self.root.update()
        pred_label = self.predict_cough(file_path)
        self.result_label.config(text=f"Predicted: {pred_label}")
        self.status_label.config(text="Ready to record")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = CoughPredictGUI()
    gui.run()

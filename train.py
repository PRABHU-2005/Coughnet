# import os
# import numpy as np
# import librosa
# from tqdm import tqdm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # ==============================
# # SETTINGS
# # ==============================
# DATA_PATH = r"E:\5TH SEM\expo\virufy"  # <-- update this if needed
# SR = 8000
# DURATION = 2  # seconds
# # N_MFCC = 40
# N_MFCC = 40

# MODEL_FILE = "cough_classifier.pkl"

# # ==============================
# # FEATURE EXTRACTION
# # ==============================
# def extract_features(file_path):
#     audio, sr = librosa.load(file_path, sr=SR)
#     max_samples = SR * DURATION
#     if len(audio) < max_samples:
#         audio = np.pad(audio, (0, max_samples - len(audio)))
#     else:
#         audio = audio[:max_samples]
#     mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
#     return mfcc.flatten()  # flatten for scikit-learn

# # ==============================
# # LOAD DATA
# # ==============================
# X = []
# y = []

# classes = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
# print("✅ Found classes:", classes)

# for label in classes:
#     folder = os.path.join(DATA_PATH, label)
#     for file in tqdm(os.listdir(folder), desc=f"Processing {label.upper()}"):
#         if file.endswith(".wav"):
#             try:
#                 feats = extract_features(os.path.join(folder, file))
#                 X.append(feats)
#                 y.append(label)
#             except Exception as e:
#                 print("⚠️ Error processing file:", file, e)

# X = np.array(X)
# y = np.array(y)

# print(f"✅ Features shape: {X.shape}, Labels shape: {y.shape}")

# # ==============================
# # LABEL ENCODING
# # ==============================
# le = LabelEncoder()
# y_enc = le.fit_transform(y)

# # ==============================
# # TRAIN MODEL
# # ==============================
# clf = RandomForestClassifier(n_estimators=200, random_state=42)
# clf.fit(X, y_enc)
# print("✅ Model trained!")

# # ==============================
# # SAVE MODEL
# # ==============================
# with open(MODEL_FILE, "wb") as f:
#     pickle.dump((clf, le), f)
# print(f"💾 Model saved as {MODEL_FILE}")
import pickle
import numpy as np
import sounddevice as sd
import wavio
import librosa

# ==============================
# SETTINGS - MUST MATCH TRAIN.PY
# ==============================
MODEL_FILE = "cough_classifier.pkl"
SR = 8000  # IMPORTANT: Match training sample rate
DURATION = 2  # IMPORTANT: Match training duration
N_MFCC = 40  # IMPORTANT: Match training MFCC count

# ==============================
# LOAD MODEL
# ==============================
with open(MODEL_FILE, "rb") as f:
    model, le = pickle.load(f)
print("✅ Model loaded!")
print(f"Model expects: {model.n_features_in_} features")

# ==============================
# FEATURE EXTRACTION - EXACT SAME AS TRAIN.PY
# ==============================
def extract_features(file_path):
    """Extract features EXACTLY like in train.py"""
    audio, sr = librosa.load(file_path, sr=SR)
    max_samples = SR * DURATION
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return mfcc.flatten()  # flatten for scikit-learn

# ==============================
# RECORD AUDIO
# ==============================
def record_audio(filename="temp.wav"):
    """Record audio for specified duration"""
    print("\nPress Enter to start recording a cough (or 'q' to quit):")
    user_input = input()
    if user_input.lower() == "q":
        return None
    
    print(f"🎤 Recording started for {DURATION} seconds at {SR} Hz...")
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    wavio.write(filename, recording, SR, sampwidth=2)
    print("✅ Recording finished.")
    return filename

# ==============================
# PREDICT WITH CONFIDENCE
# ==============================
def predict_with_confidence(file_path):
    """Predict cough type with confidence score"""
    features = extract_features(file_path)
    
    # Reshape for prediction (1 sample, n_features)
    features = features.reshape(1, -1)
    
    # Debug: Check feature size
    print(f"Extracted features: {features.shape[1]} features")
    
    # Make prediction
    pred_idx = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    
    # Get confidence score
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = proba[pred_idx] * 100
        return pred_label, confidence
    
    return pred_label, None

# ==============================
# MAIN LOOP
# ==============================
def main():
    """Main loop for cough classification"""
    print("=" * 50)
    print("🎵 COVID-19 Cough Classification System")
    print("=" * 50)
    print(f"Settings: {DURATION} seconds at {SR} Hz, {N_MFCC} MFCC features")
    print("=" * 50)
    
    while True:
        # Record audio
        audio_file = record_audio()
        
        # Check if user wants to quit
        if audio_file is None:
            print("👋 Exiting...")
            break
        
        try:
            # Predict with confidence
            prediction, confidence = predict_with_confidence(audio_file)
            
            # Display result
            if confidence:
                print(f"🩺 Predicted cough type: {prediction} ({confidence:.1f}% confidence)")
            else:
                print(f"🩺 Predicted cough type: {prediction}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            print("Please try recording again.")
            print("-" * 50)

if __name__ == "__main__":
    main()
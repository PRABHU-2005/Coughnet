# import pickle
# import numpy as np
# import sounddevice as sd
# import wavio
# import librosa

# # Configuration
# MODEL_FILE = "cough_classifier.pkl"  # or whichever model you used
# DURATION = 4
# SR = 22050
# N_MFCC = 40  # Adjust based on your training

# # Load model
# with open(MODEL_FILE, "rb") as f:
#     model, le = pickle.load(f)
# print("✅ Model loaded!")

# def record_audio(filename="temp.wav"):
#     print("\nPress Enter to start recording a cough (or 'q' to quit):")
#     user_input = input()
#     if user_input.lower() == "q":
#         return None
    
#     print(f"🎤 Recording started for {DURATION} seconds...")
#     recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
#     sd.wait()
#     wavio.write(filename, recording, SR, sampwidth=2)
#     print("✅ Recording finished.")
#     return filename

# def extract_features(file_path):
#     """Extract features matching training data"""
#     audio, sr = librosa.load(file_path, sr=SR)
    
#     # Pad or truncate to fixed length
#     max_samples = SR * DURATION
#     if len(audio) < max_samples:
#         audio = np.pad(audio, (0, max_samples - len(audio)))
#     else:
#         audio = audio[:max_samples]
    
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    
#     # Flatten to 1D array
#     features = mfcc.flatten()
    
#     # Ensure exactly 1280 features
#     target_size = 1280
#     if len(features) < target_size:
#         # Pad if less than target
#         features = np.pad(features, (0, target_size - len(features)))
#     elif len(features) > target_size:
#         # Truncate if more than target
#         features = features[:target_size]
    
#     print(f"Feature shape: {features.shape}")  # Debug print
#     return features.reshape(1, -1)

# def predict_with_confidence(file_path):
#     """Predict cough type with confidence score"""
#     features = extract_features(file_path)
#     pred_idx = model.predict(features)[0]
#     pred_label = le.inverse_transform([pred_idx])[0]
    
#     # Get prediction probabilities if your model supports it
#     if hasattr(model, 'predict_proba'):
#         proba = model.predict_proba(features)[0]
#         confidence = proba[pred_idx] * 100
#         return pred_label, confidence
#     return pred_label, None

# # Main execution
# print("=" * 50)
# print("🎵 COVID-19 Cough Classification System")
# print("=" * 50)

# while True:
#     audio_file = record_audio()
    
#     if audio_file is None:
#         print("👋 Exiting...")
#         break
    
#     try:
#         prediction, confidence = predict_with_confidence(audio_file)
        
#         if confidence:
#             print(f"🩺 Predicted cough type: {prediction} ({confidence:.1f}% confidence)")
#         else:
#             print(f"🩺 Predicted cough type: {prediction}")
        
#     except Exception as e:
#         print(f"❌ Error during prediction: {e}")
#         print("Please try recording again.")

########################################################################################
# import pickle
# import numpy as np
# import sounddevice as sd
# import wavio
# import librosa

# # ==============================
# # SETTINGS
# # ==============================
# MODEL_FILE = "cough_classifier.pkl"
# SR = 8000
# DURATION = 2
# N_MFCC = 40
# AMPLIFICATION_FACTOR = 10.0  # Amplify quiet recordings

# # ==============================
# # LOAD MODEL
# # ==============================
# with open(MODEL_FILE, "rb") as f:
#     model, le = pickle.load(f)
# print("✅ Model loaded!")
# print(f"📊 Model expects: {model.n_features_in_} features")
# print(f"📋 Classes: {le.classes_}")

# # ==============================
# # FEATURE EXTRACTION
# # ==============================
# def extract_features(file_path, debug=False):
#     """Extract features with audio normalization"""
#     audio, sr = librosa.load(file_path, sr=SR)
    
#     # Check audio energy
#     energy = np.mean(np.abs(audio))
#     if debug:
#         print(f"  Raw audio energy: {energy:.6f}")
    
#     # Normalize audio (scale to range [-1, 1])
#     if np.max(np.abs(audio)) > 0:
#         audio = audio / np.max(np.abs(audio))
    
#     # Ensure fixed length
#     max_samples = SR * DURATION
#     if len(audio) < max_samples:
#         audio = np.pad(audio, (0, max_samples - len(audio)))
#     else:
#         audio = audio[:max_samples]
    
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    
#     if debug:
#         print(f"  MFCC shape: {mfcc.shape}")
#         print(f"  MFCC mean: {np.mean(mfcc):.6f}")
#         print(f"  MFCC std: {np.std(mfcc):.6f}")
    
#     return mfcc.flatten()

# # ==============================
# # RECORD AUDIO WITH BETTER SETTINGS
# # ==============================
# def record_audio(filename="temp.wav"):
#     """Record audio with better microphone settings"""
#     print("\n" + "="*50)
#     print("Press Enter to start recording a cough (or 'q' to quit):")
#     user_input = input()
    
#     if user_input.lower() == "q":
#         return None
    
#     print(f"🎤 Recording started for {DURATION} seconds...")
#     print("   Make sure to cough LOUDLY and close to the microphone!")
    
#     # Record with higher gain
#     recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
#     sd.wait()
    
#     # Check audio level
#     audio_energy = np.mean(np.abs(recording))
#     print(f"  Audio energy: {audio_energy:.6f}")
    
#     if audio_energy < 0.01:
#         print("⚠️  WARNING: Very quiet recording! Please:")
#         print("   • Cough louder")
#         print("   • Move closer to microphone")
#         print("   • Check microphone settings")
    
#     # Amplify if too quiet
#     if audio_energy < 0.05:
#         recording = recording * 5.0
#         print("  🔊 Audio amplified for better detection")
    
#     # Save recording
#     wavio.write(filename, recording, SR, sampwidth=2)
#     print("✅ Recording finished!")
#     return filename

# # ==============================
# # PREDICT FUNCTION
# # ==============================
# def predict_cough(file_path):
#     """Predict cough type with debug info"""
#     features = extract_features(file_path, debug=True)
    
#     # Reshape for prediction
#     features = features.reshape(1, -1)
    
#     # Make prediction
#     pred_idx = model.predict(features)[0]
#     pred_label = le.inverse_transform([pred_idx])[0]
    
#     # Get probabilities
#     proba = model.predict_proba(features)[0]
#     confidence = proba[pred_idx] * 100
    
#     print("\n📊 Prediction probabilities:")
#     for i, class_name in enumerate(le.classes_):
#         print(f"  {class_name}: {proba[i]*100:.2f}%")
    
#     return pred_label, confidence

# # ==============================
# # MAIN LOOP
# # ==============================
# def main():
#     print("="*50)
#     print("🎵 COVID-19 Cough Classification System")
#     print("="*50)
#     print("⚠️  IMPORTANT:")
#     print("   • Cough CLOSELY into the microphone")
#     print("   • Cough LOUDLY and clearly")
#     print("   • Make sure microphone is not muted")
#     print("="*50)
    
#     while True:
#         audio_file = record_audio()
        
#         if audio_file is None:
#             print("\n👋 Exiting...")
#             break
        
#         try:
#             prediction, confidence = predict_cough(audio_file)
            
#             print("\n" + "="*50)
#             print("📊 PREDICTION RESULT")
#             print("="*50)
#             print(f"🩺 Predicted cough type: {prediction}")
#             print(f"🎯 Confidence: {confidence:.2f}%")
            
#             if confidence > 70:
#                 if prediction == "COVID":
#                     print("⚠️  High confidence COVID detection - please consult a doctor")
#                 else:
#                     print("✅ High confidence healthy cough detected")
#             elif confidence > 55:
#                 print("📝 Moderate confidence - consider retesting")
#             else:
#                 print("🔄 Low confidence - please record again with clearer cough")
            
#             print("="*50)
            
#         except Exception as e:
#             print(f"\n❌ Error: {e}")
#             import traceback
#             traceback.print_exc()

# # ==============================
# # TEST MICROPHONE FIRST
# # ==============================
# def test_microphone():
#     """Test if microphone is working"""
#     print("\n🔍 Testing microphone...")
#     print("Please cough or speak for 2 seconds...")
    
#     test_recording = sd.rec(int(2 * SR), samplerate=SR, channels=1)
#     sd.wait()
    
#     energy = np.mean(np.abs(test_recording))
#     print(f"Audio energy: {energy:.6f}")
    
#     if energy < 0.001:
#         print("❌ Microphone is not working or too quiet!")
#         print("\nPlease check:")
#         print("1. Windows Settings → System → Sound")
#         print("2. Check if microphone is connected")
#         print("3. Check input volume levels")
#         print("4. Try a different microphone")
#         return False
#     elif energy < 0.01:
#         print("⚠️  Microphone is working but very quiet")
#         print("   Please cough louder and closer to the mic")
#         return True
#     else:
#         print("✅ Microphone is working well!")
#         return True

# if __name__ == "__main__":
#     # Test microphone first
#     if test_microphone():
#         main()
#     else:
#         print("\n❌ Cannot proceed without working microphone")
#         print("Please fix microphone issues and try again")
# import pickle
# import numpy as np
# import sounddevice as sd
# import wavio
# import librosa

# # ==============================
# # SETTINGS
# # ==============================
# MODEL_FILE = "cough_classifier.pkl"
# SR = 8000
# DURATION = 2
# N_MFCC = 40
# SILENCE_THRESHOLD = 0.01  # Minimum energy to consider as valid cough
# CONFIDENCE_THRESHOLD = 60.0  # Minimum confidence to trust prediction

# # ==============================
# # LOAD MODEL
# # ==============================
# with open(MODEL_FILE, "rb") as f:
#     model, le = pickle.load(f)
# print("✅ Model loaded!")
# print(f"📊 Model expects: {model.n_features_in_} features")
# print(f"📋 Classes: {le.classes_}")

# # ==============================
# # FEATURE EXTRACTION
# # ==============================
# def extract_features(file_path, debug=False):
#     """Extract features with audio normalization"""
#     audio, sr = librosa.load(file_path, sr=SR)
    
#     # Check audio energy
#     energy = np.mean(np.abs(audio))
#     if debug:
#         print(f"  Raw audio energy: {energy:.6f}")
    
#     # For quiet recordings, don't normalize (keep as is)
#     if energy > SILENCE_THRESHOLD:
#         # Normalize audio to range [-1, 1] only if it's not silence
#         if np.max(np.abs(audio)) > 0:
#             audio = audio / np.max(np.abs(audio))
#         if debug:
#             print(f"  Normalized (cough detected)")
#     else:
#         if debug:
#             print(f"  No normalization (silence detected)")
    
#     # Ensure fixed length
#     max_samples = SR * DURATION
#     if len(audio) < max_samples:
#         audio = np.pad(audio, (0, max_samples - len(audio)))
#     else:
#         audio = audio[:max_samples]
    
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    
#     if debug:
#         print(f"  MFCC shape: {mfcc.shape}")
#         print(f"  MFCC mean: {np.mean(mfcc):.6f}")
#         print(f"  MFCC std: {np.std(mfcc):.6f}")
    
#     return mfcc.flatten(), energy

# # ==============================
# # RECORD AUDIO
# # ==============================
# def record_audio(filename="temp.wav"):
#     """Record audio with better microphone settings"""
#     print("\n" + "="*50)
#     print("Press Enter to start recording a cough (or 'q' to quit):")
#     user_input = input()
    
#     if user_input.lower() == "q":
#         return None
    
#     print(f"🎤 Recording started for {DURATION} seconds...")
#     print("   Make sure to cough LOUDLY and close to the microphone!")
    
#     # Record
#     recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
#     sd.wait()
#     recording = recording.flatten()
    
#     # Check audio level
#     audio_energy = np.mean(np.abs(recording))
#     print(f"  Audio energy: {audio_energy:.6f}")
    
#     # Check if recording is silence
#     if audio_energy < SILENCE_THRESHOLD:
#         print("\n❌ ERROR: No cough detected (recording is silent!)")
#         print("   Please:")
#         print("   • Cough louder")
#         print("   • Move closer to microphone")
#         print("   • Check if microphone is working")
#         print("   • Press Enter to try again")
#         return "SILENT"
    
#     # Check if recording is too quiet
#     if audio_energy < 0.05:
#         print("⚠️  WARNING: Very quiet recording - amplifying...")
#         recording = recording * 3.0
    
#     # Save recording
#     wavio.write(filename, recording, SR, sampwidth=2)
#     print("✅ Recording saved!")
#     return filename

# # ==============================
# # PREDICT FUNCTION WITH SILENCE DETECTION
# # ==============================
# def predict_cough(file_path):
#     """Predict cough type with silence detection"""
#     features, energy = extract_features(file_path, debug=True)
    
#     # Check if recording is silent
#     if energy < SILENCE_THRESHOLD:
#         print("\n❌ No cough detected in recording")
#         return "NO_COUGH", 0.0
    
#     # Reshape for prediction
#     features = features.reshape(1, -1)
    
#     # Make prediction
#     pred_idx = model.predict(features)[0]
#     pred_label = le.inverse_transform([pred_idx])[0]
    
#     # Get probabilities
#     proba = model.predict_proba(features)[0]
#     confidence = proba[pred_idx] * 100
    
#     print("\n📊 Prediction probabilities:")
#     for i, class_name in enumerate(le.classes_):
#         print(f"  {class_name}: {proba[i]*100:.2f}%")
    
#     return pred_label, confidence

# # ==============================
# # MAIN LOOP
# # ==============================
# def main():
#     print("="*50)
#     print("🎵 COVID-19 Cough Classification System")
#     print("="*50)
#     print("⚠️  IMPORTANT INSTRUCTIONS:")
#     print("   • Position microphone 2-3 inches from mouth")
#     print("   • Cough LOUDLY and clearly")
#     print("   • Make sure microphone is not muted")
#     print("   • Avoid background noise")
#     print("="*50)
    
#     valid_recordings = 0
#     covid_predictions = 0
#     healthy_predictions = 0
    
#     while True:
#         audio_file = record_audio()
        
#         if audio_file is None:
#             print("\n👋 Exiting...")
#             print(f"\n📊 Session Summary:")
#             print(f"   Total valid recordings: {valid_recordings}")
#             print(f"   COVID predictions: {covid_predictions}")
#             print(f"   HEALTHY predictions: {healthy_predictions}")
#             break
        
#         # Skip silent recordings
#         if audio_file == "SILENT":
#             continue
        
#         try:
#             prediction, confidence = predict_cough(audio_file)
            
#             print("\n" + "="*50)
#             print("📊 PREDICTION RESULT")
#             print("="*50)
            
#             # Handle low confidence or silent detection
#             if prediction == "NO_COUGH":
#                 print("❌ No valid cough detected!")
#                 print("🔄 Please record again with a clear cough")
                
#             elif confidence < CONFIDENCE_THRESHOLD:
#                 print(f"⚠️  Low confidence prediction: {confidence:.1f}%")
#                 print("🔄 Please record again for more reliable results")
#                 print(f"   Model is unsure between {le.classes_[0]} and {le.classes_[1]}")
                
#             else:
#                 # Valid prediction with good confidence
#                 valid_recordings += 1
#                 print(f"🩺 Predicted: {prediction}")
#                 print(f"🎯 Confidence: {confidence:.1f}%")
                
#                 if prediction == "COVID":
#                     covid_predictions += 1
#                     print("⚠️  COVID-19 indicators detected")
#                     print("   Please consult a healthcare professional")
#                 else:
#                     healthy_predictions += 1
#                     print("✅ Healthy cough detected")
#                     print("   No COVID-19 indicators found")
            
#             print("="*50)
            
#         except Exception as e:
#             print(f"\n❌ Error: {e}")
#             import traceback
#             traceback.print_exc()

# # ==============================
# # TEST MICROPHONE
# # ==============================
# def test_microphone():
#     """Test if microphone is working"""
#     print("\n🔍 Testing microphone...")
#     print("Please cough or speak for 2 seconds...")
    
#     test_recording = sd.rec(int(2 * SR), samplerate=SR, channels=1)
#     sd.wait()
#     test_recording = test_recording.flatten()
    
#     energy = np.mean(np.abs(test_recording))
#     max_amp = np.max(np.abs(test_recording))
    
#     print(f"\n📊 Test Results:")
#     print(f"  Audio energy: {energy:.6f}")
#     print(f"  Max amplitude: {max_amp:.6f}")
    
#     if energy < 0.001:
#         print("\n❌ Microphone is not working or too quiet!")
#         print("\nPlease check:")
#         print("1. Windows Settings → System → Sound")
#         print("2. Check if microphone is connected")
#         print("3. Check input volume levels (set to 100%)")
#         print("4. Try a different microphone")
#         return False
#     elif energy < 0.01:
#         print("\n⚠️  Microphone is working but very quiet")
#         print("   • Cough louder and closer to the mic")
#         print("   • Increase input volume in Windows")
#         return True
#     else:
#         print("\n✅ Microphone is working well!")
#         return True

# # ==============================
# # ADDITIONAL: Test with actual cough file
# # ==============================
# def test_with_sample():
#     """Optional: Test with a sample cough file if available"""
#     import os
    
#     sample_file = "sample_cough.wav"
#     if os.path.exists(sample_file):
#         print("\n🔍 Testing with sample cough file...")
#         try:
#             prediction, confidence = predict_cough(sample_file)
#             print(f"  Sample result: {prediction} ({confidence:.1f}%)")
#         except:
#             pass
#         #############################
# def predict_from_file(file_path):
#     """Function to be called from GUI"""
#     try:
#         prediction, confidence = predict_cough(file_path)
#         return prediction, confidence
#     except Exception as e:
#         return f"Error: {str(e)}", 0

# def get_model_info():
#     """Return model information"""
#     return {
#         'features': model.n_features_in_,
#         'classes': list(le.classes_)
#     }#######################
# if __name__ == "__main__":
#     # Test microphone first
#     if test_microphone():
#         # Optional: test with sample
#         test_with_sample()
#         # Run main program
#         main()
#     else:
#         print("\n❌ Cannot proceed without working microphone")
#         print("Please fix microphone issues and try again")
import pickle
import numpy as np
import sounddevice as sd
import wavio
import librosa

# ==============================
# SETTINGS
# ==============================
MODEL_FILE = "cough_classifier.pkl"
SR = 8000
DURATION = 4
MIC_DEVICE = 1
N_MFCC = 40
SILENCE_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 60.0

# ==============================
# LOAD MODEL
# ==============================
with open(MODEL_FILE, "rb") as f:
    model, le = pickle.load(f)
print("✅ Model loaded!")
print(f"📊 Model expects: {model.n_features_in_} features")
print(f"📋 Classes: {le.classes_}")

# ==============================
# FEATURE EXTRACTION (FIX ADDED HERE ONLY)
# ==============================
def extract_features(file_path, debug=False):
    audio, sr = librosa.load(file_path, sr=SR)
    
    energy = np.mean(np.abs(audio))
    if debug:
        print(f"  Raw audio energy: {energy:.6f}")
    
    if energy > SILENCE_THRESHOLD:
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        if debug:
            print(f"  Normalized (cough detected)")
    else:
        if debug:
            print(f"  No normalization (silence detected)")
    
    max_samples = SR * DURATION
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    
    if debug:
        print(f"  MFCC shape: {mfcc.shape}")
        print(f"  MFCC mean: {np.mean(mfcc):.6f}")
        print(f"  MFCC std: {np.std(mfcc):.6f}")
    
    # 🔥 ONLY FIX (feature size match)
    features = mfcc.flatten()
    expected_size = model.n_features_in_
    
    if len(features) > expected_size:
        features = features[:expected_size]
    else:
        features = np.pad(features, (0, expected_size - len(features)))
    
    return features, energy

# ==============================
# RECORD AUDIO
# ==============================
def record_audio(filename="temp.wav"):
    print("\n" + "="*50)
    print("Press Enter to start recording a cough (or 'q' to quit):")
    user_input = input()
    
    if user_input.lower() == "q":
        return None
    
    print(f"🎤 Recording started for {DURATION} seconds...")
    print("   Make sure to cough LOUDLY and close to the microphone!")
    
    recording = sd.rec( int(DURATION * SR),    samplerate=SR,    channels=1,    device=MIC_DEVICE)   
    sd.wait()
    recording = recording.flatten()
    
    audio_energy = np.mean(np.abs(recording))
    print(f"  Audio energy: {audio_energy:.6f}")
    
    if audio_energy < SILENCE_THRESHOLD:
        print("\n❌ ERROR: No cough detected (recording is silent!)")
        print("   Please:")
        print("   • Cough louder")
        print("   • Move closer to microphone")
        print("   • Check if microphone is working")
        print("   • Press Enter to try again")
        return "SILENT"
    
    if audio_energy < 0.05:
        print("⚠️  WARNING: Very quiet recording - amplifying...")
        recording = recording * 3.0
    
    wavio.write(filename, recording, SR, sampwidth=2)
    print("✅ Recording saved!")
    return filename

# ==============================
# PREDICT FUNCTION
# ==============================
def predict_cough(file_path):
    features, energy = extract_features(file_path, debug=True)
    
    if energy < SILENCE_THRESHOLD:
        print("\n❌ No cough detected in recording")
        return "NO_COUGH", 0.0
    
    features = features.reshape(1, -1)
    
    pred_idx = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    
    proba = model.predict_proba(features)[0]
    confidence = proba[pred_idx] * 100
    
    print("\n📊 Prediction probabilities:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {class_name}: {proba[i]*100:.2f}%")
    
    return pred_label, confidence

# ==============================
# MAIN LOOP
# ==============================
def main():
    print("="*50)
    print("🎵 COVID-19 Cough Classification System")
    print("="*50)
    print("⚠️  IMPORTANT INSTRUCTIONS:")
    print("   • Position microphone 2-3 inches from mouth")
    print("   • Cough LOUDLY and clearly")
    print("   • Make sure microphone is not muted")
    print("   • Avoid background noise")
    print("="*50)
    
    valid_recordings = 0
    covid_predictions = 0
    healthy_predictions = 0
    
    while True:
        audio_file = record_audio()
        
        if audio_file is None:
            print("\n👋 Exiting...")
            print(f"\n📊 Session Summary:")
            print(f"   Total valid recordings: {valid_recordings}")
            print(f"   COVID predictions: {covid_predictions}")
            print(f"   HEALTHY predictions: {healthy_predictions}")
            break
        
        if audio_file == "SILENT":
            continue
        
        try:
            prediction, confidence = predict_cough(audio_file)
            
            print("\n" + "="*50)
            print("📊 PREDICTION RESULT")
            print("="*50)
            
            if prediction == "NO_COUGH":
                print("❌ No valid cough detected!")
                print("🔄 Please record again with a clear cough")
                
            elif confidence < CONFIDENCE_THRESHOLD:
                print(f"⚠️  Low confidence prediction: {confidence:.1f}%")
                print("🔄 Please record again for more reliable results")
                print(f"   Model is unsure between {le.classes_[0]} and {le.classes_[1]}")
                
            else:
                valid_recordings += 1
                print(f"🩺 Predicted: {prediction}")
                print(f"🎯 Confidence: {confidence:.1f}%")
                
                if prediction == "COVID":
                    covid_predictions += 1
                    print("⚠️  COVID-19 indicators detected")
                else:
                    healthy_predictions += 1
                    print("✅ Healthy cough detected")
            
            print("="*50)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ==============================
# TEST MICROPHONE
# ==============================
def test_microphone():
    print("\n🔍 Testing microphone...")
    print("Please cough or speak for 2 seconds...")
    
    test_recording = sd.rec(int(2 * SR),samplerate=SR,channels=1,device=MIC_DEVICE)
    sd.wait()
    test_recording = test_recording.flatten()
    
    energy = np.mean(np.abs(test_recording))
    
    if energy < 0.001:
        print("\n❌ Microphone not working")
        return False
    else:
        print("\n✅ Microphone working")
        return True

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    if test_microphone():
        main()
    else:
        print("\n❌ Cannot proceed without working microphone")
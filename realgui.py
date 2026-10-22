import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import sounddevice as sd
import wavio
import numpy as np
from datetime import datetime
import os
import sys

# Import functions from predict.py
from predict import predict_cough, extract_features, SR, DURATION, test_microphone

class CoughClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("COVID-19 Cough Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.is_recording = False
        self.current_recording = None
        
        # Setup UI
        self.setup_ui()
        
        # Check microphone on startup
        self.check_microphone()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="🎵 COVID-19 Cough Detection System",
                               font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#34495e')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#34495e')
        left_panel.pack(side='left', fill='both', expand=True, padx=10)
        
        # Microphone status
        mic_frame = tk.LabelFrame(left_panel, text="Microphone Status", 
                                  font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        mic_frame.pack(fill='x', pady=10)
        
        self.mic_status = tk.Label(mic_frame, text="Checking...", 
                                   font=('Arial', 10), fg='#f39c12', bg='#34495e')
        self.mic_status.pack(pady=10)
        
        # Recording controls
        record_frame = tk.LabelFrame(left_panel, text="Recording", 
                                     font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        record_frame.pack(fill='x', pady=10)
        
        self.record_btn = tk.Button(record_frame, text="🎤 Start Recording",
                                    command=self.start_recording,
                                    font=('Arial', 14, 'bold'),
                                    bg='#27ae60', fg='white',
                                    height=2, width=20)
        self.record_btn.pack(pady=15)
        
        # File upload option
        upload_frame = tk.LabelFrame(left_panel, text="Or Upload Audio File", 
                                     font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        upload_frame.pack(fill='x', pady=10)
        
        upload_btn = tk.Button(upload_frame, text="📁 Upload WAV File",
                               command=self.upload_file,
                               font=('Arial', 10),
                               bg='#3498db', fg='white',
                               height=1, width=18)
        upload_btn.pack(pady=10)
        
        # Instructions
        instructions = """
        📋 How to use:
        1. Click 'Start Recording'
        2. Cough clearly into microphone
        3. Wait for analysis
        4. View results
        
        OR
        
        1. Click 'Upload WAV File'
        2. Select a recorded cough
        3. Get instant analysis
        """
        
        inst_label = tk.Label(record_frame, text=instructions,
                             font=('Arial', 9), fg='#bdc3c7', bg='#34495e',
                             justify='left')
        inst_label.pack(pady=10)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#34495e')
        right_panel.pack(side='right', fill='both', expand=True, padx=10)
        
        # Result display
        result_frame = tk.LabelFrame(right_panel, text="Analysis Result", 
                                     font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        result_frame.pack(fill='both', expand=True)
        
        self.result_label = tk.Label(result_frame, text="Ready",
                                     font=('Arial', 24, 'bold'), 
                                     fg='#95a5a6', bg='#34495e')
        self.result_label.pack(pady=30)
        
        self.confidence_label = tk.Label(result_frame, text="",
                                         font=('Arial', 12), 
                                         fg='white', bg='#34495e')
        self.confidence_label.pack()
        
        self.advice_label = tk.Label(result_frame, text="Click 'Start Recording' to begin",
                                     font=('Arial', 11), 
                                     fg='#bdc3c7', bg='#34495e',
                                     wraplength=400, justify='center')
        self.advice_label.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(result_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", 
                                   bg='#2c3e50', fg='#95a5a6',
                                   font=('Arial', 9))
        self.status_bar.pack(side='bottom', fill='x', pady=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_panel, text="Session Statistics", 
                                    font=('Arial', 10, 'bold'), fg='white', bg='#34495e')
        stats_frame.pack(fill='x', pady=10)
        
        self.stats_total = tk.Label(stats_frame, text="Total Tests: 0",
                                    font=('Arial', 10), fg='white', bg='#34495e')
        self.stats_total.pack(pady=2)
        
        self.stats_covid = tk.Label(stats_frame, text="COVID Detected: 0",
                                    font=('Arial', 10), fg='#e74c3c', bg='#34495e')
        self.stats_covid.pack(pady=2)
        
        self.stats_healthy = tk.Label(stats_frame, text="Healthy Detected: 0",
                                      font=('Arial', 10), fg='#2ecc71', bg='#34495e')
        self.stats_healthy.pack(pady=2)
        
        # Reset stats button
        reset_btn = tk.Button(stats_frame, text="Reset Stats",
                             command=self.reset_stats,
                             font=('Arial', 8), bg='#7f8c8d', fg='white')
        reset_btn.pack(pady=5)
        
        # Initialize statistics
        self.total_tests = 0
        self.covid_count = 0
        self.healthy_count = 0
    
    def check_microphone(self):
        """Check microphone in separate thread"""
        def check():
            try:
                test_recording = sd.rec(int(2 * 8000), samplerate=8000, channels=1)
                sd.wait()
                test_recording = test_recording.flatten()
                energy = np.mean(np.abs(test_recording))
                
                if energy < 0.001:
                    status = "❌ Microphone not detected or too quiet!"
                    color = '#e74c3c'
                elif energy < 0.01:
                    status = "⚠️ Microphone working but quiet"
                    color = '#f39c12'
                else:
                    status = "✅ Microphone working properly"
                    color = '#2ecc71'
                
                self.mic_status.config(text=status, fg=color)
                self.status_bar.config(text=f"Microphone: {status}")
            except Exception as e:
                self.mic_status.config(text=f"❌ Error: {str(e)}", fg='#e74c3c')
        
        threading.Thread(target=check, daemon=True).start()
    
    def start_recording(self):
        """Start recording in thread"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.record_btn.config(text="🔴 Recording...", bg='#e74c3c', state='disabled')
        self.progress.start()
        self.result_label.config(text="Recording...", fg='#f39c12')
        self.advice_label.config(text="Please cough clearly into the microphone")
        self.status_bar.config(text="Recording in progress...")
        
        threading.Thread(target=self.record_and_predict, daemon=True).start()
    
    def record_and_predict(self):
        """Record audio and predict"""
        try:
            # Record
            duration = 4  # seconds
            sr = 8000
            self.status_bar.config(text=f"Recording for {duration} seconds...")
            
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
            sd.wait()
            recording = recording.flatten()
            
            # Check energy
            energy = np.mean(np.abs(recording))
            
            if energy < 0.01:
                self.root.after(0, self.show_error, "No cough detected! Please cough louder.")
                return
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cough_{timestamp}.wav"
            
            # Amplify if needed
            if energy < 0.05:
                recording = recording * 3.0
            
            wavio.write(filename, recording, sr, sampwidth=2)
            
            # Predict using your predict.py function
            self.status_bar.config(text="Analyzing cough...")
            prediction, confidence = predict_cough(filename)
            
            # Show result
            self.root.after(0, self.show_result, prediction, confidence)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        finally:
            self.is_recording = False
            self.root.after(0, self.stop_recording_ui)
    
    def upload_file(self):
        """Upload and analyze existing audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.progress.start()
                self.status_bar.config(text="Analyzing uploaded file...")
                self.result_label.config(text="Analyzing...", fg='#f39c12')
                
                # Predict using your predict.py function
                prediction, confidence = predict_cough(filename)
                
                self.show_result(prediction, confidence)
                
            except Exception as e:
                self.show_error(f"Error analyzing file: {str(e)}")
            finally:
                self.progress.stop()
    
    def show_result(self, prediction, confidence):
        """Display prediction result"""
        self.progress.stop()
        
        # Update statistics
        self.total_tests += 1
        self.stats_total.config(text=f"Total Tests: {self.total_tests}")
        
        if prediction == "COVID":
            self.covid_count += 1
            self.stats_covid.config(text=f"COVID Detected: {self.covid_count}")
            color = '#e74c3c'
            advice = "⚠️ COVID-19 indicators detected.\n\nPlease consult a healthcare professional and consider getting tested."
            self.result_label.config(text="🩺 COVID-19", fg=color)
            
        elif prediction == "HEALTHY":
            self.healthy_count += 1
            self.stats_healthy.config(text=f"Healthy Detected: {self.healthy_count}")
            color = '#2ecc71'
            advice = "✅ Healthy cough detected.\n\nNo COVID-19 indicators found. Continue following safety guidelines."
            self.result_label.config(text="✅ HEALTHY", fg=color)
            
        else:
            color = '#e74c3c'
            advice = f"❌ Error: {prediction}\n\nPlease try again."
            self.result_label.config(text="Error", fg=color)
            self.confidence_label.config(text="")
            self.advice_label.config(text=advice)
            return
        
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        self.advice_label.config(text=advice)
        self.status_bar.config(text=f"Result: {prediction} ({confidence:.1f}% confidence)")
        
        # Change button color based on result
        if prediction == "COVID":
            self.record_btn.config(bg='#e74c3c')
            self.root.after(3000, lambda: self.record_btn.config(bg='#27ae60'))
    
    def show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.result_label.config(text="Error", fg='#e74c3c')
        self.confidence_label.config(text="")
        self.advice_label.config(text=f"❌ {error_msg}\n\nPlease try again.")
        self.status_bar.config(text=f"Error: {error_msg}")
        self.stop_recording_ui()
    
    def stop_recording_ui(self):
        """Reset UI after recording"""
        self.record_btn.config(text="🎤 Start Recording", bg='#27ae60', state='normal')
        if self.result_label.cget("text") == "Recording...":
            self.result_label.config(text="Ready", fg='#95a5a6')
            self.advice_label.config(text="Click 'Start Recording' to begin")
    
    def reset_stats(self):
        """Reset session statistics"""
        self.total_tests = 0
        self.covid_count = 0
        self.healthy_count = 0
        self.stats_total.config(text="Total Tests: 0")
        self.stats_covid.config(text="COVID Detected: 0")
        self.stats_healthy.config(text="Healthy Detected: 0")
        self.status_bar.config(text="Statistics reset")
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to exit?"):
            self.root.destroy()

# ==============================
# RUN APPLICATION
# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    app = CoughClassifierGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
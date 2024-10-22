"""
# Speaker Identification System using Machine Learning

A machine learning system that identifies speakers from voice samples using MFCC features 
and Gaussian Mixture Models (GMM). This system achieves high accuracy in speaker recognition
by analyzing unique voice characteristics.

## Hardware/Software Requirements
- Python 3.x
- NumPy
- SciPy
- python_speech_features
- scikit-learn
- Audio input device for recording
- WAV format audio files

## Library Dependencies
- numpy: For numerical operations
- scipy.io: For reading WAV files
- python_speech_features: For MFCC feature extraction
- sklearn.mixture: For GMM modeling
- warnings: For handling warnings
- os: For file operations

## Data Structure
Training_Data/
├── Speaker1/
│   ├── audio1.wav
│   └── ...
├── Speaker2/
│   ├── audio1.wav
│   └── ...
Testing_Data/
├── known_speakers/
│   ├── test1.wav
│   └── ...
└── unknown_speakers/
    ├── P1.wav
    └── ...
"""
## Code
# Import required libraries
import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

class SpeakerIdentificationSystem:
    def __init__(self, n_components=16):
        """
        Initialize the Speaker Identification System
        
        Parameters:
        n_components (int): Number of Gaussian components for GMM
        """
        self.n_components = n_components
        self.speaker_models = {}
        
    def extract_features(self, audio_path):
        """
        Extract MFCC features from audio file
        
        Parameters:
        audio_path (str): Path to the audio file
        
        Returns:
        numpy.ndarray: MFCC features
        """
        # Read audio file
        frequency, signal = wavfile.read(audio_path)
        
        # Extract MFCC features
        features = mfcc(signal, frequency,
                       winlen=0.025,      # 25ms window length
                       winstep=0.01,      # 10ms step size
                       numcep=20,         # Number of cepstral coefficients
                       nfilt=40,          # Number of filters
                       nfft=1200,         # FFT size
                       preemph=0.97,      # Pre-emphasis coefficient
                       ceplifter=22,      # Length of cepstral liftering
                       appendEnergy=True)  # Add energy feature
        
        return features
    
    def train_gmm(self, features):
        """
        Train GMM model on extracted features
        
        Parameters:
        features (numpy.ndarray): MFCC features
        
        Returns:
        GaussianMixture: Trained GMM model
        """
        # Initialize and train GMM
        gmm = GaussianMixture(n_components=self.n_components,
                             covariance_type='full',
                             max_iter=200,
                             random_state=42)
        gmm.fit(features)
        return gmm
    
    def train_speakers(self, base_path):
        """
        Train GMM models for all speakers in the training directory
        
        Parameters:
        base_path (str): Path to training data directory
        """
        print("Training speaker models...")
        
        # Iterate through speaker directories
        for speaker_dir in os.listdir(base_path):
            speaker_path = os.path.join(base_path, speaker_dir)
            
            if os.path.isdir(speaker_path):
                all_features = []
                
                # Process all audio files for current speaker
                for audio_file in os.listdir(speaker_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(speaker_path, audio_file)
                        features = self.extract_features(audio_path)
                        all_features.extend(features)
                
                if all_features:
                    print(f"Training model for {speaker_dir}")
                    # Train GMM model for speaker
                    self.speaker_models[speaker_dir] = self.train_gmm(np.array(all_features))
    
    def identify_speaker(self, audio_path):
        """
        Identify speaker from test audio file
        
        Parameters:
        audio_path (str): Path to test audio file
        
        Returns:
        str: Identified speaker name
        """
        # Extract features from test audio
        features = self.extract_features(audio_path)
        
        # Initialize variables for scoring
        max_score = float('-inf')
        identified_speaker = None
        
        # Score against all speaker models
        for speaker, model in self.speaker_models.items():
            score = np.mean(model.score_samples(features))
            if score > max_score:
                max_score = score
                identified_speaker = speaker
                
        return identified_speaker
    
    def evaluate_system(self, test_path):
        """
        Evaluate system performance on test data
        
        Parameters:
        test_path (str): Path to test data directory
        """
        print("\nTesting the system...")
        results = {'named': {}, 'unnamed': {}}
        
        # Process all test files
        for audio_file in os.listdir(test_path):
            if not audio_file.endswith('.wav'):
                continue
                
            audio_path = os.path.join(test_path, audio_file)
            identified_speaker = self.identify_speaker(audio_path)
            
            print(f"Audio file: {audio_file}")
            print(f"Identified speaker: {identified_speaker}")
            
            # Store results based on file type (named vs unnamed)
            if audio_file.startswith('P') and audio_file[1].isdigit():
                results['unnamed'][audio_file] = {
                    'identified_speaker': identified_speaker
                }
            else:
                true_speaker = audio_file.split('_')[0]
                results['named'][audio_file] = {
                    'true_speaker': true_speaker,
                    'identified_speaker': identified_speaker,
                    'correct': true_speaker in identified_speaker
                }
        
        # Calculate and display accuracy
        self._display_results(results)
        
    def _display_results(self, results):
        """
        Display evaluation results
        
        Parameters:
        results (dict): Dictionary containing evaluation results
        """
        # Calculate accuracy for named samples
        named_total = len(results['named'])
        correct_count = sum(1 for r in results['named'].values() if r['correct'])
        accuracy = (correct_count / named_total) * 100 if named_total > 0 else 0
        
        # Display results
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print("\nUnnamed Samples Identification (P1-P6):")
        for file, data in sorted(results['unnamed'].items()):
            print(f"{file}: Identified as {data['identified_speaker']}")

def main():
    """
    Main function to run the speaker identification system
    """
    # Initialize system
    speaker_system = SpeakerIdentificationSystem(n_components=16)
    
    # Set paths
    train_path = "Voice_Samples_Training"
    test_path = "Testing_Audio"
    
    # Train and evaluate
    speaker_system.train_speakers(train_path)
    speaker_system.evaluate_system(test_path)

if __name__ == "__main__":
    main()

"""
## Performance Notes

The system achieves high accuracy through:
1. Robust MFCC feature extraction
2. GMM-based speaker modeling
3. Likelihood-based speaker identification

## Troubleshooting Guide

1. Poor Recognition Accuracy:
   - Check audio quality
   - Verify training data quantity
   - Adjust GMM components
   - Check MFCC parameters

2. Processing Errors:
   - Verify file formats
   - Check file paths
   - Ensure sufficient memory
   - Verify library versions

3. Performance Issues:
   - Optimize feature extraction
   - Reduce GMM components
   - Process in batches
   - Use parallel processing

## Future Improvements

1. Real-time Processing
   - Implement streaming audio
   - Optimize processing speed
   - Add voice activity detection

2. Enhanced Features
   - Add speaker verification
   - Implement voice quality metrics
   - Add confidence scores
   - Support more audio formats

3. UI Development
   - Add GUI interface
   - Real-time visualization
   - Result reporting
   - Model management

4. Advanced Features
   - Emotion recognition
   - Gender classification
   - Age estimation
   - Language identification
"""

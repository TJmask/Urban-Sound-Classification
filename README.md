# Urban-Sound-Classification
This is a classification problem and is meant to let me have hands on exposure to audio processing in the usual classification scenario. Compared to other bodily features, voice is dynamic and complex. And voice classification has found useful applications in classifying speakers' gender, mother tongue or ethnicity (accent), emotion states, identity verification, verbal command control, and so forth. Furthermore voice classification potentially can apply to interactive-voice-response system for detecting the moods and tones of customers, thereby guessing if the calls are of complaints or complement, for example. As a consequence, digging into audio data can be beneficial. 

## Data source: UrbanSound
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes. These sound excerpts are digital audio files in .wav format.
All excerpts are taken from field recordings uploaded to www.Freesound.Org. The files are pre-sorted into ten folds (folders named fold1-fold10).
Additionally, a CSV file containing metadata information about each excerpt is also provided.
The size of the dataset is about 10GB(Big Data).

## Introduction to Feature extracting
Mel frequency cepstral coefficient (MFCC) 
MFCC divided the speech into frames (typically 20 ms for each frame), applied discrete Fourier Transformation over every frame, retained the logarithm of the amplitude spectrum, smoothed the spectrum, and applied discrete cosine transform.
Chroma features
Chroma features are an powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.
Cepstral coefficient (CC)
Cepstral coefficients are the inverse Fourier Transform representation of the log magnitude of the spectrum.
Linear prediction coefficient (LPC)
LPC consists of finding a time-based series of n-pole infinite impulse response (IIR) filters whose coefficients better adapt to the formants of a audio signal. The main idea behind LPC is that a sample of speech can be approximated as a linear combination of past audio samples.

## Feature Selection
**MFCC:**  
I extract the features on a per-frame basis using a window size of 20 ms and 50% frame overlap. I compute 40 Mel bands between 0 and 22050 Hz and keep the first 25 MFCC coefficients.

**Chroma features:**  
I used three features, chroma_cqt(compute a chromagram from a constant-Q transform.), chroma_cens(computes the chroma variant “Chroma Energy Normalized” ), chroma_stft(compute a chromagram from an STFT spectrogram or waveform). All of them are computed 40 Mel bands between 0 and 22050 Hz and keep the first 25 coefficients.

**Cepstral coefficient(CC):**  
I compute 40 Mel bands between 0 and 22050 Hz and keep the first 25 CC coefficients.


## Model Selection
Used grid-search to find the best parameters

Used cross validation on folder level to deal with overfitting

Used 4 layers in CNN model

Used CNN, SVM, KNN, and Random Forest in Ensemble model to do soft majority voting



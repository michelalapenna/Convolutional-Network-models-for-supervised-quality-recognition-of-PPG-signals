# Convolutional-Network-models-for-supervised-quality-recognition-of-PPG-signals

Photoplethysmogram (PPG) signals recover key physiological
parameters as pulse, oximetry and ECG. In this paper, we first employ an hybrid
architecture combining Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)
for the analysis of PPG signals to enable an automated quality recognition.
Then, we compare its performance to a simpler CNN architecture enriched with Kolmogorov-Arnold
Networks (KAN) layers.
Our results suggest that the usage of KAN layers is effective at reducing the number of
parameters, while also enhancing the performance of CNNs when 
equipped with standard Multilayer perceptron (MLP) layers.

This repository contains the code for preprocessing and training/evaluation of models designed to process Photoplethysmogram (PPG) signals. Specifically, we provide two main notebooks for preprocessing and training/evaluation, along with several scripts that contain the preprocessing filters and models definitions, training and test loops with grid search, and the necessary imports.

We have developed three distinct models for processing PPG signals: a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM), a CNN combined with Multilayer Perceptron (MLP), and a CNN combined with Kolmogorov-Arnold Network (KAN). Each model is trained on a labeled dataset, split into three subsets: 40% for training, 30% for validation, and 30% for testing.

The preprocessing of PPG signals is critical for optimizing the model's performance due to their sensitivity to noise, motion artifacts, and environmental factors. The preprocessing steps applied include downsampling, smoothing via a moving average filter, Butterworth low-pass filtering, and Min-Max scaling. Downsampling reduces the sampling rate of the signal while preserving important characteristics, such as heart rate peaks and troughs, with anti-aliasing applied to avoid distortion. A moving average filter is used to smooth out noise, carefully adjusting the window size to maintain signal features. The Butterworth low-pass filter addresses high-frequency noise, particularly from motion artifacts, with a cutoff frequency of 50 Hz to preserve relevant signal components. Finally, Min-Max scaling normalizes the signal amplitudes to a range of 0 to 1, ensuring each signal contributes equally during model training.

In this research work, the efficacy of employing a CNN-LSTM for identifying the quality PPG signals is established. The proposed CNN-LSTM model is shown to outperform other simpler CNN models thanks to combining CNN ability to successfully capturing spatial relationships and LSTM ability to capturing temporal ones. Combining CNN with simpler feed-forward networks, including MLP and KAN, also give good results and could be recommended for the case when computational capacity is limited.

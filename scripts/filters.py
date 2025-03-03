from imports import*

# Function to apply moving average filter
def apply_moving_average(signal, window_size):
    # Convert the signal to a numpy array if it's not already
    signal = np.array(signal)
    # Define a simple moving average kernel
    kernel = np.ones(window_size) / window_size
    # Apply the moving average filter
    smoothed_signal = convolve(signal.squeeze(), kernel, mode='same')
    return smoothed_signal

def butter_lowpass(cutoff_freq, fs, order=4):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butterworth_filter(signal, cutoff_freq, fs, order=4):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


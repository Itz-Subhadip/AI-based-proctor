import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(1000)

# Define a simple noise cancellation filter
def noise_cancellation(signal, order=3):
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        if i < order:
            filtered_signal[i] = signal[i]
        else:
            filtered_signal[i] = np.mean(signal[i-order:i])
    return filtered_signal

# Apply noise cancellation
filtered_signal = noise_cancellation(signal)

# Plot the original and filtered signals
'''plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise Cancellation')
plt.legend()
plt.grid(True)
plt.show()'''

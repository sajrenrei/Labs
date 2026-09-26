import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

fs = 1000
duration = 2.0
frequency = 1.2
noise_amplitude = 0.5


t = np.arange(0, duration, 1 / fs)


signal = np.sin(2 * np.pi * frequency * t)


plt.figure(figsize=(10, 4))
plt.plot(t, signal, label="Початковий сигнал", color="blue", linewidth=2)
plt.title("Завдання 2. Початковий (чистий) сигнал")
plt.xlabel("Час (секунди)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


noise = np.random.normal(0, noise_amplitude, len(t))
noisy_signal = signal + noise


print("--- Завдання 4. Результати статистичного аналізу ---")
print(f"Початковий сигнал -> Середнє: {np.mean(signal):.4f}, Стандартне відхилення: {np.std(signal):.4f}")
print(f"Зашумлений сигнал -> Середнє: {np.mean(noisy_signal):.4f}, Стандартне відхилення: {np.std(noisy_signal):.4f}\n")


plt.figure(figsize=(10, 4))
plt.plot(t, noisy_signal, label="Зашумлений", color="orange", alpha=0.6)
plt.plot(t, signal, label="Початковий", color="blue", linewidth=2)
plt.title("Завдання 5. Порівняння сигналів на одному графіку")
plt.xlabel("Час (секунди)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


window = 10
kernel = np.ones(window) / window
filtered_signal = np.convolve(noisy_signal, kernel, mode="same")

plt.figure(figsize=(10, 4))
plt.plot(t, noisy_signal, label="Зашумлений", color="orange", alpha=0.4)
plt.plot(t, signal, label="Початковий", color="blue", linestyle="--")
plt.plot(t, filtered_signal, label=f"Відфільтрований (вікно={window})", color="green", linewidth=2)
plt.title("Завдання 6. Згладжування сигналу ковзним середнім (вікно = 10)")
plt.xlabel("Час (секунди)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


windows = [5, 10, 25, 50]
for w in windows:
    k = np.ones(w) / w
    f_sig = np.convolve(noisy_signal, k, mode="same")

    plt.figure(figsize=(10, 3.5))
    plt.plot(t, noisy_signal, label="Зашумлений", color="orange", alpha=0.3)
    plt.plot(t, signal, label="Початковий", color="blue", linestyle="--")
    plt.plot(t, f_sig, label=f"Відфільтрований (вікно={w})", color="red", linewidth=1.8)
    plt.title(f"Завдання 7. Дослідження розміру вікна = {w}")
    plt.xlabel("Час (секунди)")
    plt.ylabel("Амплітуда")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


fft_signal = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1 / fs)

positive = frequencies >= 0
pos_freqs = frequencies[positive]
amp_spectrum = np.abs(fft_signal[positive]) * 2 / len(signal)

plt.figure(figsize=(10, 4))
plt.plot(pos_freqs, amp_spectrum, color="purple", linewidth=1.5)
plt.title("Завдання 8. Спектр частот сигналу (FFT)")
plt.xlabel("Frequency, Hz")
plt.ylabel("Amplitude")
plt.xlim(0, 10)
plt.grid(True)
plt.tight_layout()
plt.show()

peak_freq = pos_freqs[np.argmax(amp_spectrum)]
print("--- Завдання 8. Аналіз спектру ---")
print(f"Основний пік спектру знаходиться на частоті: {peak_freq:.2f} Гц\n")


freqs = np.array([1.0, 1.2, 1.5, 2.0])
hr_values = 60 * freqs

print("--- Завдання 9. Таблиця моделювання ЧСС ---")
print(f"{'Frequency, Hz':<15} | {'HR, уд/хв':<10}")
print("-" * 30)
for f, hr in zip(freqs, hr_values):
    print(f"{f:<15.1f} | {int(hr):<10}")
print("-" * 30 + "\n")

plt.figure(figsize=(10, 4))
for f in freqs:
    s = np.sin(2 * np.pi * f * t)
    plt.plot(t, s, label=f"f = {f} Hz (HR = {int(60 * f)} уд/хв)")
plt.title("Завдання 9. Моделювання зміни частоти серцевого ритму")
plt.xlabel("Час (секунди)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
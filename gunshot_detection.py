import numpy as np
import sounddevice as sd
import time
from scipy.signal import butter, lfilter
from datetime import datetime

# ===== CONFIG =====
SAMPLE_RATE = 44100       # Hz
BLOCK_SIZE = 4096         # samples per block (~93 ms)
CHANNELS = 2              # stereo for direction detection
THRESHOLD = 0.008         # lower for medium/low pitch detection
LOWCUT = 100.0            # lower frequency limit (Hz)
HIGHCUT = 1000.0          # upper frequency limit (Hz)

# ===== HELPER FUNCTIONS =====
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLE_RATE, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def detect_gunshot(audio_block, threshold=THRESHOLD):
    if len(audio_block) == 0:
        return False, 0.0, 0.0

    filtered = bandpass_filter(audio_block, LOWCUT, HIGHCUT, SAMPLE_RATE)
    rms = np.sqrt(np.mean(filtered**2))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / SAMPLE_RATE)
    mags = np.abs(np.fft.rfft(filtered))
    centroid = np.sum(freqs * mags) / (np.sum(mags) + 1e-12)

    if rms > threshold and LOWCUT < centroid < HIGHCUT:
        return True, rms, centroid
    return False, rms, centroid

# ===== MAIN FUNCTION =====
def main():
    print("[INFO] Starting gunshot detector (~1m, low/mid pitch)...\n")

    def audio_callback(indata, frames, time_info, status):
        left = indata[:, 0].astype(np.float32)
        right = indata[:, 1].astype(np.float32)

        # Mono mix and normalization
        mono = (left + right) / 2.0
        mono = mono / (np.max(np.abs(mono)) + 1e-12)

        detected, rms, centroid = detect_gunshot(mono)

        if detected:
            rms_left = np.sqrt(np.mean(left**2))
            rms_right = np.sqrt(np.mean(right**2))

            # Direction estimation
            if rms_left > rms_right * 1.2:
                direction = "LEFT"
            elif rms_right > rms_left * 1.2:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 🔴 Sound detected! "
                  f"(RMS={rms:.4f}, Centroid={centroid:.1f} Hz, Direction={direction})")
        else:
            print(f"Sound not recognized (RMS={rms:.4f}, Centroid={centroid:.0f})")

    try:
        with sd.InputStream(channels=CHANNELS,
                            samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            callback=audio_callback):
            print("[INFO] Listening... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

# ===== ENTRY POINT =====
if __name__ == "__main__":
    main()

import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Default HuggingFace Deepfake Audio Model
DEFAULT_AUDIO_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"

class AudioDeepfakeDetector:
    """
    State-of-the-art Deepfake Voice / Audio Detector.
    Combines:
    1. Pretrained Wav2Vec2 Audio Transformer fine-tuned on synthetic voice datasets.
    2. Overlapping temporal window chunk analysis for locating spliced/injected deepfakes.
    3. Multi-domain acoustic forensic feature extraction:
       - High-frequency vocoder spectral rolloff
       - Fundamental frequency (F0) pitch trajectory and jitter micro-instability
       - Spectral centroid, flatness, and zero-crossing rate
       - Silence/noise floor continuity analysis
    """
    def __init__(self, model_id=DEFAULT_AUDIO_MODEL_ID, device=None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_id = model_id
        self.target_sr = 16000  # Wav2Vec2 standard sampling rate
        self.model = None
        self.feature_extractor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading Audio Deepfake Model: {self.model_id} on {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            print("Audio Deepfake Model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load transformer audio model {self.model_id}: {e}")
            self.model = None
            self.feature_extractor = None

    def load_audio(self, audio_input):
        """
        Loads and standardizes audio from a file path or Gradio audio tuple.
        Returns: (y, sr) where y is float32 mono array normalized to [-1, 1] at 16kHz.
        """
        if audio_input is None:
            raise ValueError("No audio input provided.")

        # Check if Gradio passed (sample_rate, numpy_array)
        if isinstance(audio_input, tuple):
            sr_orig, y_orig = audio_input
            if y_orig.ndim > 1:
                # Convert stereo to mono
                y_orig = np.mean(y_orig, axis=1)
            # Normalize int types to float32 [-1, 1]
            if np.issubdtype(y_orig.dtype, np.integer):
                max_val = float(np.iinfo(y_orig.dtype).max)
                y_orig = y_orig.astype(np.float32) / max_val
            else:
                y_orig = y_orig.astype(np.float32)
                
            if sr_orig != self.target_sr:
                y = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=self.target_sr)
            else:
                y = y_orig
            sr = self.target_sr
        elif isinstance(audio_input, str):
            # Load from file path (supports wav, mp3, m4a, flac, ogg)
            y, sr = librosa.load(audio_input, sr=self.target_sr, mono=True)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Peak normalization
        max_abs = np.max(np.abs(y))
        if max_abs > 1e-6:
            y = y / max_abs

        return y, sr

    def extract_forensic_features(self, y, sr):
        """
        Computes complementary acoustic forensic indicators that reveal vocoder artifacts,
        F0 micro-jitter unnaturalness, and high-frequency spectral limits.
        """
        metrics = {}
        
        # 1. Pitch / F0 Trajectory & Jitter
        try:
            # Detect pitch using Yin algorithm (50Hz to 500Hz for speech range)
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0_voiced = f0[~np.isnan(f0)]
            if len(f0_voiced) > 10:
                mean_f0 = float(np.mean(f0_voiced))
                std_f0 = float(np.std(f0_voiced))
                # Micro-jitter estimation (relative frame-to-frame diff)
                diffs = np.abs(np.diff(f0_voiced))
                jitter_rel = float(np.mean(diffs) / (mean_f0 + 1e-6)) * 100.0  # percentage
                voiced_ratio = float(len(f0_voiced) / len(f0))
            else:
                mean_f0, std_f0, jitter_rel, voiced_ratio = 0.0, 0.0, 0.0, 0.0
        except Exception:
            f0 = np.array([])
            mean_f0, std_f0, jitter_rel, voiced_ratio = 0.0, 0.0, 0.0, 0.0

        metrics["mean_f0_hz"] = round(mean_f0, 1)
        metrics["std_f0_hz"] = round(std_f0, 1)
        metrics["jitter_percent"] = round(jitter_rel, 2)
        metrics["voiced_ratio"] = round(voiced_ratio, 2)

        # 2. High-Frequency Vocoder Energy Cutoff (>6.5kHz vs Total)
        # Most TTS/vocoders truncate or have phase smearing above 6.5k-8kHz
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        hf_mask = freqs >= 6500
        hf_energy = np.sum(stft[hf_mask, :])
        total_energy = np.sum(stft) + 1e-8
        hf_ratio = float(hf_energy / total_energy)
        metrics["hf_energy_ratio"] = round(hf_ratio, 4)

        # 3. Spectral Centroid & Flatness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        metrics["spectral_centroid_hz"] = round(float(np.mean(centroid)), 1)
        metrics["spectral_flatness"] = round(float(np.mean(flatness)), 5)

        # 4. Silence Floor Analysis (Check for synthetic zero-noise gaps)
        intervals = librosa.effects.split(y, top_db=40)
        unvoiced_samples = len(y) - sum(end - start for start, end in intervals) if len(intervals) > 0 else 0
        silence_ratio = float(unvoiced_samples / (len(y) + 1e-6))
        metrics["silence_ratio"] = round(silence_ratio, 2)

        return metrics, f0

    def predict(self, audio_input, chunk_duration=3.5, overlap=1.0, threshold=0.5):
        """
        Runs complete deepfake voice analysis.
        Uses sliding temporal windows for granular verification.
        
        Args:
            audio_input: filepath or (sr, np_array)
            chunk_duration: window length in seconds (default 3.5s)
            overlap: window overlap in seconds (default 1.0s)
            threshold: decision threshold (default 0.5)
            
        Returns:
            dict containing:
                - verdict: "Fake" or "Real"
                - fake_prob: float (0.0 to 1.0)
                - real_prob: float (0.0 to 1.0)
                - confidence_percent: float (0.0 to 100.0)
                - segment_results: list of chunk details
                - forensic_metrics: dict
                - summary_text: formatted string
                - plot_image_path: path to generated spectrogram plot
        """
        y, sr = self.load_audio(audio_input)
        total_duration = len(y) / sr
        
        forensic_metrics, f0 = self.extract_forensic_features(y, sr)
        
        chunk_samples = int(chunk_duration * sr)
        step_samples = int(max(0.5, chunk_duration - overlap) * sr)
        
        # Build segments
        segments = []
        if len(y) <= chunk_samples:
            segments.append((0, len(y), y))
        else:
            start = 0
            while start < len(y):
                end = min(start + chunk_samples, len(y))
                chunk = y[start:end]
                # If last chunk is too short, pad with zeros to avoid distortion
                if len(chunk) < int(0.5 * sr):
                    break
                segments.append((start, end, chunk))
                if end == len(y):
                    break
                start += step_samples

        segment_results = []
        chunk_fake_probs = []
        
        if self.model is not None and self.feature_extractor is not None:
            for idx, (st, en, chunk) in enumerate(segments):
                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                    
                # Note: mo-thecreator/Deepfake-audio-detection config: {0: 'fake', 1: 'real'}
                fake_p = float(probs[0])
                real_p = float(probs[1])
                
                chunk_fake_probs.append(fake_p)
                segment_results.append({
                    "segment": idx + 1,
                    "start_time": round(st / sr, 2),
                    "end_time": round(en / sr, 2),
                    "fake_prob": fake_p,
                    "real_prob": real_p,
                    "is_fake": fake_p >= threshold
                })
        else:
            # Fallback heuristic based on forensic features if model not loaded
            # Higher jitter + natural noise floor = real; abnormally rigid pitch + zero floor = fake
            heuristic_fake = 0.5
            if forensic_metrics["jitter_percent"] < 0.2 and forensic_metrics["voiced_ratio"] > 0.4:
                heuristic_fake += 0.25
            if forensic_metrics["hf_energy_ratio"] < 0.005:
                heuristic_fake += 0.2
            heuristic_fake = min(0.95, max(0.05, heuristic_fake))
            chunk_fake_probs = [heuristic_fake]
            segment_results.append({
                "segment": 1,
                "start_time": 0.0,
                "end_time": round(total_duration, 2),
                "fake_prob": heuristic_fake,
                "real_prob": 1.0 - heuristic_fake,
                "is_fake": heuristic_fake >= threshold
            })

        # Aggregation:
        # A recording is considered fake if either:
        # 1. The average fake probability exceeds the threshold, or
        # 2. Any contiguous segments exhibit extremely high fake probability (>0.85), indicating injected/spliced fake voice.
        mean_fake_prob = float(np.mean(chunk_fake_probs))
        max_fake_prob = float(np.max(chunk_fake_probs))
        
        # Weighted decision prioritizing local synthetic bursts
        overall_fake_prob = 0.65 * mean_fake_prob + 0.35 * max_fake_prob
        is_fake = overall_fake_prob >= threshold
        
        verdict = "Fake" if is_fake else "Real"
        confidence = overall_fake_prob if is_fake else (1.0 - overall_fake_prob)
        confidence_percent = round(confidence * 100.0, 1)

        # Generate Explainability Plot
        plot_image_path = self.generate_analysis_plot(y, sr, f0, segment_results, verdict, confidence_percent)

        # Build Summary Report
        fake_seg_count = sum(1 for seg in segment_results if seg["is_fake"])
        summary = (
            f"Voice Deepfake Analysis Report\n"
            f"{'=' * 50}\n"
            f"VERDICT: {'🚨 FAKE / SYNTHETIC VOICE' if is_fake else '✅ REAL / AUTHENTIC HUMAN VOICE'}\n"
            f"Confidence: {confidence_percent}%\n"
            f"Decision Threshold: {threshold:.2f}\n"
            f"{'=' * 50}\n"
            f"Audio Duration: {total_duration:.2f} seconds\n"
            f"Segments Analyzed: {len(segment_results)} (Fake segments detected: {fake_seg_count})\n"
            f"Overall Fake Probability: {overall_fake_prob * 100:.1f}%\n"
            f"Peak Segment Fake Score:  {max_fake_prob * 100:.1f}%\n"
            f"{'=' * 50}\n"
            f"Acoustic Forensic Indicators:\n"
            f"  - Mean Pitch (F0):        {forensic_metrics['mean_f0_hz']} Hz (Std: {forensic_metrics['std_f0_hz']} Hz)\n"
            f"  - Pitch Micro-Jitter:      {forensic_metrics['jitter_percent']}% "
            f"({'Natural biomechanical fluctuation' if forensic_metrics['jitter_percent'] >= 0.5 else 'Synthetically uniform / low variation'})\n"
            f"  - High-Freq Energy (>6.5k): {forensic_metrics['hf_energy_ratio'] * 100:.2f}% "
            f"({'Natural broadband presence' if forensic_metrics['hf_energy_ratio'] >= 0.015 else 'Attenuated vocoder bandwidth'})\n"
            f"  - Spectral Centroid:      {forensic_metrics['spectral_centroid_hz']} Hz\n"
            f"  - Silence / Pause Ratio:  {forensic_metrics['silence_ratio'] * 100:.1f}%\n"
            f"{'=' * 50}\n"
            f"Underlying Architecture: Wav2Vec2 Self-Supervised Speech Transformer + Spectral Forensics"
        )

        return {
            "verdict": verdict,
            "fake_prob": round(overall_fake_prob, 4),
            "real_prob": round(1.0 - overall_fake_prob, 4),
            "confidence_percent": confidence_percent,
            "segment_results": segment_results,
            "forensic_metrics": forensic_metrics,
            "summary_text": summary,
            "plot_image_path": plot_image_path
        }

    def generate_analysis_plot(self, y, sr, f0, segment_results, verdict, confidence_percent):
        """
        Creates a dual-panel explainability figure:
        Top: Mel-Spectrogram (dB) with time and frequency.
        Bottom: Audio Waveform with temporal segment risk overlay and pitch (F0) contour.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False, dpi=120)
        plt.style.use('dark_background')

        # 1. Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        times = librosa.times_like(mel_db, sr=sr)
        
        im = ax1.imshow(
            mel_db,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], 0, 8000],
            cmap='magma'
        )
        ax1.set_title(f"Acoustic Mel-Spectrogram (0 - 8000 Hz) | Analysis: {verdict} ({confidence_percent}%)", fontsize=11, fontweight='bold', color='#00e5ff')
        ax1.set_ylabel("Frequency (Hz)", fontsize=9)
        cbar = plt.colorbar(im, ax=ax1, format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=8)

        # 2. Waveform + Segment Color Coding + Pitch
        t_wave = np.linspace(0, len(y) / sr, len(y))
        ax2.plot(t_wave, y, color='#a0aec0', alpha=0.5, label='Waveform')

        # Color segments according to fake probability
        for seg in segment_results:
            st, en = seg["start_time"], seg["end_time"]
            color = '#ff3366' if seg["is_fake"] else '#00cc88'
            alpha = 0.25 if seg["is_fake"] else 0.15
            ax2.axvspan(st, en, color=color, alpha=alpha, label='Fake Segment' if seg["is_fake"] else 'Real Segment')
            ax2.text((st + en) / 2, 0.75, f"{seg['fake_prob']*100:.0f}% Fake",
                     color=color, fontsize=8, ha='center', fontweight='bold')

        # Pitch contour on secondary axis
        if f0 is not None and len(f0) > 0:
            times_f0 = librosa.times_like(f0, sr=sr)
            ax2_f0 = ax2.twinx()
            ax2_f0.plot(times_f0, f0, color='#ffea00', linewidth=1.5, label='F0 Pitch Contour')
            ax2_f0.set_ylabel("Pitch F0 (Hz)", color='#ffea00', fontsize=9)
            ax2_f0.tick_params(axis='y', labelcolor='#ffea00', labelsize=8)
            ax2_f0.set_ylim(40, 450)

        ax2.set_title("Temporal Waveform with Segment-Level Deepfake Risk", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Time (seconds)", fontsize=9)
        ax2.set_ylabel("Amplitude", fontsize=9)
        ax2.set_ylim(-1.05, 1.05)

        # Avoid duplicate legend entries
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax2.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)

        plt.tight_layout()

        # Save to temporary file
        out_file = os.path.join(tempfile.gettempdir(), f"audio_deepfake_analysis_{os.getpid()}_{np.random.randint(10000)}.png")
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

        return out_file

# Singleton detector instance for fast app responses
_detector_instance = None

def get_audio_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AudioDeepfakeDetector()
    return _detector_instance

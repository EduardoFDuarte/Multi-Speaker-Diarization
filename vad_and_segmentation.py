"""
Voice Activity Detection (VAD) and audio segmentation.
Enhanced with Silero VAD (neural network-based).
"""

import os
import numpy as np
import librosa
from scipy.io import wavfile
from collections import namedtuple
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import noisereduce as nr

Segment = namedtuple('Segment', ['start_time', 'end_time', 'audio_data'])

_silero_model = None

def get_silero_model():
    global _silero_model
    if _silero_model is None:
        print("  Loading Silero VAD model...")
        torch.set_num_threads(1)
        _silero_model = load_silero_vad()
        print("  ✓ Silero VAD model loaded")
    return _silero_model

def silero_vad_detect(
    audio_path,
    threshold=0.35,  
    min_speech_duration_ms=300,
    max_speech_duration_s=float('inf'),
    min_silence_duration_ms=150,
    window_size_samples=512,
    speech_pad_ms=100
):
    model = get_silero_model()
    wav = read_audio(audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
        visualize_probs=False
    )
    
    return speech_timestamps, 16000, wav

def vad_segment_audio(
    audio_path,
    frame_size_sec=3.0,
    hop_size_sec=1.5,
    use_silero=True,
    silero_threshold=0.35,
    save_segments=False,
    apply_noise_reduction=True
):
    print(f"Running VAD on: {audio_path}")
    print(f"  Frame size: {frame_size_sec}s, Hop size: {hop_size_sec}s")
    
    if use_silero:
        print("  Using Silero VAD (neural network-based)")
        return _vad_segment_with_silero(
            audio_path,
            frame_size_sec=frame_size_sec,
            hop_size_sec=hop_size_sec,
            threshold=silero_threshold,
            save_segments=save_segments,
            apply_noise_reduction=apply_noise_reduction
        )
    else:
        pass

def _vad_segment_with_silero(
    audio_path,
    frame_size_sec=3.0,
    hop_size_sec=1.5,
    threshold=0.35,
    save_segments=False,
    apply_noise_reduction=True
):
    sample_rate, audio_data = wavfile.read(audio_path)
    
    if sample_rate != 16000:
        print(f"  Resampling from {sample_rate}Hz to 16000Hz...")
        audio_float = audio_data.astype(float) / (np.max(np.abs(audio_data)) + 1e-8)
        audio_resampled = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
        temp_path = audio_path.replace('.wav', '_16k.wav')
        wavfile.write(temp_path, 16000, (audio_resampled * 32767).astype(np.int16))
        audio_path = temp_path
        sample_rate = 16000
        audio_data = (audio_resampled * 32767).astype(np.int16)
    
    if apply_noise_reduction:
        print("  Applying noise reduction...")
        audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.6)
    
    speech_timestamps, _, _ = silero_vad_detect(
        audio_path,
        threshold=threshold,
        min_speech_duration_ms=300,
        min_silence_duration_ms=150,
        speech_pad_ms=100
    )
    
    print(f"  Detected {len(speech_timestamps)} speech regions")
    
    segments = []
    timestamps = []
    segment_idx = 0
    
    frame_size_samples = int(frame_size_sec * sample_rate)
    hop_size_samples = int(hop_size_sec * sample_rate)
    
    for speech_region in speech_timestamps:
        region_start = speech_region['start']
        region_end = speech_region['end']
        region_start_sample = int(region_start * sample_rate)
        region_end_sample = int(region_end * sample_rate)
        
        start_sample = region_start_sample
        
        while start_sample < region_end_sample:
            end_sample = min(start_sample + frame_size_samples, region_end_sample)
            
            if (end_sample - start_sample) < frame_size_samples * 0.4:
                break
            
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < frame_size_samples:
                segment_audio = np.pad(segment_audio, (0, frame_size_samples - len(segment_audio)), mode='constant')
            
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segments.append(Segment(start_time, end_time, segment_audio))
            timestamps.append((start_time, end_time))
            
            if save_segments:
                os.makedirs("segments", exist_ok=True)
                segment_path = f"segments/seg_{segment_idx:03d}.wav"
                wavfile.write(segment_path, sample_rate, segment_audio)
            
            segment_idx += 1
            start_sample += hop_size_samples
    
    print(f"✓ Created {len(segments)} speech segments")
    if save_segments:
        print(f"  Saved segments to ./segments/ directory")

    print(f"\nTotal segments: {len(segments)}")
    print(f"Total audio duration: {timestamps[-1][1]:.2f}s")
    
    return segments, timestamps


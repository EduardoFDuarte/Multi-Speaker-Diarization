"""
Extract audio from video file using moviepy.
Outputs a mono WAV file at 16kHz for speech processing.
"""

import os
from moviepy import VideoFileClip
import numpy as np
from scipy.io import wavfile


def extract_audio_from_video(video_path, output_audio_path="audio.wav", target_sr=16000):
    """
    Extract audio from video and save as WAV file.
    
    Args:
        video_path: Path to input video file
        output_audio_path: Path to output audio file
        target_sr: Target sample rate (default: 16000 Hz)
    
    Returns:
        output_audio_path: Path to the extracted audio file
    """
    print(f"Extracting audio from: {video_path}")
    
    try:
        video = VideoFileClip(video_path)
        
        audio = video.audio
        
        if audio is None:
            raise ValueError("No audio track found in video file")
        
        # Write audio to temporary file
        temp_path = "temp_audio.wav"
        
        # Check MoviePy version and use appropriate parameters
        try:
            audio.write_audiofile(
                temp_path, 
                fps=target_sr, 
                nbytes=2, 
                codec='pcm_s16le'
            )
        except TypeError:
            audio.write_audiofile(
                temp_path, 
                fps=target_sr, 
                nbytes=2, 
                codec='pcm_s16le', 
                verbose=False, 
                logger=None
            )
        
        # Read and convert to mono if needed
        sr, audio_data = wavfile.read(temp_path)
        
        # Convert stereo to mono by averaging channels
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
        
        wavfile.write(output_audio_path, sr, audio_data)
        
        video.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"✓ Audio extracted successfully: {output_audio_path}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio_data) / sr:.2f} seconds")
        
        return output_audio_path
        
    except Exception as e:
        print(f"✗ Error extracting audio: {str(e)}")
        raise


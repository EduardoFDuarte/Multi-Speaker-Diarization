"""
Compatibility fix for torchaudio versions >= 2.1
This monkey-patches the missing functions that older SpeechBrain versions expect.
"""
import torchaudio

# Add missing functions for backward compatibility
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['sox', 'soundfile', 'ffmpeg']
    
    def get_audio_backend():
        return 'soundfile'
    
    torchaudio.list_audio_backends = list_audio_backends
    torchaudio.get_audio_backend = get_audio_backend

print("✓ Torchaudio compatibility patch applied")
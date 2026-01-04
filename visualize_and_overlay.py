"""
Video overlay with multi-speaker label support.
"""

import cv2
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
import os

def create_labeled_video_with_speakers_multi(video_path, speaker_windows, output_path):
    """
    Create video with multi-speaker labels overlaid.
    Supports multiple simultaneous speakers.
    """
    print(f"Creating labeled video: {output_path}")
    
    # Load video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Define colors for speakers
    colors = [
        (255, 100, 100),  # Blue
        (100, 255, 100),  # Green
        (100, 100, 255),  # Red
        (255, 255, 100),  # Cyan
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Yellow
        (200, 150, 100),  # Light blue
        (150, 200, 100),  # Light green
        (100, 150, 200),  # Orange
        (200, 100, 200),  # Purple
    ]
    
    frame_idx = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        # Find active speakers at this time
        active_speakers = []
        for start, end, speaker_ids in speaker_windows:
            if start <= current_time < end:
                active_speakers = speaker_ids
                break
        
        # Draw speaker labels
        if active_speakers:
            # Multi-speaker label
            if len(active_speakers) == 1:
                label = f"Speaker {active_speakers[0]}"
                color = colors[active_speakers[0] % len(colors)]
            else:
                # Multiple speakers
                speaker_str = ", ".join([str(s) for s in active_speakers])
                label = f"Speakers {speaker_str} (Overlap)"
                # Mix colors for overlapping speech
                color = tuple([
                    int(np.mean([colors[s % len(colors)][i] for s in active_speakers]))
                    for i in range(3)
                ])
            
            # Draw background rectangle
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.rectangle(frame, 
                         (10, height - 70), 
                         (20 + label_size[0], height - 20),
                         color, -1)
            
            # Draw text
            cv2.putText(frame, label, (15, height - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Add indicator for overlapping speech
            if len(active_speakers) > 1:
                cv2.putText(frame, "[OVERLAP]", (15, height - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")
    
    video.release()
    out.release()
    
    # Add audio back
    print("  Adding audio to video...")
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = AudioFileClip(video_path)
    final_clip = video_clip.with_audio(audio_clip)  
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
    
    # Cleanup
    video_clip.close()
    audio_clip.close()
    os.remove(temp_video_path)
    
    print(f"✓ Labeled video created: {output_path}")
    return output_path


# For backward compatibility with single-speaker code
def create_labeled_video_with_speakers(video_path, speaker_windows, output_path):
    """
    Wrapper for backward compatibility.
    Converts single-speaker format to multi-speaker format.
    """
    # Convert old format (start, end, speaker_id) to new format (start, end, [speaker_ids])
    multi_speaker_windows = []
    for start, end, speaker_id in speaker_windows:
        if isinstance(speaker_id, list):
            # Already multi-speaker format
            multi_speaker_windows.append((start, end, speaker_id))
        else:
            # Convert single speaker to list
            speaker_ids = [speaker_id] if speaker_id != -1 else []
            multi_speaker_windows.append((start, end, speaker_ids))
    
    return create_labeled_video_with_speakers_multi(video_path, multi_speaker_windows, output_path)

"""
Main orchestrator for Multi-Speaker Diarization Pipeline.
Supports detection of simultaneous speakers (overlapping speech).
"""

import os
import sys
import time

from extract_audio import extract_audio_from_video
from vad_and_segmentation import vad_segment_audio
from generate_embeddings import generate_embeddings_from_segments
from cluster_speakers import cluster_and_identify_speakers, save_diarization_results, analyze_speaker_distribution
from generate_metrics_visualizations import generate_all_metrics
from visualize_and_overlay import create_labeled_video_with_speakers_multi



def print_banner():
    """Print project banner."""
    print("=" * 70)
    print("  MULTI-SPEAKER DIARIZATION PIPELINE")
    print("  Overlapping Speech Detection + High Accuracy")
    print("=" * 70)
    print()


def check_input_file(video_path):
    """Check if input video file exists."""
    if not os.path.exists(video_path):
        print(f"Error: Input video not found: {video_path}")
        sys.exit(1)

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Input video found: {video_path}")
    print(f"  File size: {file_size:.2f} MB")


def run_pipeline(video_path="input_video_02.mp4",
                frame_size=3.0,
                hop_size=1.5,
                use_silero=True,
                save_segments=False,
                enable_multi_speaker=True,
                multi_speaker_threshold=0.35,
                generate_visualizations=True):
    """
    Run the complete multi-speaker diarization pipeline.

    Args:
        video_path: Path to input video
        frame_size: VAD frame size in seconds
        hop_size: VAD hop size in seconds
        use_silero: Use Silero VAD (True) or WebRTC (False)
        save_segments: Save individual audio segments
        enable_multi_speaker: Enable multi-speaker overlap detection
        multi_speaker_threshold: Threshold for overlap detection
        generate_visualizations: Generate all plots and metrics
    """
    start_time = time.time()
    print_banner()
    check_input_file(video_path)
    os.makedirs("output", exist_ok=True)

    output_files = {}

    try:
        # Step 1: Extract audio from video
        print("\n[1/6] Extracting audio from video...")
        print("-" * 70)
        audio_path = extract_audio_from_video(video_path, "audio.wav")
        output_files['audio'] = audio_path
        print()

        # Step 2: Running Voice Activity Detection
        print("[2/6] Running Voice Activity Detection...")
        print("-" * 70)
        print("  Optimized VAD parameters:")
        print("    • Threshold: 0.35")
        print("    • Min speech duration: 300ms")
        print("    • Min silence duration: 150ms")
        print("    • Speech padding: 100ms")

        segments, timestamps = vad_segment_audio(
            audio_path,
            frame_size_sec=frame_size,
            hop_size_sec=hop_size,
            use_silero=use_silero,
            silero_threshold=0.35,
            save_segments=save_segments
        )

        if len(segments) == 0:
            print("Error: No speech detected.")
            sys.exit(1)
        print()

        # Step 3: Generate speaker embeddings
        print("[3/6] Generating speaker embeddings...")
        print("-" * 70)
        embeddings = generate_embeddings_from_segments(segments)
        print()

        # Step 4: Perform multi-speaker diarization
        print("[4/6] Performing multi-speaker diarization...")
        print("-" * 70)
        print("  MULTI-SPEAKER DETECTION ENABLED")
        print(f"    • Single-stage HDBSCAN (min_cluster_size=10)")
        print(f"    • Overlapping speech detection")
        print(f"    • Multi-speaker threshold: {multi_speaker_threshold}")
        print(f"    • Two-pass noise elimination")
        print()

        speaker_timeline, speaker_windows, cluster_labels, total_speakers, multi_speaker_map = \
            cluster_and_identify_speakers(
                embeddings,
                timestamps,
                method='hdbscan',
                min_cluster_size=10,
                min_samples=2,
                smooth_window=3,
                enable_multi_speaker=enable_multi_speaker,
                multi_speaker_threshold=multi_speaker_threshold
            )

        # Step 5: Save results
        print("\n[5/6] Saving results...")
        print("-" * 70)
        timeline_path, windows_path = save_diarization_results(
            speaker_timeline,
            speaker_windows,
            output_dir="output"
        )
        output_files['timeline_csv'] = timeline_path
        output_files['windows_csv'] = windows_path

        print()
        distribution = analyze_speaker_distribution(cluster_labels, multi_speaker_map)
        for speaker, count in sorted(distribution.items()):
            print(f"  • {speaker}: {count} segments")
        print()

        
        if generate_visualizations:
            print("[5.5/6] Generating visualizations and presentation metrics...")
            print("-" * 70)
            generate_all_metrics(
                embeddings,
                cluster_labels,
                timestamps,
                speaker_timeline,
                output_dir="output"
            )

        # Step 6: Create labeled video (optional)
        if create_labeled_video_with_speakers_multi:
            print("\n[6/6] Creating labeled video with multi-speaker support...")
            print("-" * 70)
            labeled_video_path = create_labeled_video_with_speakers_multi(
                video_path,
                speaker_windows,
                output_path="output/labeled_video.mp4"
            )
            output_files['labeled_video'] = labeled_video_path
            print()

        elapsed_time = time.time() - start_time

        # Final summary
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Time: {elapsed_time:.2f}s | 👥 Speakers: {total_speakers}")
        print(f"Overlapping segments: {len(multi_speaker_map)}")
        print(f"\n Outputs in ./output/")
        print(f"   • speaker_timeline.csv - Multi-speaker segment timeline")
        print(f"   • speaker_windows.csv - Multi-speaker window timeline")
        print(f"   • speaker_diarization.png - UMAP clustering plot")
        print(f"   • speaker_timeline.png - Multi-speaker timeline (overlaps shown)")
        print(f"   • cluster_quality.png - Quality analysis")
        if generate_visualizations:
            print(f"   • speaker_dominance.png - Speaking time distribution")
            print(f"   • overlap_complexity.png - Overlap analysis")
            print(f"   • temporal_activity_heatmap.png - Activity over time")
            print(f"   • clustering_quality_metrics.png - Statistical metrics")
        if 'labeled_video' in output_files:
            print(f"   • labeled_video.mp4 - Video with multi-speaker labels")
        print("=" * 70)

        return output_files

    except Exception as e:
        print("=" * 70)
        print(f"ERROR: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    output_files = run_pipeline()
    return output_files


if __name__ == "__main__":
    main()
"""
Speaker diarization with multi-speaker support for overlapping speech detection.
Core clustering functionality using HDBSCAN or DBSCAN.
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN


def reduce_embedding_dimensions(embeddings, target_dim=64):
    """PCA-based dimensionality reduction before clustering."""
    pca = PCA(n_components=target_dim, whiten=True)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"  Reduced embeddings from {embeddings.shape[1]} to {target_dim} dimensions")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return normalize(reduced_embeddings, norm='l2')


def detect_multi_speaker_segments(embeddings_normalized, cluster_labels,
                                   multi_speaker_threshold=0.35):
    """
    Detect segments that likely contain multiple speakers.
    Returns a dict mapping segment index to list of speaker IDs.
    """
    print("  Detecting multi-speaker segments...")
    valid_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    if len(valid_clusters) < 2:
        return {}

    centroids = {}
    for cluster_id in valid_clusters:
        mask = cluster_labels == cluster_id
        centroids[cluster_id] = np.mean(embeddings_normalized[mask], axis=0)

    multi_speaker_segments = {}

    for i in range(len(cluster_labels)):
        primary_speaker = cluster_labels[i]
        if primary_speaker == -1:
            continue

        embedding = embeddings_normalized[i]
        
        distances = {}
        for cluster_id, centroid in centroids.items():
            distances[cluster_id] = cosine(embedding, centroid)

        sorted_speakers = sorted(distances.items(), key=lambda x: x[1])

        primary_dist = sorted_speakers[0][1]
        active_speakers = [sorted_speakers[0][0]]

        for speaker_id, dist in sorted_speakers[1:]:
            if dist < primary_dist + multi_speaker_threshold:
                active_speakers.append(speaker_id)

        if len(active_speakers) > 1:
            multi_speaker_segments[i] = active_speakers

    print(f"  Found {len(multi_speaker_segments)} segments with multiple speakers")
    return multi_speaker_segments


def cluster_and_identify_speakers(embeddings, timestamps,
                                   method='hdbscan',
                                   eps=0.18,
                                   min_samples=2,
                                   min_cluster_size=10,
                                   smooth_window=3,
                                   enable_multi_speaker=True,
                                   multi_speaker_threshold=0.35):
    """
    Multi-speaker clustering with overlapping speech detection.

    Returns:
        speaker_timeline: list of (start, end, speaker_ids) where speaker_ids is a list
        speaker_windows: list of (start, end, speaker_ids) for 1-second windows
        cluster_labels: primary cluster assignment for each segment
        total_speakers: total number of unique speakers detected
        multi_speaker_map: dict mapping segment index to list of active speakers
    """
    print(f"Clustering {len(embeddings)} embeddings for speaker identification...")

    embeddings_normalized = normalize(embeddings, norm='l2')

    if method == 'hdbscan':
        print(f"  Using HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            algorithm='best',
            allow_single_cluster=False,
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(embeddings_normalized)
    else:
        print(f"  Using DBSCAN: eps={eps}, min_samples={min_samples}")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clusterer.fit_predict(embeddings_normalized)

    initial_clusters = len(set(cluster_labels) - {-1})
    print(f"  Initial clustering: {initial_clusters} speakers detected")

    print("  Eliminating noise segments...")
    cluster_labels = _eliminate_all_noise(embeddings_normalized, cluster_labels)

    multi_speaker_map = {}
    if enable_multi_speaker:
        multi_speaker_map = detect_multi_speaker_segments(
            embeddings_normalized,
            cluster_labels,
            multi_speaker_threshold
        )

    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    total_speakers = len(unique_clusters)
    noise_count = np.sum(cluster_labels == -1)

    print(f"✓ Final result: {total_speakers} unique speakers detected")
    print(f"  Noise segments (unassigned): {noise_count}")
    if enable_multi_speaker:
        print(f"  Multi-speaker segments: {len(multi_speaker_map)}")

    speaker_timeline = []
    for i, (start, end) in enumerate(timestamps):
        if i in multi_speaker_map:
            speaker_ids = multi_speaker_map[i]
        else:
            speaker_ids = [int(cluster_labels[i])] if cluster_labels[i] != -1 else []
        speaker_timeline.append((start, end, speaker_ids))

    speaker_windows = _create_multi_speaker_windows(
        timestamps,
        cluster_labels,
        multi_speaker_map,
        smooth_window
    )

    _print_speaker_stats(cluster_labels, timestamps, multi_speaker_map)

    return speaker_timeline, speaker_windows, cluster_labels, total_speakers, multi_speaker_map


def _eliminate_all_noise(embeddings_normalized, cluster_labels):
    """
    Eliminate ALL noise points by assigning to nearest cluster.
    """
    noise_mask = cluster_labels == -1
    valid_clusters = set(cluster_labels) - {-1}

    if not valid_clusters or not np.any(noise_mask):
        return cluster_labels

    centroids = {}
    cluster_radii = {}

    for cluster_id in valid_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_emb = embeddings_normalized[cluster_mask]
        centroids[cluster_id] = np.mean(cluster_emb, axis=0)

        if len(cluster_emb) > 1:
            dists = [cosine(emb, centroids[cluster_id]) for emb in cluster_emb]
            cluster_radii[cluster_id] = np.percentile(dists, 95)
        else:
            cluster_radii[cluster_id] = 0.4

    for i in np.where(noise_mask)[0]:
        noise_embedding = embeddings_normalized[i]
        distances = {}

        for cluster_id, centroid in centroids.items():
            dist = cosine(noise_embedding, centroid)
            distances[cluster_id] = dist

        nearest_cluster = min(distances.keys(), key=lambda c: distances[c])
        nearest_distance = distances[nearest_cluster]
        threshold = 2.0 * cluster_radii[nearest_cluster]

        if nearest_distance <= threshold:
            cluster_labels[i] = nearest_cluster

    noise_mask = cluster_labels == -1
    for i in np.where(noise_mask)[0]:
        noise_embedding = embeddings_normalized[i]
        min_dist = float('inf')
        nearest_cluster = -1

        for cluster_id, centroid in centroids.items():
            dist = cosine(noise_embedding, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = cluster_id

        if nearest_cluster != -1:
            cluster_labels[i] = nearest_cluster

    return cluster_labels


def _create_multi_speaker_windows(timestamps, cluster_labels, multi_speaker_map, smooth_window):
    """
    Create 1-second windows with multi-speaker support.
    Each window can contain multiple active speakers.
    """
    if not timestamps:
        return []

    max_time = max([end for _, end in timestamps])
    windows = []
    current_time = 0

    while current_time < max_time:
        window_end = current_time + 1.0

        speaker_durations = {}

        for i, (start, end) in enumerate(timestamps):
            if start < window_end and end > current_time:
                overlap_duration = min(end, window_end) - max(start, current_time)

                if i in multi_speaker_map:
                    speakers = multi_speaker_map[i]
                else:
                    speakers = [cluster_labels[i]]

                for speaker_id in speakers:
                    if speaker_id not in speaker_durations:
                        speaker_durations[speaker_id] = 0
                    speaker_durations[speaker_id] += overlap_duration

        min_duration = 0.2
        active_speakers = [
            speaker_id for speaker_id, duration in speaker_durations.items()
            if duration >= min_duration and speaker_id != -1
        ]

        active_speakers = sorted(
            active_speakers,
            key=lambda s: speaker_durations[s],
            reverse=True
        )

        windows.append((current_time, window_end, active_speakers if active_speakers else []))
        current_time = window_end

    if smooth_window > 1 and len(windows) >= smooth_window:
        smoothed_windows = []
        half_window = smooth_window // 2

        for i in range(len(windows)):
            start, end, _ = windows[i]

            window_start = max(0, i - half_window)
            window_end = min(len(windows), i + half_window + 1)

            speaker_counts = {}
            for j in range(window_start, window_end):
                for speaker_id in windows[j][2]:
                    speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

            threshold = (window_end - window_start) * 0.4
            smoothed_speakers = [
                speaker_id for speaker_id, count in speaker_counts.items()
                if count >= threshold
            ]

            smoothed_windows.append((start, end, smoothed_speakers))

        windows = smoothed_windows

    merged_windows = []
    if windows:
        current_start, _, current_speakers = windows[0]

        for i in range(1, len(windows)):
            start, end, speakers = windows[i]

            if set(speakers) == set(current_speakers):
                continue 
            else:
                merged_windows.append((current_start, start, current_speakers))
                current_start = start
                current_speakers = speakers

        merged_windows.append((current_start, windows[-1][1], current_speakers))

    print(f"  Created {len(merged_windows)} multi-speaker segments from {len(windows)} windows")
    return merged_windows


def _print_speaker_stats(cluster_labels, timestamps, multi_speaker_map):
    """Print statistics for each detected speaker including overlap."""
    unique_speakers = sorted([c for c in set(cluster_labels) if c != -1])

    print("\n" + "="*70)
    print("  SPEAKER STATISTICS")
    print("="*70)

    for speaker_id in unique_speakers:
        solo_time = 0
        overlap_time = 0
        total_segments = 0

        for i, (start, end) in enumerate(timestamps):
            if i in multi_speaker_map:
                if speaker_id in multi_speaker_map[i]:
                    overlap_time += (end - start)
                    total_segments += 1
            elif cluster_labels[i] == speaker_id:
                solo_time += (end - start)
                total_segments += 1

        total_time = solo_time + overlap_time
        print(f"  Speaker {speaker_id}: {total_segments} segments, {total_time:.1f}s total")
        print(f"    Solo: {solo_time:.1f}s, Overlapping: {overlap_time:.1f}s")

    overlap_count = len(multi_speaker_map)
    if overlap_count > 0:
        overlap_duration = sum([
            timestamps[i][1] - timestamps[i][0]
            for i in multi_speaker_map.keys()
        ])
        print(f"\n  Overlapping speech: {overlap_count} segments, {overlap_duration:.1f}s total")

    noise_count = np.sum(cluster_labels == -1)
    if noise_count > 0:
        noise_time = sum([end - start for i, (start, end) in enumerate(timestamps)
                         if cluster_labels[i] == -1])
        print(f"  Unassigned: {noise_count} segments, {noise_time:.1f}s")

    print("="*70 + "\n")


def save_diarization_results(speaker_timeline, speaker_windows, output_dir="output"):
    """Save multi-speaker diarization results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    timeline_data = []
    for start, end, speaker_ids in speaker_timeline:
        speaker_str = ','.join(map(str, speaker_ids)) if speaker_ids else ''
        num_speakers = len(speaker_ids)
        timeline_data.append({
            'start_time': start,
            'end_time': end,
            'speaker_ids': speaker_str,
            'num_speakers': num_speakers
        })

    df_timeline = pd.DataFrame(timeline_data)
    timeline_path = os.path.join(output_dir, "speaker_timeline.csv")
    df_timeline.to_csv(timeline_path, index=False)
    print(f"✓ Speaker timeline saved to: {timeline_path}")

    windows_data = []
    for start, end, speaker_ids in speaker_windows:
        speaker_str = ','.join(map(str, speaker_ids)) if speaker_ids else ''
        num_speakers = len(speaker_ids)
        windows_data.append({
            'start_time': start,
            'end_time': end,
            'speaker_ids': speaker_str,
            'num_speakers': num_speakers
        })

    df_windows = pd.DataFrame(windows_data)
    windows_path = os.path.join(output_dir, "speaker_windows.csv")
    df_windows.to_csv(windows_path, index=False)
    print(f"✓ Speaker windows saved to: {windows_path}")

    return timeline_path, windows_path


def analyze_speaker_distribution(cluster_labels, multi_speaker_map=None):
    """Analyze speaker distribution including overlapping speech."""
    unique, counts = np.unique(cluster_labels, return_counts=True)
    distribution = {}

    for label, count in zip(unique, counts):
        key = 'unassigned' if label == -1 else f'speaker_{label}'
        distribution[key] = int(count)

    if multi_speaker_map:
        distribution['overlapping_segments'] = len(multi_speaker_map)

    return distribution
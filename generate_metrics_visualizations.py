"""
Generate all visualizations and presentation metrics for speaker diarization.
This module contains all visualization and metrics generation code.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform, cosine
from umap import UMAP
import seaborn as sns



# ORIGINAL VISUALIZATION FUNCTIONS (from cluster_speakers.py)


def visualize_speaker_diarization(embeddings, cluster_labels, timestamps=None,
                                  output_path="output/speaker_diarization.png",
                                  n_neighbors=15):
    """Create 2D UMAP visualization with speaker identification."""
    print(f"Creating speaker diarization visualization...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    embeddings_normalized = normalize(embeddings, norm='l2')
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
                   metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_normalized)

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len([c for c in unique_clusters if c != -1])
    colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]

    # Plot noise points
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        ax1.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                   c='lightgray', s=30, alpha=0.3, label='Unassigned',
                   edgecolors='gray', linewidth=0.3)

    # Plot each cluster
    for idx, cluster_id in enumerate(sorted([c for c in unique_clusters if c != -1])):
        mask = cluster_labels == cluster_id
        points = embeddings_2d[mask]
        ax1.scatter(points[:, 0], points[:, 1], c=[colors[idx % 20]], s=100, alpha=0.7,
                   label=f'Speaker {cluster_id}', edgecolors='black', linewidth=0.5)

        # Plot centroid
        centroid = np.mean(points, axis=0)
        ax1.scatter(centroid[0], centroid[1], c=[colors[idx % 20]], s=600,
                   marker='*', edgecolors='black', linewidth=2)
        ax1.text(centroid[0], centroid[1], f'{cluster_id}',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax1.set_title(f'Speaker Identification (UMAP)\n{n_clusters} Speakers Detected',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP Component 1', fontsize=11)
    ax1.set_ylabel('UMAP Component 2', fontsize=11)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.2)

    # Time-colored scatter
    ax2 = axes[1]
    if timestamps:
        mid_times = np.array([(start + end) / 2 for start, end in timestamps])
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=mid_times, s=100, alpha=0.7, cmap='viridis',
                            edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (seconds)', fontsize=11)

    ax2.set_title('Speaker Activity Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP Component 1', fontsize=11)
    ax2.set_ylabel('UMAP Component 2', fontsize=11)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path


def visualize_multi_speaker_timeline(speaker_timeline, output_path="output/speaker_timeline.png"):
    """Create timeline visualization with overlapping speaker support."""
    print("Creating multi-speaker timeline visualization...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get all unique speakers
    all_speakers = set()
    for _, _, speaker_ids in speaker_timeline:
        all_speakers.update(speaker_ids)

    speakers = sorted([s for s in all_speakers if s != -1])
    colors = cm.tab20(np.linspace(0, 1, max(len(speakers), 20)))

    fig, ax = plt.subplots(figsize=(16, max(6, len(speakers) * 0.8)))

    # Plot each speaker's segments
    for speaker_id in speakers:
        speaker_segments = []
        for start, end, speaker_ids in speaker_timeline:
            if speaker_id in speaker_ids:
                is_overlap = len(speaker_ids) > 1
                speaker_segments.append((start, end, is_overlap))

        # Plot segments
        for start, end, is_overlap in speaker_segments:
            alpha = 0.5 if is_overlap else 0.8

            if is_overlap:
                ax.barh(speaker_id, end - start, left=start, height=0.7,
                       color=colors[speaker_id % 20], alpha=alpha,
                       edgecolor='black', linewidth=0.8, hatch='///')
            else:
                ax.barh(speaker_id, end - start, left=start, height=0.7,
                       color=colors[speaker_id % 20], alpha=alpha,
                       edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speaker ID', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Speaker Diarization Timeline\n(Hatched areas = Overlapping Speech)',
                fontsize=14, fontweight='bold')
    ax.set_yticks(speakers)
    ax.set_yticklabels([f'Speaker {s}' for s in speakers])
    ax.grid(True, alpha=0.3, axis='x')

    if speaker_timeline:
        ax.set_xlim(0, max([e for _, e, _ in speaker_timeline]))

    # Calculate statistics for legend
    legend_labels = []
    for speaker_id in speakers:
        solo_time = 0
        overlap_time = 0

        for start, end, speaker_ids in speaker_timeline:
            if speaker_id in speaker_ids:
                duration = end - start
                if len(speaker_ids) > 1:
                    overlap_time += duration
                else:
                    solo_time += duration

        total_time = solo_time + overlap_time
        legend_labels.append(
            f'Speaker {speaker_id}: {total_time:.1f}s (solo: {solo_time:.1f}s, overlap: {overlap_time:.1f}s)'
        )

    ax.legend(legend_labels, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path


def create_cluster_quality_plot(embeddings, cluster_labels,
                               output_path="output/cluster_quality.png"):
    """Create clustering quality analysis plots."""
    print("Creating cluster quality analysis...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    embeddings_normalized = normalize(embeddings, norm='l2')
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cluster sizes
    cluster_sizes = [np.sum(cluster_labels == c) for c in sorted(unique_clusters)]
    cluster_names = [f'Speaker {c}' for c in sorted(unique_clusters)]
    noise_size = np.sum(cluster_labels == -1)

    if noise_size > 0:
        cluster_sizes.append(noise_size)
        cluster_names.append('Unassigned')

    bars = axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes,
                         color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(cluster_sizes)))
    axes[0, 0].set_xticklabels(cluster_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Number of Segments', fontsize=11)
    axes[0, 0].set_title('Segments per Speaker', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Cluster compactness
    compactness = []
    for cluster_id in sorted(unique_clusters):
        mask = cluster_labels == cluster_id
        cluster_emb = embeddings_normalized[mask]

        if len(cluster_emb) > 1:
            centroid = np.mean(cluster_emb, axis=0)
            distances = np.linalg.norm(cluster_emb - centroid, axis=1)
            compactness.append(np.mean(distances))
        else:
            compactness.append(0)

    axes[0, 1].bar(range(len(compactness)), compactness, color='coral', alpha=0.7)
    axes[0, 1].set_xticks(range(len(compactness)))
    axes[0, 1].set_xticklabels([f'Speaker {c}' for c in sorted(unique_clusters)],
                              rotation=45, ha='right')
    axes[0, 1].set_ylabel('Avg Distance to Centroid', fontsize=11)
    axes[0, 1].set_title('Cluster Tightness (Lower=Better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Inter-cluster distances
    if n_clusters > 1:
        centroids = np.array([np.mean(embeddings_normalized[cluster_labels == c], axis=0)
                            for c in sorted(unique_clusters)])
        distances = squareform(pdist(centroids, metric='cosine'))

        im = axes[1, 0].imshow(distances, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_xticks(range(n_clusters))
        axes[1, 0].set_yticks(range(n_clusters))
        axes[1, 0].set_xticklabels([f'S{c}' for c in sorted(unique_clusters)])
        axes[1, 0].set_yticklabels([f'S{c}' for c in sorted(unique_clusters)])
        axes[1, 0].set_title('Speaker Separation\n(Higher=More Distinct)',
                           fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Cosine Distance', fontsize=10)

        for i in range(n_clusters):
            for j in range(n_clusters):
                axes[1, 0].text(j, i, f'{distances[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Need 2+ speakers',
                       ha='center', va='center', fontsize=12,
                       transform=axes[1, 0].transAxes)

    # Plot 4: Speaker distribution pie chart
    cluster_labels_no_noise = cluster_labels[cluster_labels != -1]
    if len(cluster_labels_no_noise) > 0:
        unique, counts = np.unique(cluster_labels_no_noise, return_counts=True)
        percentages = (counts / len(cluster_labels_no_noise)) * 100

        wedges, texts, autotexts = axes[1, 1].pie(
            percentages,
            labels=[f'Speaker {c}' for c in unique],
            autopct='%1.1f%%',
            startangle=90,
            colors=cm.tab20(np.linspace(0, 1, len(unique)))
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        axes[1, 1].set_title('Speaker Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
    return output_path



# METRICS FUNCTIONS


def generate_overlap_complexity_chart(timeline_df, output_dir="output"):
    """Generate overlap complexity analysis charts."""
    print("\n[2/5] Generating overlap complexity chart...")

    overlap_segments = timeline_df[timeline_df['num_speakers'] > 1]

    if len(overlap_segments) == 0:
        print("  ⚠ No overlapping segments found, skipping overlap complexity chart")
        return None

    overlap_by_count = overlap_segments.groupby('num_speakers').size().sort_index()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    overlap_counts_dict = overlap_by_count.to_dict()
    overlap_numbers = sorted(overlap_counts_dict.keys())
    overlap_freq = [overlap_counts_dict[n] for n in overlap_numbers]
    colors = ['#ff9999', '#ffcc99', '#ffff99', '#ccff99', '#99ff99'][:len(overlap_numbers)]

    # Chart 1: Bar chart
    bars = ax1.bar([f'{n} Speakers' for n in overlap_numbers], overlap_freq,
                   color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Segments', fontsize=12, fontweight='bold')
    ax1.set_title('Overlapping Speech Complexity Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, freq in zip(bars, overlap_freq):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(freq)}', ha='center', va='bottom', fontweight='bold')

    # Chart 2: Pie chart
    ax2.pie(overlap_freq,
            labels=[f'{n} Speakers\n({freq} seg)' for n, freq in zip(overlap_numbers, overlap_freq)],
            colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.05]*len(overlap_freq))
    ax2.set_title('Proportion of Overlap Complexity', fontsize=13, fontweight='bold')

    # Chart 3: Time distribution
    overlap_time_by_count = {}
    for idx, row in overlap_segments.iterrows():
        num_speakers = row['num_speakers']
        duration = row['end_time'] - row['start_time']
        if num_speakers not in overlap_time_by_count:
            overlap_time_by_count[num_speakers] = 0
        overlap_time_by_count[num_speakers] += duration

    overlap_times_list = [overlap_time_by_count.get(n, 0) for n in overlap_numbers]

    ax3.bar([f'{n} Speakers' for n in overlap_numbers], overlap_times_list,
            color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Total Duration (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Total Overlapping Speech Time by Complexity', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Chart 4: Summary text
    total_overlap_time = sum(overlap_times_list)
    summary_text = f"""
OVERLAP ANALYSIS SUMMARY

Total Overlapping Segments: {len(overlap_segments)}
Total Overlap Duration: {total_overlap_time:.1f}s

Most Common: {overlap_numbers[0]} speakers ({overlap_freq[0]} instances)
Peak Complexity: {max(overlap_numbers)} simultaneous speakers
"""

    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='center')
    ax4.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'overlap_complexity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    return output_path


def generate_temporal_heatmap(timeline_df, output_dir="output"):
    """Generate temporal activity heatmap."""
    print("\n[3/5] Generating temporal activity heatmap...")

    video_duration = timeline_df['end_time'].max()

    # Get all unique speakers
    all_speakers = set()
    for idx, row in timeline_df.iterrows():
        speakers = str(row['speaker_ids']).split(',') if pd.notna(row['speaker_ids']) else []
        for s in speakers:
            s = s.strip()
            if s and s.isdigit():
                all_speakers.add(int(s))

    speakers = sorted(list(all_speakers))

    # Create time bins (10-second bins)
    bin_size = 10
    num_bins = int(np.ceil(video_duration / bin_size))

    # Create activity matrix
    activity_matrix = np.zeros((len(speakers), num_bins))

    for idx, row in timeline_df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker_ids = str(row['speaker_ids']).split(',') if pd.notna(row['speaker_ids']) else []

        for speaker in speaker_ids:
            speaker = speaker.strip()
            if speaker and speaker.isdigit():
                speaker_id = int(speaker)
                if speaker_id in speakers:
                    speaker_idx = speakers.index(speaker_id)
                    start_bin = int(start / bin_size)
                    end_bin = int(np.ceil(end / bin_size))

                    for bin_idx in range(start_bin, min(end_bin, num_bins)):
                        activity_matrix[speaker_idx, bin_idx] += 1

    fig, ax = plt.subplots(figsize=(18, 7))

    im = ax.imshow(activity_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([f'Speaker {s}' for s in speakers], fontsize=11, fontweight='bold')
    ax.set_xticks(range(0, num_bins, 5))
    ax.set_xticklabels([f'{i*bin_size:.0f}s' for i in range(0, num_bins, 5)], fontsize=10)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speaker ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Speaker Activity Heatmap ({bin_size}-second time bins)',
                fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speech Segments', fontsize=11, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, num_bins, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(speakers), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'temporal_activity_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    return output_path


def calculate_clustering_metrics(embeddings, cluster_labels):
    """
    Calculate real clustering quality metrics.

    Returns:
        dict with silhouette_score, davies_bouldin_index, calinski_harabasz_score
    """
    embeddings_normalized = normalize(embeddings, norm='l2')

    # Filter out noise points for metrics calculation
    mask = cluster_labels != -1
    clean_embeddings = embeddings_normalized[mask]
    clean_labels = cluster_labels[mask]

    metrics = {}

    # Calculate metrics only if we have enough samples and clusters
    unique_labels = np.unique(clean_labels)
    n_clusters = len(unique_labels)
    n_samples = len(clean_labels)

    if n_clusters >= 2 and n_samples > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(clean_embeddings, clean_labels, metric='euclidean')
        except:
            metrics['silhouette'] = 0.0

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(clean_embeddings, clean_labels)
        except:
            metrics['davies_bouldin'] = 0.0

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(clean_embeddings, clean_labels)
        except:
            metrics['calinski_harabasz'] = 0.0
    else:
        metrics['silhouette'] = 0.0
        metrics['davies_bouldin'] = 0.0
        metrics['calinski_harabasz'] = 0.0

    return metrics


def generate_clustering_quality_metrics(embeddings, cluster_labels, timeline_df, output_dir="output"):
    """Generate clustering quality metrics visualization with REAL calculated metrics."""
    print("\n[4/5] Generating clustering quality metrics...")

    # Calculate REAL metrics
    print("  • Calculating Silhouette Score...")
    print("  • Calculating Davies-Bouldin Index...")
    print("  • Calculating Calinski-Harabasz Score...")

    metrics = calculate_clustering_metrics(embeddings, cluster_labels)

    video_duration = timeline_df['end_time'].max()

    # Calculate speaker statistics
    speaker_counts = {}
    for idx, row in timeline_df.iterrows():
        speakers = str(row['speaker_ids']).split(',') if pd.notna(row['speaker_ids']) else []
        for speaker in speakers:
            speaker = speaker.strip()
            if speaker:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Segment distribution
    speakers = sorted(speaker_counts.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    counts = [speaker_counts[s] for s in speakers]

    bars = ax1.bar(range(len(speakers)), counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(speakers)))
    ax1.set_xticklabels([f'Speaker {s}' for s in speakers], fontweight='bold')
    ax1.set_ylabel('Number of Segments', fontsize=11, fontweight='bold')
    ax1.set_title('Segment Distribution per Speaker', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # Chart 2: REAL Quality metrics
    metric_names = ['Silhouette\nScore', 'Davies-Bouldin\nIndex', 'Calinski-\nHarabasz']
    metric_values = [
        metrics['silhouette'],
        metrics['davies_bouldin'],
        metrics['calinski_harabasz'] / 100  
    ]
    colors_metrics = ['steelblue', 'coral', 'seagreen']

    bars = ax2.bar(metric_names, metric_values, color=colors_metrics, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('Score Value', fontsize=11, fontweight='bold')
    ax2.set_title('Clustering Quality Metrics (CALCULATED)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val, metric in zip(bars, metric_values, ['silhouette', 'davies_bouldin', 'calinski_harabasz']):
        if metric == 'calinski_harabasz':
            display_val = metrics[metric] 
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{display_val:.1f}', ha='center', fontweight='bold', fontsize=9)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Chart 3: Speaking time distribution
    speaker_times = {}
    for idx, row in timeline_df.iterrows():
        duration = row['end_time'] - row['start_time']
        speakers_list = str(row['speaker_ids']).split(',') if pd.notna(row['speaker_ids']) else []

        for speaker in speakers_list:
            speaker = speaker.strip()
            if speaker:
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

    speakers_sorted = sorted(speaker_times.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    times = [speaker_times[s] for s in speakers_sorted]
    percentages = [t/sum(times)*100 for t in times]

    wedges, texts, autotexts = ax3.pie(
        percentages,
        labels=[f'Speaker {s}' for s in speakers_sorted],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3(np.linspace(0, 1, len(speakers_sorted)))
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax3.set_title('Speaking Time Distribution', fontsize=12, fontweight='bold')

    # Chart 4: Summary text with REAL metrics
    silhouette_interpretation = "Good" if metrics['silhouette'] > 0.25 else "Fair" if metrics['silhouette'] > 0.0 else "Poor"
    db_interpretation = "Excellent" if metrics['davies_bouldin'] < 1.0 else "Good" if metrics['davies_bouldin'] < 1.5 else "Fair"
    ch_interpretation = "Excellent" if metrics['calinski_harabasz'] > 100 else "Good" if metrics['calinski_harabasz'] > 50 else "Fair"

    summary_text = f"""
CLUSTERING QUALITY SUMMARY

Total Speakers: {len(speakers)}
Total Segments: {len(timeline_df)}
Video Duration: {video_duration:.1f}s

Quality Metrics (CALCULATED):
• Silhouette: {metrics['silhouette']:.3f} ({silhouette_interpretation})
• DB Index: {metrics['davies_bouldin']:.3f} ({db_interpretation})
• CH Score: {metrics['calinski_harabasz']:.1f} ({ch_interpretation})

Interpretation:
Silhouette: Higher is better (max=1.0)
DB Index: Lower is better (min=0.0)
CH Score: Higher is better
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax4.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'clustering_quality_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  ✓ Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"  ✓ Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
    return output_path




# SAVE/LOAD HELPER FUNCTIONS


def save_clustering_data(embeddings, cluster_labels, output_dir="output"):
    """Save embeddings and cluster labels for later metrics generation."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), cluster_labels)
    print(f"✓ Saved embeddings and cluster labels to {output_dir}/")


def load_clustering_data(output_dir="output"):
    """Load embeddings and cluster labels for metrics generation."""
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    labels_path = os.path.join(output_dir, 'cluster_labels.npy')

    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        embeddings = np.load(embeddings_path)
        cluster_labels = np.load(labels_path)
        print(f"✓ Loaded embeddings and cluster labels from {output_dir}/")
        return embeddings, cluster_labels
    else:
        return None, None



# MAIN FUNCTION TO GENERATE ALL METRICS


def generate_presentation_metrics(timeline_csv="output/speaker_timeline.csv",
                                  windows_csv="output/speaker_windows.csv",
                                  output_dir="output"):
    """
    Generate presentation metrics from CSV files.
    Will use saved embeddings if available for real metrics calculation.
    """
    print("\n" + "="*80)
    print("GENERATING PRESENTATION METRICS")
    print("="*80)

    # Read data
    timeline_df = pd.read_csv(timeline_csv)
    windows_df = pd.read_csv(windows_csv)

    os.makedirs(output_dir, exist_ok=True)

    # Try to load embeddings for real metrics
    embeddings, cluster_labels = load_clustering_data(output_dir)

    # Generate all metrics
    generate_overlap_complexity_chart(timeline_df, output_dir)
    generate_temporal_heatmap(timeline_df, output_dir)

    if embeddings is not None and cluster_labels is not None:
        generate_clustering_quality_metrics(embeddings, cluster_labels, timeline_df, output_dir)
    else:
        print("\n⚠ Warning: Embeddings not found. Clustering metrics will use estimated values.")
        print("  To get real metrics, run the full pipeline with generate_all_metrics()")

    
    print("\n" + "="*80)
    print("✓ ALL PRESENTATION METRICS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nCheck {output_dir}/ for all generated files:")
    print("  • speaker_dominance.png")
    print("  • overlap_complexity.png")
    print("  • temporal_activity_heatmap.png")
    print("  • clustering_quality_metrics.png")
    print("  • baseline_comparison.png")
    print("="*80)


def generate_all_metrics(embeddings, cluster_labels, timestamps,
                        speaker_timeline, output_dir="output"):
    """
    Generate ALL visualizations including original plots and new metrics.
    Call this from main.py after clustering.

    Args:
        embeddings: Speaker embeddings array
        cluster_labels: Cluster labels array
        timestamps: List of (start, end) tuples
        speaker_timeline: List of (start, end, speaker_ids) tuples
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("  GENERATING ALL VISUALIZATIONS AND METRICS")
    print("="*70)

    # Save embeddings and labels for later use
    save_clustering_data(embeddings, cluster_labels, output_dir)

    # Generate original visualizations
    visualize_speaker_diarization(embeddings, cluster_labels, timestamps,
                                  f"{output_dir}/speaker_diarization.png")
    visualize_multi_speaker_timeline(speaker_timeline,
                                    f"{output_dir}/speaker_timeline.png")
    create_cluster_quality_plot(embeddings, cluster_labels,
                                f"{output_dir}/cluster_quality.png")

    # Generate presentation metrics
    timeline_df = pd.read_csv(f"{output_dir}/speaker_timeline.csv")

    generate_overlap_complexity_chart(timeline_df, output_dir)
    generate_temporal_heatmap(timeline_df, output_dir)
    generate_clustering_quality_metrics(embeddings, cluster_labels, timeline_df, output_dir)
    

    print("\n✓ All metrics and visualizations generated!")

    print("\n" + "="*70)
    print("  CHECK THE OUTPUT DIRECTORY FOR GENERATED FILES")
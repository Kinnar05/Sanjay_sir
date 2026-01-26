import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import mne
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import glob
import pandas as pd
import seaborn as sns
from datetime import datetime
import time

# Configuration
SAMPLING_RATE = 256
WINDOW_SIZE = 5
OVERLAP = 0.5
N_CHANNELS = 19

def timestamp():
    """Return formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_time(message):
    """Print message with timestamp"""
    print(f"[{timestamp()}] {message}")

def load_edf_files(directory_path):
    """Load EDF files and classify them"""
    log_time("Loading EDF files...")
    mdd_files = []
    healthy_files = []
    
    all_files = glob.glob(os.path.join(directory_path, "*.edf"))
    
    for file in all_files:
        filename = os.path.basename(file)
        if filename.startswith('MDD'):
            mdd_files.append(file)
        elif filename.startswith('H '):
            healthy_files.append(file)
    
    log_time(f"Found {len(mdd_files)} MDD files and {len(healthy_files)} Healthy files")
    return mdd_files, healthy_files

def preprocess_eeg(raw_data, fs=256):
    """Preprocess EEG data"""
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 70.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    
    filtered_data = np.zeros_like(raw_data)
    for ch in range(raw_data.shape[0]):
        filtered_data[ch] = filtfilt(b, a, raw_data[ch])
    
    b_notch, a_notch = butter(4, [49/nyquist, 51/nyquist], btype='bandstop')
    for ch in range(filtered_data.shape[0]):
        filtered_data[ch] = filtfilt(b_notch, a_notch, filtered_data[ch])
    
    return filtered_data

def segment_data(data, window_size=4, overlap=0.25, fs=256):
    """Segment data into windows"""
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    segments = []
    n_samples = data.shape[1]
    
    start = 0
    while start + window_samples <= n_samples:
        segment = data[:, start:start + window_samples]
        segments.append(segment)
        start += step_samples
    
    return np.array(segments)

def compute_plv(signal1, signal2):
    """Compute Phase Locking Value"""
    analytic1 = signal.hilbert(signal1)
    analytic2 = signal.hilbert(signal2)
    
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    phase_diff = phase1 - phase2
    
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def compute_coherence(signal1, signal2, fs=256):
    """Compute coherence"""
    f, Cxy = signal.coherence(signal1, signal2, fs=fs, nperseg=256)
    return f, Cxy

def compute_connectivity_matrix(segment, fs=256):
    """Compute functional connectivity matrix"""
    n_channels = segment.shape[0]
    connectivity_matrix = np.zeros((n_channels, n_channels, 225))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            plv = compute_plv(segment[i], segment[j])
            f, coh = compute_coherence(segment[i], segment[j], fs)
            
            freq_mask = (f >= 0.5) & (f <= 70.0)
            coh_selected = coh[freq_mask]
            
            if len(coh_selected) != 225:
                from scipy.interpolate import interp1d
                f_interp = interp1d(np.linspace(0, 1, len(coh_selected)), coh_selected)
                coh_selected = f_interp(np.linspace(0, 1, 225))
            
            combined = (plv + coh_selected) / 2
            combined = (np.exp(combined) - 1) / (2 * np.e - 2)
            
            connectivity_matrix[i, j, :] = combined
            connectivity_matrix[j, i, :] = combined
    
    return connectivity_matrix

def extract_band_features(connectivity_matrix, band_name):
    """Extract features for specific frequency band"""
    freq_ranges = {
        'delta': (0, 35),
        'theta': (35, 75),
        'alpha': (75, 108),
        'beta': (108, 166),
        'gamma': (166, 225)
    }
    
    start_idx, end_idx = freq_ranges[band_name]
    return connectivity_matrix[:, :, start_idx:end_idx]

def calculate_metrics(y_test, y_pred):
    """Calculate accuracy, sensitivity, specificity"""
    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))
    
    sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    
    return accuracy, sensitivity, specificity

def train_ml_classifier(X_train, y_train, X_test, y_test, classifier_name):
    """Train traditional ML classifier"""
    start_time = time.time()
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    clf = classifiers[classifier_name]
    print(f"    [{timestamp()}] Training {classifier_name}...", end=' ')
    clf.fit(X_train_flat, y_train)
    y_pred = clf.predict(X_test_flat)
    
    acc, sens, spec = calculate_metrics(y_test, y_pred)
    elapsed = time.time() - start_time
    print(f"Done! Acc={acc:.2f}% (Time: {elapsed:.2f}s)")
    return acc, sens, spec

def build_band_cnn(input_shape):
    """Build CNN for specific band - Fixed Inception module"""
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: 1x1 convolution
    conv1x1 = layers.Conv2D(32, (1, 1), padding='same', activation='tanh')(inputs)
    
    # Branch 2: 3x3 convolution
    conv3x3_reduce = layers.Conv2D(32, (1, 1), padding='same', activation='tanh')(inputs)
    conv3x3 = layers.Conv2D(64, (3, 3), padding='same', activation='tanh')(conv3x3_reduce)
    
    # Branch 3: Two 3x3 convolutions (replacing 5x5)
    conv5x5_reduce = layers.Conv2D(16, (1, 1), padding='same', activation='tanh')(inputs)
    conv5x5 = layers.Conv2D(32, (3, 3), padding='same', activation='tanh')(conv5x5_reduce)
    conv5x5 = layers.Conv2D(32, (3, 3), padding='same', activation='tanh')(conv5x5)
    
    # Branch 4: MaxPooling with stride=1 to maintain spatial dimensions
    pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    pool_proj = layers.Conv2D(16, (1, 1), padding='same', activation='tanh')(pool)
    
    # Concatenate - all branches now have matching spatial dimensions (19x19)
    concat = layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool_proj])
    concat = layers.BatchNormalization()(concat)
    
    # Now reduce spatial dimensions
    x = layers.MaxPooling2D((2, 2))(concat)
    x = layers.Conv2D(64, (1, 1), activation='tanh')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Additional convolutional blocks
    for _ in range(2):
        x = layers.Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
    
    # Global Average Pooling to handle any remaining spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='tanh')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='tanh')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_cnn_for_band(X_train, y_train, X_test, y_test, band_name, epochs=50):
    """Train CNN for specific frequency band"""
    start_time = time.time()
    print(f"    [{timestamp()}] Training CNN for {band_name}...", end=' ')
    
    model = build_band_cnn(X_train.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    acc, sens, spec = calculate_metrics(y_test, y_pred)
    elapsed = time.time() - start_time
    print(f"Done! Acc={acc:.2f}% (Time: {elapsed:.2f}s)")
    return acc, sens, spec

def process_dataset(data_path):
    """Process entire dataset"""
    log_time("Starting dataset processing...")
    start_time = time.time()
    
    mdd_files, healthy_files = load_edf_files(data_path)
    
    all_segments = []
    all_labels = []
    
    # Process MDD files (EC only)
    log_time("Processing MDD files...")
    mdd_start = time.time()
    for i, file in enumerate(mdd_files):
        if 'EC' in file:
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                data = raw.get_data()[:19]
                data = preprocess_eeg(data)
                segments = segment_data(data, WINDOW_SIZE, OVERLAP)
                all_segments.extend(segments)
                all_labels.extend([1] * len(segments))
                if (i + 1) % 5 == 0:
                    log_time(f"  Processed {i + 1}/{len(mdd_files)} MDD files")
            except Exception as e:
                log_time(f"  Error processing {file}: {e}")
    mdd_elapsed = time.time() - mdd_start
    log_time(f"MDD processing completed in {mdd_elapsed:.2f}s")
    
    # Process Healthy files (EC only)
    log_time("Processing Healthy files...")
    healthy_start = time.time()
    for i, file in enumerate(healthy_files):
        if 'EC' in file:
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                data = raw.get_data()[:19]
                data = preprocess_eeg(data)
                segments = segment_data(data, WINDOW_SIZE, OVERLAP)
                all_segments.extend(segments)
                all_labels.extend([0] * len(segments))
                if (i + 1) % 5 == 0:
                    log_time(f"  Processed {i + 1}/{len(healthy_files)} Healthy files")
            except Exception as e:
                log_time(f"  Error processing {file}: {e}")
    healthy_elapsed = time.time() - healthy_start
    log_time(f"Healthy processing completed in {healthy_elapsed:.2f}s")
    
    total_elapsed = time.time() - start_time
    log_time(f"Total dataset processing time: {total_elapsed:.2f}s")
    log_time(f"Total segments: {len(all_segments)} (MDD: {sum(all_labels)}, Healthy: {len(all_labels) - sum(all_labels)})")
    
    return np.array(all_segments), np.array(all_labels)

def main():
    overall_start = time.time()
    log_time("="*80)
    log_time("EEG MDD DETECTION - TABLE 6 & FIGURE 9 GENERATION")
    log_time("="*80)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Process dataset
    data_path = "/kaggle/input/eeg-dataset"
    segments, labels = process_dataset(data_path)
    
    # Compute connectivity matrices
    log_time("\nComputing connectivity matrices...")
    conn_start = time.time()
    X = []
    for i, segment in enumerate(segments):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - conn_start
            rate = (i + 1) / elapsed
            remaining = (len(segments) - i - 1) / rate
            log_time(f"  Progress: {i + 1}/{len(segments)} ({(i+1)/len(segments)*100:.1f}%) - ETA: {remaining:.1f}s")
        conn_matrix = compute_connectivity_matrix(segment)
        X.append(conn_matrix)
    
    conn_elapsed = time.time() - conn_start
    log_time(f"Connectivity computation completed in {conn_elapsed:.2f}s ({conn_elapsed/len(segments):.3f}s per segment)")
    
    X = np.array(X)
    y = labels
    
    # Split data
    log_time("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    log_time(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Generate Table 6 and Figure 9
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    classifiers = ['LDA', 'LR', 'SVM', 'RF', 'AdaBoost', 'CNN']
    
    results = {band: {clf: {} for clf in classifiers} for band in bands}
    
    log_time("\n" + "=" * 80)
    log_time("COMPARISON OF MACHINE LEARNING METHODS ACROSS FREQUENCY BANDS")
    log_time("=" * 80)
    
    training_start = time.time()
    
    # Process each band
    for band_idx, band in enumerate(bands):
        band_start = time.time()
        log_time(f"\n[Band {band_idx + 1}/{len(bands)}] {band.upper()} Band:")
        log_time("-" * 40)
        
        # Extract band features
        extract_start = time.time()
        X_band_train = np.array([extract_band_features(x, band) for x in X_train])
        X_band_test = np.array([extract_band_features(x, band) for x in X_test])
        extract_elapsed = time.time() - extract_start
        log_time(f"  Feature extraction completed in {extract_elapsed:.2f}s")
        
        # Train ML classifiers
        for clf_name in ['LDA', 'LR', 'SVM', 'RF', 'AdaBoost']:
            acc, sens, spec = train_ml_classifier(
                X_band_train, y_train, X_band_test, y_test, clf_name
            )
            results[band][clf_name] = {
                'accuracy': acc,
                'sensitivity': sens,
                'specificity': spec
            }
        
        # Train CNN
        acc, sens, spec = train_cnn_for_band(
            X_band_train, y_train, X_band_test, y_test, band, epochs=50
        )
        results[band]['CNN'] = {
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec
        }
        
        band_elapsed = time.time() - band_start
        log_time(f"  {band.upper()} band completed in {band_elapsed:.2f}s")
    
    training_elapsed = time.time() - training_start
    log_time(f"\nTotal training time: {training_elapsed:.2f}s ({training_elapsed/len(bands):.2f}s per band)")
    
    # Print Table 6
    log_time("\n\n" + "=" * 130)
    log_time("TABLE 6: Comparison of Machine Learning Algorithms")
    log_time("=" * 130)
    
    # Create DataFrame for better formatting
    table_data = []
    for band in bands:
        row = {'Band': band.capitalize()}
        for clf in classifiers:
            row[f'{clf}_Acc'] = f"{results[band][clf]['accuracy']:.2f}"
            row[f'{clf}_Sen'] = f"{results[band][clf]['sensitivity']:.2f}"
            row[f'{clf}_Spe'] = f"{results[band][clf]['specificity']:.2f}"
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Print formatted table
    print("\nAccuracy (%):")
    print(f"{'Band':<10}", end='')
    for clf in classifiers:
        print(f"{clf:<12}", end='')
    print()
    print("-" * 80)
    for band in bands:
        print(f"{band.capitalize():<10}", end='')
        for clf in classifiers:
            print(f"{results[band][clf]['accuracy']:>6.2f}%{'':<5}", end='')
        print()
    
    print("\nSensitivity (%):")
    print(f"{'Band':<10}", end='')
    for clf in classifiers:
        print(f"{clf:<12}", end='')
    print()
    print("-" * 80)
    for band in bands:
        print(f"{band.capitalize():<10}", end='')
        for clf in classifiers:
            print(f"{results[band][clf]['sensitivity']:>6.2f}%{'':<5}", end='')
        print()
    
    print("\nSpecificity (%):")
    print(f"{'Band':<10}", end='')
    for clf in classifiers:
        print(f"{clf:<12}", end='')
    print()
    print("-" * 80)
    for band in bands:
        print(f"{band.capitalize():<10}", end='')
        for clf in classifiers:
            print(f"{results[band][clf]['specificity']:>6.2f}%{'':<5}", end='')
        print()
    
    # Create Figure 9
    log_time("\n\nGenerating Figure 9...")
    plot_start = time.time()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Accuracy by band (grouped bar)
    ax1 = fig.add_subplot(gs[0, :2])
    x_pos = np.arange(len(bands))
    width = 0.13
    
    for i, clf in enumerate(classifiers):
        values = [results[band][clf]['accuracy'] for band in bands]
        ax1.bar(x_pos + i * width, values, width, label=clf, alpha=0.8)
    
    ax1.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison Across Frequency Bands', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos + width * 2.5)
    ax1.set_xticklabels([b.capitalize() for b in bands])
    ax1.legend(fontsize=10, ncol=6, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Plot 2: Heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    heatmap_data = np.array([[results[band][clf]['accuracy'] for band in bands] for clf in classifiers])
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=60, vmax=100)
    
    ax2.set_xticks(np.arange(len(bands)))
    ax2.set_yticks(np.arange(len(classifiers)))
    ax2.set_xticklabels([b.capitalize() for b in bands], fontsize=9)
    ax2.set_yticklabels(classifiers, fontsize=9)
    ax2.set_title('Accuracy Heatmap (%)', fontsize=12, fontweight='bold')
    
    for i in range(len(classifiers)):
        for j in range(len(bands)):
            ax2.text(j, i, f'{heatmap_data[i, j]:.1f}',
                    ha="center", va="center", color="white" if heatmap_data[i, j] < 80 else "black",
                    fontsize=8, fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Sensitivity comparison
    ax3 = fig.add_subplot(gs[1, :2])
    for i, clf in enumerate(classifiers):
        values = [results[band][clf]['sensitivity'] for band in bands]
        ax3.bar(x_pos + i * width, values, width, label=clf, alpha=0.8)
    
    ax3.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sensitivity (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Sensitivity Comparison Across Frequency Bands', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos + width * 2.5)
    ax3.set_xticklabels([b.capitalize() for b in bands])
    ax3.legend(fontsize=10, ncol=6, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    # Plot 4: Line plot - trends
    ax4 = fig.add_subplot(gs[1, 2])
    for clf in classifiers:
        values = [results[band][clf]['accuracy'] for band in bands]
        ax4.plot(bands, values, marker='o', linewidth=2, markersize=8, label=clf)
    
    ax4.set_xlabel('Frequency Band', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Performance Trends', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([60, 105])
    
    # Plot 5: Specificity comparison
    ax5 = fig.add_subplot(gs[2, :2])
    for i, clf in enumerate(classifiers):
        values = [results[band][clf]['specificity'] for band in bands]
        ax5.bar(x_pos + i * width, values, width, label=clf, alpha=0.8)
    
    ax5.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Specificity (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Specificity Comparison Across Frequency Bands', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos + width * 2.5)
    ax5.set_xticklabels([b.capitalize() for b in bands])
    ax5.legend(fontsize=10, ncol=6, loc='upper left')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 105])
    
    # Plot 6: Best performer by band
    ax6 = fig.add_subplot(gs[2, 2])
    best_performers = []
    best_scores = []
    
    for band in bands:
        best_clf = max(classifiers, key=lambda c: results[band][c]['accuracy'])
        best_score = results[band][best_clf]['accuracy']
        best_performers.append(best_clf)
        best_scores.append(best_score)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(classifiers)))
    clf_colors = {clf: colors[i] for i, clf in enumerate(classifiers)}
    bar_colors = [clf_colors[clf] for clf in best_performers]
    
    bars = ax6.bar(bands, best_scores, color=bar_colors, alpha=0.8)
    ax6.set_xlabel('Frequency Band', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Best Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Best Performer by Band', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim([0, 105])
    
    for i, (bar, clf, score) in enumerate(zip(bars, best_performers, best_scores)):
        ax6.text(i, score + 1, f'{clf}\n{score:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.savefig('table6_figure9_results.png', dpi=300, bbox_inches='tight')
    plot_elapsed = time.time() - plot_start
    log_time(f"Figure generation completed in {plot_elapsed:.2f}s")
    log_time("Figure saved as 'table6_figure9_results.png'")
    plt.show()
    
    overall_elapsed = time.time() - overall_start
    log_time("\n" + "=" * 80)
    log_time(f"TOTAL EXECUTION TIME: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} minutes)")
    log_time("=" * 80)
    log_time("\nTiming Breakdown:")
    log_time(f"  - Dataset Processing: {(conn_start - overall_start):.2f}s")
    log_time(f"  - Connectivity Computation: {conn_elapsed:.2f}s")
    log_time(f"  - Model Training: {training_elapsed:.2f}s")
    log_time(f"  - Figure Generation: {plot_elapsed:.2f}s")
    log_time("\nProcessing complete!")
    
    return results

if __name__ == "__main__":
    log_time("Script started")
    results = main()
    log_time("Script completed successfully!")

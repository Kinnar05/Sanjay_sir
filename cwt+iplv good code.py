import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from datetime import datetime
mne.set_log_level('ERROR')
class OptimizedCWTExtractor:
    def __init__(self, fs=256, window_sec=5, overlap=0.5):
        self.fs = fs
        self.window_sec = window_sec
        self.overlap = overlap
        self.window_samples = int(window_sec * fs)
        self.step_samples = int(self.window_samples * (1 - overlap))
        # Frequency bands
        self.band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 70.0)
        }
        # Create CWT scales for frequency range 0.5-70 Hz
        self._setup_cwt_scales()
    def _setup_cwt_scales(self):
        wavelet = 'cmor1.5-1.0'  # Complex Morlet with center frequency ~1.0
        # Target frequencies (0.5 to 70 Hz)
        min_freq = 0.5
        max_freq = 70.0
        n_freqs = 64  # number of frequency bins
        
        # Create logarithmically spaced frequencies
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
        
        # Convert frequencies to scales for PyWavelets
        # scale = center_freq * fs / freq
        # For cmor1.5-1.0, center frequency is approximately 1.0
        center_freq = 1.0
        self.scales = center_freq * self.fs / freqs
        self.frequencies = freqs
        self.wavelet = wavelet
        
        # Get band indices
        self.band_indices = {}
        for band_name in self.band_names:
            low, high = self.bands[band_name]
            indices = np.where((self.frequencies >= low) & (self.frequencies <= high))[0]
            self.band_indices[band_name] = indices
    
    def compute_cwt_for_channel(self, signal_data):
        # Use PyWavelets cwt with complex Morlet wavelet
        cwt_coeffs, _ = pywt.cwt(
            signal_data, 
            self.scales,
            self.wavelet,
            sampling_period=1.0/self.fs
        )
        return cwt_coeffs
    
    def segment_cwt(self, cwt_matrix):
        """Segment CWT results (not raw signal!)"""
        segments = []
        n_time = cwt_matrix.shape[1]
        
        for start in range(0, n_time - self.window_samples + 1, self.step_samples):
            segment = cwt_matrix[:, start:start + self.window_samples]
            segments.append(segment)
        return segments
    def compute_iplv(self, cwt1_segment, cwt2_segment):
        phase1 = np.angle(cwt1_segment)
        phase2 = np.angle(cwt2_segment)
        # Phase difference
        phase_diff = phase1 - phase2
        # iPLV: imaginary part of mean complex phase difference
        mean_complex = np.mean(np.exp(1j * phase_diff), axis=1)
        iplv = np.abs(np.imag(mean_complex))
        return iplv
    def compute_mswc(self, cwt1_segment, cwt2_segment):
        cross = cwt1_segment * np.conj(cwt2_segment)
        auto1 = np.abs(cwt1_segment) ** 2
        auto2 = np.abs(cwt2_segment) ** 2
        denom = auto1 * auto2
        denom = np.where(denom < 1e-10, 1e-10, denom)
        
        # Squared wavelet coherence
        swc = np.clip(np.abs(cross) ** 2 / denom, 0, 1)
        
        # Mean over time
        mswc = np.mean(swc, axis=1)
        
        return mswc
    
    def fuse_iplv_mswc(self, iplv, mswc):
        summed = np.clip(iplv, 0, 1) + np.clip(mswc, 0, 1)
        result = np.zeros_like(summed)
        
        # Region 1: sum <= 1
        mask1 = summed <= 1
        result[mask1] = (np.exp(summed[mask1]) - 1) / (2 * np.e - 2)
        
        # Region 2: sum > 1
        mask2 = summed > 1
        result[mask2] = 1 - (np.exp(2 - summed[mask2]) - 1) / (2 * np.e - 2)
        return result
    
    def extract_features_from_file(self, channel_data):
        n_channels = len(channel_data)
        # Step 1: Compute CWT for all channels ONCE
        cwt_full = []
        for signal_data in channel_data:
            cwt_matrix = self.compute_cwt_for_channel(signal_data)
            cwt_full.append(cwt_matrix)
        
        # Step 2: Segment all CWT results
        cwt_segments = []
        for cwt_matrix in cwt_full:
            segments = self.segment_cwt(cwt_matrix)
            cwt_segments.append(segments)
        
        n_segments = len(cwt_segments[0])
        
        # Step 3: Extract features for each segment
        all_segment_features = []
        
        for seg_idx in range(n_segments):
            features = []
            
            # For each channel pair
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    # Get CWT segment for this pair
                    cwt_i_seg = cwt_segments[i][seg_idx]
                    cwt_j_seg = cwt_segments[j][seg_idx]
                    
                    # Compute connectivity measures
                    iplv = self.compute_iplv(cwt_i_seg, cwt_j_seg)
                    mswc = self.compute_mswc(cwt_i_seg, cwt_j_seg)
                    
                    # Fuse iPLV and MSWC
                    p_mswc = self.fuse_iplv_mswc(iplv, mswc)
                    
                    # Extract per-band features
                    for band_name in self.band_names:
                        indices = self.band_indices[band_name]
                        if len(indices) > 0:
                            features.append(np.mean(p_mswc[indices]))
                        else:
                            features.append(0.0)
            
            all_segment_features.append(np.array(features))
        
        return all_segment_features
class EEGLoader:
    def __init__(self, fs=256):
        self.fs = fs
        self.channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
            'C3', 'C4', 'Cz',
            'P3', 'P4', 'Pz',
            'T3', 'T4', 'T5', 'T6',
            'O1', 'O2'
        ]
    def load(self, filepath):
        """Load with preprocessing"""
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # Resample if needed
            if raw.info['sfreq'] != self.fs:
                raw.resample(self.fs, verbose=False)
            
            # Bandpass filter (0.5-70 Hz)
            raw.filter(0.5, 70.0, verbose=False)
            
            # Notch filter (50 Hz - power line)
            raw.notch_filter(50.0, verbose=False)
            
            return raw
        except:
            return None
    
    def extract_channels(self, raw):
        """Extract standard 19 channels"""
        available = [ch.upper() for ch in raw.ch_names]
        data = []
        
        for target in self.channels:
            for i, ch in enumerate(available):
                if target.upper() in ch or ch in target.upper():
                    data.append(raw.get_data()[i])
                    break
        
        return np.array(data) if len(data) >= 10 else None
def load_dataset(data_path, condition='EC'):
    """Load dataset file list"""
    files = list(Path(data_path).glob(f'*{condition}.edf'))
    dataset = []
    
    for f in files:
        name = f.stem
        
        # Determine label
        if 'MDD' in name:
            label = 1  # Depression
        elif 'H ' in name or '_H ' in name:
            label = 0  # Healthy
        else:
            continue
        
        # Extract subject ID
        parts = name.replace('MDD', '').replace('H', '').replace('_', '').strip().split()
        subject_id = parts[0] if parts else name[:10]
        
        dataset.append({
            'file': f,
            'label': label,
            'subject': subject_id
        })
    
    return dataset
def train_test_split_subjects(dataset, test_ratio=0.3, seed=42):
    """Subject-wise train-test split"""
    # Group by subject
    subjects = {}
    for item in dataset:
        sid = item['subject']
        if sid not in subjects:
            subjects[sid] = []
        subjects[sid].append(item)
    
    # Split subjects by class
    mdd = [sid for sid, items in subjects.items() if items[0]['label'] == 1]
    healthy = [sid for sid, items in subjects.items() if items[0]['label'] == 0]
    # Random split
    np.random.seed(seed)
    n_test_mdd = max(1, int(len(mdd) * test_ratio))
    n_test_healthy = max(1, int(len(healthy) * test_ratio))
    
    test_mdd = set(np.random.choice(mdd, n_test_mdd, replace=False))
    test_healthy = set(np.random.choice(healthy, n_test_healthy, replace=False))
    test_subjects = test_mdd | test_healthy
    
    # Create train/test sets
    train = [item for item in dataset if item['subject'] not in test_subjects]
    test = [item for item in dataset if item['subject'] in test_subjects]
    
    return train, test
def extract_features_from_dataset(dataset, loader, extractor):
    """
    OPTIMIZED: Extract features with NO duplicate CWT calculations
    """
    X, y = [], []
    skipped = 0
    
    print(f"  Processing {len(dataset)} files...")
    start_time = datetime.now()
    last_timestamp = start_time
    
    for i, item in enumerate(dataset):
        current_time = datetime.now()
        
        # 20-second timestamp
        if (current_time - last_timestamp).total_seconds() >= 20:
            elapsed = (current_time - start_time).total_seconds()
            processed = i + 1
            rate = processed / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{processed}/{len(dataset)}] {elapsed:.0f}s | "
                  f"{rate:.1f} files/min | {item['file'].name[:25]}")
            last_timestamp = current_time
        
        # Load EEG
        raw = loader.load(item['file'])
        if raw is None:
            skipped += 1
            continue
        
        # Extract channels
        channel_data = loader.extract_channels(raw)
        if channel_data is None:
            skipped += 1
            continue
        
        # CRITICAL: Extract features from ENTIRE file at once
        # This computes CWT only ONCE per channel!
        try:
            segment_features = extractor.extract_features_from_file(channel_data)
            
            # Add all segments from this file
            for features in segment_features:
                X.append(features)
                y.append(item['label'])
        
        except Exception as e:
            print(f"  ⚠ Error in {item['file'].name}: {str(e)[:50]}")
            skipped += 1
            continue
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    avg_rate = len(dataset) / (total_elapsed / 60) if total_elapsed > 0 else 0
    print(f"  [{len(dataset)}/{len(dataset)}] {total_elapsed:.0f}s | "
          f"Avg: {avg_rate:.1f} files/min | Complete")
    
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} files")
    
    return np.array(X), np.array(y)
def train_linear_svm(X_train, y_train, X_test, y_test):
    """Train Linear SVM"""
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_s)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'name': 'Linear SVM',
        'accuracy': acc * 100,
        'sensitivity': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'cm': cm
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost"""
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    clf.fit(X_train_s, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_s)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'name': 'XGBoost',
        'accuracy': acc * 100,
        'sensitivity': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'cm': cm
    }


def print_results(results):
    """Print results"""
    print(f"\n {'─'*66}")
    print(f" {results['name']:^66}")
    print(f" {'─'*66}")
    print(f"  Accuracy:    {results['accuracy']:6.2f}%")
    print(f"  Sensitivity: {results['sensitivity']:6.2f}%")
    print(f"  Specificity: {results['specificity']:6.2f}%")
    cm = results['cm']
    print(f"  CM: [{cm[0,0]:3d} {cm[0,1]:3d}]")
    print(f"      [{cm[1,0]:3d} {cm[1,1]:3d}]")
    print(f" {'─'*66}")
def main():
    """Main pipeline - OPTIMIZED with NO duplicate calculations"""
    start_time = datetime.now()
    
    # Check dependencies
    try:
        import pywt
    except ImportError:
        print("\n✗ Error: PyWavelets not installed")
        print("  Install with: pip install PyWavelets")
        return
    
    print("\n" + "="*70)
    print(" OPTIMIZED: CWT + iPLV FOR EEG DEPRESSION DETECTION")
    print("="*70)
    print("  Using PyWavelets (pywt) for CWT computation")
    print("  Key Optimization: Compute CWT ONCE per channel per file")
    print("  Then segment the CWT results (not the raw signal!)")
    print("  Advantages over SSWT:")
    print("    - Faster computation (no synchrosqueezing)")
    print("    - Simpler implementation")
    print("    - Still captures time-frequency dynamics effectively")
    print("="*70)
    print(f"  Start: {start_time.strftime('%H:%M:%S')}\n")
    
    # Data path
    DATA_PATH = '/kaggle/input/eeg-dataset/'
    
    if not Path(DATA_PATH).exists():
        print(f"✗ Error: {DATA_PATH} not found")
        print("\nMake sure:")
        print("  1. EEG dataset is available at the specified path")
        print("  2. PyWavelets is installed: pip install PyWavelets")
        return
    
    # Step 1: Load dataset
    t1 = datetime.now()
    print("STEP 1: Load dataset")
    dataset = load_dataset(DATA_PATH, 'EC')
    print(f"  ✓ {len(dataset)} files ({(datetime.now()-t1).total_seconds():.1f}s)\n")
    
    # Step 2: Train-test split
    t2 = datetime.now()
    print("STEP 2: Train-test split (subject-wise)")
    train_data, test_data = train_test_split_subjects(dataset)
    print(f"  ✓ Train: {len(train_data)}, Test: {len(test_data)} "
          f"({(datetime.now()-t2).total_seconds():.1f}s)\n")
    
    # Step 3: Extract features
    t3 = datetime.now()
    print("STEP 3: Extract CWT+iPLV features (OPTIMIZED)\n")
    loader = EEGLoader()
    extractor = OptimizedCWTExtractor()
    print("  Training:")
    t3a = datetime.now()
    X_train, y_train = extract_features_from_dataset(train_data, loader, extractor)
    t3a_time = (datetime.now()-t3a).total_seconds()
    print("\n  Test:")
    t3b = datetime.now()
    X_test, y_test = extract_features_from_dataset(test_data, loader, extractor)
    t3b_time = (datetime.now()-t3b).total_seconds()
    # Check if we got valid data
    if len(X_train) == 0 or len(X_test) == 0:
        print("\n  ✗ Error: No features extracted!")
        print("    Check that EEG files are valid and preprocessing is working")
        return 
    print(f"\n  ✓ Complete: {X_train.shape[1]} features per segment")
    print(f"    Train: {len(X_train)} segments ({t3a_time:.1f}s)")
    print(f"    Test:  {len(X_test)} segments ({t3b_time:.1f}s)\n")
    # Step 4: Train classifiers
    t4 = datetime.now()
    print("STEP 4: Train classifiers\n")
    t4a = datetime.now()
    svm_results = train_linear_svm(X_train, y_train, X_test, y_test)
    svm_time = (datetime.now()-t4a).total_seconds()
    print(f"  Linear SVM: {svm_time:.2f}s")
    print_results(svm_results)
    t4b = datetime.now()
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_time = (datetime.now()-t4b).total_seconds()
    print(f"\n  XGBoost: {xgb_time:.2f}s")
    print_results(xgb_results)
    # Summary
    total = (datetime.now()-start_time).total_seconds()
    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)
    df = pd.DataFrame([
        {
            'Model': 'Linear SVM',
            'Accuracy': f"{svm_results['accuracy']:.2f}%",
            'Sensitivity': f"{svm_results['sensitivity']:.2f}%",
            'Specificity': f"{svm_results['specificity']:.2f}%"
        },
        {
            'Model': 'XGBoost',
            'Accuracy': f"{xgb_results['accuracy']:.2f}%",
            'Sensitivity': f"{xgb_results['sensitivity']:.2f}%",
            'Specificity': f"{xgb_results['specificity']:.2f}%"
        }
    ])
    print(df.to_string(index=False))
    best = svm_results if svm_results['accuracy'] > xgb_results['accuracy'] else xgb_results
    print(f"\n  ✓ Best Model: {best['name']} ({best['accuracy']:.2f}%)")
    print(f"  ⏱ Total Time: {total:.1f}s ({total/60:.2f} min)")
    print(f"  ⏱ End: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70 + "\n")
if __name__ == "__main__":
    main()

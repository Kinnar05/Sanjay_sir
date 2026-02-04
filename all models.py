"""
EEG-Based Depression Detection: CWT+iPLV with Statistical Testing & 5 ML Models
================================================================================
MODIFIED: Added Statistical Testing + 5 ML Models to Original Pipeline

Key Features:
- CWT-based time-frequency analysis
- iPLV (instantaneous Phase Locking Value) for connectivity
- MSWC (Mean Squared Wavelet Coherence) for amplitude correlation
- Statistical Testing (t-test + FDR correction) to identify significant connections
- 5 ML Models: Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost
- Band-wise classification with significant features only
- Metrics: Accuracy, Sensitivity, Specificity per band per model

Dependencies:
- numpy, pandas, mne, scikit-learn, xgboost, pywt, scipy, statsmodels
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import pywt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from datetime import datetime
from collections import defaultdict

mne.set_log_level('ERROR')

# ============================================================================
# OPTIMIZED CWT + iPLV FEATURE EXTRACTOR
# ============================================================================

class CWTiPLVExtractor:
    """
    Extract CWT+iPLV features following the paper's methodology:
    - CWT for time-frequency decomposition
    - iPLV for phase synchronization
    - MSWC for amplitude correlation
    - Feature fusion
    """
    
    def __init__(self, fs=256, window_sec=5, overlap=0.5):
        """
        Initialize extractor
        
        Args:
            fs: Sampling frequency (256 Hz as per paper)
            window_sec: Window size in seconds (5 sec as per paper)
            overlap: Overlap ratio (0.5 = 50%)
        """
        self.fs = fs
        self.window_sec = window_sec
        self.overlap = overlap
        self.window_samples = int(window_sec * fs)
        self.step_samples = int(self.window_samples * (1 - overlap))
        
        # Frequency bands (as per paper: delta, theta, alpha, beta, gamma)
        self.band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.bands = {
            'delta': (1.0, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 45.0)
        }
        
        # Setup CWT parameters
        self._setup_cwt_scales()
    
    def _setup_cwt_scales(self):
        """Setup CWT scales to cover 1-45 Hz (as per paper)"""
        # Complex Morlet wavelet
        wavelet = 'cmor1.5-1.0'
        
        # Frequency range: 1-45 Hz
        min_freq = 1.0
        max_freq = 45.0
        n_freqs = 64
        
        # Logarithmically spaced frequencies
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
        
        # Convert to scales
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
        """
        Compute CWT for entire channel
        
        Args:
            signal_data: 1D array of EEG signal
            
        Returns:
            cwt_matrix: Complex CWT coefficients (n_scales x n_samples)
        """
        try:
            cwt_coeffs, _ = pywt.cwt(
                signal_data,
                self.scales,
                self.wavelet,
                sampling_period=1.0/self.fs
            )
            return cwt_coeffs
        except Exception as e:
            print(f"  ⚠ CWT computation error: {str(e)[:50]}")
            return None
    
    def segment_cwt(self, cwt_matrix):
        """Segment CWT results into windows"""
        if cwt_matrix is None:
            return []
        
        segments = []
        n_time = cwt_matrix.shape[1]
        
        for start in range(0, n_time - self.window_samples + 1, self.step_samples):
            segment = cwt_matrix[:, start:start + self.window_samples]
            segments.append(segment)
        
        return segments
    
    def compute_iplv(self, cwt1_segment, cwt2_segment):
        """
        Compute instantaneous Phase Locking Value (iPLV)
        
        iPLV measures phase synchronization between two signals
        """
        # Extract instantaneous phase
        phase1 = np.angle(cwt1_segment)
        phase2 = np.angle(cwt2_segment)
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # iPLV: magnitude of mean complex phase difference
        mean_complex = np.mean(np.exp(1j * phase_diff), axis=1)
        iplv = np.abs(mean_complex)
        
        return iplv
    
    def compute_mswc(self, cwt1_segment, cwt2_segment):
        """
        Compute Mean Squared Wavelet Coherence (MSWC)
        
        MSWC measures amplitude correlation
        """
        # Cross-spectrum
        cross = cwt1_segment * np.conj(cwt2_segment)
        
        # Auto-spectra
        auto1 = np.abs(cwt1_segment) ** 2
        auto2 = np.abs(cwt2_segment) ** 2
        
        # Avoid division by zero
        denom = auto1 * auto2
        denom = np.where(denom < 1e-10, 1e-10, denom)
        
        # Squared wavelet coherence
        swc = np.clip(np.abs(cross) ** 2 / denom, 0, 1)
        
        # Mean over time
        mswc = np.mean(swc, axis=1)
        
        return mswc
    
    def fuse_iplv_mswc(self, iplv, mswc):
        """
        Feature fusion: non-linear combination of iPLV and MSWC
        """
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
        """
        Extract all features from one file organized by band
        
        Returns:
            Dictionary: {band_name: [feature_vectors]} for each band
        """
        n_channels = len(channel_data)
        
        # Step 1: Compute CWT for all channels
        cwt_full = []
        for signal_data in channel_data:
            cwt_matrix = self.compute_cwt_for_channel(signal_data)
            if cwt_matrix is None:
                return {band: [] for band in self.band_names}
            cwt_full.append(cwt_matrix)
        
        # Step 2: Segment all CWT results
        cwt_segments = []
        for cwt_matrix in cwt_full:
            segments = self.segment_cwt(cwt_matrix)
            cwt_segments.append(segments)
        
        if not cwt_segments or len(cwt_segments[0]) == 0:
            return {band: [] for band in self.band_names}
        
        n_segments = len(cwt_segments[0])
        
        # Step 3: Extract features for each segment, organized by band
        band_segment_features = {band: [] for band in self.band_names}
        
        for seg_idx in range(n_segments):
            # For each band, collect features
            band_features = {band: [] for band in self.band_names}
            
            # For each channel pair (functional connectivity)
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
                            band_features[band_name].append(np.mean(p_mswc[indices]))
                        else:
                            band_features[band_name].append(0.0)
            
            # Store segment features for each band
            for band in self.band_names:
                band_segment_features[band].append(np.array(band_features[band]))
        
        return band_segment_features


# ============================================================================
# EEG DATA LOADER (Following Paper's Preprocessing)
# ============================================================================

class EEGLoader:
    """
    Load and preprocess EEG data following the paper's methodology:
    - Bandpass filter: 1-45 Hz
    - Downsample to 256 Hz
    - Artifact removal
    - Use 23 standard electrodes
    """
    
    def __init__(self, fs=256):
        self.fs = fs
        # 23 electrodes as per paper
        self.channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FT9', 'FT10',
            'T1', 'T2', 'T7', 'T8', 'C3', 'C4', 'Cz',
            'P3', 'P4', 'P7', 'P8', 'Pz', 'O1', 'O2'
        ]
    
    def load(self, filepath):
        """Load with preprocessing following paper"""
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # Resample to 256 Hz
            if raw.info['sfreq'] != self.fs:
                raw.resample(self.fs, verbose=False)
            
            # Bandpass filter: 1-45 Hz (as per paper)
            raw.filter(1.0, 45.0, verbose=False)
            
            # Notch filter (50 Hz - power line noise)
            raw.notch_filter(50.0, verbose=False)
            
            return raw
        except Exception as e:
            print(f"  ⚠ Load error: {str(e)[:50]}")
            return None
    
    def extract_channels(self, raw):
        """Extract standard channels"""
        try:
            available = [ch.upper() for ch in raw.ch_names]
            data = []
            
            for target in self.channels:
                found = False
                for i, ch in enumerate(available):
                    if target.upper() in ch or ch in target.upper():
                        data.append(raw.get_data()[i])
                        found = True
                        break
                if not found:
                    # If channel not found, pad with zeros
                    data.append(np.zeros(raw.n_times))
            
            return np.array(data) if len(data) >= 10 else None
        except Exception as e:
            print(f"  ⚠ Channel extraction error: {str(e)[:50]}")
            return None


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

def load_dataset(data_path, condition='EC'):
    """
    Load dataset with labels
    
    Labels for Depression Dataset:
    - 0: Healthy Control (H)
    - 1: Major Depressive Disorder (MDD)
    """
    files = list(Path(data_path).glob(f'*{condition}.edf'))
    dataset = []
    
    for f in files:
        name = f.stem.upper()
        
        # Determine label
        if 'MDD' in name:
            label = 1  # Depression
        elif 'H ' in name or '_H ' in name or name.startswith('H ') or name.startswith('H_'):
            label = 0  # Healthy
        else:
            continue
        
        # Extract subject ID
        parts = name.replace('MDD', '').replace('H', '').replace('_', ' ').strip().split()
        subject_id = parts[0] if parts else name[:10]
        
        dataset.append({
            'file': f,
            'label': label,
            'subject': subject_id,
            'condition': condition
        })
    
    return dataset


def subject_wise_split(dataset, test_ratio=0.3, seed=42):
    """Subject-wise train-test split (no validation for simplicity)"""
    subjects = defaultdict(list)
    for item in dataset:
        subjects[item['subject']].append(item)
    
    label_subjects = defaultdict(list)
    for sid, items in subjects.items():
        label = items[0]['label']
        label_subjects[label].append(sid)
    
    np.random.seed(seed)
    train_data, test_data = [], []
    
    for label, sids in label_subjects.items():
        n = len(sids)
        n_test = max(1, int(n * test_ratio))
        
        np.random.shuffle(sids)
        test_sids = set(sids[:n_test])
        train_sids = set(sids[n_test:])
        
        for sid, items in subjects.items():
            if sid in train_sids:
                train_data.extend(items)
            elif sid in test_sids:
                test_data.extend(items)
    
    return train_data, test_data


# ============================================================================
# FEATURE EXTRACTION (BAND-WISE)
# ============================================================================

def extract_features_from_dataset(dataset, loader, extractor, desc="Processing"):
    """Extract features organized by frequency band"""
    band_features = {band: {'X': [], 'y': []} for band in extractor.band_names}
    skipped = 0
    
    print(f"  {desc}: {len(dataset)} files...")
    start_time = datetime.now()
    last_timestamp = start_time
    
    for i, item in enumerate(dataset):
        current_time = datetime.now()
        
        # Progress every 20 seconds
        if (current_time - last_timestamp).total_seconds() >= 20:
            elapsed = (current_time - start_time).total_seconds()
            processed = i + 1
            rate = processed / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{processed}/{len(dataset)}] {elapsed:.0f}s | "
                  f"{rate:.1f} files/min | {item['file'].name[:30]}")
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
        
        # Extract features (returns dict of band features)
        try:
            file_band_features = extractor.extract_features_from_file(channel_data)
            
            # Add all segments for each band
            for band in extractor.band_names:
                for segment_features in file_band_features[band]:
                    band_features[band]['X'].append(segment_features)
                    band_features[band]['y'].append(item['label'])
        
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
    
    # Convert to numpy arrays
    for band in extractor.band_names:
        band_features[band]['X'] = np.array(band_features[band]['X'])
        band_features[band]['y'] = np.array(band_features[band]['y'])
    
    return band_features


# ============================================================================
# STATISTICAL TESTING (NEW!)
# ============================================================================

def perform_statistical_testing(X_train, y_train, band_name, alpha=0.05):
    """
    Perform t-test with FDR correction to identify significant connections
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels
        band_name: Name of frequency band
        alpha: Significance level
        
    Returns:
        significant_indices: Indices of significant features
        p_values_corrected: FDR-corrected p-values
    """
    print(f"\n  Statistical Testing for {band_name.upper()} band:")
    
    # Split by class
    X_mdd = X_train[y_train == 1]
    X_healthy = X_train[y_train == 0]
    
    print(f"    MDD samples: {len(X_mdd)}, Healthy samples: {len(X_healthy)}")
    
    n_features = X_train.shape[1]
    p_values = []
    
    # Perform t-test for each feature (channel pair)
    for feat_idx in range(n_features):
        mdd_vals = X_mdd[:, feat_idx]
        healthy_vals = X_healthy[:, feat_idx]
        
        # Check normality and variance assumptions
        try:
            _, p_levene = stats.levene(mdd_vals, healthy_vals)
            
            # Shapiro test on sample if data is large
            if len(mdd_vals) > 5000:
                mdd_sample = np.random.choice(mdd_vals, 5000, replace=False)
                healthy_sample = np.random.choice(healthy_vals, 5000, replace=False)
            else:
                mdd_sample = mdd_vals
                healthy_sample = healthy_vals
            
            _, p_shapiro_mdd = stats.shapiro(mdd_sample)
            _, p_shapiro_healthy = stats.shapiro(healthy_sample)
            
            # Use Welch's t-test if assumptions violated
            if p_levene < 0.05 or p_shapiro_mdd < 0.05 or p_shapiro_healthy < 0.05:
                _, p_val = stats.ttest_ind(mdd_vals, healthy_vals, equal_var=False)
            else:
                _, p_val = stats.ttest_ind(mdd_vals, healthy_vals, equal_var=True)
        except:
            # If statistical tests fail, use Welch's t-test
            _, p_val = stats.ttest_ind(mdd_vals, healthy_vals, equal_var=False)
        
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    
    # FDR correction
    reject, p_values_corrected = fdrcorrection(p_values, alpha=alpha)
    
    # Get significant indices
    significant_indices = np.where(reject)[0]
    
    print(f"    Total connections: {n_features}")
    print(f"    Significant connections: {len(significant_indices)} (p < {alpha})")
    if len(significant_indices) > 0:
        print(f"    Mean p-value (significant): {np.mean(p_values_corrected[significant_indices]):.6f}")
    
    return significant_indices, p_values_corrected


# ============================================================================
# MULTI-MODEL CLASSIFIER (5 ML MODELS)
# ============================================================================

def train_all_models_bandwise(X_train, y_train, X_test, y_test, band_name):
    """
    Train all 5 ML models for a specific band
    
    Models:
    1. Logistic Regression
    2. SVM
    3. Decision Tree
    4. Random Forest
    5. XGBoost
    
    Returns:
        Dictionary with results for each model
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 1. Logistic Regression
    print(f"    Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    results['Logistic Reg'] = compute_metrics(y_test, y_pred_lr)
    
    # 2. SVM
    print(f"    Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)
    results['SVM'] = compute_metrics(y_test, y_pred_svm)
    
    # 3. Decision Tree
    print(f"    Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train_s, y_train)
    y_pred_dt = dt.predict(X_test_s)
    results['Decision Tree'] = compute_metrics(y_test, y_pred_dt)
    
    # 4. Random Forest
    print(f"    Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    results['Random Forest'] = compute_metrics(y_test, y_pred_rf)
    
    # 5. XGBoost
    print(f"    Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_s, y_train)
    y_pred_xgb = xgb_model.predict(X_test_s)
    results['XGBoost'] = compute_metrics(y_test, y_pred_xgb)
    
    return results


def compute_metrics(y_true, y_pred):
    """Compute accuracy, sensitivity, specificity"""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    return {
        'accuracy': acc * 100,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'cm': cm
    }


# ============================================================================
# MAIN PIPELINE (MODIFIED)
# ============================================================================

def main():
    """Main pipeline with statistical testing"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print(" CWT+iPLV DEPRESSION DETECTION: 5 ML MODELS + STATISTICAL TESTING")
    print("="*70)
    print("  Pipeline:")
    print("  1. Extract CWT+iPLV features (band-wise)")
    print("  2. Statistical Testing (t-test + FDR correction)")
    print("  3. Use ONLY significant connections")
    print("  4. Train 5 ML Models per band:")
    print("     - Logistic Regression")
    print("     - SVM (RBF kernel)")
    print("     - Decision Tree")
    print("     - Random Forest")
    print("     - XGBoost")
    print("  5. Report: Accuracy, Sensitivity, Specificity (each model × band)")
    print("="*70)
    print(f"  Start: {start_time.strftime('%H:%M:%S')}\n")
    
    # Check dependencies
    try:
        import pywt
        from scipy import stats
        from statsmodels.stats.multitest import fdrcorrection
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("  Install with: pip install PyWavelets scipy statsmodels")
        return
    
    # Data path
    DATA_PATH = '/kaggle/input/eeg-dataset/'
    
    if not Path(DATA_PATH).exists():
        print(f"✗ Error: {DATA_PATH} not found")
        return
    
    # Step 1: Load dataset
    print("STEP 1: Load dataset (EC condition)")
    t1 = datetime.now()
    dataset = load_dataset(DATA_PATH, 'EC')
    
    if len(dataset) == 0:
        print("  ✗ No data files found")
        return
    
    n_mdd = sum(1 for item in dataset if item['label'] == 1)
    n_healthy = sum(1 for item in dataset if item['label'] == 0)
    print(f"  ✓ {len(dataset)} files (MDD: {n_mdd}, Healthy: {n_healthy})")
    print(f"    Time: {(datetime.now()-t1).total_seconds():.1f}s\n")
    
    # Step 2: Train-test split
    print("STEP 2: Subject-wise train-test split (70/30)")
    t2 = datetime.now()
    train_data, test_data = subject_wise_split(dataset)
    print(f"  ✓ Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"    Time: {(datetime.now()-t2).total_seconds():.1f}s\n")
    
    # Step 3: Extract features (band-wise)
    print("STEP 3: Extract CWT+iPLV features (band-wise)")
    t3 = datetime.now()
    
    loader = EEGLoader()
    extractor = CWTiPLVExtractor()
    
    print("\n  Training set:")
    train_features = extract_features_from_dataset(train_data, loader, extractor, "Training")
    
    print("\n  Test set:")
    test_features = extract_features_from_dataset(test_data, loader, extractor, "Test")
    
    print(f"\n  ✓ Feature extraction complete")
    print(f"    Time: {(datetime.now()-t3).total_seconds():.1f}s\n")
    
    # Step 4: Statistical Testing + Classification (per band)
    print("STEP 4: Statistical Testing + Multi-Model Classification")
    print("="*70)
    
    all_results = []
    
    for band in extractor.band_names:
        print(f"\n{'='*70}")
        print(f" BAND: {band.upper()}")
        print(f"{'='*70}")
        
        X_train_band = train_features[band]['X']
        y_train_band = train_features[band]['y']
        X_test_band = test_features[band]['X']
        y_test_band = test_features[band]['y']
        
        if len(X_train_band) == 0 or len(X_test_band) == 0:
            print(f"  ⚠ No features for {band} band, skipping...")
            continue
        
        print(f"\n  Original features: {X_train_band.shape[1]} connections")
        print(f"  Train samples: {len(X_train_band)}, Test samples: {len(X_test_band)}")
        
        # Statistical testing
        sig_indices, p_values = perform_statistical_testing(
            X_train_band, y_train_band, band, alpha=0.05
        )
        
        if len(sig_indices) == 0:
            print(f"  ⚠ No significant connections found for {band}, skipping...")
            continue
        
        # Extract only significant features
        X_train_sig = X_train_band[:, sig_indices]
        X_test_sig = X_test_band[:, sig_indices]
        
        print(f"\n  Using {len(sig_indices)} significant connections for classification")
        print(f"\n  Training all 5 ML models...")
        
        # Train all models
        band_results = train_all_models_bandwise(X_train_sig, y_train_band, 
                                                 X_test_sig, y_test_band, band)
        
        # Store results for each model
        for model_name, metrics in band_results.items():
            all_results.append({
                'Band': band.capitalize(),
                'Model': model_name,
                'N_Connections': len(sig_indices),
                'Accuracy': f"{metrics['accuracy']:.2f}%",
                'Sensitivity': f"{metrics['sensitivity']:.2f}%",
                'Specificity': f"{metrics['specificity']:.2f}%"
            })
        
        # Print band results
        print(f"\n  {'─'*66}")
        print(f"  Results for {band.upper()} band:")
        print(f"  {'─'*66}")
        print(f"  {'Model':<18} {'Accuracy':>12} {'Sensitivity':>12} {'Specificity':>12}")
        print(f"  {'-'*66}")
        for model_name, metrics in band_results.items():
            print(f"  {model_name:<18} {metrics['accuracy']:>11.2f}% "
                  f"{metrics['sensitivity']:>11.2f}% {metrics['specificity']:>11.2f}%")
        print(f"  {'─'*66}")
    
    # Final Summary
    total = (datetime.now()-start_time).total_seconds()
    
    print("\n" + "="*70)
    print(" FINAL RESULTS: ALL MODELS × ALL BANDS")
    print("="*70 + "\n")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Display grouped by band
        print(df.to_string(index=False))
        
        # Save results
        output_file = '/mnt/user-data/outputs/cwt_iplv_all_models_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n  Results saved to: {output_file}")
        
        # Additional summary: Best model per band
        print("\n" + "="*70)
        print(" BEST MODEL PER BAND (by Accuracy)")
        print("="*70 + "\n")
        
        for band in extractor.band_names:
            band_df = df[df['Band'] == band.capitalize()]
            if len(band_df) > 0:
                # Convert accuracy string to float for comparison
                band_df_copy = band_df.copy()
                band_df_copy['Acc_Val'] = band_df_copy['Accuracy'].str.rstrip('%').astype(float)
                best_row = band_df_copy.loc[band_df_copy['Acc_Val'].idxmax()]
                print(f"  {band.upper():8} → {best_row['Model']:18} "
                      f"(Acc: {best_row['Accuracy']}, "
                      f"Sens: {best_row['Sensitivity']}, "
                      f"Spec: {best_row['Specificity']})")
        
    else:
        print("  ✗ No results to display")
    
    print(f"\n  ⏱ Total Time: {total:.1f}s ({total/60:.2f} min)")
    print(f"  ⏱ End: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

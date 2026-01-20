import os
import sys
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import pywt  # PyWavelets for CWT
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow after setting environment variables
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Suppress MNE warnings
mne.set_log_level('ERROR')

def log_time(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

log_time("Libraries loaded successfully!")
log_time(f"TensorFlow version: {tf.__version__}")
log_time(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
log_time(f"PyWavelets version: {pywt.__version__}")


def inspect_edf_files(data_path, num_samples=5):
    """Inspect EDF files to check channel names"""
    log_time("="*80)
    log_time("INSPECTING EDF FILES")
    log_time("="*80)
    
    data_path = Path(data_path)
    edf_files = list(data_path.glob('*.edf'))[:num_samples]
    
    all_channels = set()
    
    for i, filepath in enumerate(edf_files):
        log_time(f"\n[{i+1}] File: {filepath.name}")
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            channels = raw.ch_names
            print(f"    Channels ({len(channels)}): {channels}")
            all_channels.update(channels)
            print(f"    Sampling rate: {raw.info['sfreq']} Hz")
            print(f"    Duration: {raw.times[-1]:.2f} seconds")
        except Exception as e:
            print(f"    Error: {e}")
    
    log_time("="*80)
    log_time(f"UNIQUE CHANNELS FOUND ACROSS ALL FILES ({len(all_channels)}):")
    log_time("="*80)
    print(sorted(all_channels))
    
    return sorted(all_channels)


class SSWTProcessor:
    """Synchrosqueezed Wavelet Transform processor for EEG signals"""
    
    def __init__(self, fs=128, window_size=30):
        self.fs = fs
        self.window_size = window_size
        
    def downsample_signal(self, signal_data, original_fs, target_fs):
        """Downsample signal to target frequency"""
        if original_fs == target_fs:
            return signal_data
        downsample_factor = int(original_fs / target_fs)
        return signal_data[::downsample_factor]
    
    def zscore_normalize(self, signal_data):
        """Z-score normalization"""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std < 1e-8:
            return signal_data - mean
        return (signal_data - mean) / std
    
    def compute_sswt(self, signal_data):
        """
        Compute Synchrosqueezed Wavelet Transform using PyWavelets
        """
        try:
            # Use PyWavelets continuous wavelet transform
            # Scales for different frequencies (0.5-64 Hz at 128 Hz sampling)
            scales = np.arange(1, 128)
            
            # Use Morlet wavelet (similar to Bump wavelet)
            wavelet = 'morl'  # Morlet wavelet
            
            # Compute CWT
            coefficients, frequencies = pywt.cwt(
                signal_data, 
                scales, 
                wavelet,
                sampling_period=1.0/self.fs
            )
            
            # Take absolute values for magnitude (synchrosqueezing approximation)
            sswt_image = np.abs(coefficients)
            
            return sswt_image
            
        except Exception as e:
            log_time(f"Error in SSWT computation: {e}")
            # Return zeros if computation fails
            return np.zeros((127, len(signal_data)))
    
    def create_sswt_image(self, sswt_data, size=(224, 224)):
        """Convert SSWT data to RGB image for CNN input"""
        if sswt_data.size == 0:
            return np.zeros((*size, 3), dtype=np.uint8)
        
        data_min = sswt_data.min()
        data_max = sswt_data.max()
        
        if data_max - data_min < 1e-8:
            normalized = np.zeros_like(sswt_data)
        else:
            normalized = ((sswt_data - data_min) / (data_max - data_min) * 255)
        
        resized = cv2.resize(normalized.astype(np.uint8), size, 
                           interpolation=cv2.INTER_LINEAR)
        rgb_image = np.stack([resized] * 3, axis=-1)
        
        return rgb_image


class EEGDataLoader:
    """Load and preprocess EEG data from .edf files"""
    
    def __init__(self, data_path, available_channels=None):
        self.data_path = Path(data_path) if data_path else None
        self.available_channels = available_channels
        
    def load_edf_file(self, filepath):
        """Load single EDF file"""
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            return raw
        except Exception as e:
            log_time(f"Error loading {filepath.name}: {e}")
            return None
    
    def parse_filename(self, filename):
        """Parse filename to extract subject info and condition"""
        name = filename.stem
        
        # Handle different patterns
        if 'MDD' in name:
            label = 1  # Depression
            parts = name.replace('MDD', '').strip().split()
            subject_id = parts[0] if parts else 'Unknown'
            condition = parts[1] if len(parts) > 1 else 'Unknown'
        elif name.startswith('H '):
            label = 0  # Healthy
            parts = name.replace('H', '').strip().split()
            subject_id = parts[0] if parts else 'Unknown'
            condition = parts[1] if len(parts) > 1 else 'Unknown'
        else:
            # Handle files with numbers like "6921959_H S15 EO.edf"
            if '_H ' in name:
                label = 0
                parts = name.split('_H ')[1].split()
                subject_id = parts[0] if parts else 'Unknown'
                condition = parts[1] if len(parts) > 1 else 'Unknown'
            else:
                label = 0
                subject_id = 'Unknown'
                condition = 'EO'
        
        return {
            'label': label,
            'subject_id': subject_id,
            'condition': condition,
            'filename': filename.name
        }
    
    def segment_signal(self, signal_data, fs, window_sec=30, overlap=0):
        """Segment signal into non-overlapping windows"""
        window_samples = int(window_sec * fs)
        step = window_samples
        
        segments = []
        for start in range(0, len(signal_data) - window_samples + 1, step):
            segment = signal_data[start:start + window_samples]
            segments.append(segment)
        
        return segments
    
    def load_dataset(self, condition='EO'):
        """Load entire dataset for specified condition"""
        if self.data_path is None:
            raise ValueError("Data path not set")
        
        all_files = list(self.data_path.glob(f'*{condition}.edf'))
        
        log_time(f"Found {len(all_files)} files with condition '{condition}'")
        
        dataset = []
        failed = 0
        
        for filepath in all_files:
            file_info = self.parse_filename(filepath)
            
            raw = self.load_edf_file(filepath)
            if raw is None:
                failed += 1
                continue
            
            file_info['raw'] = raw
            file_info['fs'] = raw.info['sfreq']
            file_info['filepath'] = filepath
            dataset.append(file_info)
        
        log_time(f"Successfully loaded: {len(dataset)}, Failed: {failed}")
        
        return dataset


class DepressionDetectionModel:
    """ResNet-based depression detection model"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build ResNet-based model with transfer learning"""
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def compile_model(self, learning_rate=0.0008):
        """Compile the model"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_data_augmentation(self, augmentation_type):
        """Get data augmentation configuration"""
        configs = {
            'model1': {},
            'model2': {'shear_range': 20},
            'model3': {'horizontal_flip': True},
            'model4': {'vertical_flip': True},
            'model5': {'zoom_range': [0.5, 2.0]},
            'model6': {'rotation_range': 30},
            'model7': {'horizontal_flip': True, 'rotation_range': 30},
            'model8': {'horizontal_flip': True, 'zoom_range': [0.5, 2.0]},
            'model9': {}
        }
        
        return configs.get(augmentation_type, {})


def create_train_test_split(dataset, test_size=20):
    """Split dataset using leave-subject-out strategy"""
    
    subjects = {}
    for item in dataset:
        sid = item['subject_id']
        if sid not in subjects:
            subjects[sid] = []
        subjects[sid].append(item)
    
    mdd_subjects = {sid: items for sid, items in subjects.items() 
                   if items[0]['label'] == 1}
    healthy_subjects = {sid: items for sid, items in subjects.items() 
                       if items[0]['label'] == 0}
    
    log_time(f"Total MDD subjects: {len(mdd_subjects)}")
    log_time(f"Total Healthy subjects: {len(healthy_subjects)}")
    
    np.random.seed(42)
    
    mdd_sids = list(mdd_subjects.keys())
    healthy_sids = list(healthy_subjects.keys())
    
    test_mdd_sids = np.random.choice(mdd_sids, 
                                     size=min(test_size//2, len(mdd_sids)), 
                                     replace=False)
    test_healthy_sids = np.random.choice(healthy_sids, 
                                         size=min(test_size//2, len(healthy_sids)), 
                                         replace=False)
    
    test_subjects_ids = list(test_mdd_sids) + list(test_healthy_sids)
    
    train_data = []
    test_data = []
    
    for sid, items in subjects.items():
        if sid in test_subjects_ids:
            test_data.extend(items)
        else:
            train_data.extend(items)
    
    return train_data, test_data, test_subjects_ids


def find_matching_channel(target_channel, available_channels):
    """Find matching channel name in available channels"""
    target_upper = target_channel.upper()
    
    # Try exact match first
    for ch in available_channels:
        if ch.upper() == target_upper:
            return ch
    
    # Try contains match
    for ch in available_channels:
        ch_upper = ch.upper()
        # Check if target is contained in channel name
        if target_upper in ch_upper:
            return ch
        # Check if channel name is contained in target
        if ch_upper in target_upper:
            return ch
    
    # Try partial match (e.g., "FP1" in "EEG FP1-REF")
    for ch in available_channels:
        ch_cleaned = ch.upper().replace('EEG', '').replace('-', '').replace('REF', '').strip()
        if target_upper == ch_cleaned:
            return ch
    
    return None


def process_channel_data(dataset, channel_name, sswt_processor):
    """Process all data for a specific channel"""
    
    X_images = []
    y_labels = []
    subject_ids = []
    
    processed = 0
    skipped = 0
    
    for item in dataset:
        raw = item['raw']
        
        # Try to find matching channel
        matched_channel = find_matching_channel(channel_name, raw.ch_names)
        
        if matched_channel is None:
            skipped += 1
            continue
        
        # Get channel data
        channel_idx = raw.ch_names.index(matched_channel)
        signal_data = raw.get_data()[channel_idx]
        
        # Downsample
        original_fs = item['fs']
        downsampled = sswt_processor.downsample_signal(
            signal_data, original_fs, sswt_processor.fs
        )
        
        # Segment into 30-second windows
        loader = EEGDataLoader(None)
        segments = loader.segment_signal(
            downsampled, sswt_processor.fs, window_sec=30
        )
        
        # Process each segment
        for segment in segments:
            normalized = sswt_processor.zscore_normalize(segment)
            sswt_data = sswt_processor.compute_sswt(normalized)
            sswt_image = sswt_processor.create_sswt_image(sswt_data)
            
            X_images.append(sswt_image)
            y_labels.append(item['label'])
            subject_ids.append(item['subject_id'])
        
        processed += 1
    
    print(f"    Processed: {processed}, Skipped: {skipped}, Total segments: {len(X_images)}")
    
    return np.array(X_images), np.array(y_labels), subject_ids


def train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                             model_config, epochs=15, batch_size=32):
    """Train model and return test accuracy"""
    
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    detector = DepressionDetectionModel()
    detector.build_model()
    detector.compile_model()
    
    aug_config = detector.get_data_augmentation(model_config)
    datagen = ImageDataGenerator(**aug_config)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, 
                     restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                         patience=2, verbose=0)
    ]
    
    history = detector.model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=batch_size),
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    loss, accuracy = detector.model.evaluate(X_test, y_test_cat, verbose=0)
    
    return accuracy * 100


def generate_accuracy_table(data_path, quick_test=False):
    """Main function to generate Table 5"""
    
    log_time("\n" + "="*80)
    log_time("LOADING EEG DATASET")
    log_time("="*80)
    
    # First, inspect files to find available channels
    available_channels = inspect_edf_files(data_path, num_samples=5)
    
    loader = EEGDataLoader(data_path, available_channels)
    dataset = loader.load_dataset(condition='EO')
    
    if len(dataset) == 0:
        log_time("ERROR: No data loaded. Check the data path.")
        return None
    
    log_time("\n" + "="*80)
    log_time("SPLITTING DATASET")
    log_time("="*80)
    
    train_data, test_data, test_subjects = create_train_test_split(dataset)
    log_time(f"\nTrain files: {len(train_data)}")
    log_time(f"Test files: {len(test_data)}")
    log_time(f"Test subjects: {len(test_subjects)}")
    
    sswt_processor = SSWTProcessor()
    
    # Use Database I standard channels
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 
                'C3', 'C4', 'Cz', 'P4', 'P3', 'Pz', 
                'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
    
    models = ['model1', 'model2', 'model3', 'model4', 'model5', 
              'model6', 'model7', 'model8', 'model9']
    
    if quick_test:
        log_time("\n*** QUICK TEST MODE ***")
        channels = ['Fp1', 'P3', 'O1']
        models = ['model1', 'model3']
    
    results = pd.DataFrame(index=channels, columns=models, dtype=float)
    
    log_time("\n" + "="*80)
    log_time("PROCESSING CHANNELS AND TRAINING MODELS")
    log_time("="*80)
    
    for i, channel in enumerate(channels):
        log_time(f"\n[{i+1}/{len(channels)}] Processing channel: {channel}")
        log_time("-" * 40)
        
        try:
            X_train, y_train, _ = process_channel_data(
                train_data, channel, sswt_processor
            )
            
            X_test, y_test, _ = process_channel_data(
                test_data, channel, sswt_processor
            )
            
            log_time(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            if len(X_train) == 0 or len(X_test) == 0:
                log_time(f"  Skipping {channel} - insufficient data")
                continue
            
            for j, model_config in enumerate(models):
                start_time = datetime.now()
                print(f"  [{j+1}/{len(models)}] Training {model_config}...", end=' ')
                
                try:
                    accuracy = train_and_evaluate_model(
                        X_train, y_train, X_test, y_test, 
                        model_config, epochs=15, batch_size=32
                    )
                    
                    results.loc[channel, model_config] = accuracy
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"Accuracy: {accuracy:.2f}% (Time: {elapsed:.1f}s)")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    results.loc[channel, model_config] = np.nan
                    
        except Exception as e:
            log_time(f"  Error processing channel {channel}: {e}")
            continue
    
    return results


def main():
    """Main execution"""
    
    start_time = datetime.now()
    
    log_time("="*80)
    log_time("EEG DEPRESSION DETECTION - TABLE 5 GENERATION")
    log_time("="*80)
    
    data_path = '/kaggle/input/eeg-dataset/'
    
    if not Path(data_path).exists():
        log_time(f"ERROR: Data path does not exist: {data_path}")
        return
    
    # Generate table (set quick_test=True for faster testing)
    results_table = generate_accuracy_table(data_path, quick_test=False)
    
    if results_table is None:
        log_time("ERROR: Failed to generate results table")
        return
    
    output_file = 'table5_accuracy_results.csv'
    results_table.to_csv(output_file)
    
    log_time("\n" + "="*80)
    log_time("RESULTS SAVED")
    log_time("="*80)
    log_time(f"File: {output_file}")
    
    log_time("\n" + "="*80)
    log_time("FINAL ACCURACY TABLE")
    log_time("="*80)
    print(results_table.to_string())
    
    log_time("\n" + "="*80)
    log_time("STATISTICS")
    log_time("="*80)
    
    print("\nAverage accuracy per channel:")
    channel_avg = results_table.mean(axis=1, skipna=True).sort_values(ascending=False)
    print(channel_avg)
    
    print("\nAverage accuracy per model:")
    model_avg = results_table.mean(axis=0, skipna=True).sort_values(ascending=False)
    print(model_avg)
    
    overall_avg = results_table.mean(skipna=True).mean()
    log_time(f"\nOverall average accuracy: {overall_avg:.2f}%")
    
    best_val = results_table.max(skipna=True).max()
    log_time(f"Best performance: {best_val:.2f}%")
    
    total_time = (datetime.now() - start_time).total_seconds()
    log_time(f"\nTotal execution time: {total_time/60:.2f} minutes")
    

if __name__ == "__main__":
    main()

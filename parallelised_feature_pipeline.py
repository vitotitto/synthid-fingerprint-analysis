"""
Parallelised Feature Extraction Pipeline

Processes images in parallel using image-level parallelisation (not window-level).

Each worker handles one complete image, extracting features from all windows serially.

This approach minimises inter-process communication overhead whilst maintaining good CPU utilisation.

The pipeline extracts the 13 features required by the watermark classifier from 
a specified colour channel (default: LAB 'a' channel) using a configurable
window grid (default: 4x4 = 16 windows per image)

Usage:
    # Extracting features from watermarked images
    python parallelised_feature_pipeline.py \\
        --input_folder ./images/watermarked \\
        --output ./features_watermarked.csv \\
        --channel a_lab \\
        --label watermarked
    
    # Extracting features from real images
    python parallelised_feature_pipeline.py \\
        --input_folder ./images/real \\
        --output ./features_real.csv \\
        --channel a_lab \\
        --label real
"""

import cv2
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Importing feature extractor from the companion module
from feature_extractor import WindowFeatureExtractor



################################################
# MODULE-LEVEL CONFIGURATION
################################################


# Global extractor instance - created once, reused by all parallel workers
# This avoids repeated initialisation overhead within each worker process
_feature_extractor = WindowFeatureExtractor()

# Supported image file extensions (case-insensitive matching handled separately)
SUPPORTED_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']

# Valid colour channel identifiers
VALID_CHANNELS = [
    'r', 'g', 'b',           # RGB colour space
    'l_lab', 'a_lab', 'b_lab',  # CIELAB colour space
    'h_hsv', 's_hsv', 'v_hsv'   # HSV colour space
]



################################################
# CHANNEL EXTRACTION
################################################


def extract_channel(image: np.ndarray, channel: str) -> np.ndarray:
    """
    Extracting a specified colour channel from a BGR image.
    
    OpenCV loads images in BGR format by default. This function handles
    conversion to the appropriate colour space and extraction of the
    requested channel.
    
    Args:
        image: BGR image array as loaded by cv2.imread()
        channel: Channel identifier string. Valid options:
                 RGB: 'r', 'g', 'b'
                 LAB: 'l_lab', 'a_lab', 'b_lab'
                 HSV: 'h_hsv', 's_hsv', 'v_hsv'
    
    Returns:
        2D numpy array containing the extracted single-channel data
    
    Raises:
        ValueError: If channel identifier is not recognised
    """
    ################################################
    # RGB channels - direct extraction from BGR
    ################################################
    
    if channel in ['r', 'g', 'b']:
        # OpenCV stores as BGR, so indices are reversed from RGB
        # B=0, G=1, R=2 in the array
        channel_map = {'b': 0, 'g': 1, 'r': 2}
        return image[:, :, channel_map[channel]]
    
    ################################################
    # CIELAB colour space channels
    ################################################
    
    elif channel in ['l_lab', 'a_lab', 'b_lab']:
        # Converting BGR to LAB colour space
        # L: lightness (0-100), a: green-red (-128 to 127), b: blue-yellow (-128 to 127)
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channel_map = {'l_lab': 0, 'a_lab': 1, 'b_lab': 2}
        return img_lab[:, :, channel_map[channel]]
    
    ################################################
    # HSV colour space channels
    ################################################
    
    elif channel in ['h_hsv', 's_hsv', 'v_hsv']:
        # Converting BGR to HSV colour space
        # H: hue (0-179 in OpenCV), S: saturation (0-255), V: value (0-255)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_map = {'h_hsv': 0, 's_hsv': 1, 'v_hsv': 2}
        return img_hsv[:, :, channel_map[channel]]
    
    ################################################
    # Invalid channel identifier
    ################################################
    
    else:
        raise ValueError(
            f"Unknown channel: '{channel}'. "
            f"Must be one of: {', '.join(VALID_CHANNELS)}"
        )



################################################
# WINDOW PARTITIONING
################################################


def split_into_windows(
    channel_data: np.ndarray, 
    grid_size: Tuple[int, int] = (4, 4)
) -> List[np.ndarray]:
    """
    Partitioning single-channel image data into a grid of non-overlapping windows.
    
    The image is divided into a regular grid where each cell becomes one window.
    Windows are returned in row-major order (left to right, top to bottom).
    
    Note: If image dimensions are not evenly divisible by the grid size,
    the remainder pixels at the right and bottom edges are discarded.
    
    Args:
        channel_data: 2D array of single-channel image data
        grid_size: Tuple of (rows, cols) specifying grid dimensions
    
    Returns:
        List of 2D arrays, one per window, in row-major order
    """
    h, w = channel_data.shape
    rows, cols = grid_size
    
    # Computing window dimensions using integer division
    # Any remainder pixels are excluded from analysis
    win_h = h // rows
    win_w = w // cols
    
    windows = []
    
    # Iterating row by row, then column by column (row-major order)
    for i in range(rows):
        for j in range(cols):
            # Computing pixel coordinates for this window
            y_start = i * win_h
            y_end = (i + 1) * win_h
            x_start = j * win_w
            x_end = (j + 1) * win_w
            
            # Extracting window region
            window = channel_data[y_start:y_end, x_start:x_end]
            windows.append(window)
    
    return windows



################################################
# SINGLE IMAGE PROCESSING
################################################


def process_single_image(
    image_path: Path,
    channel: str = 'a_lab',
    grid_size: Tuple[int, int] = (4, 4),
    label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processing a single image and extracting features from all windows.
    
    This function is designed to be called by parallel workers, with each
    worker handling one complete image. Windows within the image are
    processed serially to avoid nested parallelisation overhead.
    
    Args:
        image_path: Path to the image file
        channel: Colour channel to extract for feature analysis
        grid_size: Tuple of (rows, cols) for the window grid
        label: Optional class label string (e.g., 'watermarked', 'real')
    
    Returns:
        Dictionary containing:
            - image_id: Filename stem (without extension)
            - filename: Full filename
            - channel: Channel used for extraction
            - grid_size: Grid dimensions used
            - image_shape: Original image dimensions (H, W, C)
            - n_windows: Number of windows extracted
            - windows: List of feature dictionaries, one per window
            - label: Class label (if provided)
    
    Raises:
        ValueError: If image cannot be loaded
    """
    ################################################
    # Loading the image from disk
    ################################################
    
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    ################################################
    # Extracting the specified colour channel
    ################################################
    
    channel_data = extract_channel(img, channel)
    
    ################################################
    # Partitioning into analysis windows
    ################################################
    
    windows = split_into_windows(channel_data, grid_size)
    
    ################################################
    # Extracting features from each window
    ################################################
    
    rows, cols = grid_size
    window_features = []
    
    for window_id, window in enumerate(windows):
        # Computing window grid position from linear index
        # window_id = row * cols + col, so:
        window_row = window_id // cols
        window_col = window_id % cols
        
        # Extracting all 13 features using the feature extractor
        feats = _feature_extractor.extract_all(window)
        
        # Adding window position metadata
        feats['window_id'] = window_id
        feats['window_row'] = window_row
        feats['window_col'] = window_col
        
        window_features.append(feats)
    
    ################################################
    # Assembling the result dictionary
    ################################################
    
    result = {
        'image_id': image_path.stem,
        'filename': image_path.name,
        'channel': channel,
        'grid_size': grid_size,
        'image_shape': img.shape,
        'n_windows': len(window_features),
        'windows': window_features
    }
    
    # Adding label if provided
    if label is not None:
        result['label'] = label
    
    return result


def _process_single_image_safe(
    image_path: Path,
    channel: str,
    grid_size: Tuple[int, int],
    label: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Error-handling wrapper for process_single_image.
    
    This wrapper ensures that a single failed image does not crash the
    entire parallel batch. Errors are logged to stdout and the function
    returns None for failed images.
    
    Args:
        image_path: Path to the image file
        channel: Colour channel to extract
        grid_size: Window grid dimensions
        label: Optional class label
    
    Returns:
        Feature dictionary if successful, None if processing failed
    """
    try:
        return process_single_image(image_path, channel, grid_size, label)
    except Exception as e:
        print(f"\n  ERROR processing {image_path.name}: {e}")
        return None



################################################
# BATCH PROCESSING (PARALLEL)
################################################


def process_batch(
    input_folder: Union[str, Path],
    output_path: Union[str, Path],
    channel: str = 'a_lab',
    grid_size: Tuple[int, int] = (4, 4),
    label: Optional[str] = None,
    n_jobs: int = -1,
    output_format: str = 'both',
    verbose: bool = True
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Processing multiple images in parallel with image-level parallelisation.
    
    Each image is assigned to a separate worker process. Within each worker,
    windows are processed serially. This balances parallelisation overhead
    against CPU utilisation.
    
    Args:
        input_folder: Directory containing input images
        output_path: Base path for output files (extension added automatically)
        channel: Colour channel to extract features from
        grid_size: Tuple of (rows, cols) for window grid
        label: Optional class label for all images in this batch
        n_jobs: Number of parallel workers (-1 = all available cores)
        output_format: Output format - 'csv', 'json', or 'both'
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (csv_path, json_path) - paths are None if format not requested
    
    Raises:
        ValueError: If no images found in input folder
        RuntimeError: If all images fail to process
    """
    input_folder = Path(input_folder)
    output_path = Path(output_path)
    
    ################################################
    # Discovering image files
    ################################################
    
    image_paths = []
    
    # Collecting paths for each supported extension
    # Both lowercase and uppercase variants are checked
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(input_folder.glob(ext))
        image_paths.extend(input_folder.glob(ext.upper()))
    
    # Removing duplicates (Windows filesystem is case-insensitive)
    # Converting to set and back preserves uniqueness
    image_paths = list(set(image_paths))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        raise ValueError(f"No images found in {input_folder}")
    
    ################################################
    # Printing pipeline configuration
    ################################################
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"PARALLELISED FEATURE EXTRACTION PIPELINE")
        print(f"{'#'*60}")
        print(f"Input folder:  {input_folder}")
        print(f"Images found:  {len(image_paths)}")
        print(f"Channel:       {channel}")
        print(f"Grid size:     {grid_size[0]} x {grid_size[1]} = {grid_size[0]*grid_size[1]} windows/image")
        print(f"Label:         {label if label else 'None'}")
        print(f"Workers:       {n_jobs if n_jobs > 0 else 'All cores'}")
        print(f"Output format: {output_format}")
        print(f"{'#'*60}\n")
    
    ################################################
    # Parallel processing of images
    ################################################
    
    if verbose:
        print(f"Processing {len(image_paths)} images in parallel...")
    
    # Using joblib for process-based parallelisation
    # 'loky' backend provides robust process isolation
    # 'pre_dispatch' limits memory usage by not pre-loading all tasks
    results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=0,
        pre_dispatch='2*n_jobs'
    )(
        delayed(_process_single_image_safe)(img_path, channel, grid_size, label)
        for img_path in tqdm(
            image_paths, 
            desc=f"Extracting features ({channel})",
            disable=not verbose
        )
    )
    
    ################################################
    # Filtering failed images and reporting statistics
    ################################################
    
    # Removing None entries (failed images)
    results = [r for r in results if r is not None]
    
    if not results:
        raise RuntimeError("All images failed to process!")
    
    successful = len(results)
    failed = len(image_paths) - successful
    
    if verbose:
        print(f"\nSuccessfully processed: {successful}/{len(image_paths)} images")
        if failed > 0:
            print(f"Failed: {failed} images")
        
        # Computing extraction statistics
        total_windows = sum(r['n_windows'] for r in results)
        print(f"Total windows extracted: {total_windows}")
        print(f"Average windows/image: {total_windows/successful:.1f}")
    
    ################################################
    # Saving results to disk
    ################################################
    
    csv_path = None
    json_path = None
    
    if output_format in ['csv', 'both']:
        csv_path = _save_csv(results, output_path, channel)
        if verbose:
            print(f"\nCSV saved: {csv_path}")
    
    if output_format in ['json', 'both']:
        json_path = _save_json(results, output_path, channel)
        if verbose:
            print(f"JSON saved: {json_path}")
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'#'*60}\n")
    
    return csv_path, json_path



################################################
# CONVENIENCE API FUNCTION
################################################


def extract_features_to_csv(
    input_folder: Union[str, Path],
    output_path: Union[str, Path],
    channel: str = 'a_lab',
    label_name: Optional[str] = None,
    grid_rows: int = 4,
    grid_cols: int = 4,
    n_jobs: int = -1
) -> Path:
    """
    Simplified API for extracting features to a CSV file.
    
    This function provides a streamlined interface for the common use case
    of extracting features from a folder of images and saving to CSV format.
    It is the primary entry point for integration with the classifier module.
    
    Args:
        input_folder: Directory containing input images
        output_path: Output CSV file path
        channel: Colour channel for feature extraction (default: 'a_lab')
        label_name: Class label string (e.g., 'watermarked', 'real')
        grid_rows: Number of rows in the window grid
        grid_cols: Number of columns in the window grid
        n_jobs: Number of parallel workers (-1 = all cores)
    
    Returns:
        Path to the saved CSV file
    
    Example:
        >>> extract_features_to_csv(
        ...     input_folder='./images/watermarked',
        ...     output_path='./features.csv',
        ...     channel='a_lab',
        ...     label_name='watermarked',
        ...     grid_rows=4,
        ...     grid_cols=4
        ... )
    """
    csv_path, _ = process_batch(
        input_folder=input_folder,
        output_path=output_path,
        channel=channel,
        grid_size=(grid_rows, grid_cols),
        label=label_name,
        n_jobs=n_jobs,
        output_format='csv',
        verbose=True
    )
    
    return csv_path



################################################
# OUTPUT WRITERS
################################################


def _save_csv(
    results: List[Dict], 
    output_path: Path, 
    channel: str
) -> Path:
    """
    Saving extraction results as a flat CSV file.
    
    The output format has one row per window, with image metadata
    repeated for each window belonging to the same image. This format
    is suitable for direct loading into the classifier.
    
    Args:
        results: List of per-image result dictionaries
        output_path: Base output path (extension corrected if needed)
        channel: Channel identifier for metadata column
    
    Returns:
        Path to the saved CSV file
    """
    ################################################
    # Ensuring correct file extension
    ################################################
    
    if output_path.suffix != '.csv':
        output_path = output_path.with_suffix('.csv')
    
    ################################################
    # Flattening hierarchical results to per-window rows
    ################################################
    
    rows = []
    
    for result in results:
        image_id = result['image_id']
        filename = result['filename']
        label = result.get('label', None)
        
        for window in result['windows']:
            # Starting with image-level metadata
            row = {
                'filename': filename,
                'image_id': image_id,
                'channel': channel,
            }
            
            # Adding class label if present
            if label is not None:
                row['label'] = label
            
            # Adding window position metadata
            row['window_id'] = window['window_id']
            row['window_row'] = window['window_row']
            row['window_col'] = window['window_col']
            
            # Adding all extracted features
            # Skipping metadata keys already handled above
            for key, value in window.items():
                if key not in ['window_id', 'window_row', 'window_col']:
                    row[key] = value
            
            rows.append(row)
    
    ################################################
    # Creating DataFrame and sanitising values
    ################################################
    
    df = pd.DataFrame(rows)
    
    # Identifying feature columns (excluding metadata)
    metadata_cols = [
        'filename', 'image_id', 'channel', 'label', 
        'window_id', 'window_row', 'window_col'
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Replacing infinity and NaN values with 0.0
    # These typically arise from degenerate windows (flat regions)
    df[feature_cols] = df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    
    ################################################
    # Reordering columns for readability
    ################################################
    
    # Metadata columns first, then features
    ordered_metadata = ['filename', 'image_id', 'channel']
    if 'label' in df.columns:
        ordered_metadata.append('label')
    ordered_metadata.extend(['window_id', 'window_row', 'window_col'])
    
    remaining_cols = [c for c in df.columns if c not in ordered_metadata]
    df = df[ordered_metadata + remaining_cols]
    
    ################################################
    # Writing to disk
    ################################################
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return output_path


def _save_json(
    results: List[Dict], 
    output_path: Path, 
    channel: str
) -> Path:
    """
    Saving extraction results as a hierarchical JSON file.
    
    The output format preserves the image-window hierarchy, making it
    suitable for inspection and debugging. Each image contains a nested
    list of its window features.
    
    Args:
        results: List of per-image result dictionaries
        output_path: Base output path (extension corrected if needed)
        channel: Channel identifier for metadata
    
    Returns:
        Path to the saved JSON file
    """
    ################################################
    # Ensuring correct file extension
    ################################################
    
    if output_path.suffix == '.csv':
        output_path = output_path.with_suffix('.json')
    elif output_path.suffix != '.json':
        output_path = Path(str(output_path) + '.json')
    
    ################################################
    # Assembling output structure with metadata
    ################################################
    
    timestamp = datetime.now().isoformat()
    grid_size = results[0]['grid_size'] if results else (4, 4)
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'channel': channel,
            'grid_size': list(grid_size),
            'n_images': len(results),
            'total_windows': sum(r['n_windows'] for r in results),
            'label': results[0].get('label') if results and 'label' in results[0] else None
        },
        'images': []
    }
    
    ################################################
    # Adding per-image results
    ################################################
    
    for result in results:
        image_data = {
            'image_id': result['image_id'],
            'filename': result['filename'],
            'image_shape': list(result['image_shape']),
            'n_windows': result['n_windows'],
            'windows': result['windows']
        }
        
        if 'label' in result:
            image_data['label'] = result['label']
        
        output_data['images'].append(image_data)
    
    ################################################
    # Custom JSON encoder for NumPy types
    ################################################
    
    class NumpyEncoder(json.JSONEncoder):
        """JSON encoder that handles NumPy data types."""
        
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    ################################################
    # Writing to disk
    ################################################
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    return output_path



################################################
# COMMAND LINE INTERFACE
################################################


def main():
    """
    Command-line entry point for the feature extraction pipeline.
    
    Parses command-line arguments and invokes the batch processing function.
    """
    parser = argparse.ArgumentParser(
        description='Parallelised Feature Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extracting features from watermarked images
  python parallelised_feature_pipeline.py \\
      --input_folder ./images/watermarked \\
      --output ./features_watermarked.csv \\
      --channel a_lab \\
      --label watermarked

  # Extracting features from real images with 8 workers
  python parallelised_feature_pipeline.py \\
      --input_folder ./images/real \\
      --output ./features_real.csv \\
      --channel a_lab \\
      --label real \\
      --n_jobs 8

  # Extracting features with custom grid size
  python parallelised_feature_pipeline.py \\
      --input_folder ./images \\
      --output ./features.csv \\
      --channel b_lab \\
      --grid_rows 8 \\
      --grid_cols 8

Available channels:
  RGB:  r, g, b
  LAB:  l_lab, a_lab, b_lab
  HSV:  h_hsv, s_hsv, v_hsv
        """
    )
    
    ################################################
    # Required arguments
    ################################################
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (base name, extensions added automatically)'
    )
    
    ################################################
    # Feature extraction options
    ################################################
    
    parser.add_argument(
        '--channel',
        type=str,
        default='a_lab',
        choices=VALID_CHANNELS,
        help='Colour channel to extract features from (default: a_lab)'
    )
    
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='Class label to add to all images (e.g., "watermarked", "real")'
    )
    
    parser.add_argument(
        '--grid_rows',
        type=int,
        default=4,
        help='Number of rows in window grid (default: 4)'
    )
    
    parser.add_argument(
        '--grid_cols',
        type=int,
        default=4,
        help='Number of columns in window grid (default: 4)'
    )
    
    ################################################
    # Processing options
    ################################################
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 = all cores, 1 = serial)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['csv', 'json', 'both'],
        help='Output format (default: both)'
    )
    
    args = parser.parse_args()
    
    ################################################
    # Executing the pipeline
    ################################################
    
    try:
        csv_path, json_path = process_batch(
            input_folder=args.input_folder,
            output_path=args.output,
            channel=args.channel,
            grid_size=(args.grid_rows, args.grid_cols),
            label=args.label,
            n_jobs=args.n_jobs,
            output_format=args.format
        )
        
        print(f"Success! Features extracted and saved.")
        return 0
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

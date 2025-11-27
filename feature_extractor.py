
"""
Unified Window Feature Extractor (Streamlined)

Extracts the 13 features required by the watermark classifier from a single image window.
Features are grouped into four categories:
    - Lattice features (6): frequency domain and autocorrelation patterns
    - Angular features (3): directional energy distribution via wavelet decomposition
    - Diagonal stripe feature (1): autocorrelation along diagonal direction
    - Quality metrics (3): confidence scores and signal quality measures checks

The extraction pipeline:
    1. Preprocessing: normalisation and wavelet decomposition
    2. Lattice analysis: FFT-based frequency detection and autocorrelation
    3. Angular analysis: wavelet detail coefficients -> angular histogram
    4. Diagonal analysis: oriented autocorrelation
    5. Quality metrics: SNR, confidence scores, spatial coherence
"""

import numpy as np
from scipy import fft
from scipy import stats
from scipy.signal import find_peaks
import pywt
import warnings

warnings.filterwarnings('ignore')



################################################
# FEATURE LIST (13 features used by classifier)
################################################


REQUIRED_FEATURES = [
    # Lattice features (6)
    'lat_diag_total_frac',                    # diagonal energy fraction in frequency domain
    'lat_axis_total_frac',                    # axis-aligned energy fraction in frequency domain
    'lat_line_axis_density',                  # density of harmonic peaks along axes
    'lat_diag_over_axis_total',               # ratio of diagonal to axis energy
    'lat_ac_var_axis_over_diag',              # autocorrelation variance ratio
    'lat_p2_fraction_of_smallperiod_energy',  # period-2 energy as fraction of small periods
    
    # Angular features (3)
    'ang_entropy',                            # entropy of angular energy histogram
    'ang_spectral_flatness',                  # geometric/arithmetic mean ratio
    'ang_dip_contrast',                       # contrast at angular dips
    
    # Diagonal stripe feature (1)
    'ds_autocorr_diagonal',                   # autocorrelation along diagonal direction
    
    # Quality metrics (3)
    'cb_confidence',                          # overall checkerboard pattern confidence
    'spatial_coherence',                      # spatial pattern regularity
    'snr',                                    # signal-to-noise ratio in dB
]



################################################
# MAIN EXTRACTOR CLASS
################################################


class WindowFeatureExtractor:
    """
    Unified feature extraction for a single image window
    
    All features are computed in a single pass with shared preprocessing
    to avoid redundant calculations
    """
    
    def __init__(self, config: dict = None):
        """
        Initialising feature extractor with configuration
        
        Args:
            config: Optional dictionary with extraction parameters
                    If None, default configuration is used
        """
        self.config = config or self._default_config()
    
    
    def extract_all(self, window: np.ndarray) -> dict:
        """
        Main entry point: extracting all 13 features from a window
        
        Args:
            window: 2D numpy array (grayscale channel, typically LAB 'a' channel)
        
        Returns:
            Dictionary with all 13 features required by the classifier.
        """
        #Computing all shared/preprocessed data once
        prep = self._preprocess_window(window)
        
        #Extracting each feature family
        features = {}
        
        # Lattice features (frequency domain + autocorrelation)
        lattice_feats, autocorr, regularity_score = self._extract_lattice_features(prep)
        features.update(lattice_feats)
        
        # Angular features (wavelet-based directional analysis)
        features.update(self._extract_angular_features(prep))
        
        # Diagonal stripe features (oriented autocorrelation)
        diagonal_feats = self._extract_diagonal_features(prep)
        features.update(diagonal_feats)
        
        # Quality metrics (SNR, confidence, coherence)
        features.update(self._extract_quality_metrics(
            prep, lattice_feats, diagonal_feats, regularity_score
        ))
        
        return features
    
    
    
    ################################################
    # PREPROCESSING
    ################################################
    
    
    def _preprocess_window(self, window: np.ndarray) -> dict:
        """
        Computing all shared transforms and data structures once
        Each feature extraction function accesses what it needs
        
        Args:
            window: Raw window data (2D numpy array)
        
        Returns:
            Dictionary containing:
                - raw: original window
                - normalized: zero-mean unit-variance window
                - shape: (height, width)
                - wavelet_detail: medium-frequency detail coefficients


        """

        # validation if input is 2D (grayscale/single channel)
        if window.ndim != 2:
             raise ValueError(f"Feature extractor expects 2D array, got {window.ndim}D"
                              "Be sure you are passing a single channel (e.g., 'a' from LAB)")

        h, w = window.shape
        
        # Normalising to zero-mean, unit-variance
        # This is required for autocorrelation-based features
        mean_val = window.mean()
        std_val = window.std() + 1e-6  # small epsilon to avoid division by zero
        normalized = (window - mean_val) / std_val
        
        # Wavelet decomposition for angular features
        # Extracts medium-frequency directional information
        wavelet_detail = self._compute_wavelet_detail(window)
        
        return {
            'raw': window,
            'normalized': normalized,
            'shape': (h, w),
            'wavelet_detail': wavelet_detail,
        }
    
    
    def _compute_wavelet_detail(self, window: np.ndarray) -> np.ndarray:
        """
        Computing wavelet detail coefficients for angular analysis
        
        Uses Daubechies-4 wavelet at level 3 to extract medium-frequency
        directional information. The detail coefficients (cH, cV, cD)
        capture horizontal, vertical, and diagonal edges respectively
        
        Args:
            window: Raw window data
        
        Returns:
            Combined detail coefficients as sqrt(cH^2 + cV^2 + cD^2),
            or None if window is too small for wavelet analysis
        """
        # Minimum size requirement for 3-level wavelet decomposition
        # Each level halves the resolution, so need at least 32x32
        if window.shape[0] < 32 or window.shape[1] < 32:
            warnings.warn(
                f"Window size {window.shape} is too small for angular feature extraction "
                "(min 32x32 required). Angular features will be zeroed",
                UserWarning
            )
            return None
        
        try:
            # Performing 2D discrete wavelet transform
            # Level 3 decomposition gives coefficients at 1/8 resolution
            coeffs = pywt.wavedec2(
                window.astype(np.float32),
                wavelet=self.config['angular_wavelet'],
                level=self.config['angular_level']
            )
            
            # Extracting detail coefficients from finest level
            # cH: horizontal details (vertical edges)
            # cV: vertical details (horizontal edges)
            # cD: diagonal details (diagonal edges)
            cH, cV, cD = coeffs[-1]
            
            # Combining into single magnitude image
            # This represents total edge energy at each location
            combined_detail = np.sqrt(cH**2 + cV**2 + cD**2)
            
            return combined_detail
            
        except Exception:
            return None
    
    
    
    ################################################
    # LATTICE FEATURES (6 features)
    ################################################
    
    
    def _extract_lattice_features(self, prep: dict) -> tuple:
        """
        Extracting frequency-domain and autocorrelation lattice features.
        
        This function computes 6 features:
            - lat_diag_total_frac: total diagonal energy in frequency domain
            - lat_axis_total_frac: total axis-aligned energy in frequency domain
            - lat_diag_over_axis_total: ratio of diagonal to axis energy
            - lat_p2_fraction_of_smallperiod_energy: period-2 dominance
            - lat_line_axis_density: harmonic peak density along axes
            - lat_ac_var_axis_over_diag: autocorrelation variance ratio
        
        Args:
            prep: Preprocessed data dictionary
        
        Returns:
            Tuple of (features_dict, autocorr_array, regularity_score)
            The autocorr and regularity_score are passed to quality metrics.
        """
        window = prep['raw']
        normalized = prep['normalized']
        
        # Frequency-domain features (4 features)
        freq_feats = self._lattice_freq_features(window)
        
        # Autocorrelation features (2 features + regularity score)
        ac_feats, autocorr, regularity_score = self._lattice_autocorr_features(normalized)
        
        # Combining all lattice features
        features = {}
        features.update(freq_feats)
        features.update(ac_feats)
        
        return features, autocorr, regularity_score
    
    
    def _lattice_freq_features(self, window: np.ndarray) -> dict:
        """
        Computing frequency-domain lattice detection features.
        
        Analyses energy distribution at different periods (2-5 pixels)
        along axis-aligned and diagonal directions in Fourier space.
        
        The key insight is that checkerboard patterns produce strong
        energy at period-2 along diagonal directions, whilst other
        periodic patterns show different signatures.
        
        Args:
            window: Raw window data
        
        Returns:
            Dictionary with 4 frequency-domain features.
        """
        win = np.asarray(window, dtype=np.float32)
        h, w = win.shape
        
        # Periods to analyse (in pixels)
        periods = self.config['lattice_periods']
        
        # Angular tolerance for direction classification
        # +/- 15 degrees from target direction
        angle_tolerance = np.pi / 12
        
        # Radial tolerance as fraction of target radius
        radius_tolerance_factor = 0.25
        
        ################################################
        # Computing 2D FFT and shifting zero-frequency to centre
        ################################################
        
        F = fft.fft2(win)
        Fshift = fft.fftshift(F)
        
        # Power spectrum (magnitude squared)
        mag2 = np.abs(Fshift) ** 2
        
        # Centre coordinates in frequency space
        cy, cx = h // 2, w // 2
        
        # Creating coordinate grids for polar conversion
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        
        # Radius from centre (frequency magnitude)
        r = np.sqrt(x ** 2 + y ** 2)
        
        # Angle from centre (frequency direction)
        theta = np.arctan2(y, x)
        
        # Removing DC component and very low frequencies
        # These don't contain pattern information
        low_r = min(h, w) * 0.02
        mag2[r < low_r] = 0.0
        
        # Total energy for normalisation
        total_energy = float(mag2.sum() + 1e-12)
        
        ################################################
        # Defining direction angles
        ################################################
        
        # Axis-aligned directions: 0, 90, 180, -90 degrees
        axis_angles = [0, np.pi / 2, np.pi, -np.pi / 2]
        
        # Diagonal directions: 45, 135, -135, -45 degrees
        diag_angles = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        
        def band_energy(target_period, angles):
            """
            Computing energy in frequency band at given period and directions.
            
            Args:
                target_period: Target period in pixels
                angles: List of target angles in radians
            
            Returns:
                Tuple of (total_energy, list_of_per_angle_energies)
            """
            energies = []
            
            # Target radius in frequency space
            # r = N / period, where N is image dimension
            r0 = min(h, w) / float(target_period)
            radius_tol = r0 * radius_tolerance_factor
            
            # Radial mask: ring at target frequency
            radial_mask = np.abs(r - r0) < radius_tol
            
            if not np.any(radial_mask):
                return 0.0, []
            
            for a in angles:
                # Angular mask: wedge at target direction
                # Using complex exponential to handle angle wrapping
                ang_diff = np.abs(np.angle(np.exp(1j * (theta - a))))
                ang_mask = ang_diff < angle_tolerance
                
                # Combined mask: intersection of ring and wedge
                mask = radial_mask & ang_mask
                
                energies.append(float(mag2[mask].sum()))
            
            return sum(energies), energies
        
        ################################################
        # Computing energy at each period
        ################################################
        
        all_axis = []
        all_diag = []
        period_energies = {}
        
        for p in periods:
            e_axis, lobes_axis = band_energy(p, axis_angles)
            e_diag, lobes_diag = band_energy(p, diag_angles)
            
            # Storing per-period energies for period-2 fraction calculation
            period_energies[p] = (e_axis + e_diag) / total_energy
            
            all_axis.extend(lobes_axis)
            all_diag.extend(lobes_diag)
        
        ################################################
        # Computing final features
        ################################################
        
        feats = {}
        
        # Total axis-aligned energy fraction
        feats['lat_axis_total_frac'] = sum(all_axis) / total_energy
        
        # Total diagonal energy fraction
        feats['lat_diag_total_frac'] = sum(all_diag) / total_energy
        
        # Ratio of diagonal to axis energy
        # High values indicate diagonal-dominant patterns (checkerboard)
        if sum(all_axis) > 0:
            feats['lat_diag_over_axis_total'] = sum(all_diag) / (sum(all_axis) + 1e-12)
        else:
            feats['lat_diag_over_axis_total'] = 0.0
        
        # Period-2 dominance: fraction of small-period energy at period 2
        # Checkerboards have strong period-2 signature
        p2_energy = period_energies[2]
        all_period_energy = sum(period_energies.values()) + 1e-12
        feats['lat_p2_fraction_of_smallperiod_energy'] = float(p2_energy / all_period_energy)
        
        return {k: float(v) for k, v in feats.items()}
    
    
    def _lattice_autocorr_features(self, normalized_window: np.ndarray) -> tuple:
        """
        Computing autocorrelation-based lattice features.
        
        Autocorrelation reveals periodic structure by measuring
        self-similarity at different spatial lags. Regular patterns
        produce peaks at their fundamental period.
        
        Args:
            normalized_window: Zero-mean unit-variance window
        
        Returns:
            Tuple of (features_dict, autocorr_array, regularity_score)
        """
        win = np.asarray(normalized_window, dtype=np.float32)
        max_lag = self.config['lattice_max_lag']
        
        ################################################
        # Computing autocorrelation via FFT (Wiener-Khinchin theorem)
        # autocorr = IFFT(|FFT(signal)|^2)
        ################################################
        
        F = fft.fft2(win)
        power = np.abs(F) ** 2
        ac = fft.ifft2(power)
        ac = fft.fftshift(np.real(ac))
        
        # Normalising so that ac[centre] = 1
        cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
        centre_val = float(ac[cy, cx] + 1e-12)
        ac /= centre_val
        
        # Limiting max_lag to available range
        max_lag = min(max_lag, cy - 1, cx - 1)
        
        ################################################
        # Extracting 1D profiles along cardinal directions
        ################################################
        
        # Horizontal profile (averaging left and right)
        profile_right = ac[cy, cx:cx + max_lag + 1]
        profile_left = ac[cy, cx - max_lag:cx + 1][::-1]
        profile_horizontal = (profile_right + profile_left) / 2
        
        ################################################
        # Computing harmonic peak density along axis
        # This measures how regularly spaced the autocorr peaks are
        ################################################
        
        axis_prof = profile_horizontal
        
        # Normalising profile for peak detection
        axis_norm = axis_prof / (np.max(np.abs(axis_prof)) + 1e-12)
        
        # Finding peaks with minimum prominence
        peaks_axis, _ = find_peaks(axis_norm, prominence=0.05)
        
        # Peak density: number of peaks per unit length
        lat_line_axis_density = len(peaks_axis) / max(1, len(axis_norm))
        
        ################################################
        # Computing directional variance features
        # High variance in a direction indicates strong periodic structure
        ################################################
        
        def directional_variance(direction='axis', max_radius=16):
            """
            Computing variance of autocorrelation values along a direction.
            
            Args:
                direction: 'axis' for horizontal/vertical, 'diagonal' for 45-degree
                max_radius: Maximum distance from centre to sample
            
            Returns:
                Variance of sampled autocorrelation values
            """
            values = []
            max_radius = min(max_radius, cy, cx)
            
            if direction == 'axis':
                # Sampling along horizontal and vertical axes
                for offset in range(1, max_radius):
                    if cx + offset < ac.shape[1]:
                        values.append(abs(ac[cy, cx + offset]))
                    if cx - offset >= 0:
                        values.append(abs(ac[cy, cx - offset]))
                    if cy + offset < ac.shape[0]:
                        values.append(abs(ac[cy + offset, cx]))
                    if cy - offset >= 0:
                        values.append(abs(ac[cy - offset, cx]))
            else:
                # Sampling along diagonal directions
                for offset in range(1, max_radius):
                    if cy + offset < ac.shape[0] and cx + offset < ac.shape[1]:
                        values.append(abs(ac[cy + offset, cx + offset]))
                    if cy - offset >= 0 and cx - offset >= 0:
                        values.append(abs(ac[cy - offset, cx - offset]))
                    if cy + offset < ac.shape[0] and cx - offset >= 0:
                        values.append(abs(ac[cy + offset, cx - offset]))
                    if cy - offset >= 0 and cx + offset < ac.shape[1]:
                        values.append(abs(ac[cy - offset, cx + offset]))
            
            return np.var(values) if values else 0.0
        
        axis_var = directional_variance('axis', max_radius=16)
        diag_var = directional_variance('diagonal', max_radius=16)
        
        # Ratio of axis to diagonal variance
        # High values indicate axis-aligned patterns
        lat_ac_var_axis_over_diag = float(axis_var / (diag_var + 1e-12))
        
        ################################################
        # Computing regularity score for quality metrics
        # This measures overall pattern regularity
        ################################################
        
        # Peak count normalised to expected range
        peak_count_norm = min(len(peaks_axis) / 5.0, 1.0)
        
        # Peak density already computed
        peak_density = lat_line_axis_density
        
        # Spacing consistency: low variance in peak spacings indicates regularity
        if len(peaks_axis) > 1:
            spacings = np.diff(peaks_axis)
            spacing_consistency = 1.0 - min(np.std(spacings) / (np.mean(spacings) + 1e-12), 1.0)
        else:
            spacing_consistency = 0.0
        
        # Peak strength: mean height of detected peaks
        if len(peaks_axis) > 0:
            peak_strength = float(np.mean(np.abs(axis_norm[peaks_axis])))
        else:
            peak_strength = 0.0
        
        # Weighted combination of regularity indicators
        regularity_score = (
            0.30 * peak_count_norm +
            0.25 * peak_density +
            0.25 * spacing_consistency +
            0.20 * peak_strength
        )
        regularity_score = float(np.clip(regularity_score, 0, 1))
        
        ################################################
        # Assembling output features
        ################################################
        
        feats = {
            'lat_line_axis_density': float(lat_line_axis_density),
            'lat_ac_var_axis_over_diag': float(lat_ac_var_axis_over_diag),
        }
        
        return feats, ac, regularity_score
    
    
    ################################################
    ################################################
    # ANGULAR FEATURES (3 features)
    ################################################
    ################################################
    
    def _extract_angular_features(self, prep: dict) -> dict:
        """
        Extracting angular spectrum features via wavelet decomposition.
        
        This function computes 3 features:
            - ang_entropy: entropy of angular energy distribution
            - ang_spectral_flatness: ratio of geometric to arithmetic mean
            - ang_dip_contrast: contrast at angular dips
        
        These features measure directional energy distribution at
        medium frequencies, capturing orientation preferences in
        the image texture.
        
        Args:
            prep: Preprocessed data dictionary
        
        Returns:
            Dictionary with 3 angular features.
        """
        wavelet_detail = prep['wavelet_detail']
        
        # Returning zeros if window too small for wavelet analysis
        if wavelet_detail is None:
            return self._angular_zero_features()
        
        # Computing angular histogram from wavelet detail coefficients
        hist, angles = self._compute_angular_histogram(wavelet_detail)
        
        if hist is None:
            return self._angular_zero_features()
        
        # Extracting scalar features from histogram
        return self._angular_histogram_features(hist, angles)
    
    
    def _compute_angular_histogram(self, wavelet_detail: np.ndarray) -> tuple:
        """
        Computing angular energy histogram from wavelet detail coefficients.
        
        Performs FFT of wavelet detail, then bins energy by direction
        to create a histogram of directional energy.
        
        Args:
            wavelet_detail: Combined wavelet detail coefficients
        
        Returns:
            Tuple of (normalised_histogram, angle_centres) or (None, None) if failed.
        """
        freq_band = self.config['angular_freq_band']
        
        ################################################
        # Computing FFT of wavelet detail
        ################################################
        
        fft_result = fft.fft2(wavelet_detail)
        fft_shift = fft.fftshift(fft_result)
        magnitude = np.abs(fft_shift)
        
        h, w = wavelet_detail.shape
        
        # Creating frequency coordinate grids
        fy = fft.fftshift(fft.fftfreq(h, d=1.0))
        fx = fft.fftshift(fft.fftfreq(w, d=1.0))
        FX, FY = np.meshgrid(fx, fy)
        
        # Frequency magnitude (distance from DC)
        freq_magnitude = np.sqrt(FX**2 + FY**2)
        
        ################################################
        # Creating bandpass mask for medium frequencies
        ################################################
        
        freq_min, freq_max = freq_band[0] * 2, freq_band[1] * 2
        band_mask = (freq_magnitude >= freq_min) & (freq_magnitude <= freq_max)
        
        # Fallback if mask is too restrictive
        if not band_mask.any():
            band_mask = (freq_magnitude > 0)
        
        ################################################
        # Computing angles and creating weighted histogram
        ################################################
        
        # Angle at each frequency point
        angles = np.arctan2(FY, FX)
        
        # Converting to degrees, wrapping to [0, 180) for symmetry
        # (FFT is symmetric, so opposite directions are equivalent)
        angles_deg = np.rad2deg(angles) % 180
        
        # Extracting values within frequency band
        band_magnitudes = magnitude[band_mask]
        band_angles = angles_deg[band_mask]
        
        # Creating weighted histogram (90 bins for 2-degree resolution)
        angle_hist, angle_bins = np.histogram(
            band_angles,
            bins=90,
            range=(0, 180),
            weights=band_magnitudes
        )
        
        # Bin centres for feature extraction
        angle_centres = (angle_bins[:-1] + angle_bins[1:]) / 2
        
        # Normalising histogram to sum to 1
        angle_hist_norm = angle_hist / (angle_hist.sum() + 1e-10)
        
        return angle_hist_norm, angle_centres
    
    
    def _angular_histogram_features(self, hist: np.ndarray, angles: np.ndarray) -> dict:
        """
        Extracting scalar features from angular histogram.
        
        Args:
            hist: Normalised angular histogram (90 bins)
            angles: Bin centre angles in degrees
        
        Returns:
            Dictionary with 3 angular features.
        """
        features = {}
        
        ################################################
        # Computing dip contrast
        # Measures energy ratio between diagonal and intermediate angles
        ################################################
        
        def get_energy(target, window=5):
            """Getting total histogram energy within window of target angle."""
            mask = np.abs(angles - target) < window
            return hist[mask].sum() if mask.any() else 0
        
        # Energy at key angles
        e_90 = get_energy(90)    # vertical
        e_135 = get_energy(135)  # diagonal
        e_110 = get_energy(110)  # intermediate
        
        # Dip contrast: ratio of edge angles to intermediate
        # High contrast indicates distinct directional preferences
        features['ang_dip_contrast'] = (e_90 + e_135) / (2 * e_110 + 1e-10)
        
        ################################################
        # Computing entropy
        # Low entropy = energy concentrated in few directions
        # High entropy = energy spread across all directions
        ################################################
        
        # Shannon entropy: H = -sum(p * log(p))
        features['ang_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        ################################################
        # Computing spectral flatness
        # Ratio of geometric mean to arithmetic mean
        # Flatness = 1 for uniform distribution, < 1 for peaked
        ################################################
        
        # Geometric mean (using scipy.stats.gmean)
        # Adding small epsilon to avoid log(0)
        geometric_mean = stats.gmean(hist + 1e-10)
        arithmetic_mean = np.mean(hist) + 1e-10
        
        features['ang_spectral_flatness'] = geometric_mean / arithmetic_mean
        
        return features
    
    
    def _angular_zero_features(self) -> dict:
        """Returning zero features when window too small for angular analysis."""
        return {
            'ang_dip_contrast': 0.0,
            'ang_entropy': 0.0,
            'ang_spectral_flatness': 0.0,
        }
    
    
    
    ################################################
    # DIAGONAL STRIPE FEATURES (1 feature)
    ################################################
    
    
    def _extract_diagonal_features(self, prep: dict) -> dict:
        """
        Extracting diagonal stripe detection features.
        
        This function computes 1 feature:
            - ds_autocorr_diagonal: maximum autocorrelation along diagonals
        
        Uses autocorrelation along diagonal directions to detect
        stripe-like patterns at 45-degree angles.
        
        Args:
            prep: Preprocessed data dictionary
        
        Returns:
            Dictionary with 1 diagonal feature.
        """
        window = prep['raw']
        
        ################################################
        # Computing autocorrelation
        ################################################
        
        autocorr = self._compute_autocorrelation(window)
        ac_h, ac_w = autocorr.shape
        cy, cx = ac_h // 2, ac_w // 2
        
        ################################################
        # Sampling autocorrelation along diagonal directions
        ################################################
        
        # Four diagonal directions at 45-degree intervals
        diag_offsets = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        diag_profile = []
        for d in range(1, min(cy, cx)):
            vals = []
            for dy, dx in diag_offsets:
                y = cy + dy * d
                x = cx + dx * d
                if 0 <= y < ac_h and 0 <= x < ac_w:
                    vals.append(autocorr[y, x])
            if vals:
                diag_profile.append(np.mean(vals))
        
        diag_profile = np.array(diag_profile)
        
        # Maximum absolute autocorrelation along diagonals
        # High values indicate strong diagonal structure
        autocorr_strength = float(np.max(np.abs(diag_profile))) if diag_profile.size > 0 else 0.0
        
        return {
            'ds_autocorr_diagonal': autocorr_strength,
        }
    
    
    def _compute_autocorrelation(self, window: np.ndarray) -> np.ndarray:
        """
        Computing fast FFT-based autocorrelation.
        
        Uses Wiener-Khinchin theorem: autocorr = IFFT(|FFT(x)|^2)
        
        Args:
            window: Raw window data
        
        Returns:
            Normalised autocorrelation array with centre at (h/2, w/2)
        """
        # Normalising input
        mean_val = window.mean()
        std_val = window.std() + 1e-10
        window_norm = (window - mean_val) / std_val
        
        # Wiener-Khinchin theorem
        f_transform = fft.fft2(window_norm)
        power_spectrum = np.abs(f_transform) ** 2
        autocorr = fft.ifft2(power_spectrum)
        
        # Shifting so centre (zero lag) is at array centre
        autocorr = fft.fftshift(np.real(autocorr))
        
        # Normalising to [-1, 1] range
        max_val = np.max(np.abs(autocorr)) + 1e-10
        
        return autocorr / max_val
    
    
    
    ################################################
    # QUALITY METRICS (3 features)
    ################################################
    
    
    def _extract_quality_metrics(
        self,
        prep: dict,
        lattice_feats: dict,
        diagonal_feats: dict,
        regularity_score: float
    ) -> dict:
        """
        Computing quality metrics for overall pattern assessment.
        
        This function computes 3 features:
            - snr: signal-to-noise ratio in decibels
            - cb_confidence: checkerboard pattern confidence score
            - spatial_coherence: spatial pattern regularity
        
        These metrics provide overall assessments of pattern quality
        and signal strength, combining information from multiple sources.
        
        Args:
            prep: Preprocessed data dictionary
            lattice_feats: Already extracted lattice features
            diagonal_feats: Already extracted diagonal features
            regularity_score: Pattern regularity score from autocorrelation
        
        Returns:
            Dictionary with 3 quality metrics.
        """
        window = prep['raw']
        
        ################################################
        # Computing SNR (signal-to-noise ratio)
        ################################################
        
        # Signal power, variance of image intensity
        signal_power = np.var(window)
        
        # Noise power, estimated from high-frequency FFT content
        f_transform = fft.fft2(window.astype(np.float32))
        f_shift = fft.fftshift(f_transform)
        
        h, w = window.shape
        centre_y, centre_x = h // 2, w // 2
        
        # Creating distance grid from centre
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
        max_distance = np.sqrt(centre_x**2 + centre_y**2)
        distance_norm = distance / max_distance
        
        # High-frequency mask,  outer 70% of spectrum
        # This region is dominated by noise in natural images
        noise_mask = distance_norm > 0.3
        noise_spectrum = np.abs(f_shift * noise_mask)
        noise_power = np.mean(noise_spectrum**2)
        
        # SNR in decibels, 10 * log10(signal / noise)
        snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        ################################################
        # Computing checkerboard confidence
        # Combines period-2 frequency energy with autocorrelation regularity
        ################################################
        
        p2_frac = lattice_feats.get('lat_p2_fraction_of_smallperiod_energy', 0.0)
        
        # Weighted combination, frequency evidence + autocorrelation regularity
        cb_confidence = (0.6 * p2_frac) + (0.4 * regularity_score)
        cb_confidence = float(np.clip(cb_confidence, 0, 1))
        
        ################################################
        # Computing spatial coherence
        # Measures overall spatial pattern regularity
        ################################################
        
        # Combining multiple coherence indicators
        diag_energy = lattice_feats.get('lat_diag_total_frac', 0.0)
        ds_autocorr = diagonal_feats.get('ds_autocorr_diagonal', 0.0)
        
        # Weighted combination of coherence measures
        spatial_coherence = np.clip(
            diag_energy * 0.5 + ds_autocorr * 0.5,
            0, 1
        )
        
        return {
            'snr': float(snr),
            'cb_confidence': float(cb_confidence),
            'spatial_coherence': float(spatial_coherence),
        }
    
    
    
    ################################################
    # CONFIGURATION
    ################################################
    
    
    def _default_config(self) -> dict:
        """
        Default configuration for feature extraction.
        
        Returns:
            Dictionary with default parameters:
                - lattice_periods: periods to analyse (2-5 pixels)
                - lattice_max_lag: maximum autocorrelation lag
                - angular_wavelet: wavelet type for decomposition
                - angular_level: wavelet decomposition level
                - angular_freq_band: frequency band for angular analysis
        """
        return {
            'lattice_periods': (2, 3, 4, 5),
            'lattice_max_lag': 20,
            'angular_wavelet': 'db4',
            'angular_level': 3,
            'angular_freq_band': (1/60, 1/30),
        }
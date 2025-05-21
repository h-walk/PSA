"""
Core spectral energy density (SED) data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SED:
    sed: np.ndarray
    freqs: np.ndarray
    k_points: np.ndarray  # Magnitudes of k-vectors for 1D path plots
    k_vectors: np.ndarray  # Full 3D k-vectors
    phase: Optional[np.ndarray] = None
    is_complex: bool = True  # Indicates if sed attribute holds complex amplitudes or intensities

    @property
    def intensity(self) -> np.ndarray:
        return np.sum(np.abs(self.sed)**2, axis=-1).astype(np.float32)

    def gather_3d(self, intensity_thresh_rel: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        intensity_map = self.intensity
        if intensity_map.size == 0:
            return (np.array([], dtype=np.float32), np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        max_intensity = np.max(intensity_map) if intensity_map.size > 0 else 0
        actual_thresh = intensity_thresh_rel * max_intensity if max_intensity > 0 else 0.0
        kx_vals, ky_vals, freq_vals, amp_vals = [], [], [], []
        
        if self.freqs is None or self.k_vectors is None:
            logger.warning("SED.gather_3d: Frequencies or k_vectors are None, cannot gather.")
            return (np.array(kx_vals, dtype=np.float32), np.array(ky_vals, dtype=np.float32),
                    np.array(freq_vals, dtype=np.float32), np.array(amp_vals, dtype=np.float32))

        pos_freq_mask = self.freqs >= 0
        plot_freqs = self.freqs[pos_freq_mask]
        plot_intensity_map = intensity_map[pos_freq_mask, :]

        for i, freq in enumerate(plot_freqs):
            if i < plot_intensity_map.shape[0]: 
                for j, k_vec in enumerate(self.k_vectors):
                    if j < plot_intensity_map.shape[1]: 
                        amp = plot_intensity_map[i, j]
                        if amp >= actual_thresh and k_vec.size >= 2:
                            kx_vals.append(k_vec[0])
                            ky_vals.append(k_vec[1]) 
                            freq_vals.append(freq)
                            amp_vals.append(amp)
                    else:
                        logger.debug(f"SED.gather_3d: k-point index {j} out of bounds for intensity data (max: {plot_intensity_map.shape[1]-1}).")
                        break 
            else:
                logger.debug(f"SED.gather_3d: Frequency index {i} out of bounds for intensity data (max: {plot_intensity_map.shape[0]-1}).")
                break

        return (
            np.array(kx_vals, dtype=np.float32), np.array(ky_vals, dtype=np.float32),
            np.array(freq_vals, dtype=np.float32), np.array(amp_vals, dtype=np.float32)
        )

    def save(self, base_path: Path):
        base_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(base_path.with_suffix('.sed.npy'), self.sed)
        np.save(base_path.with_suffix('.freqs.npy'), self.freqs)
        np.save(base_path.with_suffix('.k_points.npy'), self.k_points)
        np.save(base_path.with_suffix('.k_vectors.npy'), self.k_vectors)
        if self.phase is not None:
            np.save(base_path.with_suffix('.phase.npy'), self.phase)
        logger.info(f"SED data saved: {base_path.name}.*.npy")

    @staticmethod
    def load(base_path: Path) -> 'SED':
        required_suffixes = ['.sed.npy', '.freqs.npy', '.k_points.npy', '.k_vectors.npy']
        if not all((base_path.with_suffix(s)).exists() for s in required_suffixes):
            raise FileNotFoundError(f"Required SED files missing for base: {base_path.name}")

        sed_val = np.load(base_path.with_suffix('.sed.npy'))
        freqs_val = np.load(base_path.with_suffix('.freqs.npy'))
        k_points_val = np.load(base_path.with_suffix('.k_points.npy'))
        k_vectors_val = np.load(base_path.with_suffix('.k_vectors.npy'))
        phase_val = None
        phase_file = base_path.with_suffix('.phase.npy')
        if phase_file.exists():
            try:
                phase_val = np.load(phase_file)
            except Exception as e:
                logger.warning(f"Could not load phase data from {phase_file.name}: {e}")
        return SED(sed_val, freqs_val, k_points_val, k_vectors_val, phase=phase_val) 
# preprocess_lstm.py - 순수 전처리 함수들
import numpy as np
from scipy.signal import butter, filtfilt, get_window, welch, find_peaks
from typing import Tuple, List, Optional, Dict

# ===== LSTM 전처리용 계산 함수들 =====
def extract_phase_derivative(z_tau: np.ndarray, fs: float, normalize: bool = False) -> np.ndarray:
    """복소 신호에서 위상 미분 추출 + 정규화"""
    phase = np.unwrap(np.angle(z_tau))
    dphi = np.gradient(phase) * fs
    
    if normalize and len(dphi) > 1:
        # Z-score 정규화 (평균=0, 표준편차=1)
        mean_dphi = np.mean(dphi)
        std_dphi = np.std(dphi)
        if std_dphi > 1e-8:  # 표준편차가 0에 가까우면 정규화 생략
            dphi = (dphi - mean_dphi) / std_dphi
    
    return dphi.astype(np.float32)

def bandpass_filter(signal: np.ndarray, fs: float, f1: float, f2: float, order: int = 2) -> np.ndarray:
    """대역통과 필터"""
    b, a = butter(order, [f1, f2], btype='band', fs=fs)
    return filtfilt(b, a, signal).astype(np.float32)

def compute_spectrum_features(signal: np.ndarray, fs: float, pad_factor: int,
                             fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """심박 대역의 로그 파워 스펙트럼 특징 추출"""
    N = len(signal)
    nfft = 1 << int(np.ceil(np.log2(max(1, N * pad_factor))))
    windowed = signal * np.hanning(N)
    X = np.fft.rfft(windowed, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=1/fs)
    mask = (freq >= fmin) & (freq <= fmax)
    freq_band = freq[mask]
    power = np.abs(X[mask])**2
    log_power = np.log10(power + 1e-12)
    features = (log_power - log_power.mean()) / (log_power.std() + 1e-8)
    return features.astype(np.float32), freq_band.astype(np.float32)

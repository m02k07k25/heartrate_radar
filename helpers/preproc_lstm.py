# preprocess_lstm.py - 순수 전처리 함수들
import numpy as np
import os
import torch
from scipy.signal import butter, filtfilt, get_window, welch, find_peaks
from typing import Tuple, List, Optional, Dict
from helpers.preproc_signal import range_axis_m

# ===== LSTM 전처리용 계산 함수들 =====
def extract_phase_derivative(z_tau: np.ndarray, fs: float, normalize: bool = False, 
                           apply_hpf: bool = True, hpf_freq: float = 0.4) -> np.ndarray:
    """복소 신호에서 위상 미분 추출 + 전처리 필터링
    
    Args:
        z_tau: 복소 신호
        fs: 샘플링 주파수
        normalize: Z-score 정규화 여부
        apply_hpf: HPF 적용 여부 (전처리/안정화)
        hpf_freq: HPF 차단 주파수 (Hz)
    """
    # 1단계: 위상 추출
    phase = np.unwrap(np.angle(z_tau)).astype(np.float64)

    # 2단계: 위상에 HPF 적용 (전처리/안정화)
    if apply_hpf:
        from scipy.signal import butter, sosfiltfilt
        
        # HPF 설계 (2차 Butterworth)
        nyquist = fs / 2.0
        hpf_norm = hpf_freq / nyquist
        sos_hpf = butter(2, hpf_norm, btype='high', output='sos')
        
        # HPF 적용 (위상에) - 내장 패딩 사용
        phase = sosfiltfilt(sos_hpf, phase, padtype='odd')
    
    # 3단계: 위상 미분
    dphi = np.gradient(phase) * fs
    
    # 4단계: 정규화 (선택적)
    if normalize and len(dphi) > 1:
        # Z-score 정규화 (평균=0, 표준편차=1)
        mean_dphi = np.mean(dphi)
        std_dphi = np.std(dphi)
        if std_dphi > 1e-8:  # 표준편차가 0에 가까우면 정규화 생략
            dphi = (dphi - mean_dphi) / std_dphi
    
    return dphi.astype(np.float32)

def compute_harmonic_features(dphi_bp: np.ndarray, dphi_hpf: np.ndarray, fs: float, nfft: int = 2048) -> Tuple[float, float]:
    """하모닉 스칼라 피처 계산 (Top-2 피크의 H2) - 고BPM 지원
    
    Args:
        dphi_bp: BPF 적용된 위상미분 신호
        dphi_hpf: HPF만 적용된 위상미분 신호
        fs: 샘플링 주파수
        nfft: FFT 크기
        
    Returns:
        (H2_top1, H2_top2): Top-1, Top-2 피크의 2f vs f 하모닉 비율
    """
    try:
        # FFT 계산
        fft_bp = np.fft.fft(dphi_bp, n=nfft)
        fft_hpf = np.fft.fft(dphi_hpf, n=nfft)
        freqs = np.fft.fftfreq(nfft, 1/fs)
        
        # 파워 스펙트럼
        pow_bp = np.abs(fft_bp) ** 2
        pow_hpf = np.abs(fft_hpf) ** 2
        
        # 양의 주파수만 사용
        pos = freqs > 0
        f = freqs[pos]
        Pbp = pow_bp[pos]  # BPF 파워
        Ph = pow_hpf[pos]  # HPF 파워
        
        # HR 대역 마스크 (피크 찾기에만 사용)
        hr_mask = (f >= 0.8) & (f <= 3.0)
        if not np.any(hr_mask):
            return 0.0, 0.0
            
        # Top-2 피크 찾기 (이웃 중복 방지)
        from scipy.signal import find_peaks
        
        # 피크 간 최소 거리 (약 0.2 Hz)
        min_distance = max(1, int(0.2 / (f[1] - f[0])))
        
        # HR 대역에서 피크 찾기
        peaks, _ = find_peaks(Pbp[hr_mask], distance=min_distance)
        
        if peaks.size == 0:
            # 피크가 없으면 상위 2개 선택
            peak_indices = np.argsort(Pbp[hr_mask])[-2:][::-1]
        else:
            # 피크가 있으면 상위 2개 피크 선택
            peak_values = Pbp[hr_mask][peaks]
            order = np.argsort(peak_values)[-2:][::-1]
            peak_indices = peaks[order]
        
        # HR 대역 주파수와 파워
        f_hr = f[hr_mask]
        P_hr = Pbp[hr_mask]
        
        h2_values = []
        for i in peak_indices[:2]:
            f0 = f_hr[i]
            
            # 3점 포물선 보간 (QIFFT 흉내)
            if 0 < i < len(P_hr) - 1:
                y1, y2, y3 = P_hr[i-1], P_hr[i], P_hr[i+1]
                delta = 0.5 * (y3 - y1) / (2*y2 - y1 - y3 + 1e-12)
                f0 += delta * (f_hr[1] - f_hr[0])
            
            # f에서의 파워 (BPF)
            Pf0 = np.interp(f0, f, Pbp)
            
            # 2f에서의 파워 (HPF) - 전체 양의 주파수에서 보간
            P2f = np.interp(2*f0, f, Ph)
            
            # H2 비율 계산
            if Pf0 > 1e-12:
                h2_ratio = P2f / Pf0
                h2_values.append(float(h2_ratio))
            else:
                h2_values.append(0.0)
        
        # 길이 맞춤 (부족하면 0으로 패딩)
        while len(h2_values) < 2:
            h2_values.append(0.0)
        
        return h2_values[0], h2_values[1]
            
    except Exception as e:
        print(f"하모닉 피처 계산 중 에러: {e}")
        return 0.0, 0.0

def compute_energy_features(dphi_bp: np.ndarray, fs: float, nfft: int = 2048) -> Tuple[float, float, float]:
    """에너지 기반 피처 계산 (E_lo, E_hi, SNR_hr)
    
    Args:
        dphi_bp: BPF 적용된 위상미분 신호
        fs: 샘플링 주파수
        nfft: FFT 크기
        
    Returns:
        (E_lo_norm, SNR_hr): 정규화된 하군집 에너지, HR 대역 SNR
    """
    try:
        # FFT 계산
        fft_result = np.fft.fft(dphi_bp, n=nfft)
        freqs = np.fft.fftfreq(nfft, 1/fs)
        power_spectrum = np.abs(fft_result) ** 2
        
        # 양의 주파수만 사용
        pos_freqs = freqs[:nfft//2]
        pos_power = power_spectrum[:nfft//2]
        
        # 하군집 (0.95-1.35 Hz) 에너지
        lo_mask = (pos_freqs >= 0.95) & (pos_freqs <= 1.35)
        E_lo = np.sum(pos_power[lo_mask]) if np.any(lo_mask) else 0.0
        
        # 상군집 (1.45-1.75 Hz) 에너지
        hi_mask = (pos_freqs >= 1.45) & (pos_freqs <= 1.75)
        E_hi = np.sum(pos_power[hi_mask]) if np.any(hi_mask) else 0.0
        
        # 정규화된 하군집 에너지
        epsilon = 1e-10
        E_lo_norm = E_lo / (E_lo + E_hi + epsilon)
        
        # HR 대역 SNR 계산
        hr_mask = (pos_freqs >= 0.8) & (pos_freqs <= 3.0)
        if np.any(hr_mask):
            hr_power = pos_power[hr_mask]
            max_power = np.max(hr_power)
            median_power = np.median(hr_power)
            SNR_hr = max_power / (median_power + epsilon)
        else:
            SNR_hr = 0.0
        
        return float(E_lo_norm), float(SNR_hr)
            
    except Exception as e:
        print(f"에너지 피처 계산 중 에러: {e}")
        return 0.0, 0.0

# ===== 데이터 로딩 및 처리 함수들 =====

def calculation(file_path: str, fs_adc: float, num_samples: int, pad_ft: int, b_hz: float) -> Tuple[int, np.ndarray]:
    """레이더 데이터에서 최적 거리 bin과 위상 신호 추출 (전처리된 데이터 또는 원본 데이터 모두 지원)"""
    data = np.load(file_path, allow_pickle=True)
    
    # 새로운 전처리된 데이터 형식인지 확인
    if isinstance(data, np.ndarray) and data.ndim == 0 and isinstance(data.item(), dict):
        # 전처리된 데이터 (딕셔너리 형태)
        processed_dict = data.item()
        fc_bin = processed_dict['fc_bin']
        z_tau = processed_dict['z_tau']
        
        # 거리 축 계산 후 fc_bin의 거리[m] 표기
        rng_axis, _, _ = range_axis_m(fs_adc, num_samples, pad_ft, b_hz)
        dist_m = float(rng_axis[int(fc_bin)]) if 0 <= int(fc_bin) < len(rng_axis) else float('nan')
        print(f"[CALCULATION] 전처리된 데이터 로드: fc_bin={fc_bin} ({dist_m:.2f} m)")
        
        return int(fc_bin), z_tau
    raise ValueError(f"지원하지 않는 데이터 형식: type={type(data)}, shape={getattr(data, 'shape', 'N/A')}")

def find_matching_files(data_dir: str, answer_dir: str) -> List[Tuple[str, str]]:
    """폴더에서 매칭되는 데이터-정답 파일 쌍을 찾기"""
    if not os.path.exists(data_dir) or not os.path.exists(answer_dir):
        return []
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    answer_files = [f for f in os.listdir(answer_dir) if f.endswith('.csv')]
    
    # 번호로 매칭
    pairs = []
    for data_file in data_files:
        # 파일명에서 번호 추출 (예: "5.npy" -> "5")
        try:
            data_num = os.path.splitext(data_file)[0]
            answer_file = f"{data_num}.csv"
            
            if answer_file in answer_files:
                data_path = os.path.join(data_dir, data_file)
                answer_path = os.path.join(answer_dir, answer_file)
                pairs.append((data_path, answer_path))
        except Exception:
            continue
    
    # 번호순으로 정렬
    pairs.sort(key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
    return pairs

def load_ground_truth(path: str) -> Optional[np.ndarray]:
    """ECG 정답 파일 로드 (CSV 형식)"""
    if not os.path.exists(path):
        return None
    timestamps: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # CSV 헤더 건너뛰기
            if line.startswith("t_s"):
                continue
            # 쉼표로 구분
            parts = line.split(",")
            try:
                timestamps.append(float(parts[0]))  # 첫 번째 열이 타임스탬프
            except (ValueError, IndexError):
                continue
    return np.array(timestamps, dtype=np.float32) if timestamps else None

def causal_window_hr_from_rr(gt_times: np.ndarray, centers: np.ndarray, win: float) -> np.ndarray:
    """창 경계 편향 제거된 RR 기반 HR 계산 (GPT 제안 방식)

    Args:
        gt_times: ECG R-peak 타임스탬프 배열 (초)
        centers: 예측 센터 타임스탬프 (초)
        win: 창 길이 (초)

    Returns:
        hr_labels: 각 센터에 대한 HR 값 배열
    """
    t = np.asarray(gt_times, float)
    if len(t) < 2:
        return np.full(len(centers), 90.0, dtype=np.float32)

    rr = np.diff(t)
    # 리프랙토리/범위 필터링
    valid = (rr > 0.30) & (rr < 2.50)
    t = t[np.r_[True, valid]]
    rr = rr[valid]

    if len(rr) == 0:
        return np.full(len(centers), 90.0, dtype=np.float32)

    hr = 60.0 / rr
    mid = 0.5 * (t[1:] + t[:-1])  # RR 중앙 타임스탬프

    out = []
    for c in centers:
        s, e = c - win, c  # 인과적 창
        # 각 RR 구간과 [s,e] 겹치는 비율을 가중치로 사용
        start = np.maximum(s, t[:-1])
        end = np.minimum(e, t[1:])
        w = np.clip(end - start, 0, None) / rr
        m = w > 0
        if np.any(m):
            weighted_hr = np.sum(w[m] * hr[m]) / np.sum(w[m])
            # NaN 체크 및 범위 제한
            if np.isnan(weighted_hr) or weighted_hr < 30 or weighted_hr > 200:
                weighted_hr = 90.0  # 기본값
        else:
            weighted_hr = 90.0  # 기본값
        out.append(weighted_hr)

    return np.array(out, dtype=np.float32)

def create_bpm_labels(gt_times: np.ndarray, z_tau: np.ndarray,
                     win_frames: int, hop_frames: int,
                     frame_repetition_time_s: float, window_sec: float = 3.0) -> np.ndarray:
    """ECG 타임스탬프로부터 구간의 평균 BPM 라벨 생성 (창 경계 편향 제거 - GPT 제안 방식)"""
    try:
        # 예측 센터 타임스탬프 계산 (z_tau 길이에 맞춤)
        centers = []
        i = win_frames
        while i < len(z_tau):
            centers.append(i * frame_repetition_time_s)
            i += hop_frames
        centers = np.array(centers)

        # GPT의 causal_window_hr_from_rr 함수 사용
        return causal_window_hr_from_rr(gt_times, centers, window_sec)

    except Exception as e:
        print(f"BPM 라벨 생성 에러: {e}")
        raise e

def create_training_data(all_z_tau: List[np.ndarray], all_gt_times: List[np.ndarray], 
                        predictor, win_frames: int, hop_frames: int, 
                        frame_repetition_time_s: float, validation_split: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """모든 파일에서 학습/검증 데이터 생성 (파일 단위 분할)"""
    print("학습 데이터 생성 중...")
    
    # 파일별로 특징과 라벨 수집
    file_features = []
    file_labels = []
    
    for z_tau, gt_times in zip(all_z_tau, all_gt_times):
        bpm_labels = create_bpm_labels(gt_times, z_tau, win_frames, hop_frames, frame_repetition_time_s, 3.0)
        file_feature_list = []
        file_label_list = []
        window_idx = 0
        
        for i in range(win_frames, len(z_tau), hop_frames):
            if window_idx >= len(bpm_labels):
                break
                
            window = z_tau[i-win_frames:i]
            # 기존의 process_window 메서드 재사용
            window_features = predictor.process_window(window)
            
            file_feature_list.append(window_features)
            file_label_list.append(bpm_labels[window_idx])
            window_idx += 1
        
        # 파일별로 특징과 라벨 저장
        if file_feature_list:  # 빈 파일이 아닌 경우만
            file_features.append(np.array(file_feature_list, dtype=np.float32))
            file_labels.append(np.array(file_label_list, dtype=np.float32))
    
    print(f"데이터 생성 완료: {len(file_features)}개 파일")
    
    # 구간마다 골고루 랜덤 선택으로 검증 데이터 분할
    num_files = len(file_features)
    num_val_files = max(1, int(round(num_files * validation_split)))
    num_val_files = min(num_val_files, num_files)
    
    # 20개 단위 구간으로 나누어 각 구간에서 랜덤하게 선택 (1~20, 21~40, 41~60, ...)
    bucket_size = 5
    val_indices = []
    
    # 각 구간에서 선택할 개수 계산 (라운드로빈)
    num_buckets = (num_files + bucket_size - 1) // bucket_size  # 올림 계산
    selections_per_bucket = [0] * num_buckets
    
    # 필요한 검증 파일 수를 구간에 골고루 분배
    for i in range(num_val_files):
        bucket_idx = i % num_buckets
        selections_per_bucket[bucket_idx] += 1
    
    # 각 구간에서 랜덤하게 선택
    rng = np.random.default_rng(42)  # 재현성을 위해 별도의 Generator 사용
    for bucket_idx in range(num_buckets):
        start_idx = bucket_idx * bucket_size
        end_idx = min(start_idx + bucket_size, num_files)
        bucket_files = list(range(start_idx, end_idx))
        
        # 이 구간에서 선택할 개수만큼 랜덤 샘플링
        num_selections = selections_per_bucket[bucket_idx]
        if num_selections > 0 and len(bucket_files) > 0:
            selected = rng.choice(bucket_files,
                                      size=min(num_selections, len(bucket_files)), 
                                      replace=False)
            val_indices.extend(selected)
    
    # 훈련 인덱스는 검증 인덱스를 제외한 나머지
    train_indices = [i for i in range(num_files) if i not in set(val_indices)]
    
    # 훈련 데이터 결합
    train_features = []
    train_labels = []
    for idx in train_indices:
        train_features.extend(file_features[idx])
        train_labels.extend(file_labels[idx])
    
    # 검증 데이터 결합
    val_features = []
    val_labels = []
    for idx in val_indices:
        val_features.extend(file_features[idx])
        val_labels.extend(file_labels[idx])
    
    # numpy 배열로 변환
    X_train = np.array(train_features, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.float32)
    X_val = np.array(val_features, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.float32)

    # 라벨 정규화 통계 (훈련 세트 기준)
    predictor.label_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
    predictor.label_std = float(np.std(y_train) + 1e-6) if len(y_train) > 0 else 1.0
    print(f"라벨 정규화 통계: mean={predictor.label_mean:.2f}, std={predictor.label_std:.2f}")

    # 라벨 표준화
    if predictor.label_std > 0:
        y_train = (y_train - predictor.label_mean) / predictor.label_std
        y_val = (y_val - predictor.label_mean) / predictor.label_std
    
    print(f"파일 단위 데이터 분할 완료 (구간별 골고루 랜덤 샘플링):")
    print(f"  검증 파일 인덱스: {sorted(val_indices)}")
    print(f"  훈련 파일: {len(train_indices)}개, 윈도우: {len(X_train)}개")
    print(f"  검증 파일: {len(val_indices)}개, 윈도우: {len(X_val)}개")
    
    return (torch.from_numpy(X_train), torch.from_numpy(y_train), 
            torch.from_numpy(X_val), torch.from_numpy(y_val))

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

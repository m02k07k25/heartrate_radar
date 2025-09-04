# lstm_bpm.py - BPM 회귀 모델 시스템
import os
import sys
# 스크립트를 어떤 위치에서 실행하더라도 프로젝트 루트를 import 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# CuBLAS 결정적 동작을 위해 환경변수 설정 (torch/CUDA 초기화 이전)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
from sklearn.model_selection import train_test_split
from helpers.func_plot import plot_training_curves, plot_test_results
from helpers.preproc_lstm import (
    extract_phase_derivative,
)
from helpers.preproc_signal import range_axis_m
from helpers.radar_config import FS_ADC, PAD_FT, B_HZ, NUM_SAMPLES, FRAME_REPETITION_TIME_S, FS_FRAME
from typing import Tuple, List, Optional, Dict

# ===== 학습 파라미터 =====
EPOCHS = 1000                 # 에포크
LEARNING_RATE = 1e-4  # 학습률 증가로 다양성 향상          # 직접 BPM 예측용 낮은 학습률 (과적합 방지)
HIDDEN_DIM = 128
NUM_LAYERS = 2                # LSTM 레이어 2층 및 드롭아웃 적용

VALIDATION_SPLIT = 0.25       # 검증 데이터 비율 (20%로 줄임)
EARLY_STOP_PATIENCE = 200     # 얼리 스탑 인내심 (에포크) - 더 여유롭게
EARLY_STOP_MIN_DELTA = 5e-5   # 최소 개선 임계값 - 더 관대하게

# ===== 스케줄러 파라미터 =====
SCHEDULER_FACTOR = 0.5        # 학습률 감소 비율
SCHEDULER_PATIENCE = 25       # 스케줄러 인내심 (에포크) - 더 빠른 학습률 감소
SCHEDULER_MIN_LR = 5e-6       # 최소 학습률

# ===== 신호 처리 파라미터 =====
FS          = FS_FRAME        # 프레임레이트 (Hz) - radar_config에서 가져옴
WIN_FRAMES  = int(8.0 * FS)   # 8초 윈도우 = 288 프레임
HOP_FRAMES  = int(0.5 * FS)   # 0.5초 홉 = FS/2 프레임
FMIN, FMAX  = 0.8, 3.0      # 심박 대역 [Hz] (48-180 BPM에 대응) - BPF 적용
FEATURE_DIM = 18              # 1D CNN으로 압축할 특징 차원 -> 홉수

# ===== 부드러움 제약 파라미터 =====
SMOOTH_LAMBDA = 0.01           # 부드러움 제약 강도 더 증가 (데이터 손실의 5-10% 수준으로)

# ===== 경로 설정 =====
TRAIN_DATA_DIR = "record3/train/data/"
TRAIN_ANSWER_DIR = "record3/train/answer/"
TEST_DATA_DIR = "record3/test/data/"
TEST_ANSWER_DIR = "record3/test/answer/"

# ===== 시드 고정 (재현성) =====
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===== 데이터 처리 함수들 (preproc_lstm.py에서 import) =====
from helpers.preproc_lstm import (
    calculation, find_matching_files, load_ground_truth,
    create_bpm_labels, create_training_data,
    FileGroupedDataset, FileBatchSampler
)

# ===== 1D CNN + LSTM BPM 회귀 모델 정의 =====
class BPMRegressionModel(nn.Module):
    """BPM 예측을 위한 1D CNN + LSTM 회귀 모델"""
    
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        
        # 1D CNN: 7채널 입력 (다양한 관점의 신호)
        self.conv1d = nn.Sequential(
                nn.Conv1d(7, 64, kernel_size=3, padding=1),  # 7채널: dφ_BPF, dφ, H2_top1, SNR_hr, E_lo_norm, |z|_HPF, PSD_conf
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),   # (N,64,1)
                nn.Flatten(1),              # (N,64)
                nn.LayerNorm(64),   # ★ 한 줄
            )
        
        # LSTM: 시계열 패턴 학습
        self.lstm = nn.LSTM(
            input_size=64,  # 위상미분만
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=0.0 if num_layers == 1 else 0.3,  # 단일 레이어에서는 dropout 비활성화
            batch_first=True
        )
        
        # ===== 근본적 문제 해결: 모델 구조 단순화 =====
        # 회귀 헤드 단순화 (과적합 방지, 예측 안정성 향상)
        # 회귀: LSTM 출력을 BPM으로 변환 (평균 수렴 방지용 더 큰 용량)
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),  # 128 -> 64 (더 큰 용량)
            nn.ReLU(),
            nn.Dropout(0.3),  # 드롭아웃 증가로 과적합 방지
            nn.Linear(hidden // 2, hidden // 4),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.2),  # 드롭아웃 증가
            nn.Linear(hidden // 4, 1),  # 32 -> 1 (BPM 값 직접 출력)
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        # 마지막 레이어를 더 큰 범위로 재초기화 (다양성 증가)
        with torch.no_grad():
            self.regressor[-1].weight.normal_(0, 0.2)  # 가중치 분산 증가
            self.regressor[-1].bias.normal_(0, 0.1)    # 바이어스도 랜덤 초기화
        
        # 모델 구조 출력
        print(f"BPM 회귀 모델 구조:")
        print(f"  1D CNN: 7채널({input_dim}) -> 64 (Conv1d 1개)")
        print(f"    채널: [dφ_BPF, dφ, H2_top1, SNR_hr, E_lo_norm, |z|_HPF, PSD_conf]")
        print(f"  LSTM: 64 -> {hidden} (layers={num_layers})")
        print(f"  Regressor: {hidden} -> {hidden//2} -> {hidden//4} -> 1 (강화된 BPM 출력)")
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _calculate_train_labels_mean(self) -> float:
        """훈련 라벨의 평균값을 계산하여 바이어스 초기화에 사용"""
        try:
            # 훈련 데이터 디렉토리에서 모든 정답 파일을 읽어서 BPM 평균 계산
            train_answer_dir = TRAIN_ANSWER_DIR
            if not os.path.exists(train_answer_dir):
                print("[WARNING] 훈련 정답 디렉토리를 찾을 수 없습니다. 기본값 80.0 사용")
                return 80.0
            
            all_bpms = []
            answer_files = [f for f in os.listdir(train_answer_dir) if f.endswith('.csv')]
            
            for answer_file in answer_files:
                answer_path = os.path.join(train_answer_dir, answer_file)
                try:
                    gt_times = load_ground_truth(answer_path)
                    if gt_times is not None and len(gt_times) > 1:
                        # 시간 간격으로 BPM 계산
                        time_span = gt_times[-1] - gt_times[0]
                        if time_span > 0:
                            bpm = 60.0 * (len(gt_times) - 1) / time_span
                            if 30 <= bpm <= 200:  # 유효한 BPM 범위
                                all_bpms.append(bpm)
                except Exception as e:
                    raise e
            
            if all_bpms:
                mean_bpm = np.mean(all_bpms)
                print(f"[INFO] 훈련 라벨 평균 계산 완료: {len(all_bpms)}개 파일, 평균 BPM: {mean_bpm:.2f}")
                return float(mean_bpm)
            else:
                raise ValueError("유효한 BPM 데이터를 찾을 수 없습니다.")
                
        except Exception as e:
            raise e

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: (batch, seq_len, 7, feat_dim) - 7채널: [dφ_BPF, dφ, H2_top1, SNR_hr, E_lo_norm, |z|_HPF, PSD_conf]
            hidden: LSTM hidden state
        Returns:
            bpm_pred: (batch, 1) - 예측된 BPM 값
            hidden: 업데이트된 hidden state
        """
        # # x: (B, T, F)
        # mu  = x.mean(dim=1, keepdim=True)                 # (B, 1, F)  시간 평균
        # std = x.std(dim=1, keepdim=True) + 1e-6           # (B, 1, F)
        # x = (x - mu) / std                                # (B, T, F)

        batch_size, seq_len, channels, feat_dim = x.shape  # (batch, seq_len, 6, feat_dim)
        
        # 1D CNN 적용을 위해 차원 재배열: (batch, seq_len, 6, feat_dim) -> (batch * seq_len, 6, feat_dim)
        x_reshaped = x.view(-1, channels, feat_dim)
        # 1D CNN 특징 추출: (batch * seq_len, 6, feat_dim) -> (batch * seq_len, 64, 1)
        conv_out = self.conv1d(x_reshaped)
        # LSTM 입력을 위해 차원 재배열: (batch * seq_len, 64, 1) -> (batch, seq_len, 64)
        conv_out = conv_out.view(batch_size, seq_len, 64)
        
        # LSTM 처리
        lstm_out, hidden = self.lstm(conv_out, hidden)

        # 가중 평균 (권장, 라벨과 정확히 일치)
        step_sec = HOP_FRAMES / FS                    # 0.5s
        M = max(1, int(round(3.0 / step_sec)))        # 라벨 창 3.0초에 해당하는 0.5초에 해당하는 스텝 수 (Hop=0.5s → 6)
        M = min(M, lstm_out.size(1))

        sel = lstm_out[:, -M:, :]                     # (B, M, H)
        # 뒤로 갈수록(라벨 끝에 가까울수록) 조금 더 가중
        weights = torch.arange(1, M + 1, device=lstm_out.device, dtype=lstm_out.dtype)
        weights = weights / weights.sum()
        last_output = (sel * weights.view(1, M, 1)).sum(dim=1)  # (B, H)
        # w = torch.tensor([0.2, 0.3, 0.5], device=lstm_out.device)
        # last_output = (lstm_out[:, -3:, :] * w.view(1, 3, 1)).sum(dim=1)
        
        # 직접 BPM 회귀 (선형 출력, 클리핑 제거)
        bpm_pred = self.regressor(last_output)  # (batch, 1)
        
        return bpm_pred, hidden

# ===== BPM 예측 시스템 =====
class BPMPredictor:
    """디렉터리 기반 BPM 회귀 예측 시스템"""
    
    def __init__(self, train_data_dir: str, train_answer_dir: str):
        """
        Args:
            train_data_dir: 학습 데이터 폴더 경로
            train_answer_dir: 학습 정답 폴더 경로
        """
        self.train_data_dir = train_data_dir
        self.train_answer_dir = train_answer_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 라벨 정규화 통계 (훈련 시 계산)
        self.label_mean: Optional[float] = None
        self.label_std: Optional[float] = None
        print(f"디바이스: {self.device}")
        print(f"학습 데이터 경로: {train_data_dir}")
        print(f"학습 정답 경로: {train_answer_dir}")

    def _apply_pre_filtering(self, signal_for_psd: np.ndarray, signal_for_notch: np.ndarray, fs: float) -> np.ndarray:
        """프리필터링: 이상치 주파수 notch 제거 (안정성 개선 버전)

        Args:
            signal_for_psd: HPF만 적용한 dφ 신호 (PSD 검출용)
            signal_for_notch: BPF 적용된 dφ 신호 (notch 적용용)
            fs: 샘플링 주파수

        Welch PSD로 HR 대역 이상치 탐지 후 선택적 notch 적용:
        - 4초 세그먼트, 75% overlap (해상도 0.25Hz)
        - 로컬 median 대비 z-score > 6.0 (보수적)
        - 연속 2개 이상 지속되는 이상치만 선택
        - 최강 HR 피크 1개 주변 ±0.20Hz 보호 (정교)
        - 최대 1개 주파수에 notch (Q=30, 효과적이면 2개까지)
        """
        try:
            from scipy.signal import welch, iirnotch, tf2sos, sosfiltfilt
            from scipy.ndimage import median_filter

            # Welch PSD 계산 (4초 세그먼트, 75% overlap, 해상도 0.25Hz)
            nperseg = min(int(4.0 * fs), len(signal_for_psd))  # 4초 세그먼트 (안전하게 길이 제한)
            noverlap = int(0.75 * nperseg)  # 75% overlap

            f, Pxx = welch(signal_for_psd, fs=fs, window='hann',
                          nperseg=nperseg, noverlap=noverlap,
                          detrend='constant', scaling='density')

            # HR 대역 추출 (0.8-3 Hz)
            hr_mask = (f >= 0.8) & (f <= 3.0)
            f_hr = f[hr_mask]
            P_hr = Pxx[hr_mask]

            if len(P_hr) == 0:
                raise ValueError("HR 대역 데이터 없음")

            # 이상치 탐지 (로컬 median 대비 z-score/MAD)
            bg = median_filter(P_hr, size=min(5, len(P_hr)), mode='nearest')
            mad = np.median(np.abs(P_hr - bg)) + 1e-9
            z_scores = (P_hr - bg) / mad

            # 이상치 조건: z-score > 6.0, 연속 2개 이상 (보수적)
            outlier_mask = z_scores > 6.0

            # 연속성 체크 (2개 이상 연속)
            consecutive_count = 0
            sustained_outliers = []

            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    consecutive_count += 1
                else:
                    if consecutive_count >= 2:  # 2개 이상 연속
                        sustained_outliers.extend(range(i - consecutive_count, i))
                    consecutive_count = 0

            # 마지막 그룹 처리
            if consecutive_count >= 2:
                sustained_outliers.extend(range(len(outlier_mask) - consecutive_count, len(outlier_mask)))

            # 중복 제거 및 정렬
            sustained_outliers = sorted(list(set(sustained_outliers)))

            # 피크 보호: 최강 HR 피크 1개 주변 ±0.20Hz 보호 (정교)
            protect_mask = np.zeros(len(f_hr), dtype=bool)
            if len(P_hr) > 0:
                # 최강 피크 1개만 보호 (가장 큰 1개)
                peak_idx = np.argmax(P_hr)
                peak_freq = f_hr[peak_idx]
                # 피크 주변 ±0.20 Hz 보호
                peak_protect = (f_hr >= peak_freq - 0.20) & (f_hr <= peak_freq + 0.20)
                protect_mask |= peak_protect

            # 보호 구간 제외
            sustained_outliers = [idx for idx in sustained_outliers if not protect_mask[idx]]

            # 개수 제한: 최대 1개 notch (효과적이면 2개까지 허용)
            sustained_outliers = sustained_outliers[:1]  # 기본 1개, 효과적이면 2개까지

            # notch 적용 (signal_for_notch에 적용)
            filtered_signal = signal_for_notch.copy()
            for idx in sustained_outliers:
                notch_freq = f_hr[idx]
                w0 = notch_freq / (fs / 2.0)  # 정규화된 주파수

                # IIR notch 필터 (Q=30으로 좁게)
                b, a = iirnotch(w0, Q=30.0)
                sos = tf2sos(b, a)

                # 오프라인 필터링 (sosfiltfilt)
                filtered_signal = sosfiltfilt(sos, filtered_signal)

            return filtered_signal

        except Exception as e:
            print(f"프리필터링 에러: {e}")
            return signal_for_notch  # 에러시 원본 반환

    def _compute_psd_confidence(self, signal_segment: np.ndarray, fs: float) -> float:
        """PSD 기반 신호 신뢰도 계산 (프리필터링 적용된 세그먼트용)

        Args:
            signal_segment: 0.5초 세그먼트 신호 (dphi_bp_segment)
            fs: 샘플링 주파수

        Returns:
            confidence: 0-1 사이 신뢰도 (1 = 고품질 HR 신호)
        """
        try:
            from scipy.signal import welch

            # 세그먼트가 너무 짧으면 기본값 반환
            if len(signal_segment) < 32:  # 최소 32샘플
                return 0.5

            # Welch PSD 계산 (세그먼트 크기에 맞게)
            nperseg = min(64, len(signal_segment))  # 최대 64샘플 세그먼트
            noverlap = nperseg // 2  # 50% overlap

            f, Pxx = welch(signal_segment, fs=fs, window='hann',
                          nperseg=nperseg, noverlap=noverlap,
                          detrend='constant', scaling='density')

            # HR 대역 (0.8-3 Hz) 추출
            hr_mask = (f >= 0.8) & (f <= 3.0)
            if not np.any(hr_mask):
                return 0.3  # HR 대역 없음

            f_hr = f[hr_mask]
            P_hr = Pxx[hr_mask]

            # 신뢰도 계산 지표들
            if len(P_hr) > 0:
                # 1. HR 대역 피크 강도 (전체 파워 대비)
                hr_power = np.sum(P_hr)
                total_power = np.sum(Pxx) + 1e-10
                hr_ratio = hr_power / total_power

                # 2. 피크 날카로움 (최대/평균 비율)
                peak_sharpness = np.max(P_hr) / (np.mean(P_hr) + 1e-10)

                # 3. 잡음 바닥 레벨 (하위 25% 분위수)
                noise_floor = np.percentile(P_hr, 25)

                # 종합 신뢰도 (0-1 스케일)
                confidence = hr_ratio * np.tanh(peak_sharpness / 10.0) * (1.0 - noise_floor / (np.max(P_hr) + 1e-10))

                # 범위 제한
                confidence = np.clip(confidence, 0.0, 1.0)

                return float(confidence)
            else:
                return 0.3

        except Exception as e:
            print(f"PSD 신뢰도 계산 에러: {e}")
            return 0.5  # 중간값 반환

    def load_all_training_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """모든 학습 파일 로딩"""
        pairs = find_matching_files(self.train_data_dir, self.train_answer_dir)
        
        if not pairs:
            raise FileNotFoundError("학습 데이터 파일을 찾을 수 없습니다.")
        
        print(f"=== 다중 파일 데이터 로딩 ===")
        print(f"학습 데이터 {len(pairs)}개 파일 발견:")
        all_z_tau = []
        all_gt_times = []
        
        for i, (data_path, answer_path) in enumerate(pairs):
            file_num = os.path.splitext(os.path.basename(data_path))[0]
            print(f"  {i+1}. 파일 {file_num}: {data_path}")
            
            # 데이터 로딩
            fc_bin, z_tau = calculation(data_path, FS_ADC, NUM_SAMPLES, PAD_FT, B_HZ)
            gt_times = load_ground_truth(answer_path)
            
            if gt_times is not None:
                all_z_tau.append(z_tau)
                all_gt_times.append(gt_times)
                avg_bpm = 60.0 * len(gt_times) / (gt_times[-1] - gt_times[0])
                print(f"     - 거리 bin: {fc_bin}, 비트: {len(gt_times)}개, 평균 BPM: {avg_bpm:.1f}")
            else:
                print(f"     - ⚠️ 정답 파일 로딩 실패")
        
        print(f"총 {len(all_z_tau)}개 파일 로딩 완료")
        
        # BPM 분포 분석
        all_bpms = []
        for gt_times in all_gt_times:
            if len(gt_times) > 1:
                avg_bpm = 60.0 * len(gt_times) / (gt_times[-1] - gt_times[0])
                all_bpms.append(avg_bpm)
        
        if all_bpms:
            print(f"\n=== 학습 데이터 BPM 분포 ===")
            print(f"평균: {np.mean(all_bpms):.1f} BPM")
            print(f"범위: {np.min(all_bpms):.1f} - {np.max(all_bpms):.1f} BPM")
            print(f"표준편차: {np.std(all_bpms):.1f} BPM")

            # BPM 분포 분석 (근본적 문제 진단)
            bpm_counts, bpm_bins = np.histogram(all_bpms, bins=10)
            print(f"BPM 히스토그램:")
            for i in range(len(bpm_counts)):
                bin_start = bpm_bins[i]
                bin_end = bpm_bins[i+1]
                count = bpm_counts[i]
                print(f"  {bin_start:.0f}-{bin_end:.0f} BPM: {count}개")

            # 데이터 품질 지표
            print(f"\n데이터 품질 지표:")
            print(f"- 총 샘플 수: {len(all_bpms)}")
            print(f"- IBI 범위 내 샘플 비율: {(len(all_bpms)/len(all_bpms))*100:.1f}%")  # 라벨 노이즈 게이트 후
            print(f"- BPM 변동성: {(np.std(all_bpms)/np.mean(all_bpms))*100:.1f}% CV")
            
        return all_z_tau, all_gt_times
    
    def process_window(self, window: np.ndarray) -> np.ndarray:
        """8초 윈도우를 8개 구간으로 나누어 위상미분 신호를 직접 사용 (개선된 신호 처리 순서)"""
        
        # ===== 개선된 신호 처리 순서 =====
        # Before: z_tau(복소) → BPF(0.8-3 Hz) → angle() → unwrap() → 미분
        # After:  z_tau(복소) → angle() → unwrap() → HPF(0.3 Hz) → 미분 → BPF(0.8-3 Hz)
        
        try:
            # 1단계: 복소 신호에서 위상 추출 (HPF 전처리만, 정규화는 1초 구간별로)
            dphi_full = extract_phase_derivative(window, FS, apply_hpf=True, hpf_freq=0.3, normalize=False)
            
            # 2단계: 위상 신호에 BPF(0.8-3 Hz) 적용
            from scipy.signal import butter, sosfiltfilt, welch, iirnotch, tf2sos
            from scipy.ndimage import median_filter
            from helpers.preproc_lstm import compute_harmonic_features, compute_energy_features
            
            low_freq = FMIN   # Hz (48 BPM)
            high_freq = FMAX  # Hz (180 BPM)
            nyquist = FS / 2.0
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # 2차 Butterworth BPF 계수 계산 (SOS 형태)
            sos_bpf = butter(2, [low_norm, high_norm], btype='band', output='sos')
            dphi_bp = sosfiltfilt(sos_bpf, dphi_full, padtype='odd')

            # 2.5단계: 프리필터링 (이상치 주파수 notch 제거)
            # PSD 검출은 HPF만 적용한 dphi_full로, notch 적용은 dphi_bp에
            dphi_bp = self._apply_pre_filtering(dphi_full, dphi_bp, FS)
            
            # 복소 신호 크기 추출 (|z|_HPF) - 내장 패딩 사용
            z_magnitude = np.abs(window)
            
            # HPF 설계 (2차 Butterworth, 0.3 Hz)
            hpf_freq = 0.3  # Hz
            hpf_norm = hpf_freq / nyquist
            sos_hpf = butter(2, hpf_norm, btype='high', output='sos')
            
            # HPF 적용 (크기에) - 내장 패딩 사용
            z_magnitude_hpf = sosfiltfilt(sos_hpf, z_magnitude, padtype='odd')
            
            # 3단계: 8초 전체에서 하모닉/에너지 피처 1회 계산
            h2_top1, h2_top2 = compute_harmonic_features(dphi_bp, dphi_full, FS)
            E_lo_norm, SNR_hr = compute_energy_features(dphi_bp, FS)
            
            # 4단계: 6채널 입력 생성 [dφ_BPF, dφ, H2_top1, SNR_hr, E_lo_norm, |z|_HPF]
            sub_intervals = WIN_FRAMES // HOP_FRAMES  # 16개 구간
            sub_window_size = HOP_FRAMES              # 18프레임 = 0.5초
            sub_features = []
            
            # 3초 롤링 컨텍스트 정규화 (라벨 창과 동기화)
            span = int(round(3.0 * FS))  # 3초 = 108프레임
            
            # 3초 컨텍스트 기반 Robust 정규화 함수
            def rznorm_ctx(x, ctx):
                med = np.median(ctx)
                mad = np.median(np.abs(ctx - med)) + 1e-6
                return (x - med) / mad
            
            for i in range(sub_intervals):
                start_idx = i * sub_window_size
                end_idx = (i + 1) * sub_window_size if i < sub_intervals - 1 else len(dphi_bp)
                
                # 3초 롤링 컨텍스트 (직전 3초 구간)
                ctx_start = max(0, end_idx - span)
                ctx_slice_bp = dphi_bp[ctx_start:end_idx]
                ctx_slice_dph = dphi_full[ctx_start:end_idx]
                ctx_slice_mag = z_magnitude_hpf[ctx_start:end_idx]
                
                # 각 구간에서 6채널 추출 및 3초 컨텍스트 정규화
                dphi_bp_segment = dphi_bp[start_idx:end_idx]
                dphi_segment = dphi_full[start_idx:end_idx]
                z_mag_segment = z_magnitude_hpf[start_idx:end_idx]
                
                # 3초 컨텍스트 기반 Robust 정규화
                dphi_bp_segment = rznorm_ctx(dphi_bp_segment, ctx_slice_bp)
                dphi_segment = rznorm_ctx(dphi_segment, ctx_slice_dph)
                z_mag_segment = rznorm_ctx(z_mag_segment, ctx_slice_mag)
                
                if (len(dphi_bp_segment) == FEATURE_DIM and 
                    len(dphi_segment) == FEATURE_DIM and 
                    len(z_mag_segment) == FEATURE_DIM):
                    
                    # 7채널 스택: [dφ_BPF, dφ, H2_top1, SNR_hr, E_lo_norm, |z|_HPF, PSD_conf]
                    # 전역 특성들을 시간축에 따라 약간의 변동 추가 (표준편차 0 방지)
                    h2_top1_channel = np.full(FEATURE_DIM, h2_top1, dtype=np.float32)
                    h2_top1_channel += np.random.normal(0, 0.01, FEATURE_DIM).astype(np.float32)  # 작은 노이즈 추가
                    
                    snr_hr_channel = np.full(FEATURE_DIM, SNR_hr, dtype=np.float32)
                    snr_hr_channel += np.random.normal(0, 0.01, FEATURE_DIM).astype(np.float32)  # 작은 노이즈 추가
                    
                    e_lo_norm_channel = np.full(FEATURE_DIM, E_lo_norm, dtype=np.float32)
                    e_lo_norm_channel += np.random.normal(0, 0.01, FEATURE_DIM).astype(np.float32)  # 작은 노이즈 추가

                    # PSD 기반 신뢰도 계산 (직전 3초 컨텍스트로 계산)
                    psd_confidence = self._compute_psd_confidence(ctx_slice_bp, FS)
                    psd_conf_channel = np.full(FEATURE_DIM, psd_confidence, dtype=np.float32)
                    psd_conf_channel += np.random.normal(0, 0.01, FEATURE_DIM).astype(np.float32)  # 작은 노이즈 추가

                    seven_channel = np.stack([
                        dphi_bp_segment,      # 채널 0: dφ_BPF (3초 컨텍스트 정규화)
                        dphi_segment,         # 채널 1: dφ (3초 컨텍스트 정규화)
                        h2_top1_channel,      # 채널 2: H2_top1 (8초 전체에서 계산)
                        snr_hr_channel,       # 채널 3: SNR_hr (8초 전체에서 계산)
                        e_lo_norm_channel,    # 채널 4: E_lo_norm (8초 전체에서 계산)
                        z_mag_segment,        # 채널 5: |z|_HPF (3초 컨텍스트 정규화)
                        psd_conf_channel      # 채널 6: PSD_conf (프리필터링 신뢰도)
                    ], axis=0)
                    
                    sub_features.append(seven_channel.astype(np.float32))
                else:
                    print(f"위상미분 구간 크기 불일치: {len(dphi_bp_segment)} != {FEATURE_DIM}")
                    raise ValueError(f"위상미분 구간 크기 불일치: {len(dphi_bp_segment)} != {FEATURE_DIM}")

        except Exception as e:
            print(f"위상미분 추출 중 에러 발생: {e}")
            raise e
        
        # (16, 6, FEATURE_DIM)으로 결합
        feats = np.array(sub_features, dtype=np.float32)  # (16, 6, 36)
        # 3초 롤링 컨텍스트 정규화로 라벨 창과 동기화, 하모닉/에너지 피처는 8초 전체에서 1회 계산
        return feats
    
    def train_on_multiple_files(self, all_z_tau: List[np.ndarray], all_gt_times: List[np.ndarray], batch_size: int):
        """여러 파일로 DataLoader 배치 학습 (검증 및 얼리 스탑 포함)"""
        # ===== 4) 입력 특성(전처리) 점검 =====
        first_features = self.process_window(all_z_tau[0][:WIN_FRAMES])
        print(f"특징 차원: {first_features.shape}")

        # 입력 특성 분포 분석
        f = first_features  # numpy array
        print("=== 입력 특성 분석 ===")
        print(f"first_features shape: {f.shape}")
        print(f"global mean/std: {f.mean():.4f}, {f.std():.4f}")
        print(f"min/max: {f.min():.4f}, {f.max():.4f}")

        # 채널별 분포 분석 (7개 채널)
        for ch in range(f.shape[1]):
            ch_mean = f[:, ch, :].mean()
            ch_std = f[:, ch, :].std()
            print(f"ch{ch} mean/std: {ch_mean:.4f}, {ch_std:.4f}")
        
        self.model = BPMRegressionModel(input_dim=FEATURE_DIM).to(self.device)
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 파라미터 그룹: 마지막 회귀 레이어에 더 큰 학습률 (항등성으로 분리)
        last_layer_params = list(self.model.regressor[-1].parameters())
        last_param_ids = {id(p) for p in last_layer_params}
        other_params = [p for p in self.model.parameters() if id(p) not in last_param_ids]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": LEARNING_RATE},
            {"params": last_layer_params, "lr": LEARNING_RATE},
        ])
        
        # 학습/검증 데이터 생성 및 DataLoader 설정
        (X_train, y_train, X_val, y_val,
         train_file_features, train_file_labels,
         val_file_features, val_file_labels) = create_training_data(
            all_z_tau, all_gt_times, self, WIN_FRAMES, HOP_FRAMES,
            FRAME_REPETITION_TIME_S, VALIDATION_SPLIT
        )
        
        # 라벨 정규화 통계 계산 후 Huber Loss의 beta 조정
        beta_bpm = 3.0                             # 2~4 BPM 중 하나로 튜닝
        beta_std = beta_bpm / self.label_std        # z-스케일 임계치
        
        # criterion 객체는 더 이상 사용하지 않음 (F.smooth_l1_loss 직접 호출)
        print(f"Huber Loss beta 조정: {beta_bpm} BPM -> {beta_std:.3f} (z-scale)")
        print(f"SmoothL1Loss 직접 호출로 브로드캐스트 경고 제거")
        
        # 학습률 스케줄러 추가 (검증 손실 기반)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',           # 검증 손실 최소화
            factor=SCHEDULER_FACTOR,  # 학습률 감소 비율
            patience=SCHEDULER_PATIENCE,  # 인내심
            min_lr=SCHEDULER_MIN_LR,     # 최소 학습률
        )

        print(f"\n=== 배치 학습 시작 (에포크: {EPOCHS}, 배치크기: {batch_size}) ===")
        print(f"검증 데이터 비율: {VALIDATION_SPLIT*100:.0f}%, 얼리 스탑 인내심: {EARLY_STOP_PATIENCE} 에포크")
        print(f"스케줄러: ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
        
        # ===== 파일별 그룹화된 데이터셋 생성 (부드러움 제약용) =====
        train_grouped_dataset = FileGroupedDataset(train_file_features, train_file_labels)
        val_grouped_dataset = FileGroupedDataset(val_file_features, val_file_labels)

        # 파일별 배치 샘플러 생성 (각 배치가 동일 파일의 연속 샘플만 포함)
        train_sampler = FileBatchSampler(
            file_lengths=train_grouped_dataset.file_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle_files=True,  # 파일 순서만 랜덤하게 섞음
            seed=42
        )

        val_sampler = FileBatchSampler(
            file_lengths=val_grouped_dataset.file_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle_files=True,  # 검증에서도 파일 순서 랜덤화
            seed=42
        )

        train_dataloader = DataLoader(
            train_grouped_dataset,
            batch_sampler=train_sampler,  # batch_sampler 사용
            num_workers=0,  # Windows에서 안정적
            pin_memory=torch.cuda.is_available()
        )

        val_dataloader = DataLoader(
            val_grouped_dataset,
            batch_sampler=val_sampler,  # batch_sampler 사용
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"DataLoader 생성: 훈련 {len(train_grouped_dataset)}개, 검증 {len(val_grouped_dataset)}개 윈도우")
        print(f"파일별 배치 생성: 훈련 {len(train_sampler)}개, 검증 {len(val_sampler)}개 배치")
        
        # 얼리 스탑 변수들
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_loss_history = []
        val_loss_history = []
        train_mae_history = []
        val_mae_history = []

        for epoch in range(EPOCHS):
            # 훈련 단계
            self.model.train()
            train_loss = train_mae = batch_count = 0
            
            for features, labels in train_dataloader:
                features = features.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                bpm_pred, _ = self.model(features, None)
                
                # SNR + PSD + 극단 BPM 가중치 적용
                pred = bpm_pred.squeeze()
                labels_squeezed = labels.squeeze()

                # 차원 일치 보장 (브로드캐스트 경고 제거)
                pred_flat = pred.view(-1)        # (B,)
                labels_flat = labels_squeezed.view(-1)    # (B,)
                
                # SmoothL1Loss 직접 호출로 브로드캐스트 경고 제거
                base = F.smooth_l1_loss(pred_flat, labels_flat, reduction='none', beta=beta_std)  # (B,) 형태의 개별 샘플 loss

                # SNR 기반 가중치 계산 (채널 3: SNR_hr)
                snr = features[:, :, 3, :].mean(dim=(1,2))  # (B,) - 배치별 평균 SNR
                snr_w = (snr - snr.median()) / (snr.std() + 1e-6)
                snr_w = snr_w.clamp(-1, 2) * 0.2 + 1.0  # 0.8~1.4 범위

                # PSD 신뢰도 기반 가중치 계산 (채널 6: PSD_conf)
                psd_conf = features[:, :, 6, :].mean(dim=(1,2))  # (B,) - 배치별 평균 PSD 신뢰도
                psd_w = 0.5 + 0.5 * psd_conf  # 0.5~1.0 범위 (신뢰도가 높을수록 가중치 증가)

                # 실제 BPM으로 변환하여 BPM 가중치 계산
                true_bpm = labels * self.label_std + self.label_mean
                bpm_w = torch.ones_like(true_bpm)
                bpm_w = bpm_w + 0.6 * (true_bpm < 75).float() + 0.2 * (true_bpm > 95).float()

                # ===== 부드러움 제약 추가 (z-score 스케일 통일) =====
                # 라벨과 같은 z-score 스케일에서 부드러움 제약 계산
                if pred.shape[0] > 1:  # 배치에 여러 샘플이 있는 경우만
                    # z-score 스케일에서 연속 예측값들의 차이 계산
                    diff = pred[1:] - pred[:-1]  # pred는 이미 z-score 정규화된 값
                    smooth_loss = torch.abs(diff).mean()  # L1 노름
                else:
                    smooth_loss = torch.tensor(0.0, device=pred.device)

                # 최종 결합 가중치 (브로드캐스트 버그 수정)
                # 모든 가중치를 (B,) 형태로 유지해서 element-wise 곱
                snr_w = snr_w.view(-1)      # (B,) 확실히 보장
                psd_w = psd_w.view(-1)      # (B,) 확실히 보장
                bpm_w = bpm_w.view(-1)      # (B,) 확실히 보장

                combined_w = bpm_w * snr_w * psd_w   # (B,), element-wise 곱
                base = base.view(-1)   # base도 (B,) 확실히 보장

                data_loss = (combined_w * base).mean()

                # ===== 2) data_loss, combined_w, base 분포 확인 =====
                if epoch == 0 and batch_count < 1:
                    print("=== 손실 구성 요소 분석 ===")
                    print(f"base mean/std: {base.mean().item():.4f}, {base.std().item():.4f}")
                    print(f"combined_w mean/std: {combined_w.mean().item():.4f}, {combined_w.std().item():.4f}")
                    print(f"data_loss: {data_loss.item():.4f}")

                # 부드러움 제약 결합
                loss = data_loss + SMOOTH_LAMBDA * smooth_loss

                # ===== 1) 그래디언트/파라미터 업데이트 확인 =====
                if epoch == 0 and batch_count < 1:
                    # 파라미터 노름 저장 (업데이트 전)
                    before_norm = sum(p.data.norm().item() for p in self.model.parameters() if p.requires_grad)
                    print(f"before_step_norm: {before_norm:.4f}")

                loss.backward()

                # 그래디언트 노름 계산
                if epoch == 0 and batch_count < 1:
                    total_grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_grad_norm += float(p.grad.data.norm().cpu().item())
                    print(f"grad norm: {total_grad_norm:.4f}")

                optimizer.step()

                # 파라미터 노름 변화 확인
                if epoch == 0 and batch_count < 1:
                    after_norm = sum(p.data.norm().item() for p in self.model.parameters() if p.requires_grad)
                    param_diff = abs(after_norm - before_norm)
                    print(f"after_step_norm: {after_norm:.4f}, param_diff: {param_diff:.4f}")

                # 디버깅: 예측값 분포 분석 (첫 에포크만)
                if epoch == 0 and batch_count < 1:
                    print(f"[부드러움] 데이터 손실: {data_loss.item():.3f}, 부드러움 손실(L2,z-score): {smooth_loss.item():.3f}, λ: {SMOOTH_LAMBDA}")
                    print(f"[부드러움] 배치 크기: {pred.shape[0]}, 부드러움 적용: {'예' if pred.shape[0] > 1 else '아니오'}")

                    # 예측값 분포 분석
                    pred_mean = pred.mean().item()
                    pred_std = pred.std().item()
                    pred_min = pred.min().item()
                    pred_max = pred.max().item()
                    print(f"[예측 분포] z-score: 평균={pred_mean:.3f}, 표준편차={pred_std:.3f}, 범위=[{pred_min:.3f}, {pred_max:.3f}]")

                    # 라벨 분포 분석
                    labels_mean = labels.mean().item()
                    labels_std = labels.std().item()
                    labels_min = labels.min().item()
                    labels_max = labels.max().item()
                    print(f"[라벨 분포] z-score: 평균={labels_mean:.3f}, 표준편차={labels_std:.3f}, 범위=[{labels_min:.3f}, {labels_max:.3f}]")

                    print(f"[라벨 정규화] mean: {self.label_mean:.3f}, std: {self.label_std:.3f}")
                # MAE는 BPM 단위로 계산 (역정규화)
                if (self.label_mean is not None) and (self.label_std is not None):
                    pred_bpm = bpm_pred.squeeze() * self.label_std + self.label_mean
                    true_bpm = labels * self.label_std + self.label_mean
                    mae = F.l1_loss(pred_bpm, true_bpm)

                    # BPM 단위 예측값 분포도 분석 (첫 에포크만)
                    if epoch == 0 and batch_count < 1:
                        pred_bpm_mean = pred_bpm.mean().item()
                        pred_bpm_std = pred_bpm.std().item()
                        pred_bpm_min = pred_bpm.min().item()
                        pred_bpm_max = pred_bpm.max().item()
                        print(f"[예측 분포] BPM: 평균={pred_bpm_mean:.1f}, 표준편차={pred_bpm_std:.1f}, 범위=[{pred_bpm_min:.1f}, {pred_bpm_max:.1f}]")

                        true_bpm_mean = true_bpm.mean().item()
                        true_bpm_std = true_bpm.std().item()
                        true_bpm_min = true_bpm.min().item()
                        true_bpm_max = true_bpm.max().item()
                        print(f"[실제 분포] BPM: 평균={true_bpm_mean:.1f}, 표준편차={true_bpm_std:.1f}, 범위=[{true_bpm_min:.1f}, {true_bpm_max:.1f}]")
                else:
                    mae = F.l1_loss(bpm_pred.squeeze(), labels)
                
                train_loss += loss.item()
                train_mae += mae.item()
                batch_count += 1
            
            avg_train_loss = train_loss / max(1, batch_count)
            avg_train_mae = train_mae / max(1, batch_count)
            
            # 검증 단계
            self.model.eval()
            val_loss = val_mae = val_batch_count = 0
            
            with torch.no_grad():
                for features, labels in val_dataloader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    bpm_pred, _ = self.model(features, None)
                    
                    # ===== 검증 단계: 순수 데이터 손실만 계산 (부드러움 제약 제외) =====
                    # 검증에서는 모델 학습이 일어나지 않으므로 부드러움 제약 제외
                    pred_val = bpm_pred.squeeze()  # 예측값
                    labels_val = labels.squeeze()  # 라벨

                    # 차원 일치 보장 (브로드캐스트 경고 제거)
                    pred_val_flat = pred_val.view(-1)        # (B,)
                    labels_val_flat = labels_val.view(-1)    # (B,)
                    
                    # SmoothL1Loss 직접 호출로 브로드캐스트 경고 제거
                    base_loss = F.smooth_l1_loss(pred_val_flat, labels_val_flat, reduction='mean', beta=beta_std)  # 스칼라 손실

                    # 최종 검증 손실 (부드러움 제약 제외)
                    loss = base_loss

                    # 디버깅: 검증 단계 순수 데이터 손실 출력 (첫 배치만)
                    if batch_count == 0:
                        print(f"[검증] 순수 데이터 손실: {loss:.3f} (부드러움 제약 제외)")
                    # MAE는 BPM 단위로 계산 (역정규화)
                    if (self.label_mean is not None) and (self.label_std is not None):
                        pred_bpm = bpm_pred.squeeze() * self.label_std + self.label_mean
                        true_bpm = labels * self.label_std + self.label_mean
                        mae = F.l1_loss(pred_bpm, true_bpm)
                    else:
                        mae = F.l1_loss(bpm_pred.squeeze(), labels)
                    
                    val_loss += loss.item()
                    val_mae += mae.item()
                    val_batch_count += 1
            
            avg_val_loss = val_loss / max(1, val_batch_count)
            avg_val_mae = val_mae / max(1, val_batch_count)
            
            # 스케줄러 스텝 (검증 손실 기반)
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 학습률 변경 감지 및 출력
            if current_lr != old_lr:
                print(f"🔽 학습률 감소: {old_lr:.2e} → {current_lr:.2e}")
            
            # 얼리 스탑 체크
            if avg_val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"Epoch {epoch+1}/{EPOCHS} | 훈련 Loss: {avg_train_loss:.6f} MAE: {avg_train_mae:.4f} | 검증 Loss: {avg_val_loss:.6f} MAE: {avg_val_mae:.4f} | LR: {current_lr:.2e} | 🎯 새로운 최고 성능!")
            else:
                patience_counter += 1
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{EPOCHS} | 훈련 Loss: {avg_train_loss:.6f} MAE: {avg_train_mae:.4f} | 검증 Loss: {avg_val_loss:.6f} MAE: {avg_val_mae:.4f} | LR: {current_lr:.2e} | 인내심: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            # 얼리 스탑 조건 확인
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n\n!!! 얼리 스탑! {EARLY_STOP_PATIENCE} 에포크 동안 개선 없음")
                print(f"최고 검증 Loss: {best_val_loss:.6f} (Epoch {epoch+1-EARLY_STOP_PATIENCE})")
                break
            
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_mae_history.append(avg_train_mae)
            val_mae_history.append(avg_val_mae)
        
        # 최고 성능 모델로 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"최고 성능 모델로 복원 완료 (검증 Loss: {best_val_loss:.6f})")
        
        # 모델 저장
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/bpm_regressor.pt")
        print("\n모델 저장 완료: checkpoints/bpm_regressor.pt")

        # 학습 과정 시각화
        plot_training_curves(train_loss_history, val_loss_history, train_mae_history, val_mae_history)
    
    def _predict_windows(self, z_tau: np.ndarray) -> List[float]:
        """윈도우별 BPM 예측 (공통 함수)"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(WIN_FRAMES, len(z_tau), HOP_FRAMES):
                window = z_tau[i-WIN_FRAMES:i]
                features = self.process_window(window)
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                bpm_pred, _ = self.model(x, None)
                y = bpm_pred.cpu().item()
                # 역정규화(라벨 정규화가 존재하는 경우)
                if (self.label_mean is not None) and (self.label_std is not None):
                    y = y * self.label_std + self.label_mean
                predictions.append(y)
                
        return predictions
    
    def predict_bpm(self, z_tau: np.ndarray) -> Tuple[List[float], List[float]]:
        """BPM 예측 (시간 정보 포함)"""
        predictions = self._predict_windows(z_tau)
        times = [i / FS for i in range(WIN_FRAMES, len(z_tau), HOP_FRAMES)][:len(predictions)]
        return predictions, times
    
    def calculate_epoch_metrics(self, z_tau: np.ndarray, gt_times: np.ndarray) -> Dict:
        """에포크별 성능 지표 계산"""
        
        # 예측 BPM
        pred_bpms = self._predict_windows(z_tau)
        
        # 정답 BPM (이미 실제 BPM 값) - 창 경계 편향 제거된 계산
        true_bpms = create_bpm_labels(gt_times, z_tau, WIN_FRAMES, HOP_FRAMES, FRAME_REPETITION_TIME_S, 3.0)
        
        if len(pred_bpms) == 0 or len(true_bpms) == 0:
            return {"rmse": 0.0, "mae": 0.0, "avg_pred_bpm": 0.0, "avg_true_bpm": 0.0}
        
        # 길이 맞춤 및 메트릭 계산
        min_len = min(len(pred_bpms), len(true_bpms))
        pred_array = np.array(pred_bpms[:min_len])
        true_array = np.array(true_bpms[:min_len])
        
        # 훈련/검증과 동일한 방식으로 MAE 계산 (PyTorch F.l1_loss 사용)
        pred_tensor = torch.tensor(pred_array, dtype=torch.float32)
        true_tensor = torch.tensor(true_array, dtype=torch.float32)
        
        # 차원 일치 보장 (브로드캐스트 경고 제거)
        pred_flat = pred_tensor.view(-1)        # (N,)
        true_flat = true_tensor.view(-1)        # (N,)
        
        # F.l1_loss로 MAE 계산 (훈련/검증과 동일)
        mae_tensor = F.l1_loss(pred_flat, true_flat, reduction='mean')
        
        return {
            "rmse": float(np.sqrt(np.mean((pred_array - true_array) ** 2))),
            "mae": float(mae_tensor.item()),  # PyTorch F.l1_loss 사용
            "avg_pred_bpm": float(np.mean(pred_array)),
            "avg_true_bpm": float(np.mean(true_array))
        }
    
    def evaluate_predictions(self, z_tau: np.ndarray, gt_times: np.ndarray) -> None:
        """예측 성능을 구간별로 상세 평가"""
 
        pred_bpms, window_times = self.predict_bpm(z_tau)
        window_sec = WIN_FRAMES * FRAME_REPETITION_TIME_S  # 8초 창
        true_bpms = create_bpm_labels(gt_times, z_tau, WIN_FRAMES, HOP_FRAMES, FRAME_REPETITION_TIME_S, window_sec)
        
        # 길이 맞춤
        min_len = min(len(pred_bpms), len(true_bpms))
        pred_bpms = pred_bpms[:min_len]
        true_bpms = true_bpms[:min_len]
        window_times = window_times[:min_len]
        
        print(f"\n=== 구간별 BPM 예측 성능 평가 ===")
        
        # 원하는 구간 길이(초)와 현재 홉 간격으로부터 구간 당 윈도우 개수 계산
        target_interval_sec = 4
        step_seconds = HOP_FRAMES / FS  # 0.5초
        total_duration = len(z_tau) / FS
        interval_size = max(1, int(round(target_interval_sec / step_seconds)))
        for start_idx in range(0, len(pred_bpms), interval_size):
            end_idx = min(start_idx + interval_size, len(pred_bpms))
            
            interval_pred = pred_bpms[start_idx:end_idx]
            interval_true = true_bpms[start_idx:end_idx]
            interval_times = window_times[start_idx:end_idx]
            
            if len(interval_pred) == 0:
                continue
                
            # 구간 내 통계 (훈련/검증과 동일한 방식으로 MAE 계산)
            pred_mean = float(np.mean(interval_pred))
            true_mean = float(np.mean(interval_true))
            pred_std = float(np.std(interval_pred))
            true_std = float(np.std(interval_true))
            
            # PyTorch F.l1_loss로 MAE 계산 (훈련/검증과 동일)
            pred_tensor = torch.tensor(interval_pred, dtype=torch.float32)
            true_tensor = torch.tensor(interval_true, dtype=torch.float32)
            pred_flat = pred_tensor.view(-1)
            true_flat = true_tensor.view(-1)
            mae = float(F.l1_loss(pred_flat, true_flat, reduction='mean').item())
            
            rmse = float(np.sqrt(np.mean((np.array(interval_pred) - np.array(interval_true))**2)))
            
            start_time = float(interval_times[0])
            end_time = float(interval_times[-1])
            # 마지막 구간은 실제 총 길이를 반영해 표시 (예: 60.0초)
            if end_idx == len(pred_bpms):
                end_time = min(end_time + step_seconds, total_duration)
            
            # 한 줄 요약 출력 (MAE 추가)
            print(f"구간 {start_time:.1f}~{end_time:.1f}초 | 예측 {pred_mean:.2f}±{pred_std:.2f} | 실제 {true_mean:.2f}±{true_std:.2f} | RMSE {rmse:.2f} BPM | MAE {mae:.2f} BPM")
            # print()
    
    def test_on_multiple_files(self, test_data_dir: str, test_answer_dir: str):
        """여러 테스트 파일로 평가"""
        test_pairs = find_matching_files(test_data_dir, test_answer_dir)
        
        if not test_pairs:
            print("테스트 데이터 파일을 찾을 수 없습니다.")
            return
        
        print(f"\n=== 테스트 데이터 평가 ({len(test_pairs)}개 파일) ===")
        
        all_rmse = []
        all_mae = []
        test_losses = []
        test_maes = []
        
        for i, (test_data_path, test_answer_path) in enumerate(test_pairs):
            file_num = os.path.splitext(os.path.basename(test_data_path))[0]
            print(f"\n파일 {file_num} 테스트:")
            
            # 테스트 데이터 직접 로딩
            fc_bin, test_z_tau = calculation(test_data_path, FS_ADC, NUM_SAMPLES, PAD_FT, B_HZ)
            test_gt_times = load_ground_truth(test_answer_path)
            
            if test_gt_times is not None:
                # 성능 계산
                metrics = self.calculate_epoch_metrics(test_z_tau, test_gt_times)
                
                print(f"  RMSE: {metrics['rmse']:.2f} BPM")
                print(f"  MAE: {metrics['mae']:.2f} BPM")
                print(f"  예측 평균: {metrics['avg_pred_bpm']:.2f}")
                print(f"  실제 평균: {metrics['avg_true_bpm']:.2f}")
                
                # 구간별 상세 평가 추가
                self.evaluate_predictions(test_z_tau, test_gt_times)
                
                all_rmse.append(metrics['rmse'])
                all_mae.append(metrics['mae'])
                
                # 테스트 loss 계산 (Huber Loss 사용)
                test_loss = self.calculate_test_loss(test_z_tau, test_gt_times)
                test_losses.append(test_loss)
                test_maes.append(metrics['mae'])
            else:
                print(f"  ⚠️ 정답 파일 로딩 실패")
        
        # 전체 평균 성능
        if all_rmse:
            avg_rmse = np.mean(all_rmse)
            avg_mae = np.mean(all_mae)
            std_rmse = np.std(all_rmse)
            std_mae = np.std(all_mae)
            
            print(f"\n=== 전체 테스트 성능 ===")
            print(f"평균 RMSE: {avg_rmse:.2f} ± {std_rmse:.2f} BPM")
            print(f"평균 MAE: {avg_mae:.2f} ± {std_mae:.2f} BPM")
            print(f"테스트 파일 수: {len(all_rmse)}개")
            
            # 테스트셋 loss 그래프 생성
            if test_losses:
                plot_test_results(test_losses, test_maes, all_rmse)
    
    def run(self, test_data_dir=None, test_answer_dir=None, batch_size=64):  # 배치 사이즈 증가 (과적합 방지)
        """전체 실행"""
        # 1. 모든 학습 데이터 로딩
        all_z_tau, all_gt_times = self.load_all_training_data()
        
        # 2. 다중 파일 배치 학습
        self.train_on_multiple_files(all_z_tau, all_gt_times, batch_size)
        
        # 3. 테스트 (옵션)
        if test_data_dir and test_answer_dir:
            self.test_on_multiple_files(test_data_dir, test_answer_dir)
        else:
            print("\n💡 테스트 데이터가 제공되지 않았습니다.")
            print("   사용법: predictor.run(test_data_dir='path/to/test/', test_answer_dir='path/to/test_answer/')")
    
    def calculate_test_loss(self, z_tau: np.ndarray, gt_times: np.ndarray) -> float:
        """테스트 데이터에 대한 loss 계산"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        # Huber Loss 사용 (훈련과 동일한 beta, 개별 샘플 loss)
        beta_std = 3.0 / self.label_std if self.label_std else 3.0
        # criterion 객체는 더 이상 사용하지 않음 (F.smooth_l1_loss 직접 호출)
        
        with torch.no_grad():
            for i in range(WIN_FRAMES, len(z_tau), HOP_FRAMES):
                window = z_tau[i-WIN_FRAMES:i]
                features = self.process_window(window)
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                bpm_pred, _ = self.model(x, None)
                
                # 정답 BPM 계산
                t_end = i * FRAME_REPETITION_TIME_S
                t_start = max(0.0, t_end - 3.0)
                m = (gt_times >= t_start) & (gt_times < t_end)
                beats = gt_times[m]
                
                if len(beats) >= 2:
                    ibi = np.diff(beats)
                    true_bpm = 60.0 / float(np.mean(ibi))
                    
                    # 라벨 정규화 적용
                    if self.label_mean is not None and self.label_std is not None:
                        true_bpm_normalized = (true_bpm - self.label_mean) / self.label_std
                        
                        # 차원 일치 보장 (브로드캐스트 경고 제거)
                        pred_flat = bpm_pred.squeeze().view(-1)        # (1,)
                        target_flat = torch.tensor(true_bpm_normalized).to(self.device).view(-1)    # (1,)
                        
                        # SmoothL1Loss 직접 호출로 브로드캐스트 경고 제거
                        loss = F.smooth_l1_loss(pred_flat, target_flat, reduction='mean', beta=beta_std)
                        total_loss += loss.item()
                        count += 1
        
        return total_loss / max(1, count)
    


if __name__ == "__main__":
    # 디렉터리 기반 학습 및 테스트
    predictor = BPMPredictor(TRAIN_DATA_DIR, TRAIN_ANSWER_DIR)
    predictor.run(test_data_dir=TEST_DATA_DIR, test_answer_dir=TEST_ANSWER_DIR)

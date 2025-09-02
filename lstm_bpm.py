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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from helpers.preproc_lstm import (
    extract_phase_derivative,
)
from helpers.preproc_signal import range_axis_m
from helpers.radar_config import FS_ADC, PAD_FT, B_HZ, NUM_SAMPLES
from typing import Tuple, List, Optional, Dict

# ===== 학습 파라미터 =====
EPOCHS = 1000                   # 에포크
LEARNING_RATE = 1e-4          # 직접 BPM 예측용 낮은 학습률
HIDDEN_DIM = 256
NUM_LAYERS = 1                # LSTM 레이어 2층 및 드롭아웃 적용

VALIDATION_SPLIT = 0.25       # 검증 데이터 비율 (20%로 줄임)
EARLY_STOP_PATIENCE = 100      # 얼리 스탑 인내심 (에포크) - 더 여유롭게
EARLY_STOP_MIN_DELTA = 1e-5   # 최소 개선 임계값 - 더 관대하게

# ===== 스케줄러 파라미터 =====
SCHEDULER_FACTOR = 0.5        # 학습률 감소 비율
SCHEDULER_PATIENCE = 60       # 스케줄러 인내심 (에포크)
SCHEDULER_MIN_LR = 1e-5       # 최소 학습률

# ===== 신호 처리 파라미터 =====
FS          = 36.0            # 프레임레이트 (Hz)
WIN_FRAMES  = int(8.0 * FS)   # 8초 윈도우 = 288 프레임
HOP_FRAMES  = 18*2            # 1초 홉 18*2프레임
# FMIN, FMAX  = 0.5, 3.33       # 심박 대역 [Hz] (30-200 BPM에 대응)
# PAD_FACTOR  = 8               # FFT 패딩 (주파수 해상도 향상)
FEATURE_DIM = 36              # 1D CNN으로 압축할 특징 차원

# ===== 경로 설정 =====
TRAIN_DATA_DIR = "record2/train/data/"
TRAIN_ANSWER_DIR = "record2/train/answer/"
TEST_DATA_DIR = "record2/test/data/"
TEST_ANSWER_DIR = "record2/test/answer/"

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

# ===== 파일 처리 함수들 =====

def calculation(file_path: str) -> Tuple[int, np.ndarray]:
    """레이더 데이터에서 최적 거리 bin과 위상 신호 추출 (전처리된 데이터 또는 원본 데이터 모두 지원)"""
    data = np.load(file_path, allow_pickle=True)
    
    # 새로운 전처리된 데이터 형식인지 확인
    if isinstance(data, np.ndarray) and data.ndim == 0 and isinstance(data.item(), dict):
        # 전처리된 데이터 (딕셔너리 형태)
        processed_dict = data.item()
        fc_bin = processed_dict['fc_bin']
        z_tau = processed_dict['z_tau']
        
        # 거리 축 계산 후 fc_bin의 거리[m] 표기
        rng_axis, _, _ = range_axis_m(FS_ADC, NUM_SAMPLES, PAD_FT, B_HZ)
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

def create_bpm_labels(gt_times, z_tau, window_sec=3.0):
    """ECG 타임스탬프로부터 구간의 평균 BPM 라벨 생성
    
    IBI(Inter-Beat Interval) 기반으로 정확한 BPM 계산:
    - 구간 내 비트 2개 이상: IBI 평균으로 BPM 계산 (60 / mean_IBI)
    """
    T = len(z_tau); labels = []
    for i in range(WIN_FRAMES, T, HOP_FRAMES):
        t_end = i / FS; t_start = max(0.0, t_end - window_sec)
        # 창 안 비트
        m = (gt_times >= t_start) & (gt_times < t_end)
        beats = gt_times[m]
        bpm = 60.0
        if len(beats) >= 2:
            ibi = np.diff(beats)          # 창 내부 IBI들
            bpm = 60.0 / float(np.mean(ibi))
        elif len(beats) == 1:
            # bpm = 0
            raise ValueError("비트가 1개 이하입니다.")
        labels.append(bpm)  # 정규화 없이 직접 BPM 값 사용
    return np.array(labels, dtype=np.float32)

# ===== 데이터 생성 함수 =====
def create_training_data(all_z_tau: List[np.ndarray], all_gt_times: List[np.ndarray], predictor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """모든 파일에서 학습/검증 데이터 생성 (파일 단위 분할)"""
    print("학습 데이터 생성 중...")
    
    # 파일별로 특징과 라벨 수집
    file_features = []
    file_labels = []
    
    for z_tau, gt_times in zip(all_z_tau, all_gt_times):
        bpm_labels = create_bpm_labels(gt_times, z_tau)
        file_feature_list = []
        file_label_list = []
        window_idx = 0
        
        for i in range(WIN_FRAMES, len(z_tau), HOP_FRAMES):
            if window_idx >= len(bpm_labels):
                break
                
            window = z_tau[i-WIN_FRAMES:i]
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
    num_val_files = max(1, int(round(num_files * VALIDATION_SPLIT)))
    num_val_files = min(num_val_files, num_files)
    
    # 20개 단위 구간으로 나누어 각 구간에서 랜덤하게 선택 (1~20, 21~40, 41~60, ...)
    bucket_size = 20
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

# ===== 1D CNN + LSTM BPM 회귀 모델 정의 =====
class BPMRegressionModel(nn.Module):
    """BPM 예측을 위한 1D CNN + LSTM 회귀 모델"""
    
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        
        # 1D CNN: 주파수 특징 압축 및 추출 (LN을 CNN 끝단에 적용)
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(64),
            # nn.Conv1d(64, 64, kernel_size=3, padding=1),  # 출력 채널을 64로 증가
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (N,64,1)
            nn.Flatten(1),              # (N,64)
            nn.LayerNorm(64),   # ★ 한 줄
        )
        
        # LSTM: 시계열 패턴 학습
        self.lstm = nn.LSTM(
            input_size=64,  # CNN 출력 차원 
            hidden_size=hidden,
            num_layers=num_layers,
            # dropout=0.3,   # 층 사이 드롭아웃 (num_layers>1에서만 작동)
            batch_first=True
        )
        
        # 회귀 헤드 (BPM 예측) - 직접 BPM 값 출력 (30-200)
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 4, hidden // 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 16, 1),  # BPM 값 직접 출력
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        # 마지막 레이어를 적절한 범위로 재초기화
        with torch.no_grad():
            self.regressor[-1].weight.normal_(0, 0.05)
            self.regressor[-1].bias.zero_()
        
        # 모델 구조 출력
        print(f"BPM 회귀 모델 구조:")
        print(f"  1D CNN: {input_dim} -> 64")
        print(f"  LSTM: 64 -> {hidden} (layers={num_layers})")
        print(f"  Regressor: {hidden} -> 1 (직접 BPM 출력)")
        
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
                    print(f"[WARNING] {answer_file} 처리 중 오류: {e}")
                    continue
            
            if all_bpms:
                mean_bpm = np.mean(all_bpms)
                print(f"[INFO] 훈련 라벨 평균 계산 완료: {len(all_bpms)}개 파일, 평균 BPM: {mean_bpm:.2f}")
                return float(mean_bpm)
            else:
                print("[WARNING] 유효한 BPM 데이터를 찾을 수 없습니다. 기본값 80.0 사용")
                return 80.0
                
        except Exception as e:
            print(f"[WARNING] 훈련 라벨 평균 계산 중 오류: {e}. 기본값 80.0 사용")
            return 80.0
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: (batch, seq_len, feat_dim)
            hidden: LSTM hidden state
        Returns:
            bpm_pred: (batch, 1) - 예측된 BPM 값
            hidden: 업데이트된 hidden state
        """
        # # x: (B, T, F)
        # mu  = x.mean(dim=1, keepdim=True)                 # (B, 1, F)  시간 평균
        # std = x.std(dim=1, keepdim=True) + 1e-6           # (B, 1, F)
        # x = (x - mu) / std                                # (B, T, F)

        batch_size, seq_len, feat_dim = x.shape
        
        # 1D CNN 적용을 위해 차원 재배열: (batch, seq_len, feat_dim) -> (batch * seq_len, 1, feat_dim)
        x_reshaped = x.view(-1, 1, feat_dim)
        
        # 1D CNN 특징 추출: (batch * seq_len, 1, feat_dim) -> (batch * seq_len, 64, 1)
        conv_out = self.conv1d(x_reshaped)
        
        # LSTM 입력을 위해 차원 재배열: (batch * seq_len, 64, 1) -> (batch, seq_len, 64)
        conv_out = conv_out.squeeze(-1).view(batch_size, seq_len, 64)
        
        # LSTM 처리
        lstm_out, hidden = self.lstm(conv_out, hidden)
        # lstm_out, hidden = self.lstm(x, hidden)

        # 마지막 시점의 출력만 사용 (시퀀스 전체를 고려한 BPM 예측)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        # last_output = lstm_out.mean(dim=1)  # (batch, hidden_dim)
        
        # 직접 BPM 회귀 (선형 출력, 클리핑 제거)
        bpm_pred = self.regressor(last_output)  # (batch, 1)
        
        # 클리핑을 제거하여 모델이 자유롭게 학습하도록 함
        # 손실 계산에서만 범위 제한 적용
        
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
            fc_bin, z_tau = calculation(data_path)
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
            
        return all_z_tau, all_gt_times
    
    def process_window(self, window: np.ndarray) -> np.ndarray:
        """8초 윈도우를 8개 구간으로 나누어 위상미분 신호를 직접 사용 (FFT 없음)"""
        # 8초 윈도우를 8개 서브구간으로 분할 (각 1초, 36프레임)
        sub_intervals = WIN_FRAMES // HOP_FRAMES
        sub_window_size = HOP_FRAMES
        sub_features = []
        
        for i in range(sub_intervals):
            start_idx = i * sub_window_size
            end_idx = (i + 1) * sub_window_size if i < sub_intervals - 1 else len(window)
            
            # 서브윈도우 추출
            sub_window = window[start_idx:end_idx]
            
            # 서브윈도우가 너무 작으면 이전 특징 재사용
            if len(sub_window) < 5:  # 최소 5프레임 필요
                print(f"서브윈도우 크기 부족: {len(sub_window)}")
                raise ValueError(f"서브윈도우 크기 부족: {len(sub_window)}")
            
            # 위상미분 신호를 FFT 없이 직접 사용
            try:
                # 위상미분 추출 (시간 영역 신호, 13프레임) - BPF 없음
                dphi = extract_phase_derivative(sub_window, FS)
                
                # 차원 맞춤: dphi를 FEATURE_DIM으로 조정
                if len(dphi) >= FEATURE_DIM:
                    # 다운샘플링으로 압축
                    step = len(dphi) / FEATURE_DIM
                    compressed = np.zeros(FEATURE_DIM, dtype=np.float32)
                    for j in range(FEATURE_DIM):
                        start_feat = int(j * step)
                        end_feat = int((j + 1) * step)
                        if end_feat > len(dphi):
                            end_feat = len(dphi)
                        if start_feat < end_feat:
                            compressed[j] = np.mean(dphi[start_feat:end_feat])
                    sub_features.append(compressed)
                else:
                    # 제로패딩으로 차원 확장
                    result = np.zeros(FEATURE_DIM, dtype=np.float32)
                    result[:len(dphi)] = dphi
                    sub_features.append(result)
            except Exception as e:
                print(f"에러 발생: {e}")
                raise e
        
        # # 16개 서브구간을 하나의 배열로 결합
        # return np.array(sub_features, dtype=np.float32)  # shape: (16, FEATURE_DIM)
         # (8, FEATURE_DIM)으로 결합
        feats = np.array(sub_features, dtype=np.float32)  # (8, 36)
        # 서브구간(행)별 z-score 정규화 제거: 크기/스케일 정보 보존
        return feats
    

    
    def train_on_multiple_files(self, all_z_tau: List[np.ndarray], all_gt_times: List[np.ndarray], batch_size: int = 32):
        """여러 파일로 DataLoader 배치 학습 (검증 및 얼리 스탑 포함)"""
        first_features = self.process_window(all_z_tau[0][:WIN_FRAMES])
        print(f"특징 차원: {first_features.shape}")
        
        self.model = BPMRegressionModel(input_dim=FEATURE_DIM).to(self.device)
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 파라미터 그룹: 마지막 회귀 레이어에 더 큰 학습률 (항등성으로 분리)
        last_layer_params = list(self.model.regressor[-1].parameters())
        last_param_ids = {id(p) for p in last_layer_params}
        other_params = [p for p in self.model.parameters() if id(p) not in last_param_ids]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": LEARNING_RATE},
            {"params": last_layer_params, "lr": LEARNING_RATE * 5.0},
        ])
        
        # 학습/검증 데이터 생성 및 DataLoader 설정
        X_train, y_train, X_val, y_val = create_training_data(all_z_tau, all_gt_times, self)
        
        # 라벨 정규화 통계 계산 후 Huber Loss의 beta 조정
        beta_bpm = 3.0                             # 2~4 BPM 중 하나로 튜닝
        beta_std = beta_bpm / self.label_std        # z-스케일 임계치
        
        criterion = torch.nn.SmoothL1Loss(beta=beta_std)
        print(f"Huber Loss beta 조정: {beta_bpm} BPM -> {beta_std:.3f} (z-scale)")
        # criterion = torch.nn.SmoothL1Loss(beta=4.0)  # Huber loss
        
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
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 에포크마다 자동 셔플링
            num_workers=0,  # Windows에서 안정적
            pin_memory=torch.cuda.is_available(),
            drop_last=False # 마지막 배치 버리지 않음
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 검증은 셔플 불필요
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        print(f"DataLoader 생성: 훈련 {len(train_dataset)}개, 검증 {len(val_dataset)}개 윈도우")
        
        # 얼리 스탑 변수들
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            # 훈련 단계
            self.model.train()
            train_loss = train_mae = batch_count = 0
            
            for features, labels in train_dataloader:
                features = features.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                bpm_pred, _ = self.model(features, None)
                
                # loss = F.mse_loss(bpm_pred.squeeze(), labels)
                loss = criterion(bpm_pred.squeeze(), labels) # Huber loss (정규화 라벨 기준)
                # MAE는 BPM 단위로 계산 (역정규화)
                if (self.label_mean is not None) and (self.label_std is not None):
                    pred_bpm = bpm_pred.squeeze() * self.label_std + self.label_mean
                    true_bpm = labels * self.label_std + self.label_mean
                    mae = F.l1_loss(pred_bpm, true_bpm)
                else:
                    mae = F.l1_loss(bpm_pred.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                
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
                    labels = labels.squeeze().to(self.device)
                    
                    bpm_pred, _ = self.model(features, None)
                    
                    # 훈련과 동일한 Huber Loss 사용 (정규화 라벨 기준)
                    loss = criterion(bpm_pred.squeeze(), labels)
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
        
        # 최고 성능 모델로 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"최고 성능 모델로 복원 완료 (검증 Loss: {best_val_loss:.6f})")
        
        # 모델 저장
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/bpm_regressor.pt")
        print("\n모델 저장 완료: checkpoints/bpm_regressor.pt")
    
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
        
        # 정답 BPM (이미 실제 BPM 값)
        true_bpms = create_bpm_labels(gt_times, z_tau)
        
        if len(pred_bpms) == 0 or len(true_bpms) == 0:
            return {"rmse": 0.0, "mae": 0.0, "avg_pred_bpm": 0.0, "avg_true_bpm": 0.0}
        
        # 길이 맞춤 및 메트릭 계산
        min_len = min(len(pred_bpms), len(true_bpms))
        pred_array = np.array(pred_bpms[:min_len])
        true_array = np.array(true_bpms[:min_len])
        
        return {
            "rmse": float(np.sqrt(np.mean((pred_array - true_array) ** 2))),
            "mae": float(np.mean(np.abs(pred_array - true_array))),
            "avg_pred_bpm": float(np.mean(pred_array)),
            "avg_true_bpm": float(np.mean(true_array))
        }
    
    def evaluate_predictions(self, z_tau: np.ndarray, gt_times: np.ndarray) -> None:
        """예측 성능을 구간별로 상세 평가 (현재 사용 안함)"""
 
        pred_bpms, window_times = self.predict_bpm(z_tau)
        true_bpms = create_bpm_labels(gt_times, z_tau)
        
        # 길이 맞춤
        min_len = min(len(pred_bpms), len(true_bpms))
        pred_bpms = pred_bpms[:min_len]
        true_bpms = true_bpms[:min_len]
        window_times = window_times[:min_len]
        
        print(f"\n=== 구간별 BPM 예측 성능 평가 ===")
        
        # 원하는 구간 길이(초)와 현재 홉 간격으로부터 구간 당 윈도우 개수 계산
        target_interval_sec = 4
        step_seconds = HOP_FRAMES / FS
        total_duration = len(z_tau) / FS
        interval_size = max(1, int(round(target_interval_sec / step_seconds)))
        for start_idx in range(0, len(pred_bpms), interval_size):
            end_idx = min(start_idx + interval_size, len(pred_bpms))
            
            interval_pred = pred_bpms[start_idx:end_idx]
            interval_true = true_bpms[start_idx:end_idx]
            interval_times = window_times[start_idx:end_idx]
            
            if len(interval_pred) == 0:
                continue
                
            # 구간 내 통계
            pred_mean = float(np.mean(interval_pred))
            true_mean = float(np.mean(interval_true))
            pred_std = float(np.std(interval_pred))
            true_std = float(np.std(interval_true))
            rmse = float(np.sqrt(np.mean((np.array(interval_pred) - np.array(interval_true))**2)))
            
            start_time = float(interval_times[0])
            end_time = float(interval_times[-1])
            # 마지막 구간은 실제 총 길이를 반영해 표시 (예: 60.0초)
            if end_idx == len(pred_bpms):
                end_time = min(end_time + step_seconds, total_duration)
            
            # 한 줄 요약 출력 
            print(f"구간 {start_time:.1f}~{end_time:.1f}초 | 예측 {pred_mean:.2f}±{pred_std:.2f} | 실제 {true_mean:.2f}±{true_std:.2f} | RMSE {rmse:.2f} BPM")
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
        
        for i, (test_data_path, test_answer_path) in enumerate(test_pairs):
            file_num = os.path.splitext(os.path.basename(test_data_path))[0]
            print(f"\n파일 {file_num} 테스트:")
            
            # 테스트 데이터 직접 로딩
            fc_bin, test_z_tau = calculation(test_data_path)
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
    
    def run(self, test_data_dir=None, test_answer_dir=None, batch_size=64):
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

if __name__ == "__main__":
    # 디렉터리 기반 학습 및 테스트
    predictor = BPMPredictor(TRAIN_DATA_DIR, TRAIN_ANSWER_DIR)
    predictor.run(test_data_dir=TEST_DATA_DIR, test_answer_dir=TEST_ANSWER_DIR)

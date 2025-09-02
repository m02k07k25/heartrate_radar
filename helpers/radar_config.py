# -*- coding: utf-8 -*-
"""
radar_config.py - 레이더 시스템 공통 설정 및 상수

이 파일은 모든 레이더 관련 모듈에서 공통으로 사용하는 상수들을 정의합니다.
레이더 설정을 변경할 때는 이 파일만 수정하면 됩니다.
"""

import numpy as np
from typing import Tuple

# ===== 물리 상수 =====
C_LIGHT = 299_792_458.0                    # 빛의 속도 [m/s]

# ===== 레이더 하드웨어 설정 =====
# FMCW 시퀀스 설정
FS_FRAME = 36.0                            # 프레임레이트 [Hz] (목표값)
CHIRP_REPETITION_TIME_S = 434e-6           # Chirp 반복 시간 [s]
NUM_CHIRPS = 64                            # 프레임당 Chirp 수
TDM_MIMO = False                           # TDM MIMO 사용 여부

# Chirp 설정
START_FREQUENCY_HZ = 58e9                  # 시작 주파수 [Hz] (58 GHz)
END_FREQUENCY_HZ = 63.5e9                  # 종료 주파수 [Hz] (63.5 GHz)
SAMPLE_RATE_HZ = 2e6                       # ADC 샘플레이트 [Hz] (2 MHz)
NUM_SAMPLES = 512                          # Chirp당 샘플 수

# RX/TX 설정
RX_MASK = 7                                # RX 안테나 마스크 (0b111 = 3개 RX)
TX_MASK = 1                                # TX 안테나 마스크
TX_POWER_LEVEL = 31                        # TX 전력 레벨
LP_CUTOFF_HZ = 300_000                     # 저역 통과 필터 차단 주파수 [Hz]
HP_CUTOFF_HZ = 20_000                      # 고역 통과 필터 차단 주파수 [Hz]
IF_GAIN_DB = 28                            # IF 증폭기 이득 [dB]

# ===== 파생 상수 =====
# 주파수 관련
B_HZ = END_FREQUENCY_HZ - START_FREQUENCY_HZ  # FMCW 대역폭 [Hz] (5.5 GHz)
F0_HZ = 0.5 * (START_FREQUENCY_HZ + END_FREQUENCY_HZ)  # 중심 주파수 [Hz] (~60.75 GHz)
LAMBDA = C_LIGHT / F0_HZ                   # 파장 [m] (~4.93 mm)

# 시간 관련
FS_ADC = SAMPLE_RATE_HZ                    # ADC 샘플레이트 [Hz] (2 MHz)
FRAME_REPETITION_TIME_S = 1.0 / FS_FRAME   # 프레임 반복 시간 [s] (파생값)
FRAME_HZ = FS_FRAME                         # 프레임레이트 별칭

# 샘플링 관련
N_ADC = NUM_SAMPLES                        # ADC 샘플 수
N_CHIRP = NUM_CHIRPS                       # Chirp 수

# ===== 신호 처리 설정 =====
# FFT 설정
WIN_FT = "hann"                            # Fast-time 윈도우 함수
PAD_FT = 2                                 # Fast-time FFT zero-padding 배수

# 거리 설정
RMIN = 0.2                                 # 최소 탐지 거리 [m]
RMAX = 2.0                                 # 최대 탐지 거리 [m]

# 신호 품질 설정
FUSE_M = 2                                 # fc 선정 시 ±M bin 융합 폭
ALPHA_BR = 0.2                             # 점수에서 호흡 가중치

# ===== RX 안테나 위치 설정 =====
D = 0.5 * LAMBDA                           # RX 간격 [m] (λ/2)
RX_POS_M = np.array([0.0, D, 2*D], dtype=np.float64)  # [x0, x1, x2] on a line

# ===== 데이터 수집 설정 =====
TARGET_FRAMES = 60 * 36                    # 목표 프레임 수 (60초 × 36 Hz = 2160 프레임)

# ===== 유틸리티 함수 =====
def get_radar_config() -> dict:
    """레이더 설정을 딕셔너리로 반환"""
    return {
        'frame_repetition_time_s': FRAME_REPETITION_TIME_S,
        'chirp_repetition_time_s': CHIRP_REPETITION_TIME_S,
        'num_chirps': NUM_CHIRPS,
        'tdm_mimo': TDM_MIMO,
        'start_frequency_hz': START_FREQUENCY_HZ,
        'end_frequency_hz': END_FREQUENCY_HZ,
        'sample_rate_hz': SAMPLE_RATE_HZ,
        'num_samples': NUM_SAMPLES,
        'rx_mask': RX_MASK,
        'tx_mask': TX_MASK,
        'tx_power_level': TX_POWER_LEVEL,
        'lp_cutoff_hz': LP_CUTOFF_HZ,
        'hp_cutoff_hz': HP_CUTOFF_HZ,
        'if_gain_db': IF_GAIN_DB,
        'b_hz': B_HZ,
        'f0_hz': F0_HZ,
        'lambda': LAMBDA,
        'fs_adc': FS_ADC,
        'fs_frame': FS_FRAME,
        'rmin': RMIN,
        'rmax': RMAX,
        'fuse_m': FUSE_M,
        'alpha_br': ALPHA_BR,
        'target_frames': TARGET_FRAMES
    }

def print_radar_config():
    """레이더 설정을 출력"""
    config = get_radar_config()
    print("=== 레이더 시스템 설정 ===")
    for key, value in config.items():
        if isinstance(value, float):
            if value >= 1e6:
                print(f"{key:25s}: {value/1e6:.1f} M")
            elif value >= 1e3:
                print(f"{key:25s}: {value/1e3:.1f} k")
            else:
                print(f"{key:25s}: {value}")
        else:
            print(f"{key:25s}: {value}")
    print("=" * 30)

if __name__ == "__main__":
    print_radar_config()

# preprocessing.py - 레이더 데이터 전처리 함수들 + 캘리브레이션 적용
# python -m helpers.preproc_signal 으로 실행

import numpy as np
from scipy.signal import butter, filtfilt, get_window, welch
from typing import Tuple, List, Optional, Dict

# ===== 공통 설정 임포트 =====
from .radar_config import *

def fast_time_fft(iq_cube, win_name="hann", pad_fact=2):
    """
    시간 축에 FFT를 해서 거리 주파수로 변경
    입력: iq_cube shape = (F, RX, C, N) (복소)
    출력: Rfft_rxmean_T shape = (range_bins, τ=F) (복소), 그리고 N(샘플/치프)
    """
    # Numpy 배열로 변환 + 복소수로 통일
    X = np.ascontiguousarray(np.asarray(iq_cube, dtype=np.complex64))
    F, RX, C, N = X.shape
    # 윈도우 적용 (스펙트럼 누설 현상을 줄이기 위해)
    win = get_window(win_name, N).astype(np.float32) 
    nfft = pad_fact * N

    # 복소 FFT → 양의 주파수만 음의 주파수 사용 X (효율)
    Xw = X * win[None, None, None, :]
    Rfft_full = np.fft.fft(Xw, n=nfft, axis=-1)     # 시간 축인 Xw -> 주파수 축으로 변환
    # Rfft_full shape = (F, RX, C, nfft)
    Rfft = Rfft_full[..., : nfft//2 + 1]            # (F, RX, C, nbin), 양의 주파수만 사용

    # 프레임 내 chirp 평균 → (F, RX, nbin)
    Rfft_cavg = Rfft.mean(axis=2)                   # C(chirp) 축 평균
    return Rfft_cavg, N     # (F, RX, nbin) + N(샘플/처프)

def range_axis_m(fs_adc, n_samp, pad_fact, B_hz):
    '''
    흉벽 거리 축 생성
    '''
    T = n_samp / fs_adc
    S = B_hz / T                      # chirp slope
    nfft = pad_fact * n_samp          # FFT할 데이터 길이 (프레임수 * 패딩) = 2048
    fb_full = np.fft.fftfreq(nfft, d=1/fs_adc)  # ±대역
    fb = fb_full[: nfft//2 + 1]                 # 0..Nyquist
    rng = (C_LIGHT * fb) / (2 * S)                   # [m]
    return rng.astype(np.float64), float(S), int(nfft)

def remove_static_clutter(Rft_bins_T):
    """
    고정 클러터 제거 = 같은 range bin에서 τ-평균(복소) 제거
    입력: (nbin, F) 거리-시간 데이터 (복소)
    출력: (nbin, F) 에서 slow-time 평균 제거된 신호 + (nbin,) 평균(클러터 성분)
    """
    Xdc = Rft_bins_T.mean(axis=1, keepdims=True)
    return (Rft_bins_T - Xdc).astype(np.complex64), Xdc.squeeze()

def band_power_welch(x: np.ndarray, fs: float, f1: float, f2: float) -> float:
    """
    Welch's method를 사용하여 신호의 지정된 주파수 대역의 총 파워를 계산합니다.
    시간축(slow-time) 시계열 x에 대해 Welch 방법으로 PSD(전력스펙트럼밀도)를 구하고,
    주파수 구간 [f1, f2] Hz에 해당하는 PSD를 적분하여 "대역 파워"를 반환합니다.

    Args:
        x (np.ndarray): 입력 시계열 신호.
        fs (float): 신호의 샘플링 주파수 (Hz).
        f1 (float): 관심 대역의 시작 주파수 (Hz).
        f2 (float): 관심 대역의 끝 주파수 (Hz).
    Returns:
        float: 계산된 대역 파워.
    """
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 512))
    m = (f >= f1) & (f <= f2)
    return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0

def phase_consistency(phi: np.ndarray) -> float:
    """
    위상 시퀀스(phi)의 '원형 일관성' 지표.
    입력은 이미 드리프트 제거(HPF 등)된 위상(언랩)이어야 함.
    값 범위: 0~1 (주기적/안정할수록 1에 가까움)
    """
    u = np.exp(1j * phi)        # 위상을 단위원 복소수로 투영
    return float(np.abs(np.mean(u)))

def select_fc_by_phase_score(
    R_bar: np.ndarray, 
    rng_axis: np.ndarray, 
    fs_frame: float, 
    rmin: float, 
    rmax: float, 
    fuse_M: int, 
    alpha_br: float
 ) -> Tuple[int, np.ndarray]:
    """
    위상 신호 품질 점수를 기반으로 최적의 거리 bin(fc)을 자동으로 선택합니다.

    이 함수는 각 거리 bin의 신호를 평가하여 호흡/심박과 같은 생체 신호가
    가장 뚜렷하게 나타나는 최적의 위치를 찾습니다.

    Args:
        R_bar (np.ndarray): 정지 클러터가 제거된 거리-시간 맵. shape=(nbin, F), 복소수.
        rng_axis (np.ndarray): 각 bin에 해당하는 거리 축 (m).
        fs_frame (float): 프레임레이트 (슬로우타임 샘플링 주파수, Hz).
        rmin (float): 탐색할 최소 거리 (m).
        rmax (float): 탐색할 최대 거리 (m).
        fuse_M (int): 신호 품질 향상을 위해 현재 bin의 주변 ±M개 bin을 융합할 폭.
        alpha_br (float): 최종 점수 계산 시 호흡 신호 파워에 대한 가중치.

    Returns:
        Tuple[int, np.ndarray]: (최적의 bin 인덱스, 해당 bin의 융합된 복소수 시계열 신호)
    """
    B, F = R_bar.shape
    # 1. 지정된 거리(rmin ~ rmax) 내의 후보 bin 인덱스를 찾습니다.
    cand = np.where((rng_axis >= rmin) & (rng_axis <= rmax))[0]
    if cand.size == 0:
        raise RuntimeError("탐색 거리 구간 안에 유효한 range bin이 없습니다.")

    # 2. 신호 품질 평가를 위한 기준 노이즈 레벨을 계산합니다.
    mag = np.abs(R_bar[cand])
    noise = np.median(mag)

    best_score, best_bin, best_z = -np.inf, None, None
    # 3. 위상 신호의 저주파 드리프트(느린 움직임, 0.05Hz)를 제거
    bhp05, ahp05 = butter(2, 0.05/(fs_frame/2), btype='highpass')  # breathing용
    bhp30, ahp30 = butter(2, 0.30/(fs_frame/2), btype='highpass')  # HR/일관성용 (호흡 필요 X)

    # 4. 모든 후보 bin을 순회하며 품질 점수를 계산합니다.
    for b in cand:
        # 4-1. 주변 bin(±fuse_M)의 신호를 가중 평균하여 융합 신호(z)를 생성합니다. (SNR 향상)
        lo, hi = max(0, b - fuse_M), min(B, b + fuse_M + 1) # 인덱스 경계 처리
        sliceR = R_bar[lo:hi]  # (<=2M+1, F), 주변 신호 슬라이스
        w = np.mean(np.abs(sliceR), axis=1)**1.5 # 신호가 강한 bin에 더 높은 가중치 부여
        w = w / (w.sum() + 1e-12) # 가중치 정규화
        z = (sliceR.T @ w).astype(np.complex64)  # (F,), 가중 평균된 최종 신호

        # 4-2. 융합 신호(z)에서 위상(phi)을 추출하고 필터링하여 드리프트를 제거합니다.
        phi = np.unwrap(np.angle(z)) # 2π 래핑 현상 보정
        # phi_d = filtfilt(bhp, ahp, phi) 

        # 제로-페이즈 필터링으로 시간 지연 없이 드리프트 제거
        phi_d05 = filtfilt(bhp05, ahp05, phi)
        phi_d35 = filtfilt(bhp30, ahp30, phi)

        # 4-3. 품질 점수 계산을 위한 지표들을 추출합니다.
        # - P_br: 호흡 대역(0.1-0.5Hz)의 파워
        # - P_hr: 심박 대역(0.8-3.0Hz)의 파워
        # - C: 위상 안정성/일관성 (주기적일수록 1에 가까워짐)
        # - snr_ok: 신호가 노이즈보다 충분히 강한지 여부
        P_br = band_power_welch(phi_d05, fs_frame, 0.10, 0.50)
        P_hr = band_power_welch(phi_d35, fs_frame, 0.80, 3.00)
        C = phase_consistency(phi_d35)
        snr_ok = (np.mean(np.abs(z)) > noise * 1.5)

        # 4-4. 최종 점수 계산: 심박/호흡 파워가 높고, 위상이 안정적이며, SNR이 좋을수록 높은 점수
        score = (P_hr + alpha_br * P_br) * C * (1.0 if snr_ok else 0.0)
        
        # 4-5. 최고 점수를 갱신합니다.
        if score > best_score:
            best_score, best_bin, best_z = score, int(b), z

    return best_bin, best_z.astype(np.complex64)

# ===== RX 캘리브레이션 관련 함수들 =====
# run_calibration.py에서 캘리브레이션 함수들을 import하여 사용
try:
    from .run_calibration import get_calibration_for_fast_fft, apply_calibration_for_fast_fft
    CALIBRATION_AVAILABLE = True
    print("[FAST_FFT] 캘리브레이션 모듈 로드 성공")
except ImportError as e:
    print(f"[FAST_FFT] 캘리브레이션 모듈 로드 실패: {e}")
    raise ImportError("run_calibration.py를 찾을 수 없습니다. 캘리브레이션 기능이 필요합니다.")

def process_radar_data(iq_cube: np.ndarray, use_calibration: bool = True) -> np.ndarray:
    """
    collect_dataset_new.py에서 사용할 레이더 데이터 전처리 함수
    
    Args:
        iq_cube: 원본 레이더 데이터 (F, RX, C, N)
        use_calibration: RX 캘리브레이션 사용 여부
    
    Returns:
        processed_data: 전처리된 데이터 (축소된 크기)
    """
    try:
        # 0. RX 캘리브레이션 적용 (선택적)
        calibrated_iq = iq_cube
        if use_calibration and CALIBRATION_AVAILABLE:
            calibrated_iq = apply_calibration_for_fast_fft(iq_cube)
            if calibrated_iq is not iq_cube:  # 캘리브레이션이 적용된 경우
                print(f"[FAST_FFT] RX 캘리브레이션 적용됨")
            else:
                raise ValueError("캘리브레이션 적용 실패")
        elif use_calibration and not CALIBRATION_AVAILABLE:
            print(f"[FAST_FFT] 캘리브레이션 기능 비활성화됨")
        
        # 1. fast-time FFT → 거리 변환
        Rft, N_samp = fast_time_fft(calibrated_iq, win_name=WIN_FT, pad_fact=PAD_FT)  # (F, RX, nbin)

        # 2. 거리 축 생성
        rng_axis, _, _ = range_axis_m(FS_ADC, N_samp, PAD_FT, B_HZ)

        # 3. RX 차원 평균화 (거리-시간 형태로 변환)
        if use_calibration:
            if CALIBRATION_AVAILABLE:
                # 캘리브레이션이 적용된 경우: 가중 평균
                cal_vec, _ = get_calibration_for_fast_fft()
                if cal_vec is not None:
                    # 보정 계수의 역수를 가중치로 사용 (보정된 신호의 품질 반영)
                    weights = 1.0 / np.abs(cal_vec).astype(np.float32)
                    weights = weights / weights.sum()  # 정규화
                    Rft_dt = np.average(Rft, axis=1, weights=weights)  # (F, nbin)
                    print(f"[FAST_FFT] RX 가중 평균 적용 (캘리브레이션 기반)")
                else:
                    raise ValueError("캘리브레이션 데이터가 존재하지 않습니다. 캘리브레이션을 먼저 수행하세요.")
            else:
                raise ValueError("캘리브레이션 모듈을 찾을 수 없습니다. run_calibration.py가 필요합니다.")
        else:
            raise ValueError("캘리브레이션 모듈을 찾을 수 없습니다. run_calibration.py가 필요합니다.")

        Rft_dt = Rft_dt.T  # (nbin, F) 형태로 변환

        # 4. 정지 클러터 제거
        R_bar, _ = remove_static_clutter(Rft_dt)

        # 5. 최적 거리 bin 자동 선정 및 복소 슬로우타임 신호 추출
        fc_bin, z_tau = select_fc_by_phase_score(
            R_bar, rng_axis, fs_frame=FRAME_HZ,
            rmin=RMIN, rmax=RMAX, fuse_M=FUSE_M, alpha_br=ALPHA_BR
        )
        
        # 6. 반환: 선택된 거리 bin 인덱스와 해당 위치의 시계열 신호
        # 원본 대비 크기가 대폭 축소됨 (F x RX x C x N) -> (F,) 복소수
        result = {
            'fc_bin': fc_bin,
            'z_tau': z_tau,
            'calibration_applied': use_calibration and CALIBRATION_AVAILABLE and calibrated_iq is not iq_cube,
            'shape_info': {
                'original_shape': iq_cube.shape,
                'processed_shape': z_tau.shape,
                'compression_ratio': np.prod(iq_cube.shape) / np.prod(z_tau.shape)
            }
        }
        
        cal_status = "적용됨" if result['calibration_applied'] else "없음"
        print(f"[FAST_FFT] 전처리 완료: fc_bin={fc_bin}({rng_axis[fc_bin]:.2f}m), 캘리브레이션={cal_status}, 압축비={result['shape_info']['compression_ratio']:.1f}:1")
        
        return result
        
    except Exception as e:
        print(f"[FAST_FFT] 전처리 오류: {e}")
        raise e


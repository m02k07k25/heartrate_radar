# run_calibration.py - RX 캘리브레이션 통합 시스템
# python -m helpers.run_calibration 으로 실행
"""
RX 캘리브레이션을 실행하고 관리하는 통합 시스템입니다.
- 캘리브레이션 데이터 수집 및 계산
- 캘리브레이션 파일 저장/로드
- fast_fft.py에서 사용할 캘리브레이션 적용 함수들
"""

import os
import sys
import json
import time
import datetime as dt
import numpy as np
from typing import Optional, Tuple, Dict
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

# ===== 레이더 설정 (radar_config.py에서 가져옴) =====
from helpers.radar_config import *

RADAR_CFG = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=FRAME_REPETITION_TIME_S,
    chirp_repetition_time_s=CHIRP_REPETITION_TIME_S,
    num_chirps=NUM_CHIRPS,
    tdm_mimo=TDM_MIMO,
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=START_FREQUENCY_HZ,
        end_frequency_Hz=END_FREQUENCY_HZ,
        sample_rate_Hz=SAMPLE_RATE_HZ,
        num_samples=NUM_SAMPLES,
        rx_mask=RX_MASK,
        tx_mask=TX_MASK,
        tx_power_level=TX_POWER_LEVEL,
        lp_cutoff_Hz=LP_CUTOFF_HZ,
        hp_cutoff_Hz=HP_CUTOFF_HZ,
        if_gain_dB=IF_GAIN_DB,
    ),
)
# ======================================================

# ── HEALTH HELPERS ───────────────────────────────────────────────────────────
def _rx_phase_coherence(Z: np.ndarray, ref_idx: int = 0) -> float:
    """
    Z: (F, RX) at one bin. 시간에 따른 RX 간 위상차의 '안정도'(0~1).
    ref_idx 대비 각 RX의 위상차가 시간 축으로 얼마나 일관적인지 평균.
    """
    phi = np.angle(Z)                           # (F,RX)
    dphi = phi - phi[:, [ref_idx]]              # ref 대비
    C = np.abs(np.mean(np.exp(1j * dphi), axis=0))  # (RX,)
    if C.size <= 1:
        return 1.0
    return float(np.mean(np.delete(C, ref_idx)))    # ref 제외 평균

def _phase_bias_deg_against_ref(Z: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """
    Z: (F,RX). ref 대비 RX별 '상수 위상 오프셋'(deg)을 추정해 반환 (길이 RX).
    """
    r = np.mean(Z * np.conj(Z[:, [ref_idx]]), axis=0)   # (RX,)
    return np.degrees(np.angle(r)).astype(np.float32)

def _alignment_factor(Z: np.ndarray) -> float:
    """
    Z: (F,RX). 공간 정렬도(0~1). |Σ z_m| / Σ |z_m| 를 프레임 평균.
    1에 가까울수록 동상합 준비가 잘 된 상태.
    """
    eps = 1e-12
    num = np.abs(np.sum(Z, axis=1))            # (F,)
    den = np.sum(np.abs(Z), axis=1) + eps      # (F,)
    return float(np.mean(num / den))

def _coherent_gain_db(Z: np.ndarray) -> float:
    """
    Z: (F,RX). 코히어런트 합 이득[dB] ≈ 10log10(E{|Σz|^2} / E{Σ|z|^2}).
    """
    eps = 1e-12
    num = np.mean(np.abs(np.sum(Z, axis=1))**2)           # E{|Σz|^2}
    den = np.mean(np.sum(np.abs(Z)**2, axis=1)) + eps     # E{Σ|z|^2}
    return float(10.0 * np.log10((num / den) + eps))

# ===== 캘리브레이션 데이터 수집 =====
def capture_calibration_data(duration_s: float = 15.0) -> np.ndarray:
    """캘리브레이션용 레이더 데이터 수집"""
    device = None
    try:
        dev_list = DeviceFmcw.get_list()
        if not dev_list:
            raise RuntimeError("레이더 장치를 찾을 수 없습니다.")
        device = DeviceFmcw(uuid=dev_list[0])

        seq = device.create_simple_sequence(RADAR_CFG)
        device.set_acquisition_sequence(seq)

        print(f"[CALIBRATION] 캘리브레이션 시작: {duration_s}초 동안 데이터 수집")
        
        frames = []
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_s:
            frame_data = device.get_next_frame()
            frame = frame_data[0]  # (RX, C, N)
            frames.append(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[CALIBRATION] {frame_count} 프레임 수집됨 ({elapsed:.1f}s)")
                
    except Exception as e:
        print(f"[CALIBRATION] 캡처 오류: {e}")
        raise e
    finally:
        if device is not None:
            try:
                device.stop_acquisition()
            except:
                pass

    iq = np.stack(frames, axis=0)      # (F, RX, C, N)
    if not np.iscomplexobj(iq):
        iq = iq.astype(np.float32).astype(np.complex64)
    
    print(f"[CALIBRATION] 수집 완료: {iq.shape}, {len(frames)} 프레임")
    return iq

# ===== DSP 유틸리티 함수들 =====
def range_axis_m(fs_adc: float, N: int, pad_fact: int, B_hz: float) -> Tuple[np.ndarray, float, int]:
    """한쪽(one-sided) FFT bin에 대응하는 range 축을 생성"""
    nfft = pad_fact * N
    nbin = nfft//2 + 1
    dR = C_LIGHT / (2.0 * B_hz)
    rng = np.arange(nbin, dtype=np.float32) * dR
    return rng, dR, nbin

def range_fft_cavg(iq_cube: np.ndarray, win_name: str = "hann", pad_fact: int = 2) -> np.ndarray:
    """fast-time FFT 후 프레임 내 chirp 축을 코히어런트 평균 → (F,RX,nbin)"""
    from scipy.signal import get_window
    
    X = np.ascontiguousarray(np.asarray(iq_cube, dtype=np.complex64))
    F, RX, C, N = X.shape
    win = get_window(win_name, N).astype(np.float32)
    nfft = pad_fact * N
    R = np.fft.fft(X * win[None,None,None,:], n=nfft, axis=-1)[..., : nfft//2 + 1]
    Rfft_cavg = R.mean(axis=2)  # chirp 축 평균
    return Rfft_cavg.astype(np.complex64)

# ===== 캘리브레이션 계산 =====
def estimate_rx_calibration_complex_ls(
    Rfft_cavg: np.ndarray,
    ref_rx: int = 2,
    target_bin: Optional[int] = None,
    target_range_m: Optional[float] = None,
    pad_fact: int = 2,
    fs_adc: float = FS_ADC,
    B_hz: float  = B_HZ,
    eps: float = 1e-12,
    clip_gain: Optional[float] = 6.0,
    rwin_m: Optional[Tuple[float, float]] = (0.4, 2.5),  # 사람/타깃 거리창 [m]
    min_bin: int = 1,  # 0-bin(DC) 제외
    auto_ref: bool = False,  # 참조 RX 자동 선택
) -> Tuple[np.ndarray, int]:
    """
    프레임들에 대해 최소제곱으로 RX 보정 벡터 c(진폭+위상)를 추정:
        c_m = (Σ_t X_ref(t) * conj(X_m(t))) / (Σ_t |X_m(t)|^2)
    ref 채널은 1+0j로 고정. (스케일 기준)

    개선사항:
    - 0-bin(DC) 제외: 정지 클러터/루프백 피함
    - 거리창 적용: 사람/타깃 거리 범위만 고려
    - 참조 RX 자동 선택: 가장 SNR 좋은 채널을 기준으로
    """
    F, RX, nbin = Rfft_cavg.shape
    rng_axis, _, _ = range_axis_m(fs_adc, N_ADC, pad_fact, B_hz)

    # 후보 bin 집합 만들기 (거리창 + 0-bin 제외)
    if target_bin is None:
        if target_range_m is None:
            # 거리창 내 후보 bin들
            if rwin_m is None:
                cand = np.arange(min_bin, nbin)
            else:
                mask = (rng_axis >= rwin_m[0]) & (rng_axis <= rwin_m[1])
                cand = np.where(mask)[0]
                cand = cand[cand >= min_bin]

            if cand.size == 0:
                print(f"[CALIBRATION] 거리창 {rwin_m}m에 후보 bin 없음, 전체 범위에서 선택")
                cand = np.arange(min_bin, nbin)

            # 후보 중 가장 강한 bin 선택
            mag = np.abs(Rfft_cavg[:, :, cand]).mean(axis=(0, 1))  # (len(cand),)
            target_bin = int(cand[int(np.argmax(mag))])
            print(f"[CALIBRATION] 선택된 타깃 bin: {target_bin} (거리: {rng_axis[target_bin]:.2f}m)")
        else:
            target_bin = int(np.argmin(np.abs(rng_axis - float(target_range_m))))
            print(f"[CALIBRATION] 지정된 거리 {target_range_m}m에 해당하는 bin: {target_bin}")

    X = Rfft_cavg[:, :, target_bin]   # (F, RX)

    # ✅ 참조 RX 자동 선택 (가장 SNR 좋은 채널)
    if auto_ref:
        snr_per_rx = np.abs(X).mean(axis=0)  # 각 RX의 평균 진폭
        ref_rx = int(np.argmax(snr_per_rx))
        print(f"[CALIBRATION] 자동 선택된 참조 RX: {ref_rx} (SNR: {snr_per_rx})")

    Xref = X[:, ref_rx]               # (F,)
    num = (Xref[:, None] * np.conj(X)).sum(axis=0)     # (RX,)
    den = (np.abs(X)**2).sum(axis=0) + eps             # (RX,)
    c = (num / den).astype(np.complex64)

    # 기준 채널 고정
    c[ref_rx] = np.complex64(1.0 + 0j)

    # 과도 이득 안정화(옵션)
    if clip_gain is not None:
        g = np.abs(c); scale = np.minimum(1.0, float(clip_gain) / (g + eps))
        c = (c * scale.astype(c.dtype))

    return c, int(target_bin)

# ===== 캘리브레이션 저장/로드 =====
def save_calibration_npz(path: str, cal_vec: np.ndarray, meta: dict) -> None:
    """캘리브레이션을 .npz 파일로 저장"""
    np.savez(path,
             rx_cal=np.asarray(cal_vec, dtype=np.complex64).reshape(-1),
             meta=json.dumps(meta))

def load_calibration_npz(path: str = "rx_calibration.npz") -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """캘리브레이션 파일을 로드"""
    if not os.path.exists(path):
        print(f"[CALIBRATION] 캘리브레이션 파일 없음: {path}")
        return None, None
    
    try:
        z = np.load(path, allow_pickle=True)
        cal_vec = z["rx_cal"].astype(np.complex64)
        meta = json.loads(str(z["meta"]))
        print(f"[CALIBRATION] 캘리브레이션 로드 성공: {path}")
        return cal_vec, meta
    except Exception as e:
        print(f"[CALIBRATION] 캘리브레이션 로드 오류: {e}")
        return None, None

# ===== 캘리브레이션 적용 =====
def apply_rx_calibration_to_iq(iq_cube: np.ndarray, cal_vec: np.ndarray) -> np.ndarray:
    """
    원본 IQ 데이터에 RX 캘리브레이션을 적용합니다.

    Args:
        iq_cube: (F, RX, C, N) 원본 데이터
        cal_vec: (RX,) 캘리브레이션 벡터
    
    Returns:
        calibrated_iq: (F, RX, C, N) 보정된 데이터
    """
    cal = np.asarray(cal_vec, dtype=np.complex64).reshape(1, -1, 1, 1)  # (1, RX, 1, 1)
    return (iq_cube * cal).astype(np.complex64)

def apply_rx_calibration_to_spectrum(Rfft_cavg: np.ndarray, cal_vec: np.ndarray) -> np.ndarray:
    """
    FFT 스펙트럼 데이터에 RX 캘리브레이션을 적용합니다.

    Args:
        Rfft_cavg: (F, RX, nbin) FFT 스펙트럼 데이터
        cal_vec: (RX,) 캘리브레이션 벡터

    Returns:
        calibrated_spectrum: (F, RX, nbin) 보정된 스펙트럼 데이터
    """
    cal = np.asarray(cal_vec, dtype=np.complex64).reshape(1, -1, 1)  # (1, RX, 1)
    return (Rfft_cavg * cal).astype(np.complex64)

# --- Angle steering helpers ---
def _estimate_angle_from_pair(Rfft_cavg_cal: np.ndarray, k_bin: int,
                              rx_i: int, rx_j: int,
                              d_i: float, d_j: float,
                              lam: float) -> float:
    Xi = Rfft_cavg_cal[:, rx_i, k_bin].mean()
    Xj = Rfft_cavg_cal[:, rx_j, k_bin].mean()
    dphi = np.angle(Xi * np.conj(Xj))
    d = float(d_j - d_i)
    return float(np.arcsin(np.clip((lam/(2*np.pi*d)) * dphi, -1.0, 1.0)))

def _steering_weights(theta_rad: float, d_list, lam: float) -> np.ndarray:
    k = 2*np.pi/lam
    d = np.asarray(d_list, dtype=np.float64)
    return np.exp(-1j * k * d * np.sin(theta_rad)).astype(np.complex64)

def steering_weights(theta_rad: float, d_list, lam: float) -> np.ndarray:
    """각도 θ로 스티어링한 빔 가중치 w_m = exp(-j k d_m sinθ)"""
    return _steering_weights(theta_rad, d_list, lam)

def steer_and_sum(Rfft_cavg_cal: np.ndarray, k_bin: int,
                  theta_rad: float | None = None,
                  d_list=None, lam: float = None) -> tuple[np.ndarray, float]:
    """
    보정 후 (F,RX,nbin)에서 선택 bin(k_bin)을 각도 θ로 스티어링해서 합산.
    θ가 None이면 RX0/RX2 위상차로 근사 추정.
    Returns: (y[F], theta_used)
    """
    if d_list is None:
        d_list = RX_POS_M
    if lam is None:
        lam = LAMBDA
    if theta_rad is None:
        theta_rad = _estimate_angle_from_pair(Rfft_cavg_cal, k_bin,
                                              rx_i=0, rx_j=2,
                                              d_i=d_list[0], d_j=d_list[-1],
                                              lam=lam)
    w = _steering_weights(theta_rad, d_list, lam)             # (RX,)
    y = (Rfft_cavg_cal[:, :, k_bin] * w[None, :]).sum(axis=1) # (F,)
    return y.astype(np.complex64), float(theta_rad)
    
def apply_rx_calibration_to_spectrum(Rfft_cavg: np.ndarray, cal_vec: np.ndarray) -> np.ndarray:
    """
    (F, RX, nbin) 스펙트럼에 RX 보정 벡터를 브로드캐스트 곱.
    """
    cal = np.asarray(cal_vec, dtype=np.complex64).reshape(-1)
    assert Rfft_cavg.shape[1] == cal.shape[0], "RX 개수 불일치"
    return (Rfft_cavg * cal[None, :, None]).astype(np.complex64)

# ===== 메인 캘리브레이션 파이프라인 =====
def run_calibration_and_save(
    out_path: str = "rx_calibration.npz",
    duration_s: float = 10.0,
    ref_rx: int = 0,
    pad_fact: int = 2,
    target_range_m: Optional[float] = None
) -> dict:
    """
    1) 1분간 프레임 수집
    2) fast-time FFT & chirp 평균 → (F,RX,nbin)
    3) 복소(진폭+위상) 최소제곱으로 보정벡터 c 추정
    4) 파일 저장(.npz) + HEALTH 체크(보정 효과 가시화)
    """
    # 1) 캡처
    iq_cube = capture_calibration_data(duration_s=duration_s)      # (F,RX,C,N)

    # 2) FFT+chirp 평균
    Rfft_cavg = range_fft_cavg(iq_cube, win_name="hann", pad_fact=pad_fact)  # (F,RX,nbin)

    # 3) 보정 벡터 추정(복소 이득+위상)
    cal_vec, used_bin = estimate_rx_calibration_complex_ls(
        Rfft_cavg,
        ref_rx=ref_rx,
        target_bin=None,                 # 자동 선택(가장 강한 bin)
        target_range_m=target_range_m,   # 지정 시 그 근처 bin 선택
        pad_fact=pad_fact,
        eps=1e-12,
        clip_gain=6.0,
        rwin_m=(0.4, 2.5),               # 사람/타깃 거리창 [m]
        min_bin=1,                       # 0-bin(DC) 제외
        auto_ref=True                    # 참조 RX 자동 선택
    )

    # 실제 사용된 ref 인덱스 추정(보정에서 ref는 1+0j로 남음)
    cand = np.where(np.isclose(cal_vec.real, 1.0) & np.isclose(cal_vec.imag, 0.0))[0]
    ref_used = int(cand[0]) if cand.size else int(ref_rx)

    # 4) (데모) 보정 적용 → 스티어링합
    R_cal = apply_rx_calibration_to_spectrum(Rfft_cavg, cal_vec)   # (F,RX,nbin)
    y_steered, theta_used = steer_and_sum(R_cal, used_bin, theta_rad=None,
                                          d_list=RX_POS_M, lam=LAMBDA)
    print(f"  steering θ ≈ {np.degrees(theta_used):.1f}°  (bin={used_bin}, F={len(y_steered)})")

    # --- [HEALTH CHECK] 보정 효과 가시화 ---
    Z0 = Rfft_cavg[:, :, used_bin]                    # (F,RX) 보정 전
    Z1 = R_cal[:, :, used_bin]                        # (F,RX) 보정 후

    # 4-1) 시간적 위상 일관성(지표 특성상 오프셋과 무관)
    C_before = _rx_phase_coherence(Z0, ref_idx=ref_used)
    C_after  = _rx_phase_coherence(Z1, ref_idx=ref_used)

    # 4-2) 상수 위상 오프셋(phase bias) 전/후
    bias_before_deg = _phase_bias_deg_against_ref(Z0, ref_idx=ref_used)  # (RX,)
    bias_after_deg  = _phase_bias_deg_against_ref(Z1, ref_idx=ref_used)  # (RX,)

    # 4-3) 공간 정렬도(Alignment factor) 전/후
    AF_before = _alignment_factor(Z0)
    AF_after  = _alignment_factor(Z1)

    # 4-4) 코히어런트 합 이득 전/후
    gain0_db = _coherent_gain_db(Z0)
    gain1_db = _coherent_gain_db(Z1)

    print(f"[HEALTH] RX phase coherence: before={C_before:.3f} → after={C_after:.3f}")
    print(f"[HEALTH] phase bias before(deg): {np.round(bias_before_deg, 2)}")
    print(f"[HEALTH] phase bias after (deg): {np.round(bias_after_deg,  2)}")
    print(f"[HEALTH] alignment factor:      before={AF_before:.3f} → after={AF_after:.3f}")
    print(f"[HEALTH] coherent-sum gain:     before={gain0_db:.2f} dB → after={gain1_db:.2f} dB")

    # 5) 저장 (HEALTH 메타 포함)
    meta = dict(
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
        ref_rx=int(ref_used),
        used_bin=int(used_bin),
        fs_adc=float(FS_ADC),
        B_hz=float(B_HZ),
        n_adc=int(N_ADC),
        n_chirp=int(N_CHIRP),
        frame_rate=float(FS_FRAME),
        pad_fact=int(pad_fact),
        f0_hz=float(F0_HZ),
        lambda_m=float(LAMBDA),
        rx_pos_m=list(map(float, RX_POS_M)),
        duration_s=float(duration_s),
        target_range_m=None if target_range_m is None else float(target_range_m),
        theta_demo_deg=float(np.degrees(theta_used)),
        health=dict(
            coherence_before=float(C_before),
            coherence_after=float(C_after),
            phase_bias_deg_before=list(map(float, bias_before_deg)),
            phase_bias_deg_after=list(map(float, bias_after_deg)),
            alignment_factor_before=float(AF_before),
            alignment_factor_after=float(AF_after),
            coherent_gain_db_before=float(gain0_db),
            coherent_gain_db_after=float(gain1_db),
        ),
    )
    save_calibration_npz(out_path, cal_vec, meta)

    # 간단 로그
    print(f"[OK] Calibration saved -> {out_path}")
    print(f"  used_bin={used_bin}  ref_rx={ref_used}  |cal|={np.abs(cal_vec)}  ∠cal(deg)={np.rad2deg(np.angle(cal_vec))}")

    return meta


# ===== fast_fft.py에서 사용할 함수들 =====
def get_calibration_for_fast_fft() -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """fast_fft.py에서 호출할 캘리브레이션 로드 함수"""
    return load_calibration_npz("rx_calibration.npz")

def apply_calibration_for_fast_fft(iq_cube: np.ndarray) -> np.ndarray:
    """fast_fft.py에서 호출할 캘리브레이션 적용 함수"""
    cal_vec, _ = get_calibration_for_fast_fft()
    if cal_vec is not None:
        return apply_rx_calibration_to_iq(iq_cube, cal_vec)
    return iq_cube

# ===== 메인 실행 =====
def main():
    print("=== RX 캘리브레이션 시작 ===\n")
    
    # 캘리브레이션 실행 (60초 동안 데이터 수집)
    try:
        meta = run_calibration_and_save(
            out_path="rx_calibration.npz",
            duration_s=10.0,  # 60초 동안 수집
            ref_rx=0,         # RX0을 기준으로 설정
            target_range_m=None  # 자동으로 가장 강한 신호 위치 선택
        )
        
        print(f"\n✅ 캘리브레이션 완료!")
        print(f"📁 저장 위치: rx_calibration.npz")
        print(f"📊 메타데이터:")
        for key, value in meta.items():
            print(f"   {key}: {value}")
            
        print(f"\n💡 이제 collect_dataset_new.py 실행 시 자동으로 캘리브레이션이 적용됩니다.")
        
    except Exception as e:
        print(f"❌ 캘리브레이션 실패: {e}")
        print(f"\n🔧 문제 해결 방법:")
        print(f"   1. 레이더 장치가 연결되어 있는지 확인")
        print(f"   2. 다른 프로그램이 레이더를 사용하지 않는지 확인")
        print(f"   3. 레이더 앞에 반사체(사람, 벽 등)가 있는지 확인")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



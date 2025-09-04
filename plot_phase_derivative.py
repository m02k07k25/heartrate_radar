#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
z_tau 위상 미분 시각화
60초 전체 데이터에서 위상 변화와 미분값을 그래프로 표시
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, sosfiltfilt
import pandas as pd
import os



def get_fs(data_dict, default=36.0):
    """데이터에서 실제 fs 추출"""
    if 'fs' in data_dict:
        return float(data_dict['fs'])
    if 'frame_repetition_time_s' in data_dict and data_dict['frame_repetition_time_s'] > 0:
        return 1.0 / float(data_dict['frame_repetition_time_s'])
    return default

def ecg_avg_bpm_from_csv(path):
    """CSV에서 ECG 평균 BPM 계산 (형식 자동 판별)"""
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]

    # 1) bpm 컬럼이 있으면 그대로 평균
    if any('bpm' in c for c in cols):
        bpm_col = df[[c for c in df.columns if 'bpm' in c.lower()][0]].to_numpy(float)
        return float(np.nanmean(bpm_col))

    # 2) 타임스탬프(peak times)만 있으면 diff로 계산
    #    후보 컬럼명: t, t_s, time, timestamp …
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ['t', 'time'])]
    if len(time_candidates) >= 1 and (df.shape[1] == 1 or len(time_candidates) >= 1):
        t = df[time_candidates[0]].to_numpy(float)
        t = t[np.isfinite(t)]
        if t.size >= 2:
            duration = t[-1] - t[0]
            return 60.0 * (t.size - 1) / duration

    # 3) 그 밖(예: raw ECG 시계열) → 여기서는 표시 생략
    return None

def median_welch(signal, fs, win_len=8.0, hop=0.5, nfft_mul=4):
    """윈도우 기반 중앙값 Welch로 비정상 구간 무력화"""
    N = len(signal)
    L = int(round(win_len * fs))
    H = int(round(hop * fs))
    starts = np.arange(0, max(1, N - L + 1), H)
    psds = []
    f_ref = None
    
    for s in starts:
        seg = signal[s:s+L]
        if len(seg) < L: 
            break
        # 충분한 해상도 확보
        nperseg = L
        noverlap = int(0.75 * nperseg)
        nfft = 1 << (int(np.ceil(np.log2(nperseg))) + nfft_mul.bit_length()-1)  # 대략 ×4
        f, P = welch(seg, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     nfft=nfft, window='hann', scaling='density')
        if f_ref is None: 
            f_ref = f
        psds.append(P)
    
    if not psds:
        return welch(signal, fs=fs, nperseg=min(len(signal), int(fs*60)),
                     noverlap=int(0.75*min(len(signal), int(fs*60))),
                     window='hann', scaling='density')
    
    psd_med = np.median(np.vstack(psds), axis=0)
    return f_ref, psd_med

def pick_hr_with_harmonics_improved(f_hpf, psd_hpf, f_bpf, psd_bpf, fmin=0.8, fmax=3.0):
    """개선된 하모닉-합 스코어로 HR 주파수 선택"""
    # BPF PSD에서 HR 대역 후보 찾기
    mask_bpf = (f_bpf >= fmin) & (f_bpf <= fmax)
    f_hr = f_bpf[mask_bpf]
    P_bpf = psd_bpf[mask_bpf]
    
    # 지역 봉우리 찾기 (간단한 피크 검출)
    peaks = []
    peak_scores = []
    
    for i in range(1, len(f_hr)-1):
        if P_bpf[i] > P_bpf[i-1] and P_bpf[i] > P_bpf[i+1]:
            peaks.append(f_hr[i])
            peak_scores.append(P_bpf[i])
    
    if not peaks:
        # 피크가 없으면 단순 최대값
        i = np.argmax(P_bpf)
        return float(f_hr[i]), float(P_bpf[i])
    
    # 하모닉-합 스코어 계산
    def interp_hpf(x):
        return np.interp(x, f_hpf, psd_hpf, left=0.0, right=0.0)
    
    def interp_bpf(x):
        return np.interp(x, f_bpf, psd_bpf, left=0.0, right=0.0)
    
    final_scores = []
    for f_peak in peaks:
        # S(f) = P_BPF(f) + 0.6*P_HPF(2f) - 0.2*P_BPF(0.5f)
        score = interp_bpf(f_peak) + 0.6 * interp_hpf(2.0 * f_peak) - 0.2 * interp_bpf(0.5 * f_peak)
        final_scores.append(score)
    
    # 최고 스코어 선택
    best_idx = np.argmax(final_scores)
    return float(peaks[best_idx]), float(final_scores[best_idx])

def welch_fft(signal, fs, nperseg=None, noverlap=None, nfft=None):
    """Welch 방법으로 FFT 및 PSD 계산 (고해상도)"""
    if nperseg is None:
        nperseg = min(len(signal), int(fs * 60))  # 60초 전체에 가깝게
    if noverlap is None:
        noverlap = int(nperseg * 0.75)  # 75% 오버랩
    if nfft is None:
        # 해상도 보강(제로패딩)
        import math
        pow2 = 1 << (int(math.ceil(math.log2(nperseg))) + 2)  # ×4 zero-pad
        nfft = max(pow2, nperseg)
    
    # FFT 및 PSD 계산
    f, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, window='hann', scaling='density')
    return f, psd

def extract_phase_derivative(z_tau, fs):
    """위상 미분 추출 (올바른 파이프라인)"""
    # 올바른 파이프라인: z_tau → angle → unwrap → HPF → gradient → BPF
    
    # 1) 위상 계산 + 언랩
    phi = np.unwrap(np.angle(z_tau)).astype(np.float64)
    
    # 2) 드리프트 제거: 0.30 Hz 하이패스 (낮은 차수, SOS)
    sos_hp = butter(2, 0.30, btype='highpass', fs=fs, output='sos')
    phi_hp = sosfiltfilt(sos_hp, phi)
    
    # 3) 위상 미분 (rad/s)
    dphi = np.gradient(phi_hp) * fs
    
    # 4) 심박 대역 BPF는 **미분 신호에** 적용
    sos_bp = butter(4, [0.8, 3.0], btype='band', fs=fs, output='sos')
    dphi_bp = sosfiltfilt(sos_bp, dphi)
    
    # 시간축 생성
    t = np.arange(len(phi)) / fs
    
    return t, phi, phi_hp, t, dphi, dphi_bp

def plot_phase_derivative(data_file, answer_file, fs_frame=36.0):
    """위상과 미분값 시각화"""
    
    # 데이터 로드
    data_dict = np.load(data_file, allow_pickle=True).item()
    z_tau = data_dict['z_tau']
    
    # 실제 fs 추출 (하드코딩 방지)
    fs_frame = get_fs(data_dict, fs_frame)
    print(f"실제 fs: {fs_frame:.2f} Hz")
    
    # 위상 미분 추출 (올바른 파이프라인)
    t, phi, phi_hp, t_diff, dphi, dphi_bp = extract_phase_derivative(z_tau, fs_frame)
    
    # 그래프 생성 (3개)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # 1. 위상 변화 (원본 vs HPF 적용)
    ax1.plot(t, phi, 'b-', linewidth=0.8, alpha=0.6, label='Original Phase')
    ax1.plot(t, phi_hp, 'r-', linewidth=1.2, alpha=0.8, label='HPF Filtered (0.3 Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_title('z_tau Phase Change: Original vs HPF Filtered (60s)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)
    ax1.legend()
    
    # 2. 위상 미분 (HPF만 vs HPF+BPF)
    ax2.plot(t_diff, dphi, 'b-', linewidth=0.8, alpha=0.6, label='HPF Only (0.3 Hz)')
    ax2.plot(t_diff, dphi_bp, 'r-', linewidth=1.2, alpha=0.8, label='HPF + BPF (0.8-3.0 Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase Derivative (rad/s)')
    ax2.set_title('Phase Derivative: HPF vs HPF+BPF')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 60)
    ax2.legend()
    
    # 3. FFT 주파수 도메인 (HPF만 vs HPF+BPF)
    # FFT 계산 (중앙값 Welch 사용)
    f_original, psd_original = median_welch(dphi, fs_frame)
    f_filtered, psd_filtered = median_welch(dphi_bp, fs_frame)
    
    ax3.plot(f_original, psd_original, 'b-', linewidth=1.0, alpha=0.7, label='HPF Only (0.3 Hz)')
    ax3.plot(f_filtered, psd_filtered, 'r-', linewidth=1.5, alpha=0.8, label='HPF + BPF (0.8-3.0 Hz)')
    ax3.set_xlabel('Frequency (Hz) / Heart Rate (BPM)')
    ax3.set_ylabel('Power Spectral Density (rad²/s²/Hz)')
    # 주파수 해상도 계산
    nperseg = int(8.0 * fs_frame)  # 8초 윈도우
    delta_f = fs_frame / nperseg
    
    ax3.set_title(f'Frequency Domain: Phase Derivative PSD\nfs={fs_frame:.1f} Hz, N={len(dphi_bp)}, Δf≈{delta_f:.3f} Hz')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)  # 0-10 Hz 범위
    ax3.set_ylim(0, max(psd_original.max(), psd_filtered.max()) * 1.1)
    
    # x축 눈금을 더 세밀하게 설정
    ax3.set_xticks(np.arange(0, 10.1, 0.2))  # 0.2 Hz 간격으로 눈금
    ax3.tick_params(axis='x', rotation=45)  # x축 라벨 회전
    
    # 개선된 하모닉-합으로 HR 주파수 선택
    # 원본 신호의 HR 주파수
    max_freq_original, max_score_original = pick_hr_with_harmonics_improved(f_original, psd_original, f_original, psd_original)
    
    # BPF 필터링된 신호의 HR 주파수
    max_freq_filtered, max_score_filtered = pick_hr_with_harmonics_improved(f_original, psd_original, f_filtered, psd_filtered)
    
    # 고주파 우선 선택 로직 (f_hi vs f_lo 경쟁)
    if max_freq_filtered > 0:
        # 고주파(1.5-1.7 Hz)와 저주파(1.0-1.3 Hz) 구분
        if max_freq_filtered >= 1.5:  # 고주파 후보
            # 저주파 대안 찾기
            low_freq_candidates = f_filtered[(f_filtered >= 1.0) & (f_filtered <= 1.3)]
            if len(low_freq_candidates) > 0:
                # 가장 강한 저주파 후보 찾기
                low_freq_mask = np.isin(f_filtered, low_freq_candidates)
                low_freq_psd = psd_filtered[low_freq_mask]
                best_low_freq = low_freq_candidates[np.argmax(low_freq_psd)]
                best_low_score = np.max(low_freq_psd)
                
                # 고주파 우선 선택 조건: S(f_hi) ≥ 0.9·S(f_lo)
                if max_score_filtered >= 0.9 * best_low_score:
                    print(f"고주파 우선 선택: {max_freq_filtered:.3f} Hz ({max_freq_filtered*60:.1f} BPM)")
                else:
                    print(f"저주파 선택: {best_low_freq:.3f} Hz ({best_low_freq*60:.1f} BPM)")
                    max_freq_filtered = best_low_freq
                    max_score_filtered = best_low_score
    
    # 최고 성분을 그래프에 표시 (하모닉-합 결과 + BPM 표시)
    if max_freq_original > 0:
        bpm_original = max_freq_original * 60
        ax3.plot(max_freq_original, max_score_original, 'bo', markersize=8, 
                label=f'Original HR Peak: {max_freq_original:.3f} Hz ({bpm_original:.1f} BPM)')
    if max_freq_filtered > 0:
        bpm_filtered = max_freq_filtered * 60
        ax3.plot(max_freq_filtered, max_score_filtered, 'ro', markersize=8, 
                label=f'BPF HR Peak: {max_freq_filtered:.3f} Hz ({bpm_filtered:.1f} BPM)')
    
    # 심박수 대역 표시 (PSD에서만)
    ax3.axvspan(0.8, 3.0, alpha=0.2, color='green', label='Heart Rate Band (48-180 BPM)')
    
    # ECG 평균 BPM 정보만 print (그래프에는 표시하지 않음)
    avg_bpm = ecg_avg_bpm_from_csv(answer_file)
    if avg_bpm is not None:
        f_ecg = avg_bpm / 60.0  # BPM → Hz
        print(f"ECG 평균 BPM: {avg_bpm:.1f} BPM ({f_ecg:.3f} Hz)")
        
        # f_est/f_ecg 비율 계산 (축척 오류 검증)
        if max_freq_filtered > 0:
            ratio = max_freq_filtered / f_ecg
            print(f"f_est/f_ecg 비율: {ratio:.3f} (≈1.0이 정상, 0.72면 26↔36 축척 혼선)")
    else:
        print("ECG BPM 계산 불가 (CSV 형식 확인 필요)")
    
    ax3.legend()
    
    # 최고 성분 정보 출력
    print(f"\n=== Peak Frequency Analysis ===")
    if max_freq_original > 0:
        print(f"Original signal peak (HR band): {max_freq_original:.3f} Hz ({max_freq_original*60:.1f} BPM)")
    if max_freq_filtered > 0:
        print(f"BPF filtered signal peak (HR band): {max_freq_filtered:.3f} Hz ({max_freq_filtered*60:.1f} BPM)")
    
    # BPF 검증: 0.8 Hz 미만 구간 평균 파워 확인
    low_freq_mask = f_filtered < 0.8
    if np.any(low_freq_mask):
        low_freq_power = np.mean(psd_filtered[low_freq_mask])
        print(f"BPF 검증 - 0.8 Hz 미만 평균 파워: {low_freq_power:.2f} rad²/s²/Hz")
        if low_freq_power > 10:  # 임계값
            print("⚠️  경고: BPF가 저주파를 제대로 차단하지 못함")
    
    plt.tight_layout()
    # plt.show()  # GUI 없이 실행하므로 주석 처리
    
    # plots 폴더에 저장 (데이터 소스에 따라 하위 폴더 생성)
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 데이터 소스 확인 (train/test/한글폴더 등)
    if "train" in data_file:
        sub_dir = "train"
    elif "test" in data_file:
        sub_dir = "test"
    elif "보관" in data_file or "임시" in data_file:
        sub_dir = "backup_temp"
    elif "격리" in data_file:
        sub_dir = "isolation"
    else:
        # 경로에서 폴더명 추출하여 사용
        path_parts = data_file.split(os.sep)
        if len(path_parts) >= 3:
            # record3/폴더명/data 구조에서 폴더명 추출
            sub_dir = path_parts[-3]  # record3 다음 폴더명
        else:
            sub_dir = "unknown"
    
    # 하위 폴더 생성
    output_dir = os.path.join(plots_dir, sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일명 생성 (확장자 제거 후 _phase_derivative.png 추가)
    filename = os.path.basename(data_file).replace('.npy', '_phase_derivative.png')
    output_file = os.path.join(output_dir, filename)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph saved: {output_file}")
    plt.close()  # 메모리 해제
    
    # 통계 정보 출력
    print(f"\n=== Statistics ===")
    print(f"Total time: {t[-1]:.1f}s")
    print(f"Phase range: {phi.min():.2f} ~ {phi.max():.2f} rad")
    print(f"HPF phase derivative range: {dphi.min():.2f} ~ {dphi.max():.2f} rad/s")
    print(f"HPF+BPF phase derivative range: {dphi_bp.min():.2f} ~ {dphi_bp.max():.2f} rad/s")
    # HR 대역 평균값 계산
    hr_mask = (f_original >= 0.8) & (f_original <= 3.0)
    if np.any(hr_mask):
        hr_mean_original = np.mean(psd_original[hr_mask])
        hr_mean_filtered = np.mean(psd_filtered[hr_mask])
        print(f"HPF only HR band mean: {hr_mean_original:.2f} rad/s")
        print(f"HPF+BPF HR band mean: {hr_mean_filtered:.2f} rad/s")

def main():
    """메인 함수 - 폴더 전체 처리"""
    print("=== z_tau 위상 미분 시각화 (폴더 전체) ===")
    
    # 데이터 폴더 경로
    data_folder = "record3/train/data"
    answer_folder = "record3/train/answer"
    
    if not os.path.exists(data_folder):
        raise ValueError(f"데이터 폴더를 찾을 수 없습니다: {data_folder}")
    
    if not os.path.exists(answer_folder):
        raise ValueError(f"답안 폴더를 찾을 수 없습니다: {answer_folder}")
    
    # .npy 파일 목록 가져오기 (raw_* 파일 제외)
    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy') and not f.startswith('raw_')]
    
    # 숫자 순서대로 정렬 (1, 2, 3, 4, 5, 11, 14, 24, 25, 26)
    def natural_sort_key(filename):
        # .npy 확장자 제거하고 숫자 부분만 추출
        name = filename.replace('.npy', '')
        try:
            return int(name)
        except ValueError:
            return float('inf')  # 숫자가 아닌 경우 맨 뒤로
    
    npy_files.sort(key=natural_sort_key)
    
    print(f"발견된 데이터 파일 수: {len(npy_files)} (raw_* 파일 제외)")
    print(f"데이터 폴더: {data_folder}")
    print(f"답안 폴더: {answer_folder}")
    print("=" * 50)
    
    # ECG 정답 결과를 저장할 딕셔너리
    ecg_results = {}
    
    # 각 파일 처리
    for i, npy_file in enumerate(npy_files, 1):
        data_file = os.path.join(data_folder, npy_file)
        answer_file = os.path.join(answer_folder, npy_file.replace('.npy', '.csv'))
        
        print(f"\n[{i}/{len(npy_files)}] 처리 중: {npy_file}")
        
        if os.path.exists(answer_file):
            try:
                # ECG BPM 정보 추출
                avg_bpm = ecg_avg_bpm_from_csv(answer_file)
                if avg_bpm is not None:
                    ecg_results[npy_file.replace('.npy', '')] = avg_bpm
                
                plot_phase_derivative(data_file, answer_file)
                print(f"✅ 성공: {npy_file}")
            except Exception as e:
                print(f"❌ 오류: {npy_file} - {str(e)}")
        else:
            print(f"⚠️  답안 파일 없음: {answer_file}")
    
    print("\n" + "=" * 50)
    print("모든 파일 처리 완료!")
    
    # ECG 정답 결과를 딕셔너리 형태로 출력
    print("\n=== ECG 정답 BPM 요약 ===")
    print("ecg_answers = {")
    for key, value in ecg_results.items():
        print(f"    '{key}': {value:.1f},")
    print("}")
    print(f"\n총 {len(ecg_results)}개 파일 처리됨")

if __name__ == "__main__":
    main()

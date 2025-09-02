# -*- coding: utf-8 -*-
# acquisition.py - BGT60TR13C & ECG 동시 수집 (단독 수집 금지, 레이더 60초 기준 종료)
# python -m helpers.collect_data

import os
import time
import threading
import csv, re
import numpy as np
import serial

from ifxradarsdk.fmcw import DeviceFmcw
# from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from .preproc_signal import process_radar_data
from .run_calibration import RADAR_CFG

from scipy.signal import butter, filtfilt, welch

# ===================== 사용자 설정 =====================
from helpers.radar_config import *

DATA_DIR = "record2/train/data"     # 레이더 저장 (.npy)
ANS_DIR  = "record2/train/answer"   # ECG 저장 (.csv)
ECG_PORT = "COM6"
ECG_BAUD = 1_000_000

# ECG 피크 검출 파라미터
THRESH_V = 1.65
RESET_V  = 0.1
MIN_INTERVAL_MS = 300
# ======================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANS_DIR,  exist_ok=True)

# 다음 파일 번호 (레이더 기준)
existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
nums = []
for f in existing:
    try:
        nums.append(int(os.path.splitext(f)[0]))
    except:
        pass
NEXT_NUM = 1 if not nums else max(nums) + 1

# 이벤트들
radar_ready = threading.Event()
ecg_ready   = threading.Event()
start_event = threading.Event()
stop_event  = threading.Event()    # 정상 종료(레이더 1560 프레임 완료)
fatal_event = threading.Event()    # 어느 쪽이든 오류 발생 시

# 공유 시작 시간 (시간축 동기화용)
t_start = None

def wait_for_start_or_fatal(poll_sec=0.01):
    # start_event가 켜질 때까지 대기하되, fatal이면 즉시 탈출
    while not start_event.is_set():
        if fatal_event.is_set():
            return False
        time.sleep(poll_sec)
    return True

def radar_thread():
    """
    레이더: 정확히 TARGET_FRAMES 수집 -> N.npy 저장
    단, fatal_event 발생 시 즉시 중단 및 저장하지 않음
    """
    device = None
    aborted = False
    try:
        dev_list = DeviceFmcw.get_list()
        if not dev_list:
            print("[RADAR] 레이더 장치를 찾을 수 없습니다.")
            fatal_event.set()
            return
        device = DeviceFmcw(uuid=dev_list[0])

        seq = device.create_simple_sequence(RADAR_CFG)
        device.set_acquisition_sequence(seq)

        chirp_loop = seq.loop.sub_sequence.contents
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx = bin(chirp.rx_mask).count("1")
        num_chirps = chirp_loop.loop.num_repetitions
        num_samples = chirp.num_samples

        # 정확한 프레임 수를 알고 있으니 사전할당
        data_buffer = np.zeros((TARGET_FRAMES, num_rx, num_chirps, num_samples), dtype=np.complex64)

        radar_ready.set()  # 초기화 완료 신호
        if not wait_for_start_or_fatal():
            aborted = True
            return

        print(f"[RADAR] 수집 시작 (목표 {TARGET_FRAMES} frames)…")
        for frm in range(TARGET_FRAMES):
            if fatal_event.is_set():
                aborted = True
                break
            frame_data = device.get_next_frame()
            frame = frame_data[0]
            data_buffer[frm] = frame
            if (frm + 1) % 50 == 0:
                print(f"[RADAR] ... {frm + 1}/{TARGET_FRAMES} frames")

        if aborted:
            print("[RADAR] 중단됨(오류 신호 수신). 저장하지 않습니다.")
            return

        # 레이더 정상 완료 -> ECG 정상 종료 신호
        stop_event.set()
        print("[RADAR] 레이더 정상 완료 -> ECG 정상 종료 신호")

        # 원본 데이터 저장 (처프 차원 평균하여 용량 절감)
        try:
            print(f"[RADAR] 원본 데이터 저장 시작... (원본 크기: {data_buffer.shape})")
            
            # 처프 차원을 평균하여 용량 절감
            # data_buffer shape: (frames, rx, chirps, samples)
            # 처프 차원을 평균하여 (frames, rx, samples)로 압축
            compressed_data = np.mean(data_buffer, axis=2, dtype=np.complex64)
            
            # 원본 데이터를 raw_N.npy 형태로 저장
            raw_out_npy = os.path.join(DATA_DIR, f"raw_{NEXT_NUM}.npy")
            np.save(raw_out_npy, compressed_data)
            print(f"[RADAR] 원본 데이터 저장 완료 -> {raw_out_npy}")
            
            # 전처리된 데이터도 함께 저장 (기존 방식 유지)
            print(f"[RADAR] 전처리 시작...")
            processed_data = process_radar_data(data_buffer)

            # === 품질 알림: 위상 일관성 C, HR-SNR(dB) 계산 ===
            z_tau = processed_data.get('z_tau', None)
            if z_tau is not None:
                fs = 1.0 / RADAR_CFG.frame_repetition_time_s
                # 0.30 Hz HPF로 드리프트/호흡 억제 → HR 품질 평가
                bhp, ahp = butter(2, 0.30/(fs/2), btype='highpass')
                phi = np.unwrap(np.angle(z_tau)).astype(np.float32)
                phi_d = filtfilt(bhp, ahp, phi)
                C = float(np.abs(np.mean(np.exp(1j*phi_d))))
                # Welch 기반 HR-SNR 계산 (HR: 0.8–3.0 Hz, Noise: 3.5–5.0 Hz)
                nper = min(len(phi_d), int(fs*8))
                nover = nper//2
                f, Pxx = welch(phi_d, fs=fs, nperseg=nper, noverlap=nover)
                hr_band = (f >= 0.8) & (f <= 3.0)
                nz_band = (f >= 3.5) & (f <= 5.0)
                snr_db = 10.0*np.log10((Pxx[hr_band].sum()+1e-12)/(Pxx[nz_band].sum()+1e-12))
                print(f"[QUALITY] fc_bin={processed_data.get('fc_bin')}  C={C:.2f}  HR-SNR={snr_db:.1f} dB")
                if (C < 0.30) or (snr_db < 6.0):
                    print("---------------⚠️⚠️⚠️-------------")
                    print("[QUALITY] ⚠️⚠️⚠️ 품질 낮음: 자세/거리/각도/환경(반사체) 조정 권장")

            # 전처리된 데이터 저장
            out_npy = os.path.join(DATA_DIR, f"{NEXT_NUM}.npy")
            np.save(out_npy, processed_data)
            print(f"[RADAR] 전처리 완료 및 저장 -> {out_npy}")
            print(f"[RADAR] 전처리 압축비: {processed_data['shape_info']['compression_ratio']:.1f}:1")
            
        except Exception as e:
            print(f"[RADAR] 데이터 저장 오류: {e}")
            raise e
    except Exception as e:
        print("[RADAR] 오류:", e)
        aborted = True
        fatal_event.set()
    finally:
        if device is not None:
            try:
                device.stop_acquisition()
            except:
                pass
        print("[RADAR] 종료")

def ecg_thread():
    """
    ECG: 레이더 시작과 동시에 시작, 레이더가 TARGET_FRAMES 완료하여 stop_event가 켜지면 정상 종료.
    fatal_event가 켜지면 즉시 중단하고 임시파일 삭제.
    """
    ser = None
    tmp_path = os.path.join(ANS_DIR, f"{NEXT_NUM}.csv.tmp")
    final_path = os.path.join(ANS_DIR, f"{NEXT_NUM}.csv")
    aborted = False
    try:
        ser = serial.Serial(ECG_PORT, ECG_BAUD, timeout=0.05)
        ecg_ready.set()

        if not wait_for_start_or_fatal():
            aborted = True
            return

        # 공유 시작 시간 사용 (시간축 동기화)
        with open(tmp_path, "w", newline="") as pf:
            w = csv.writer(pf)
            w.writerow(["t_s", "voltage"])
            print("[ECG] 수집 시작 (레이더 완료 시까지)…")

            last_peak_ms = -1e9
            armed = True

            # 레이더 정상 완료(stop_event) 또는 오류(fatal_event)까지 대기하며 수집
            while True:
                if fatal_event.is_set():
                    aborted = True
                    break
                if stop_event.is_set():
                    break

                line = ser.readline().decode("utf-8", errors="ignore")
                if not line:
                    continue
                parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(parts) < 2:
                    continue

                v0, v1 = float(parts[0]), float(parts[1])
                t_s = time.perf_counter() - t_start

                if armed and (v1 > THRESH_V) and ((t_s * 1000.0 - last_peak_ms) >= MIN_INTERVAL_MS):
                    w.writerow([f"{t_s:.3f}", f"{v1:.3f}"])
                    pf.flush()
                    last_peak_ms = t_s * 1000.0
                    armed = False
                elif v1 <= RESET_V:
                    armed = True

        if aborted:
            print("[ECG] 중단됨(오류 신호 수신). 임시파일 삭제.")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
            return

        # 정상 완료 -> 최종 파일명으로 원자적 교체
        os.replace(tmp_path, final_path)
        print(f"[ECG] 저장 완료 -> {final_path}")

    except Exception as e:
        print("[ECG] 오류:", e)
        aborted = True
        fatal_event.set()
        # 임시파일 정리
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
    finally:
        if ser is not None:
            try:
                ser.close()
            except:
                pass
        print("[ECG] 종료")

if __name__ == "__main__":
    th_radar = threading.Thread(target=radar_thread, name="radar")
    th_ecg   = threading.Thread(target=ecg_thread,   name="ecg")

    th_radar.start()
    th_ecg.start()

    # 둘 다 준비될 때까지 대기 (타임아웃을 적절히 줘서 영원히 안 기다리게)
    WAIT_SEC = 15
    t_wait0 = time.perf_counter()
    while True:
        if fatal_event.is_set():
            break
        if radar_ready.is_set() and ecg_ready.is_set():
            # 두 장치 모두 준비 완료 -> 동시에 시작
            print("[MAIN] 두 장치 준비 완료. 동시 시작!")
            t_start = time.perf_counter()  # 공유 시작 시간 설정
            start_event.set()
            break
        if time.perf_counter() - t_wait0 > WAIT_SEC:
            print("[MAIN] 준비 대기 타임아웃. 전체 중단.")
            fatal_event.set()
            break
        time.sleep(0.01)

    # 만약 시작 전에 fatal이면, 시작 신호 내지 않고 종료 유도
    # 레이더가 정상 종료되면 stop_event를 셋함 -> ECG도 종료

    th_radar.join()
    th_ecg.join()

    if fatal_event.is_set():
        print("[MAIN] 오류로 인해 수집 실패(단독 수집/부분 저장 없음).")
    else:
        print("[MAIN] 모든 수집 정상 완료.")

인피니언 BGT60TR13C fmcw레이더를 사용하여 원격으로 심박수 측정. 
Accurate Heart Rate Measurement Across Various Body Postures Using FMCW Radar 논문을 기반으로 작성.
해당 논문의 마지막 PSO 최적화 대신 LSTM 사용 목표
helpers.collect_data -> 데이터 수집
helpers.preproc_signal -> 전처리 관련 함수 모음
helpers.preproc_lstm -> lstm에서 사용할 위상 미분 관련 전처리 함수 모음
helpers.radar_config -> 레이더 파라미터 (고정값)
helpers.run_calibration -> 레이더 RX 캘리브레이션 생성
lstm_bpm -> lstm을 사용하여 해당 시점의 bpm 예측 (main)

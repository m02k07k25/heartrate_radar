# func_plot.py - 그래프 생성 및 시각화 함수들
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # PyQt5 대신 TkAgg 사용

from typing import List

# 경고 억제 설정
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')



def plot_training_curves(train_loss_history: List[float], val_loss_history: List[float], 
                        train_mae_history: List[float], val_mae_history: List[float], 
                        save_dir: str = "plots"):
    """훈련과 검증에 대한 Loss와 MAE 그래프 생성"""
    os.makedirs(save_dir, exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Loss 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

        # MAE 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(train_mae_history, label='Training MAE')
        plt.plot(val_mae_history, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'mae_plot.png'))
        plt.close()
    
    print(f"훈련/검증 그래프 저장 완료: {save_dir}/loss_plot.png, {save_dir}/mae_plot.png")

def plot_test_results(test_losses: List[float], test_maes: List[float], test_rmses: List[float], 
                     save_dir: str = "plots"):
    """테스트 결과를 그래프로 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 테스트 파일별 Loss 그래프
        axes[0].plot(range(1, len(test_losses) + 1), test_losses, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Test File Number')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Test Loss by File')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(1, len(test_losses) + 1))
        
        # 테스트 파일별 MAE 그래프
        axes[1].plot(range(1, len(test_maes) + 1), test_maes, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Test File Number')
        axes[1].set_ylabel('MAE (BPM)')
        axes[1].set_title('Test MAE by File')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(1, len(test_maes) + 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"테스트 결과 그래프 저장 완료: {save_dir}/test_results.png")

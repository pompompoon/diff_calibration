import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import os

class SaccadeAnalyzer:
    def __init__(self, df):
        """
        サッカード分析クラスの初期化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            分析対象のデータフレーム
        """
        self.df = df
        self.df['timestamp'] = pd.to_numeric(self.df['timestamp'])
        self.process_data()
    
    def process_data(self):
        """データの前処理とベースとなる計算を行う"""
        # 時間差分の計算
        self.dt = np.diff(self.df['timestamp']) / 1000  # msからsに変換
        
        # 位置の差分を計算
        self.dx = np.diff(self.df['RotatedEyeCenterX_DrawPointerX_Diff'])
        self.dy = np.diff(self.df['RotatedEyeCenterY_DrawPointerY_Diff'])
        
        # 速度の計算
        self.velocity = np.sqrt(self.dx**2 + self.dy**2) / self.dt
        
        # 加速度の計算
        self.acceleration = np.diff(self.velocity) / self.dt[:-1]
        
        # 躍度の計算
        self.jerk = np.diff(self.acceleration) / self.dt[:-2]
    
    def calculate_basic_metrics(self):
        """基本的な統計量を計算"""
        metrics = {
            'mean_velocity': np.mean(self.velocity),
            'max_velocity': np.max(self.velocity),
            'mean_acceleration': np.mean(self.acceleration),
            'max_acceleration': np.max(self.acceleration),
            'mean_jerk': np.mean(self.jerk),
            'max_jerk': np.max(self.jerk)
        }
        return metrics
    
    def detect_saccades(self, velocity_threshold=30):
        """サッカードを検出する"""
        self.saccade_mask = self.velocity > velocity_threshold
        saccade_count = np.sum(self.saccade_mask)
        
        # サッカード中の移動距離を計算
        saccade_distance = np.sum(self.velocity[self.saccade_mask] * self.dt[self.saccade_mask])
        
        # 固視時間を計算
        fixation_time = np.sum(self.dt[~self.saccade_mask]) * 1000  # sからmsに変換
        
        return {
            'saccade_count': saccade_count,
            'saccade_distance': saccade_distance,
            'fixation_time': fixation_time
        }
    
    def calculate_direction_changes(self, angle_threshold=np.pi/4):
        """方向転換の回数を計算"""
        angles = np.arctan2(self.dy, self.dx)
        angle_diff = np.diff(angles)
        
        # -πとπの境界をまたぐ場合の補正
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi,
                            np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff))
        
        direction_changes = np.sum(np.abs(angle_diff) > angle_threshold)
        direction_change_freq = direction_changes / (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]) * 1000
        
        return {
            'direction_changes': direction_changes,
            'direction_change_frequency': direction_change_freq
        }
    
    def calculate_fft_metrics(self):
        """FFT解析を行い、高周波成分の強度を計算"""
        # FFTの計算
        fft_result = fft(self.velocity)
        freq = np.fft.fftfreq(len(self.velocity), d=np.mean(self.dt))
        
        # 正の周波数成分のみを使用
        pos_freq_mask = freq > 0
        power_spectrum = np.abs(fft_result[pos_freq_mask])
        freq = freq[pos_freq_mask]
        
        # 高周波成分（10Hz以上）の強度を計算
        high_freq_power = np.sum(power_spectrum[freq > 10])
        
        return {
            'high_freq_power': high_freq_power,
            'freq': freq,
            'power_spectrum': power_spectrum
        }
    
    def analyze(self):
        """すべての分析を実行して結果を返す"""
        results = {}
        
        # 基本的な統計量
        results.update(self.calculate_basic_metrics())
        
        # サッカード関連の指標
        results.update(self.detect_saccades())
        
        # 方向転換の分析
        results.update(self.calculate_direction_changes())
        
        # FFT解析
        results.update(self.calculate_fft_metrics())
        
        return results
    
    def plot_profiles(self, output_dir):
        """速度、加速度、躍度のプロファイルをプロット"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 速度プロファイル
        ax1.plot(self.df['timestamp'][1:], self.velocity)
        ax1.set_title('Velocity Profile')
        ax1.set_ylabel('Velocity (deg/s)')
        
        # 加速度プロファイル
        ax2.plot(self.df['timestamp'][2:], self.acceleration)
        ax2.set_title('Acceleration Profile')
        ax2.set_ylabel('Acceleration (deg/s²)')
        
        # 躍度プロファイル
        ax3.plot(self.df['timestamp'][3:], self.jerk)
        ax3.set_title('Jerk Profile')
        ax3.set_ylabel('Jerk (deg/s³)')
        ax3.set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, 'saccade_profiles.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    # 入力ファイルと出力ディレクトリのパス
    base_dir = r'G:\共有ドライブ\GAP_長寿研\user\iwamoto\20241009101605_1'
    input_file = os.path.join(base_dir, 'cali_diff1.csv')
    output_dir = base_dir
    
    # データの読み込み
    df = pd.read_csv(input_file)
    
    # 分析の実行
    analyzer = SaccadeAnalyzer(df)
    results = analyzer.analyze()
    
    # 結果を表示
    print("\nサッカード分析結果:")
    print(f"平均速度: {results['mean_velocity']:.2f} deg/s")
    print(f"最大速度: {results['max_velocity']:.2f} deg/s")
    print(f"平均加速度: {results['mean_acceleration']:.2f} deg/s²")
    print(f"最大加速度: {results['max_acceleration']:.2f} deg/s²")
    print(f"平均躍度: {results['mean_jerk']:.2f} deg/s³")
    print(f"最大躍度: {results['max_jerk']:.2f} deg/s³")
    print(f"サッカード回数: {results['saccade_count']}")
    print(f"サッカード中の移動距離: {results['saccade_distance']:.2f} deg")
    print(f"固視時間: {results['fixation_time']:.2f} ms")
    print(f"方向転換回数: {results['direction_changes']}")
    print(f"方向転換頻度: {results['direction_change_frequency']:.2f} 回/s")
    print(f"高周波成分の強度: {results['high_freq_power']:.2f}")
    
    # 結果をCSVファイルに保存
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_dir, 'saccade_analysis_results.csv'), index=False)
    
    # プロファイルのプロット
    analyzer.plot_profiles(output_dir)

if __name__ == '__main__':
    main()
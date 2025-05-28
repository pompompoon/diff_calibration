
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import os
import glob
from extractioncondition.sensitivityfilter import SensitivityFilter
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
        self.df['timestamp'] = pd.to_numeric(self.df['timestamp'])  # timestampカラムを数値に変換
        self.process_data()
    
    def process_data(self):
        """データの前処理とベースとなる計算を行う"""
        # 時間差分の計算
        self.dt = np.diff(self.df['timestamp']) / 1000  # msからsに変換
        
        # 位置の差分を計算
        self.dx = np.diff(self.df['EyeCenterAngleX'])
        self.dy = np.diff(self.df['EyeCenterAngleY'])
        
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
    
    def plot_profiles(self, output_dir, base_name):
        """速度、加速度、躍度のプロファイルをプロット"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 速度プロファイル
        ax1.plot(self.df['timestamp'][1:], self.velocity)
        ax1.set_title('Velocity Profile')
        ax1.set_ylabel('Velocity (units/s)')
        
        # 加速度プロファイル
        ax2.plot(self.df['timestamp'][2:], self.acceleration)
        ax2.set_title('Acceleration Profile')
        ax2.set_ylabel('Acceleration (units/s²)')
        
        # 躍度プロファイル
        ax3.plot(self.df['timestamp'][3:], self.jerk)
        ax3.set_title('Jerk Profile')
        ax3.set_ylabel('Jerk (units/s³)')
        ax3.set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_profiles.png')
        plt.savefig(plot_path)
        plt.close()

def preprocess_data(df):
    """
    データの前処理を行う関数
    - 距離の計算
    - カラム名の調整
    """
    # マンハッタン距離の計算
    manhattan_distances = np.abs(df['EyeCenterAngleX']) + np.abs(df['EyeCenterAngleY'])
    df['manhattan_distance'] = manhattan_distances
    manhattan_stats = {
        'manhattan_distance_sum': manhattan_distances.sum(),
        'manhattan_distance_mean': manhattan_distances.mean()
    }
    
    # ユークリッド距離の計算
    euclidean_distances = np.sqrt(
        df['EyeCenterAngleX']**2 + 
        df['EyeCenterAngleY']**2
    )
    df['euclidean_distance'] = euclidean_distances
    euclidean_stats = {
        'euclidean_distance_sum': euclidean_distances.sum(),
        'euclidean_distance_mean': euclidean_distances.mean()
    }
    
    # TimeStampカラムの名前を確認し、必要に応じて変更
    if 'TimeStamp' in df.columns:
        df['timestamp'] = df['TimeStamp']
    elif 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp']
    else:
        raise ValueError("タイムスタンプのカラムが見つかりません")
    
    return df, manhattan_stats, euclidean_stats

def analyze_multiple_calibrations(base_dir, output_dir):
    """
    複数のキャリブレーションデータを分析する関数
    
    Parameters:
    -----------
    base_dir : str
        GAPデータが格納されているベースディレクトリ
    output_dir : str
        結果を出力するディレクトリ
    """
    # calibrationファイルを含むすべてのフォルダを取得
    folders = sorted(glob.glob(os.path.join(base_dir, "2024*")))
    
    # 結果を格納するリスト
    all_results = []
    
    print(f"合計{len(folders)}個のフォルダを処理します。")
    
    # 各フォルダ内のcalibrationファイルを処理
    for i, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        id_num = folder_name.split('_')[-1]  # フォルダ名から ID を抽出
        
        # calibrationファイルを検索
        calib_files = glob.glob(os.path.join(folder, "calibration_*.csv"))
        
        for calib_file in calib_files:
            try:
                print(f"\n処理中 ({i}/{len(folders)}): {folder_name}")
                
                # 個別のファイルを分析
                results = analyze_eye_movements(calib_file, output_dir)
                
                # ID を追加
                results['ID'] = int(id_num)
                
                # freq と power_spectrum を文字列として保存
                results['freq'] = str(results['freq'])
                results['power_spectrum'] = str(results['power_spectrum'])
                
                # 結果をリストに追加
                all_results.append(results)
                
            except Exception as e:
                print(f"エラー - {os.path.basename(calib_file)}: {str(e)}")
    
    # 結果をデータフレームに変換
    if all_results:
        # DataFrameに変換
        combined_results = pd.DataFrame(all_results)
        
        # IDでソート
        combined_results = combined_results.sort_values('ID')
        
        # 列の順序を指定（IDを最初に）
        cols = ['ID'] + [col for col in combined_results.columns if col != 'ID']
        combined_results = combined_results[cols]
        
        # CSVファイルとして保存
        output_file = os.path.join(output_dir, 'combined_saccade_analysis.csv')
        combined_results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n分析完了: {len(all_results)}件の結果を保存しました")
        print(f"保存先: {output_file}")
    
    return combined_results

def analyze_eye_movements(input_file, output_dir):
    """
    眼球運動データを分析する関数
    
    Parameters:
    -----------
    input_file : str
        入力ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    """
    # データの読み込みと前処理
    df = pd.read_csv(input_file)
    df, manhattan_stats, euclidean_stats = preprocess_data(df)
    
    # 分析の実行
    analyzer = SaccadeAnalyzer(df)
    results = analyzer.analyze()
    
    # 距離の統計量を結果に追加
    results.update(manhattan_stats)
    results.update(euclidean_stats)
    
    return results

# 実行コード
if __name__ == "__main__":
    # ベースディレクトリ（GAPデータフォルダ）とアウトプットディレクトリを指定
    base_dir = r"G:\共有ドライブ\GAP_長寿研\GAPデータ"
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\data"
    
    # 複数のcalibrationデータを分析
    # combined_results = analyze_multiple_calibrations(base_dir, output_dir)


    # 感度の閾値を設定（必要に応じて変更可能）
    sensitivity_threshold = 15
    
    # 複数のcalibrationデータを分析（感度フィルター付き）
    combined_results = analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold)
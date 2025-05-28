import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import label
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
        self.df['timestamp'] = pd.to_numeric(self.df['TimeStamp'])  # timestampカラムを数値に変換
        
        # -100の値を除外（コピーではなく元のデータフレームを変更）
        self.df = self.df.loc[(self.df['EyeCenterAngleX'] != -100) & (self.df['EyeCenterAngleY'] != -100)].copy()
        
        self.process_data()
    
    def process_data(self):
        """データの前処理とベースとなる計算を行う"""
        # 時間差分の計算
        self.dt = np.diff(self.df['timestamp']) / 1000  # msからsに変換
        
        # 位置の差分を計算 (角度データを使用)
        self.dx = np.diff(self.df['EyeCenterAngleX'])
        self.dy = np.diff(self.df['EyeCenterAngleY'])
        
        # 速度の計算 (deg/s)
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
    
    def detect_microsaccades(self):
        """
        マイクロサッカードを検出する
        
        条件:
        - 眼球回転角速度: 10 ~ 99 deg/s
        - 移動量: ≤ 1 deg
        - 継続時間: 10 ~ 30 ms
        - 発生頻度: 0.2 ~ 3 Hz
        """
        # 速度条件に基づくマスク (10 ~ 99 deg/s)
        velocity_mask = (self.velocity >= 10) & (self.velocity <= 99)
        
        # 連続したTrueの領域を特定
        labeled_regions, num_regions = label(velocity_mask)
        region_properties = {}
        
        microsaccades = []
        microsaccade_durations = []
        microsaccade_amplitudes = []
        microsaccade_peak_velocities = []
        
        # 各領域を調査
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) == 0:
                continue
            
            # 継続時間 (ms)
            start_time = self.df['timestamp'].iloc[region_indices[0]]
            end_time = self.df['timestamp'].iloc[min(region_indices[-1] + 1, len(self.df) - 1)]
            duration = end_time - start_time
            
            # 継続時間条件 (10 ~ 30 ms)
            if 10 <= duration <= 30:
                # 移動量計算 (deg)
                start_x = self.df['EyeCenterAngleX'].iloc[region_indices[0]]
                start_y = self.df['EyeCenterAngleY'].iloc[region_indices[0]]
                end_x = self.df['EyeCenterAngleX'].iloc[min(region_indices[-1] + 1, len(self.df) - 1)]
                end_y = self.df['EyeCenterAngleY'].iloc[min(region_indices[-1] + 1, len(self.df) - 1)]
                
                amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                
                # 移動量条件 (≤ 1 deg)
                if amplitude <= 1:
                    peak_velocity = np.max(self.velocity[region_indices])
                    
                    # マイクロサッカードとして記録
                    microsaccades.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'amplitude': amplitude,
                        'peak_velocity': peak_velocity
                    })
                    
                    microsaccade_durations.append(duration)
                    microsaccade_amplitudes.append(amplitude)
                    microsaccade_peak_velocities.append(peak_velocity)
        
        # 総記録時間 (s)
        total_time = (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]) / 1000
        
        # 発生頻度 (Hz)
        if total_time > 0:
            frequency = len(microsaccades) / total_time
        else:
            frequency = 0
        
        # マイクロサッカードの統計
        microsaccade_stats = {
            'microsaccade_count': len(microsaccades),
            'microsaccade_frequency': frequency,
            'microsaccade_mean_duration': np.mean(microsaccade_durations) if microsaccade_durations else 0,
            'microsaccade_mean_amplitude': np.mean(microsaccade_amplitudes) if microsaccade_amplitudes else 0,
            'microsaccade_mean_velocity': np.mean(microsaccade_peak_velocities) if microsaccade_peak_velocities else 0,
        }
        
        return microsaccade_stats, microsaccades
    
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
        
        # マイクロサッカードの検出
        microsaccade_stats, _ = self.detect_microsaccades()
        results.update(microsaccade_stats)
        
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
        plot_path = os.path.join(output_dir, f'{base_name}_profiles.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_microsaccades(self, output_dir, base_name):
        """マイクロサッカードを視覚化"""
        _, microsaccades = self.detect_microsaccades()
        
        if not microsaccades:
            print(f"マイクロサッカードが検出されませんでした: {base_name}")
            return
        
        # マイクロサッカードのハイライト用のマスクを生成
        microsaccade_mask = np.zeros_like(self.velocity, dtype=bool)
        
        for ms in microsaccades:
            # タイムスタンプから対応するインデックスを見つける
            start_idx = np.where(self.df['timestamp'] >= ms['start_time'])[0][0] - 1
            end_idx = np.where(self.df['timestamp'] >= ms['end_time'])[0][0] - 1
            
            # インデックス範囲を確認
            if start_idx >= 0 and end_idx < len(microsaccade_mask):
                microsaccade_mask[start_idx:end_idx+1] = True
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 速度プロファイルとマイクロサッカードのハイライト
        times = self.df['timestamp'][1:].values
        ax1.plot(times, self.velocity, 'b-', alpha=0.6)
        
        # マイクロサッカード領域をハイライト
        for ms in microsaccades:
            ax1.axvspan(ms['start_time'], ms['end_time'], color='r', alpha=0.3)
        
        ax1.set_title('Velocity Profile with Microsaccades Highlighted')
        ax1.set_ylabel('Velocity (deg/s)')
        ax1.axhline(y=10, color='g', linestyle='--', alpha=0.5)  # 下限閾値
        ax1.axhline(y=99, color='g', linestyle='--', alpha=0.5)  # 上限閾値
        ax1.set_ylim(0, min(max(self.velocity)*1.1, 150))  # 表示範囲調整
        
        # 眼球運動軌跡
        ax2.plot(self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 'b-', alpha=0.6)
        
        # マイクロサッカードの開始点と終了点をプロット
        for ms in microsaccades:
            start_idx = np.where(self.df['timestamp'] >= ms['start_time'])[0][0]
            end_idx = np.where(self.df['timestamp'] >= ms['end_time'])[0][0]
            
            # 開始点と終了点をハイライト
            ax2.plot(self.df['EyeCenterAngleX'].iloc[start_idx], self.df['EyeCenterAngleY'].iloc[start_idx], 'go')
            ax2.plot(self.df['EyeCenterAngleX'].iloc[end_idx], self.df['EyeCenterAngleY'].iloc[end_idx], 'ro')
            
            # マイクロサッカードの軌跡をハイライト
            ax2.plot(self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1], 
                    self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1], 
                    'r-', linewidth=2)
        
        ax2.set_title('Eye Movement Trajectory with Microsaccades Highlighted')
        ax2.set_xlabel('Horizontal Eye Angle (deg)')
        ax2.set_ylabel('Vertical Eye Angle (deg)')
        ax2.axis('equal')
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_microsaccades.png')
        plt.savefig(plot_path)
        plt.close()

def preprocess_data(df):
    """
    データの前処理を行う関数
    - 距離の計算
    - カラム名の調整
    - -100の値を除外
    """
    # -100の値を除外（コピーを作成して警告を回避）
    df = df.loc[(df['EyeCenterAngleX'] != -100) & (df['EyeCenterAngleY'] != -100)].copy()
    
    # マンハッタン距離の計算 (角度データを使用)
    manhattan_distances = np.abs(df['EyeCenterAngleX']) + np.abs(df['EyeCenterAngleY'])
    df.loc[:, 'manhattan_distance'] = manhattan_distances
    manhattan_stats = {
        'manhattan_distance_sum': manhattan_distances.sum(),
        'manhattan_distance_mean': manhattan_distances.mean()
    }
    
    # ユークリッド距離の計算 (角度データを使用)
    euclidean_distances = np.sqrt(
        df['EyeCenterAngleX']**2 + 
        df['EyeCenterAngleY']**2
    )
    df.loc[:, 'euclidean_distance'] = euclidean_distances
    euclidean_stats = {
        'euclidean_distance_sum': euclidean_distances.sum(),
        'euclidean_distance_mean': euclidean_distances.mean()
    }
    
    # TimeStampカラムの名前を確認し、必要に応じて変更
    if 'TimeStamp' in df.columns:
        df.loc[:, 'timestamp'] = df['TimeStamp']
    elif 'timestamp' in df.columns:
        df.loc[:, 'timestamp'] = df['timestamp']
    else:
        raise ValueError("タイムスタンプのカラムが見つかりません")
    
    return df, manhattan_stats, euclidean_stats

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

def analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold=15):
    """
    複数のキャリブレーションデータを分析する関数（感度フィルター付き）
    
    Parameters:
    -----------
    base_dir : str
        GAPデータが格納されているベースディレクトリ
    output_dir : str
        結果を出力するディレクトリ
    sensitivity_threshold : int
        LessThanSensitivityの閾値
    """
    # 感度フィルターを初期化
    sensitivity_filter = SensitivityFilter(threshold=sensitivity_threshold)
    
    # calibrationファイルを含むすべてのフォルダを取得
    folders = sorted(glob.glob(os.path.join(base_dir, "2024*")))
    
    # 結果を格納するリスト
    all_results = []
    skipped_folders = []
    
    print(f"合計{len(folders)}個のフォルダを処理します。")
    
    # 各フォルダ内のcalibrationファイルを処理
    for i, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        
        # InspectionDateAndIdをフォルダ名から抽出（例: 20241009101605_1 -> 20241009101605_1）
        inspection_date_and_id = folder_name
        
        # 感度チェック
        should_process, sensitivity_count = sensitivity_filter.check_result_file(folder)
        
        if not should_process:
            print(f"\nスキップ ({i}/{len(folders)}): {folder_name}")
            print(f"理由: LessThanSensitivity合計 = {sensitivity_count} > {sensitivity_threshold}")
            skipped_folders.append({
                'folder': folder_name,
                'sensitivity_count': sensitivity_count
            })
            continue
            
        # calibrationファイルを検索
        calib_files = glob.glob(os.path.join(folder, "calibration_*.csv"))
        
        for calib_file in calib_files:
            try:
                print(f"\n処理中 ({i}/{len(folders)}): {folder_name}")
                print(f"LessThanSensitivity合計: {sensitivity_count}")
                
                # 個別のファイルを分析
                results = analyze_eye_movements(calib_file, output_dir)
                
                # InspectionDateAndIdと感度情報を追加
                results['InspectionDateAndId'] = inspection_date_and_id
                results['less_than_sensitivity_count'] = sensitivity_count
                
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
        
        # InspectionDateAndIdでソート
        combined_results = combined_results.sort_values('InspectionDateAndId')
        
        # 列の順序を指定（InspectionDateAndIdを最初に）
        cols = ['InspectionDateAndId', 'less_than_sensitivity_count'] + [col for col in combined_results.columns if col not in ['InspectionDateAndId', 'less_than_sensitivity_count']]
        combined_results = combined_results[cols]
        
        # CSVファイルとして保存（閾値を含むファイル名）
        output_file = os.path.join(output_dir, f'combined_EyeCenterAngle_microsaccade_analysis_{sensitivity_threshold}.csv')
        combined_results.to_csv(output_file, index=False, encoding='utf-8')
        
        # スキップされたフォルダの情報を保存（閾値を含むファイル名）
        if skipped_folders:
            skipped_df = pd.DataFrame(skipped_folders)
            skipped_file = os.path.join(output_dir, f'skipped_folders_{sensitivity_threshold}.csv')
            skipped_df.to_csv(skipped_file, index=False, encoding='utf-8')
        
        print(f"\n分析完了:")
        print(f"- 処理された結果: {len(all_results)}件")
        print(f"- スキップされたフォルダ: {len(skipped_folders)}件")
        print(f"保存先:")
        print(f"- 分析結果: {output_file}")
        if skipped_folders:
            print(f"- スキップ情報: {skipped_file}")

# 実行コード
if __name__ == "__main__":
    # ベースディレクトリ（GAPデータフォルダ）とアウトプットディレクトリを指定
    base_dir = r"G:\共有ドライブ\GAP_長寿研\GAPデータ"
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\data"
    
    # 感度の閾値を設定
    sensitivity_threshold = 15
    
    # 複数のcalibrationデータを分析（感度フィルター付き）
    combined_results = analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold)
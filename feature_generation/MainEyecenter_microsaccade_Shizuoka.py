import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import label
from scipy.fft import fft
import matplotlib.pyplot as plt
import os
import glob
import re
from extractioncondition.sensitivityfilter import SensitivityFilter

class SaccadeAnalyzer:
    def __init__(self, df):
        """
        サッカード分析クラスの初期化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            分析対象のデータフレーム（前処理済み）
        """
        self.df = df
        
        # このコンストラクタでは、dfは既に前処理済みと想定
        # (timestampは数値型、-100の値は除外済み、開始5秒間のデータも除外済み)
        
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
    
    def calculate_gaze_endpoint_std(self):
        """視線の終点（X,Y）の標準偏差を計算"""
        std_x = np.std(self.df['EyeCenterAngleX'])
        std_y = np.std(self.df['EyeCenterAngleY'])
        
        return {
            'gaze_endpoint_std_x': std_x,
            'gaze_endpoint_std_y': std_y,
            'gaze_endpoint_std_combined': np.sqrt(std_x**2 + std_y**2)  # 合成標準偏差
        }
    
    def detect_saccades(self, velocity_threshold=30):
        """
        サッカードを検出する
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
        """
        # 速度条件のみでサッカードを検出
        self.saccade_mask = self.velocity > velocity_threshold
        saccade_count = np.sum(self.saccade_mask)
        
        # サッカード中の移動距離を計算
        saccade_distance = np.sum(self.velocity[self.saccade_mask] * self.dt[self.saccade_mask])
        
        # 固視時間を計算
        fixation_time = np.sum(self.dt[~self.saccade_mask]) * 1000  # sからmsに変換
        
        # サッカードの軌跡情報を計算
        saccade_trajectories = self.calculate_saccade_trajectories(velocity_threshold)
        
        saccade_metrics = {
            'saccade_count': saccade_count,
            'saccade_distance': saccade_distance,
            'fixation_time': fixation_time
        }
        
        # 軌跡の統計情報を追加
        if saccade_trajectories:
            peak_velocities = [t['peak_velocity'] for t in saccade_trajectories]
            trajectory_metrics = {
                'saccade_mean_amplitude': np.mean([t['amplitude'] for t in saccade_trajectories]),
                'saccade_max_amplitude': max([t['amplitude'] for t in saccade_trajectories]) if saccade_trajectories else 0,
                'saccade_mean_duration': np.mean([t['duration'] for t in saccade_trajectories]),
                'saccade_mean_peak_velocity': np.mean(peak_velocities),
                'saccade_peak_velocity_std': np.std(peak_velocities),  # 最大速度の標準偏差を追加
                'saccade_curvature_index': np.mean([t['curvature_index'] for t in saccade_trajectories]) if saccade_trajectories else 0
            }
            saccade_metrics.update(trajectory_metrics)
        
        return saccade_metrics
    
    def detect_microsaccades(self, velocity_threshold_low=10, velocity_threshold_high=100, 
                            amplitude_threshold=1, duration_low_ms=10, duration_high_ms=30):
        """
        マイクロサッカードを検出する
        
        Parameters:
        -----------
        velocity_threshold_low : float
            マイクロサッカード検出の最小速度閾値 (deg/s)
        velocity_threshold_high : float
            マイクロサッカード検出の最大速度閾値 (deg/s)
        amplitude_threshold : float
            マイクロサッカードの最大振幅閾値 (deg)
        duration_low_ms : float
            マイクロサッカードの最小継続時間 (ms)
        duration_high_ms : float
            マイクロサッカードの最大継続時間 (ms)
            
        Returns:
        --------
        dict : マイクロサッカードの統計情報
        """
        # 速度条件でマイクロサッカード候補を検出
        microsaccade_mask = (self.velocity >= velocity_threshold_low) & (self.velocity < velocity_threshold_high)
        
        # 連続したTrueの領域（マイクロサッカード候補）を特定
        labeled_regions, num_regions = label(microsaccade_mask)
        
        microsaccade_trajectories = []
        total_recording_time_ms = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]  # 総記録時間（ms）
        
        # 各マイクロサッカード候補領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # 開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1  # データポイントとして次のインデックスも含める
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 開始と終了の時間・位置
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx]
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx, len(self.df) - 1)]
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 継続時間（ms）
            duration = end_time - start_time
            
            # マイクロサッカード条件の検証
            if (amplitude <= amplitude_threshold and 
                duration_low_ms <= duration <= duration_high_ms):
                
                # マイクロサッカード中の最大速度
                peak_velocity = np.max(self.velocity[region_indices])
                
                # 軌跡データを収集
                if end_idx + 1 <= len(self.df):
                    trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
                    trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
                else:
                    trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:].values
                    trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:].values
                
                # 曲率インデックス（軌跡の長さ / 直線距離）
                path_length = 0
                for j in range(len(trajectory_x) - 1):
                    path_length += np.sqrt((trajectory_x[j+1] - trajectory_x[j])**2 + 
                                          (trajectory_y[j+1] - trajectory_y[j])**2)
                
                curvature_index = path_length / amplitude if amplitude > 0 else 0
                
                microsaccade_info = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'amplitude': amplitude,
                    'peak_velocity': peak_velocity,
                    'curvature_index': curvature_index,
                    'trajectory_x': trajectory_x,
                    'trajectory_y': trajectory_y
                }
                
                microsaccade_trajectories.append(microsaccade_info)
        
        # マイクロサッカードの発生頻度（Hz）を計算
        microsaccade_count = len(microsaccade_trajectories)
        microsaccade_frequency = microsaccade_count / (total_recording_time_ms / 1000)  # msからsに変換
        
        # マイクロサッカードの統計情報を収集
        if microsaccade_trajectories:
            amplitudes = [m['amplitude'] for m in microsaccade_trajectories]
            durations = [m['duration'] for m in microsaccade_trajectories]
            peak_velocities = [m['peak_velocity'] for m in microsaccade_trajectories]
            
            microsaccade_metrics = {
                'microsaccade_count': microsaccade_count,
                'microsaccade_frequency': microsaccade_frequency,
                'microsaccade_mean_amplitude': np.mean(amplitudes),
                'microsaccade_max_amplitude': np.max(amplitudes),
                'microsaccade_mean_duration': np.mean(durations),
                'microsaccade_mean_peak_velocity': np.mean(peak_velocities),
                'microsaccade_peak_velocity_std': np.std(peak_velocities),
                'microsaccade_main_sequence_ratio': np.mean([m['peak_velocity'] / m['amplitude'] 
                                                           if m['amplitude'] > 0 else 0 
                                                           for m in microsaccade_trajectories])
            }
        else:
            microsaccade_metrics = {
                'microsaccade_count': 0,
                'microsaccade_frequency': 0,
                'microsaccade_mean_amplitude': 0,
                'microsaccade_max_amplitude': 0,
                'microsaccade_mean_duration': 0,
                'microsaccade_mean_peak_velocity': 0,
                'microsaccade_peak_velocity_std': 0,
                'microsaccade_main_sequence_ratio': 0
            }
        
        # 元の処理用にマイクロサッカードの軌跡情報も保持
        self.microsaccade_trajectories = microsaccade_trajectories
        
        return microsaccade_metrics
    
    def calculate_saccade_trajectories(self, velocity_threshold=30):
        """
        サッカードの軌跡情報を計算する
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
            
        軌跡の特徴量:
        - amplitude: サッカードの振幅（角度）
        - duration: サッカードの継続時間（ms）
        - peak_velocity: 最大速度（deg/s）
        - curvature_index: 曲率インデックス（直線からのずれ）
        """
        # 速度条件のみでサッカードを検出
        saccade_mask = self.velocity > velocity_threshold
        
        # 連続したTrueの領域（サッカード）を特定
        labeled_regions, num_regions = label(saccade_mask)
        
        saccade_trajectories = []
        
        # 各サッカード領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # サッカードの開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1  # データポイントとして次のインデックスも含める
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 開始と終了の時間・位置
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx]
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx, len(self.df) - 1)]
            
            # サッカード中の軌跡情報
            if end_idx + 1 <= len(self.df):
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
            else:
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:].values
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 継続時間（ms）
            duration = end_time - start_time
            
            # サッカード中の最大速度
            peak_velocity = np.max(self.velocity[region_indices]) if len(region_indices) > 0 else 0
            
            # 曲率インデックス（軌跡の長さ / 直線距離）
            path_length = 0
            for j in range(len(trajectory_x) - 1):
                path_length += np.sqrt((trajectory_x[j+1] - trajectory_x[j])**2 + 
                                       (trajectory_y[j+1] - trajectory_y[j])**2)
            
            curvature_index = path_length / amplitude if amplitude > 0 else 0
            
            saccade_info = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'amplitude': amplitude,
                'peak_velocity': peak_velocity,
                'curvature_index': curvature_index,
                'trajectory_x': trajectory_x,
                'trajectory_y': trajectory_y
            }
            
            saccade_trajectories.append(saccade_info)
        
        return saccade_trajectories
    
    def calculate_amplitude_based_metrics(self, velocity_threshold=30, amplitude_bins=[5, 10, 20, 30, 40, 50]):
        """
        振幅の範囲ごとにサッカードの速度を分析
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
        amplitude_bins : list
            分析する振幅の範囲（度）
            
        Returns:
        --------
        dict : 各振幅範囲におけるサッカードの平均速度と最大速度
        """
        # サッカードの軌跡情報を取得
        saccade_trajectories = self.calculate_saccade_trajectories(velocity_threshold)
        
        # 結果を格納する辞書
        amplitude_metrics = {}
        
        # 各振幅範囲のビンを定義 (例: 0-5度, 5-10度, ...)
        bin_ranges = [(0, amplitude_bins[0])]
        for i in range(len(amplitude_bins)-1):
            bin_ranges.append((amplitude_bins[i], amplitude_bins[i+1]))
        bin_ranges.append((amplitude_bins[-1], float('inf')))
        
        # 各振幅範囲ごとに速度を計算
        for i, (min_amp, max_amp) in enumerate(bin_ranges):
            bin_name = f"amplitude_{min_amp}_{max_amp}" if max_amp != float('inf') else f"amplitude_over_{min_amp}"
            
            # この振幅範囲に該当するサッカード
            bin_saccades = [s for s in saccade_trajectories if min_amp <= s['amplitude'] < max_amp]
            
            if bin_saccades:
                peak_velocities = [s['peak_velocity'] for s in bin_saccades]
                amplitude_metrics[f"{bin_name}_count"] = len(bin_saccades)
                amplitude_metrics[f"{bin_name}_mean_velocity"] = np.mean(peak_velocities)
                amplitude_metrics[f"{bin_name}_max_velocity"] = np.max(peak_velocities)
            else:
                amplitude_metrics[f"{bin_name}_count"] = 0
                amplitude_metrics[f"{bin_name}_mean_velocity"] = 0
                amplitude_metrics[f"{bin_name}_max_velocity"] = 0
        
        return amplitude_metrics
    
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
    
    def analyze(self, velocity_threshold=30):
        """すべての分析を実行して結果を返す"""
        results = {}
        
        # 基本的な統計量
        results.update(self.calculate_basic_metrics())
        
        # 視線の終点の標準偏差
        results.update(self.calculate_gaze_endpoint_std())
        
        # サッカード関連の指標（指定された速度閾値を使用）
        results.update(self.detect_saccades(velocity_threshold=velocity_threshold))
        
        # マイクロサッカード関連の指標
        results.update(self.detect_microsaccades(
            velocity_threshold_low=10, 
            velocity_threshold_high=100, 
            amplitude_threshold=1, 
            duration_low_ms=10, 
            duration_high_ms=30
        ))
        
        # 振幅ごとのサッカード速度分析
        results.update(self.calculate_amplitude_based_metrics(velocity_threshold=velocity_threshold))
        
        # 方向転換の分析
        results.update(self.calculate_direction_changes())
        
        # FFT解析
        results.update(self.calculate_fft_metrics())
        
        # 使用した速度閾値を結果に追加
        results['velocity_threshold'] = velocity_threshold
        
        return results


def preprocess_data(df, exclude_initial_seconds=5):
    """
    データの前処理を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        前処理するデータフレーム
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数 (デフォルト: 5秒)
        
    Returns:
    --------
    df : pandas.DataFrame
        前処理されたデータフレーム
    manhattan_stats : dict
        マンハッタン距離の統計情報
    euclidean_stats : dict
        ユークリッド距離の統計情報
    """
    # timestampカラムを数値型に変換
    if 'TimeStamp' in df.columns:
        df.loc[:, 'timestamp'] = pd.to_numeric(df['TimeStamp'])
    elif 'timestamp' in df.columns:
        df.loc[:, 'timestamp'] = pd.to_numeric(df['timestamp'])
    else:
        raise ValueError("タイムスタンプのカラムが見つかりません")
    
    # 最初のタイムスタンプを取得
    initial_timestamp = df['timestamp'].iloc[0]
    
    # 開始からX秒後のタイムスタンプを計算 (ミリ秒単位)
    cutoff_timestamp = initial_timestamp + (exclude_initial_seconds * 1000)
    
    # X秒以降のデータのみをフィルタリング
    df = df[df['timestamp'] >= cutoff_timestamp].copy()
    
    # データが残っているか確認
    if df.empty:
        raise ValueError(f"{exclude_initial_seconds}秒間のフィルタリング後にデータが残っていません")
    
    print(f"最初の{exclude_initial_seconds}秒間のデータを除外: {initial_timestamp}ms から {cutoff_timestamp}ms")
    print(f"残りのデータポイント数: {len(df)}")
    
    # -100の値を除外
    df = df.loc[(df['EyeCenterAngleX'] != -100) & (df['EyeCenterAngleY'] != -100)].copy()
    
    # マンハッタン距離の計算
    manhattan_distances = np.abs(df['EyeCenterAngleX']) + np.abs(df['EyeCenterAngleY'])
    df.loc[:, 'manhattan_distance'] = manhattan_distances
    manhattan_stats = {
        'manhattan_distance_sum': manhattan_distances.sum(),
        'manhattan_distance_mean': manhattan_distances.mean()
    }
    
    # ユークリッド距離の計算
    euclidean_distances = np.sqrt(
        df['EyeCenterAngleX']**2 + 
        df['EyeCenterAngleY']**2
    )
    df.loc[:, 'euclidean_distance'] = euclidean_distances
    euclidean_stats = {
        'euclidean_distance_sum': euclidean_distances.sum(),
        'euclidean_distance_mean': euclidean_distances.mean()
    }
    
    return df, manhattan_stats, euclidean_stats

def analyze_eye_movements(input_file, output_dir, inspection_id, velocity_threshold=30, exclude_initial_seconds=5):
    """
    眼球運動データを分析する関数
    
    Parameters:
    -----------
    input_file : str
        入力ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    inspection_id : str
        検査ID（InspectionDateAndIdとして使用）
    velocity_threshold : float
        サッカード検出の速度閾値 (deg/s)
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数
    """
    # データの読み込みと前処理
    df = pd.read_csv(input_file)
    df, manhattan_stats, euclidean_stats = preprocess_data(df, exclude_initial_seconds=exclude_initial_seconds)
    
    # 分析の実行
    analyzer = SaccadeAnalyzer(df)
    results = analyzer.analyze(velocity_threshold=velocity_threshold)
    
    # 距離の統計量を結果に追加
    results.update(manhattan_stats)
    results.update(euclidean_stats)
    
    # InspectionDateAndIdを追加
    results['InspectionDateAndId'] = inspection_id
    
    return results

def extract_inspection_id_from_path(path):
    """
    パスから検査IDを抽出する関数
    
    Parameters:
    -----------
    path : str
        パス文字列
        
    Returns:
    --------
    str
        抽出された検査ID（例: "20240201094540_20855"）
    """
    # パスの区切り文字を統一
    normalized_path = path.replace('\\', '/')
    
    # 正規表現でIDを抽出
    match = re.search(r'/(\d{14}_\d{5})(?:/|$)', normalized_path)
    if match:
        return match.group(1)
    
    # バックスラッシュ区切りでも試す
    match = re.search(r'\\(\d{14}_\d{5})(?:\\|$)', path)
    if match:
        return match.group(1)
    
    # パス全体がIDの場合
    if re.match(r'\d{14}_\d{5}$', path):
        return path
    
    return None

def load_metadata(metadata_path):
    """
    メタデータファイルを読み込み、パスとIDの対応関係を作成する
    
    Parameters:
    -----------
    metadata_path : str
        メタデータファイルのパス
        
    Returns:
    --------
    dict
        パスからIDへのマッピング辞書
    """
    if not os.path.exists(metadata_path):
        print(f"メタデータファイルが見つかりません: {metadata_path}")
        return {}
    
    # 様々なエンコーディングを試す
    encodings = ['utf-8', 'shift-jis', 'cp932', 'latin1', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            print(f"エンコーディング {encoding} でメタデータファイルを読み込み中...")
            metadata_df = pd.read_csv(metadata_path, encoding=encoding)
            
            # パスからIDへのマッピングを作成
            path_to_id = {}
            
            for _, row in metadata_df.iterrows():
                if 'ins_path' in row and pd.notna(row['ins_path']):
                    path = row['ins_path']
                    
                    # パスから検査IDを抽出
                    inspection_id = extract_inspection_id_from_path(path)
                    if inspection_id:
                        # 元のパスをキーとしてIDをマッピング
                        path_to_id[path] = inspection_id
                        
                        # 末尾の検査IDだけをキーとしてもマッピング
                        path_to_id[inspection_id] = inspection_id
            
            print(f"メタデータから{len(path_to_id)}個のIDマッピングを読み込みました")
            return path_to_id
            
        except Exception as e:
            print(f"エンコーディング {encoding} での読み込みに失敗: {str(e)}")
    
    print("すべてのエンコーディングでメタデータ読み込みに失敗しました。空のマッピングを返します。")
    return {}

# Fixed section for the analyze_shizuoka_2023_data function
def analyze_shizuoka_2023_data(base_dir, output_dir, metadata_path, velocity_threshold=30, exclude_initial_seconds=5):
    """
    2023年の静岡コホートデータを分析する関数
    
    Parameters:
    -----------
    base_dir : str
        基本ディレクトリ（GAP_Analysisが含まれるディレクトリ）
    output_dir : str
        出力ディレクトリ
    metadata_path : str
        メタデータファイルのパス
    velocity_threshold : float
        サッカード検出の速度閾値 (deg/s)
    exclude_initial_seconds : float
        試験開始から除外する秒数
    """
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist, creating: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # メタデータを読み込む
    path_to_id = load_metadata(metadata_path)
    
    # 2023年の静岡コホートディレクトリ
    cohort_dir = os.path.join(base_dir, "共有ドライブ", "GAP_Analysis", "Data", "GAP2_ShizuokaCohort", "2023")
    
    # パスが正しいか確認
    print(f"検索対象パス: {cohort_dir}")
    
    if not os.path.exists(cohort_dir):
        print(f"指定されたディレクトリが存在しません: {cohort_dir}")
        
        # 代替パスを試す
        alt_dir = r"G:\共有ドライブ\GAP_Analysis\Data\GAP2_ShizuokaCohort\2023"
        print(f"代替パスを試します: {alt_dir}")
        
        if os.path.exists(alt_dir):
            print(f"代替パスが見つかりました。こちらを使用します。")
            cohort_dir = alt_dir
        else:
            print(f"代替パスも存在しません。処理を中止します。")
            return
    
    # 被験者IDパターン
    id_pattern = re.compile(r'\d{14}_\d{5}')
    
    # すべての被験者フォルダを取得
    subject_dirs = [d for d in glob.glob(os.path.join(cohort_dir, "*")) 
                   if os.path.isdir(d) and id_pattern.match(os.path.basename(d))]
    
    print(f"検索対象ディレクトリ: {cohort_dir}")
    print(f"合計{len(subject_dirs)}個の被験者フォルダを処理します")
    
    # 結果を格納するリスト
    all_results = []
    
    # 各被験者フォルダを処理
    for i, subject_dir in enumerate(subject_dirs, 1):
        subject_id = os.path.basename(subject_dir)
        
        # calibrationファイルを検索
        calib_files = glob.glob(os.path.join(subject_dir, "calibration_*.csv"))
        
        if not calib_files:
            print(f"スキップ ({i}/{len(subject_dirs)}): {subject_id} - キャリブレーションファイルなし")
            continue
        
        print(f"処理中 ({i}/{len(subject_dirs)}): {subject_id} - {len(calib_files)}個のファイル")
        
        # InspectionDateAndIdを取得
        # まずmetadataからパスを検索
        inspection_id = None
        
        # 完全なパスを試す
        full_path = subject_dir.replace('\\', '/')
        if full_path in path_to_id:
            inspection_id = path_to_id[full_path]
        # バックスラッシュパスを試す
        elif subject_dir in path_to_id:
            inspection_id = path_to_id[subject_dir]
        # IDだけを試す
        elif subject_id in path_to_id:
            inspection_id = path_to_id[subject_id]
        else:
            # メタデータにない場合はフォルダ名をそのまま使用
            inspection_id = subject_id
            
        print(f"  使用するID: {inspection_id}")
        
        for calib_file in calib_files:
            try:
                # 眼球運動データを分析
                results = analyze_eye_movements(
                    calib_file,
                    output_dir,
                    inspection_id,
                    velocity_threshold=velocity_threshold,
                    exclude_initial_seconds=exclude_initial_seconds
                )
                
                # 追加情報を結果に加える
                results['exclude_initial_seconds'] = exclude_initial_seconds
                
                # freq と power_spectrum を文字列として保存
                # Convert numpy arrays to lists before converting to strings to avoid issues
                if 'freq' in results and hasattr(results['freq'], 'tolist'):
                    results['freq'] = str(results['freq'].tolist())
                else:
                    results['freq'] = str(results.get('freq', ''))
                    
                if 'power_spectrum' in results and hasattr(results['power_spectrum'], 'tolist'):
                    results['power_spectrum'] = str(results['power_spectrum'].tolist())
                else:
                    results['power_spectrum'] = str(results.get('power_spectrum', ''))
                
                # ファイル名を追加
                results['file_name'] = os.path.basename(calib_file)
                
                # 結果をリストに追加
                all_results.append(results)
                
                print(f"  処理成功: {os.path.basename(calib_file)}")
                
            except Exception as e:
                print(f"  エラー - {os.path.basename(calib_file)}: {str(e)}")
                import traceback
                print(traceback.format_exc())  # Print full traceback for debugging
    
    # 結果をデータフレームに変換
    if all_results:
        print(f"\n結果の保存処理を開始: {len(all_results)}件のデータ")
        try:
            # DataFrameに変換
            combined_results = pd.DataFrame(all_results)
            
            # DataFrameの内容を確認
            print(f"DataFrame columns: {combined_results.columns.tolist()}")
            print(f"DataFrame shape: {combined_results.shape}")
            
            # InspectionDateAndIdでソート
            combined_results = combined_results.sort_values('InspectionDateAndId')
            
            # 列の順序を指定（InspectionDateAndIdを最初に）
            cols = ['InspectionDateAndId', 'file_name'] + [col for col in combined_results.columns 
                                                         if col not in ['InspectionDateAndId', 'file_name']]
            combined_results = combined_results[cols]
            
            # CSVファイルとして保存
            output_file = os.path.join(output_dir, f'shizuoka2023_EyeCenterAngle_saccade_analysis_vel{velocity_threshold}_excl{exclude_initial_seconds}s_angle_micro_1.csv')
            print(f"保存先ファイル: {output_file}")
            
            # Try with explicit encoding
            combined_results.to_csv(output_file, index=False, encoding='utf-8')
            
            # 保存されたかどうか確認
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"ファイル保存成功: サイズ {file_size} バイト")
            else:
                print(f"ファイルが保存されていません。パスと権限を確認してください: {output_file}")
                
            print(f"\n分析完了:")
            print(f"- 処理された結果: {len(all_results)}件")
            print(f"保存先: {output_file}")
        except Exception as e:
            print(f"結果の保存中にエラーが発生しました: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
    else:
        print("処理可能なデータが見つかりませんでした。")

# Modified execution code
if __name__ == "__main__":
    # パス設定 - 絶対パスを使用
    base_dir = r"G:"  # G:ドライブのルート
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\feature_generation\fdata"
    metadata_path = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\feature_generation\fdata\shizuoka2023_moca_metadata_u.csv"
    
    # パスの存在確認
    print(f"メタデータファイルの存在確認: {os.path.exists(metadata_path)}")
    print(f"出力ディレクトリの存在確認: {os.path.exists(output_dir)}")
    
    # パラメータ設定
    velocity_threshold = 30         # サッカード検出の速度閾値 (deg/s)
    exclude_initial_seconds = 5     # 試験開始時刻から除外する秒数
    
    # Try alternative output directory if the primary one doesn't exist or is inaccessible
    if not os.path.exists(output_dir) or not os.access(output_dir, os.W_OK):
        print(f"出力ディレクトリがアクセスできないか存在しません: {output_dir}")
        alt_output_dir = os.path.join(os.path.expanduser("~"), "Documents", "EyeTrackingResults")
        print(f"代替出力ディレクトリを試します: {alt_output_dir}")
        os.makedirs(alt_output_dir, exist_ok=True)
        output_dir = alt_output_dir
    
    # Run analysis
    analyze_shizuoka_2023_data(base_dir, output_dir, metadata_path, velocity_threshold, exclude_initial_seconds)
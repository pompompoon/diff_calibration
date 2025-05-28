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
            trajectory_metrics = {
                'saccade_mean_amplitude': np.mean([t['amplitude'] for t in saccade_trajectories]),
                'saccade_max_amplitude': max([t['amplitude'] for t in saccade_trajectories]) if saccade_trajectories else 0,
                'saccade_mean_duration': np.mean([t['duration'] for t in saccade_trajectories]),
                'saccade_mean_peak_velocity': np.mean([t['peak_velocity'] for t in saccade_trajectories]),
                'saccade_curvature_index': np.mean([t['curvature_index'] for t in saccade_trajectories]) if saccade_trajectories else 0
            }
            saccade_metrics.update(trajectory_metrics)
        
        return saccade_metrics
    
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
        
        # サッカード関連の指標（速度閾値30deg/sのみを使用）
        results.update(self.detect_saccades(velocity_threshold=30))
        
        # 方向転換の分析
        results.update(self.calculate_direction_changes())
        
        # FFT解析
        results.update(self.calculate_fft_metrics())
        
        return results
    
    def plot_saccade_trajectories(self, output_dir, base_name, velocity_threshold=30):
        """サッカードの軌跡をプロット"""
        # サッカードの軌跡を検出
        saccade_trajectories = self.calculate_saccade_trajectories(velocity_threshold)
        
        if not saccade_trajectories:
            print(f"サッカードが検出されませんでした: {base_name}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 全体の眼球運動軌跡
        ax.plot(self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 'b-', alpha=0.3, label='全体の軌跡')
        
        # 各サッカードの軌跡
        for i, saccade in enumerate(saccade_trajectories):
            ax.plot(saccade['trajectory_x'], saccade['trajectory_y'], 'r-', linewidth=2, 
                    label=f'サッカード {i+1}' if i == 0 else "")
            
            # 始点と終点をマーク
            ax.plot(saccade['trajectory_x'][0], saccade['trajectory_y'][0], 'go', label='開始点' if i == 0 else "")
            ax.plot(saccade['trajectory_x'][-1], saccade['trajectory_y'][-1], 'mo', label='終了点' if i == 0 else "")
        
        ax.set_title('サッカード軌跡の可視化')
        ax.set_xlabel('水平方向の眼球角度 (deg)')
        ax.set_ylabel('垂直方向の眼球角度 (deg)')
        ax.axis('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_saccades.png')
        plt.savefig(plot_path)
        plt.close()
    
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

def preprocess_data(df):
    """
    データの前処理を行う関数
    - 最初の5秒間のデータを除外
    - -100の値を除外
    - 距離の計算
    - カラム名の調整
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
    
    # 開始から5秒後のタイムスタンプを計算 (ミリ秒単位)
    cutoff_timestamp = initial_timestamp + 5000
    
    # 5秒以降のデータのみをフィルタリング
    df = df[df['timestamp'] >= cutoff_timestamp].copy()
    
    # データが残っているか確認
    if df.empty:
        raise ValueError("5秒間のフィルタリング後にデータが残っていません")
    
    print(f"最初の5秒間のデータを除外: {initial_timestamp}ms から {cutoff_timestamp}ms")
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
    
    # 軌跡のプロット生成
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    analyzer.plot_saccade_trajectories(output_dir, base_name, velocity_threshold=30)
    analyzer.plot_profiles(output_dir, base_name)
    
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
        output_file = os.path.join(output_dir, f'combined_EyeCenterAngle_saccade_analysis_{sensitivity_threshold}.csv')
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
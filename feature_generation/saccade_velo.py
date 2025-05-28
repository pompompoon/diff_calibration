import numpy as np
import pandas as pd
from scipy.ndimage import label
import os
import glob
from extractioncondition.sensitivityfilter import SensitivityFilter

class SaccadeFeaturesCalculator:
    def __init__(self, df):
        """
        サッカード特徴量計算クラスの初期化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            分析対象のデータフレーム（前処理済み）
        """
        self.df = df
        self.process_data()
    
    def process_data(self):
        """データの前処理とベースとなる計算を行う"""
        # 時間差分の計算
        self.dt = np.diff(self.df['timestamp']) / 1000  # msからsに変換
        
        # 位置の差分を計算
        self.dx = np.diff(self.df['EyeCenterAngleX'])
        self.dy = np.diff(self.df['EyeCenterAngleY'])
        
        # 各方向の速度と合成速度の計算
        self.velocity_x = self.dx / self.dt
        self.velocity_y = self.dy / self.dt
        self.velocity = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
        # タイムスタンプ（先頭の値を引いて0からの相対時間に変換）
        self.timestamps = self.df['timestamp'].values[1:] - self.df['timestamp'].values[0]
    
    def detect_saccades(self, velocity_threshold=30):
        """
        サッカードを検出する
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
            
        Returns:
        --------
        list : サッカードの情報リスト
        """
        # 速度条件のみでサッカードを検出
        saccade_mask = self.velocity > velocity_threshold
        
        # 連続したTrueの領域（サッカード）を特定
        labeled_regions, num_regions = label(saccade_mask)
        
        saccades = []
        
        # 各サッカード領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # サッカードの開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1  # データポイントとして次のインデックスも含める
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df) - 1:  # -1は速度データの長さ調整
                end_idx = len(self.df) - 2
            
            # 開始と終了の時間・位置
            start_time = self.timestamps[start_idx]
            end_time = self.timestamps[min(end_idx, len(self.timestamps) - 1)]
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx + 1]  # 速度との添え字合わせのため+1
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx + 1]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx + 1, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx + 1, len(self.df) - 1)]
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # タイムスタンプ、速度データの抽出
            timestamps = self.timestamps[region_indices]
            velocities = self.velocity[region_indices]
            velocities_x = self.velocity_x[region_indices]
            velocities_y = self.velocity_y[region_indices]
            
            # 最大速度と加速度
            peak_velocity = np.max(velocities)
            peak_velocity_idx = np.argmax(velocities)
            
            # 加速時間と減速時間
            acceleration_time = timestamps[peak_velocity_idx] - timestamps[0] if peak_velocity_idx > 0 else 0
            deceleration_time = timestamps[-1] - timestamps[peak_velocity_idx] if peak_velocity_idx < len(timestamps) - 1 else 0
            
            # 加速・減速の非対称性比率（skewness）
            skewness = acceleration_time / (acceleration_time + deceleration_time) if (acceleration_time + deceleration_time) > 0 else 0.5
            
            # 速度プロファイルの形状特徴（4次モーメント、歪度、尖度）
            velocity_mean = np.mean(velocities)
            velocity_std = np.std(velocities)
            
            if len(velocities) > 2 and velocity_std > 0:
                velocity_skewness = np.mean(((velocities - velocity_mean) / velocity_std) ** 3)
                velocity_kurtosis = np.mean(((velocities - velocity_mean) / velocity_std) ** 4) - 3  # 正規分布からの偏差
            else:
                velocity_skewness = 0
                velocity_kurtosis = 0
            
            # 軌道の曲率（直線からの逸脱度）
            path_length = 0
            for j in range(len(region_indices) - 1):
                dx = self.df['EyeCenterAngleX'].iloc[region_indices[j] + 1 + 1] - self.df['EyeCenterAngleX'].iloc[region_indices[j] + 1]
                dy = self.df['EyeCenterAngleY'].iloc[region_indices[j] + 1 + 1] - self.df['EyeCenterAngleY'].iloc[region_indices[j] + 1]
                path_length += np.sqrt(dx**2 + dy**2)
            
            curvature_index = path_length / amplitude if amplitude > 0 else 0
            
            # サッカード情報をまとめる
            saccade_info = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'amplitude': amplitude,
                'peak_velocity': peak_velocity,
                'mean_velocity': velocity_mean,
                'std_velocity': velocity_std,
                'acceleration_time': acceleration_time,
                'deceleration_time': deceleration_time,
                'skewness': skewness,
                'velocity_skewness': velocity_skewness,
                'velocity_kurtosis': velocity_kurtosis,
                'curvature_index': curvature_index,
                'timestamps': timestamps,
                'velocities': velocities,
                'velocities_x': velocities_x,
                'velocities_y': velocities_y
            }
            
            saccades.append(saccade_info)
        
        return saccades
    
    def calculate_features_by_amplitude(self, angle_thresholds=[5, 10, 20, 30, 40, 50], tolerance=2.5, velocity_threshold=30):
        """
        指定された角度ごとにサッカードの特徴量を計算する
        
        Parameters:
        -----------
        angle_thresholds : list
            抽出する目標角度のリスト (deg)
        tolerance : float
            許容誤差 (deg)
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
            
        Returns:
        --------
        dict : 角度ごとの特徴量の辞書
        """
        # サッカードを検出
        saccades = self.detect_saccades(velocity_threshold=velocity_threshold)
        
        # 角度ごとの特徴量を格納する辞書
        features_by_amplitude = {}
        
        # 各振幅に対して
        for angle in angle_thresholds:
            # 振幅が角度範囲内のサッカードをフィルタリング
            filtered_saccades = [s for s in saccades if abs(s['amplitude'] - angle) <= tolerance]
            
            # サッカードがなければ空の結果を追加
            if not filtered_saccades:
                features_by_amplitude[angle] = {
                    'angle': angle,
                    'count': 0,
                    'mean_duration': 0,
                    'std_duration': 0,
                    'mean_amplitude': 0,
                    'std_amplitude': 0,
                    'mean_peak_velocity': 0,
                    'std_peak_velocity': 0,
                    'mean_velocity': 0,
                    'std_velocity': 0,
                    'mean_acceleration_time': 0,
                    'std_acceleration_time': 0,
                    'mean_deceleration_time': 0,
                    'std_deceleration_time': 0,
                    'mean_skewness': 0,
                    'std_skewness': 0,
                    'mean_velocity_skewness': 0,
                    'std_velocity_skewness': 0,
                    'mean_velocity_kurtosis': 0,
                    'std_velocity_kurtosis': 0,
                    'mean_curvature_index': 0,
                    'std_curvature_index': 0
                }
                continue
            
            # 各特徴量の平均と標準偏差を計算
            features = {
                'angle': angle,
                'count': len(filtered_saccades),
                'mean_duration': np.mean([s['duration'] for s in filtered_saccades]),
                'std_duration': np.std([s['duration'] for s in filtered_saccades]),
                'mean_amplitude': np.mean([s['amplitude'] for s in filtered_saccades]),
                'std_amplitude': np.std([s['amplitude'] for s in filtered_saccades]),
                'mean_peak_velocity': np.mean([s['peak_velocity'] for s in filtered_saccades]),
                'std_peak_velocity': np.std([s['peak_velocity'] for s in filtered_saccades]),
                'mean_velocity': np.mean([s['mean_velocity'] for s in filtered_saccades]),
                'std_velocity': np.mean([s['std_velocity'] for s in filtered_saccades]),
                'mean_acceleration_time': np.mean([s['acceleration_time'] for s in filtered_saccades]),
                'std_acceleration_time': np.std([s['acceleration_time'] for s in filtered_saccades]),
                'mean_deceleration_time': np.mean([s['deceleration_time'] for s in filtered_saccades]),
                'std_deceleration_time': np.std([s['deceleration_time'] for s in filtered_saccades]),
                'mean_skewness': np.mean([s['skewness'] for s in filtered_saccades]),
                'std_skewness': np.std([s['skewness'] for s in filtered_saccades]),
                'mean_velocity_skewness': np.mean([s['velocity_skewness'] for s in filtered_saccades]),
                'std_velocity_skewness': np.std([s['velocity_skewness'] for s in filtered_saccades]),
                'mean_velocity_kurtosis': np.mean([s['velocity_kurtosis'] for s in filtered_saccades]),
                'std_velocity_kurtosis': np.std([s['velocity_kurtosis'] for s in filtered_saccades]),
                'mean_curvature_index': np.mean([s['curvature_index'] for s in filtered_saccades]),
                'std_curvature_index': np.std([s['curvature_index'] for s in filtered_saccades])
            }
            
            features_by_amplitude[angle] = features
            
            # 速度データを保存（最大10個まで）
            for i, saccade in enumerate(filtered_saccades[:10]):
                for j, vel in enumerate(saccade['velocities']):
                    time_key = f'time_{i}_{j}'
                    vel_key = f'velocity_{i}_{j}'
                    features[time_key] = saccade['timestamps'][j]
                    features[vel_key] = vel
        
        return features_by_amplitude

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
    
    return df

def calculate_saccade_features(input_file, output_dir, angle_thresholds=[5, 10, 20, 30, 40, 50], tolerance=2.5, velocity_threshold=30, exclude_initial_seconds=5):
    """
    サッカードの特徴量を計算する関数
    
    Parameters:
    -----------
    input_file : str
        入力ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    angle_thresholds : list
        抽出する目標角度のリスト (deg)
    tolerance : float
        許容誤差 (deg)
    velocity_threshold : float
        サッカード検出の速度閾値 (deg/s)
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数
    """
    # データの読み込みと前処理
    df = pd.read_csv(input_file)
    df = preprocess_data(df, exclude_initial_seconds=exclude_initial_seconds)
    
    # 特徴量計算クラスの初期化
    calculator = SaccadeFeaturesCalculator(df)
    
    # 角度ごとの特徴量計算
    features = calculator.calculate_features_by_amplitude(
        angle_thresholds=angle_thresholds, 
        tolerance=tolerance, 
        velocity_threshold=velocity_threshold
    )
    
    # ファイル名からベース名を取得
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 特徴量をデータフレームに変換
    features_list = [features[angle] for angle in angle_thresholds]
    features_df = pd.DataFrame(features_list)
    
    # CSVとして保存
    output_file = os.path.join(output_dir, f'{base_name}_saccade_features.csv')
    features_df.to_csv(output_file, index=False)
    
    return features_df

def process_multiple_participants(base_dir, output_dir, angle_thresholds=[5, 10, 20, 30, 40, 50], tolerance=2.5, velocity_threshold=30, sensitivity_threshold=15, exclude_initial_seconds=5):
    """
    複数の参加者データを処理する関数
    
    Parameters:
    -----------
    base_dir : str
        GAPデータが格納されているベースディレクトリ
    output_dir : str
        結果を出力するディレクトリ
    angle_thresholds : list
        抽出する目標角度のリスト (deg)
    tolerance : float
        許容誤差 (deg)
    velocity_threshold : float
        サッカード検出の速度閾値 (deg/s)
    sensitivity_threshold : int
        LessThanSensitivityの閾値
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数
    """
    # 感度フィルターを初期化
    sensitivity_filter = SensitivityFilter(threshold=sensitivity_threshold)
    
    # calibrationファイルを含むすべてのフォルダを取得
    folders = sorted(glob.glob(os.path.join(base_dir, "2024*")))
    
    # 結果を格納するリスト
    all_features = []
    skipped_folders = []
    
    print(f"合計{len(folders)}個のフォルダを処理します。")
    
    # 各フォルダ内のcalibrationファイルを処理
    for i, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        
        # InspectionDateAndIdをフォルダ名から抽出
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
                
                # 参加者専用の出力ディレクトリを作成
                participant_output_dir = os.path.join(output_dir, inspection_date_and_id)
                os.makedirs(participant_output_dir, exist_ok=True)
                
                # サッカード特徴量計算
                features_df = calculate_saccade_features(
                    calib_file, participant_output_dir, 
                    angle_thresholds=angle_thresholds,
                    tolerance=tolerance,
                    velocity_threshold=velocity_threshold,
                    exclude_initial_seconds=exclude_initial_seconds
                )
                
                # 特徴量に参加者情報を追加
                features_df['InspectionDateAndId'] = inspection_date_and_id
                features_df['less_than_sensitivity_count'] = sensitivity_count
                features_df['file_name'] = os.path.basename(calib_file)
                
                all_features.append(features_df)
                
            except Exception as e:
                print(f"エラー - {os.path.basename(calib_file)}: {str(e)}")
    
    # 結果をマージ
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # 列の順序を指定（参加者IDと角度を最初に）
        cols = ['InspectionDateAndId', 'less_than_sensitivity_count', 'file_name', 'angle', 'count'] + [
            col for col in combined_features.columns if col not in [
                'InspectionDateAndId', 'less_than_sensitivity_count', 'file_name', 'angle', 'count'
            ]
        ]
        combined_features = combined_features[cols]
        
        # CSVファイルとして保存
        output_file = os.path.join(output_dir, f'combined_saccade_features_sens{sensitivity_threshold}_vel{velocity_threshold}.csv')
        combined_features.to_csv(output_file, index=False)
        
        # 速度プロファイルの時系列データを除いた要約統計量のみのファイルも保存
        summary_cols = [col for col in cols if not (col.startswith('time_') or col.startswith('velocity_'))]
        summary_df = combined_features[summary_cols]
        summary_file = os.path.join(output_dir, f'summary_saccade_features_sens{sensitivity_threshold}_vel{velocity_threshold}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # スキップされたフォルダの情報を保存
        if skipped_folders:
            skipped_df = pd.DataFrame(skipped_folders)
            skipped_file = os.path.join(output_dir, f'skipped_folders_{sensitivity_threshold}.csv')
            skipped_df.to_csv(skipped_file, index=False)
        
        print(f"\n処理完了:")
        print(f"- 処理された参加者数: {len(set(combined_features['InspectionDateAndId']))}人")
        print(f"- スキップされたフォルダ: {len(skipped_folders)}件")
        print(f"保存先:")
        print(f"- 詳細特徴量: {output_file}")
        print(f"- 要約統計量: {summary_file}")
        if skipped_folders:
            print(f"- スキップ情報: {skipped_file}")

# 実行コード
if __name__ == "__main__":
    # ベースディレクトリ（GAPデータフォルダ）とアウトプットディレクトリを指定
    base_dir = r"G:\共有ドライブ\GAP_長寿研\GAPデータ"
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\saccade_features"
    
    # パラメータ設定
    angle_thresholds = [5, 10, 20, 30, 40, 50]  # 抽出する角度のリスト (deg)
    tolerance = 2.5                # 角度の許容誤差 (deg)
    velocity_threshold = 30        # サッカード検出の速度閾値 (deg/s)
    sensitivity_threshold = 15     # 感度閾値
    exclude_initial_seconds = 5    # 試験開始時刻から除外する秒数
    
    # 複数の参加者データを処理
    process_multiple_participants(
        base_dir, output_dir, 
        angle_thresholds=angle_thresholds,
        tolerance=tolerance,
        velocity_threshold=velocity_threshold,
        sensitivity_threshold=sensitivity_threshold,
        exclude_initial_seconds=exclude_initial_seconds)
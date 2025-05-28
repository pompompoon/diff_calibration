import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import os
import glob
from extractioncondition.sensitivityfilter import SensitivityFilter

class EyeMovementAnalyzer:
    def __init__(self, df):
        """
        眼球運動分析クラスの初期化
        
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
        
        # 速度の計算（合成速度とX,Y方向の速度）
        self.velocity_x = self.dx / self.dt
        self.velocity_y = self.dy / self.dt
        self.velocity = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
        # 加速度の計算
        self.acceleration_x = np.diff(self.velocity_x) / self.dt[:-1]
        self.acceleration_y = np.diff(self.velocity_y) / self.dt[:-1]
        self.acceleration = np.sqrt(self.acceleration_x**2 + self.acceleration_y**2)
        
        # 躍度の計算
        self.jerk_x = np.diff(self.acceleration_x) / self.dt[:-2]
        self.jerk_y = np.diff(self.acceleration_y) / self.dt[:-2]
        self.jerk = np.sqrt(self.jerk_x**2 + self.jerk_y**2)
    
    def calculate_metrics(self):
        """基本的な統計量を計算"""
        metrics = {
            # 合成速度の統計量
            'mean_velocity': np.mean(self.velocity),
            'max_velocity': np.max(self.velocity),
            'std_velocity': np.std(self.velocity),
            
            # X方向の速度の統計量
            'mean_velocity_x': np.mean(self.velocity_x),
            'max_velocity_x': np.max(np.abs(self.velocity_x)),
            'std_velocity_x': np.std(self.velocity_x),
            
            # Y方向の速度の統計量
            'mean_velocity_y': np.mean(self.velocity_y),
            'max_velocity_y': np.max(np.abs(self.velocity_y)),
            'std_velocity_y': np.std(self.velocity_y),
            
            # 合成加速度の統計量
            'mean_acceleration': np.mean(self.acceleration),
            'max_acceleration': np.max(self.acceleration),
            'std_acceleration': np.std(self.acceleration),
            
            # X方向の加速度の統計量
            'mean_acceleration_x': np.mean(self.acceleration_x),
            'max_acceleration_x': np.max(np.abs(self.acceleration_x)),
            'std_acceleration_x': np.std(self.acceleration_x),
            
            # Y方向の加速度の統計量
            'mean_acceleration_y': np.mean(self.acceleration_y),
            'max_acceleration_y': np.max(np.abs(self.acceleration_y)),
            'std_acceleration_y': np.std(self.acceleration_y),
            
            # 合成躍度の統計量
            'mean_jerk': np.mean(self.jerk),
            'max_jerk': np.max(self.jerk),
            'std_jerk': np.std(self.jerk),
            
            # X方向の躍度の統計量
            'mean_jerk_x': np.mean(self.jerk_x),
            'max_jerk_x': np.max(np.abs(self.jerk_x)),
            'std_jerk_x': np.std(self.jerk_x),
            
            # Y方向の躍度の統計量
            'mean_jerk_y': np.mean(self.jerk_y),
            'max_jerk_y': np.max(np.abs(self.jerk_y)),
            'std_jerk_y': np.std(self.jerk_y),
            
            # 視線位置の統計量
            'gaze_endpoint_std_x': np.std(self.df['EyeCenterAngleX']),
            'gaze_endpoint_std_y': np.std(self.df['EyeCenterAngleY']),
            'gaze_endpoint_std_combined': np.sqrt(np.std(self.df['EyeCenterAngleX'])**2 + np.std(self.df['EyeCenterAngleY'])**2)
        }
        
        return metrics
    
    def analyze(self):
        """すべての分析を実行して結果を返す"""
        results = {}
        
        # 基本的な統計量
        results.update(self.calculate_metrics())
        
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

def analyze_eye_movements(input_file, output_dir, exclude_initial_seconds=5):
    """
    眼球運動データを分析する関数
    
    Parameters:
    -----------
    input_file : str
        入力ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数
    """
    # データの読み込みと前処理
    df = pd.read_csv(input_file)
    df, manhattan_stats, euclidean_stats = preprocess_data(df, exclude_initial_seconds=exclude_initial_seconds)
    
    # 分析の実行
    analyzer = EyeMovementAnalyzer(df)
    results = analyzer.analyze()
    
    # 距離の統計量を結果に追加
    results.update(manhattan_stats)
    results.update(euclidean_stats)

    return results

def analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold=15, exclude_initial_seconds=5):
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
    exclude_initial_seconds : float
        試験開始時刻から除外する秒数
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
                results = analyze_eye_movements(calib_file, output_dir,
                                              exclude_initial_seconds=exclude_initial_seconds)
                
                # 分析パラメータと識別情報を追加
                results['InspectionDateAndId'] = inspection_date_and_id
                results['less_than_sensitivity_count'] = sensitivity_count
                results['exclude_initial_seconds'] = exclude_initial_seconds
                
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
        
        # 特定の列のみを保持する
        required_columns = [
            'InspectionDateAndId', 
            'less_than_sensitivity_count',
            'mean_velocity', 
            'max_velocity', 
            'std_velocity',
            'mean_velocity_x', 
            'max_velocity_x', 
            'std_velocity_x',
            'mean_velocity_y', 
            'max_velocity_y', 
            'std_velocity_y',
            'mean_acceleration', 
            'max_acceleration', 
            'std_acceleration',
            'mean_acceleration_x', 
            'max_acceleration_x', 
            'std_acceleration_x',
            'mean_acceleration_y', 
            'max_acceleration_y', 
            'std_acceleration_y',
            'mean_jerk', 
            'max_jerk', 
            'std_jerk',
            'mean_jerk_x', 
            'max_jerk_x', 
            'std_jerk_x',
            'mean_jerk_y', 
            'max_jerk_y', 
            'std_jerk_y',
            'manhattan_distance_sum', 
            'manhattan_distance_mean',
            'euclidean_distance_sum', 
            'euclidean_distance_mean',
            'gaze_endpoint_std_x',
            'gaze_endpoint_std_y',
            'gaze_endpoint_std_combined',
            'exclude_initial_seconds'
        ]
        
        # 存在する列のみをフィルタリング
        available_columns = [col for col in required_columns if col in combined_results.columns]
        combined_results = combined_results[available_columns]
        
        # CSVファイルとして保存（閾値と除外時間を含むファイル名）
        output_file = os.path.join(output_dir, f'combined_EyeCenterAngle_analysis_sens{sensitivity_threshold}_excl{exclude_initial_seconds}s.csv')
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
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\feature_generation\fdata"
    
    # パラメータ設定
    sensitivity_threshold = 15      # 感度閾値
    exclude_initial_seconds = 5     # 試験開始時刻から除外する秒数
    
    # 複数のcalibrationデータを分析
    combined_results = analyze_multiple_calibrations(base_dir, output_dir, 
                                                   sensitivity_threshold=sensitivity_threshold,
                                                   exclude_initial_seconds=exclude_initial_seconds)
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
    
    def analyze(self, velocity_threshold=30):
        """すべての分析を実行して結果を返す"""
        results = {}
        
        # 基本的な統計量
        results.update(self.calculate_basic_metrics())
        
        # 視線の終点の標準偏差
        results.update(self.calculate_gaze_endpoint_std())
        
        # サッカード関連の指標（指定された速度閾値を使用）
        results.update(self.detect_saccades(velocity_threshold=velocity_threshold))
        
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
                results['freq'] = str(results['freq'])
                results['power_spectrum'] = str(results['power_spectrum'])
                
                # ファイル名を追加
                results['file_name'] = os.path.basename(calib_file)
                
                # 結果をリストに追加
                all_results.append(results)
                
            except Exception as e:
                print(f"  エラー - {os.path.basename(calib_file)}: {str(e)}")
    
    # 結果をデータフレームに変換
    if all_results:
        # DataFrameに変換
        combined_results = pd.DataFrame(all_results)
        
        # InspectionDateAndIdでソート
        combined_results = combined_results.sort_values('InspectionDateAndId')
        
        # 列の順序を指定（InspectionDateAndIdを最初に）
        cols = ['InspectionDateAndId', 'file_name'] + [col for col in combined_results.columns 
                                                     if col not in ['InspectionDateAndId', 'file_name']]
        combined_results = combined_results[cols]
        
        # CSVファイルとして保存
        output_file = os.path.join(output_dir, f'shizuoka2023_EyeCenterAngle_saccade_analysis_vel{velocity_threshold}_excl{exclude_initial_seconds}s.csv')
        combined_results.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n分析完了:")
        print(f"- 処理された結果: {len(all_results)}件")
        print(f"保存先: {output_file}")
    else:
        print("処理可能なデータが見つかりませんでした。")

# 実行コード
if __name__ == "__main__":
    # パス設定 - 絶対パスを使用
    base_dir = r"G:"  # G:ドライブのルート
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\feature_generation\fdata"
    metadata_path = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\feature_generation\fdata\moca_metadata.csv"
    
    # パスの存在確認
    print(f"メタデータファイルの存在確認: {os.path.exists(metadata_path)}")
    print(f"出力ディレクトリの存在確認: {os.path.exists(output_dir)}")
    
    # パラメータ設定
    velocity_threshold = 30         # サッカード検出の速度閾値 (deg/s)
    exclude_initial_seconds = 5     # 試験開始時刻から除外する秒数
    
    try:
        # 2023年の静岡コホートデータを分析
        analyze_shizuoka_2023_data(
            base_dir, 
            output_dir,
            metadata_path,
            velocity_threshold=velocity_threshold,
            exclude_initial_seconds=exclude_initial_seconds
        )
    except Exception as e:
        print(f"実行中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
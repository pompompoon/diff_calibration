import os
import glob
import pandas as pd

class SensitivityFilter:
    def __init__(self, threshold=15):
        """
        感度フィルタークラスの初期化
        
        Parameters:
        -----------
        threshold : int
            LessThanSensitivityの合計閾値（これを超えると除外）
        """
        self.threshold = threshold
        
    def check_result_file(self, folder_path):
        """
        resultファイルを読み込み、LessThanSensitivityの合計を計算する
        
        Parameters:
        -----------
        folder_path : str
            結果ファイルが格納されているフォルダのパス
            
        Returns:
        --------
        bool : キャリブレーションファイルを処理するかどうか
        int : LessThanSensitivityの合計
        """
        try:
            # resultファイルを検索
            result_files = glob.glob(os.path.join(folder_path, "result_*.csv"))
            
            if not result_files:
                print(f"Warning: No result files found in {folder_path}")
                return True, 0  # resultファイルがない場合は処理を続行
            
            # 最新のresultファイルを使用
            latest_result = max(result_files, key=os.path.getctime)
            
            # resultファイルを読み込む
            df = pd.read_csv(latest_result)
            
            # LessThanSensitivityの合計を計算
            less_than_sensitivity_count = df['LessThanSensitivity'].sum()
            
            # 閾値との比較
            should_process = less_than_sensitivity_count <= self.threshold
            
            return should_process, less_than_sensitivity_count
            
        except Exception as e:
            print(f"Error processing result file in {folder_path}: {str(e)}")
            return True, 0  # エラーの場合は処理を続行
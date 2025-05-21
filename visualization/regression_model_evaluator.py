import os
import numpy as np
import pandas as pd
import japanize_matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionModelEvaluator:
    """
    回帰モデルの評価指標を計算し、結果をファイルに出力するクラス。
    訓練精度、汎化精度、その他の評価指標を算出します。
    """
    
    def __init__(self, output_dir=None):
        """
        RegressionModelEvaluatorクラスの初期化
        
        Parameters:
        -----------
        output_dir : str, default=None
            評価結果を保存する出力ディレクトリのパス。
            Noneの場合は結果を保存しません。
        """
        self.output_dir = output_dir
        
        # 出力ディレクトリの作成
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"出力ディレクトリを作成しました: {self.output_dir}")
    
    def set_output_dir(self, output_dir):
        """
        出力ディレクトリを設定または変更する
        
        Parameters:
        -----------
        output_dir : str
            新しい出力ディレクトリのパス
        """
        self.output_dir = output_dir
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"出力ディレクトリを作成しました: {self.output_dir}")
    
    def _handle_nan_values(self, X, y, set_name="データ"):
        """
        NaN値をチェックして処理する内部メソッド
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            特徴量データ
        y : array-like
            ターゲットデータ
        set_name : str, default="データ"
            データセットの名前（ログメッセージ用）
            
        Returns:
        --------
        tuple
            処理後の (X, y)
        """
        # pandas.Seriesをnumpy配列に変換
        if isinstance(y, pd.Series):
            y = y.values
        
        # NaN値の処理
        valid_mask = ~np.isnan(y)
        
        if not np.all(valid_mask):
            print(f"警告: {set_name}のターゲットに{np.sum(~valid_mask)}個のNaN値が含まれています。これらを除外します。")
            y = y[valid_mask]
            
            if isinstance(X, pd.DataFrame):
                X = X.iloc[valid_mask]
            else:
                X = X[valid_mask]
        
        return X, y
    
    def calculate_metrics(self, results, model_name='model'):
        """
        回帰モデルの評価指標を計算する
        
        Parameters:
        -----------
        results : dict
            モデル学習結果を含む辞書。以下のキーが必要:
            - 'model': 学習済みモデル
            - 'X_train': 訓練データの特徴量
            - 'y_train': 訓練データのターゲット
            - 'X_test': テストデータの特徴量
            - 'true_values': テストデータのターゲット（y_test）
            - 'predictions': テストデータに対する予測値（オプション）
        model_name : str, default='model'
            モデル名（ファイル名の一部として使用）
        
        Returns:
        --------
        dict
            計算された評価指標の辞書
        """
        print("\n訓練精度、汎化精度、評価指標を算出します...")
        
        # 結果辞書から必要なデータを取得
        model = results.get('model')
        X_train = results.get('X_train')
        y_train = results.get('y_train')
        X_test = results.get('X_test')
        y_test = results.get('true_values')
        
        # NaN値の処理
        X_train, y_train = self._handle_nan_values(X_train, y_train, "訓練データ")
        X_test, y_test = self._handle_nan_values(X_test, y_test, "テストデータ")
        
        # 詳細な評価指標を計算
        metrics = {}
        
        try:
            # 訓練データでの予測
            y_train_pred = model.predict(X_train)
            
            # NaN値の処理
            train_pred_valid_mask = ~np.isnan(y_train_pred)
            if not np.all(train_pred_valid_mask):
                print(f"警告: 訓練データの予測に{np.sum(~train_pred_valid_mask)}個のNaN値が含まれています。これらを除外します。")
                y_train_pred = y_train_pred[train_pred_valid_mask]
                y_train_for_metrics = y_train[train_pred_valid_mask]
            else:
                y_train_for_metrics = y_train
            
            # 訓練データでの評価指標
            metrics['train_mse'] = mean_squared_error(y_train_for_metrics, y_train_pred)
            metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
            metrics['train_mae'] = mean_absolute_error(y_train_for_metrics, y_train_pred)
            metrics['train_r2'] = r2_score(y_train_for_metrics, y_train_pred)
            
            # テストデータでの予測
            y_test_pred = results.get('predictions')
            if y_test_pred is None:
                y_test_pred = model.predict(X_test)
            
            # NaN値の処理
            test_pred_valid_mask = ~np.isnan(y_test_pred)
            if not np.all(test_pred_valid_mask):
                print(f"警告: テストデータの予測に{np.sum(~test_pred_valid_mask)}個のNaN値が含まれています。これらを除外します。")
                y_test_pred = y_test_pred[test_pred_valid_mask]
                y_test_for_metrics = y_test[test_pred_valid_mask]
            else:
                y_test_for_metrics = y_test
            
            # テストデータでの評価指標
            metrics['test_mse'] = mean_squared_error(y_test_for_metrics, y_test_pred)
            metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
            metrics['test_mae'] = mean_absolute_error(y_test_for_metrics, y_test_pred)
            metrics['test_r2'] = r2_score(y_test_for_metrics, y_test_pred)
            metrics['test_spearman_corr'] = spearman_correlation(y_test_for_metrics, y_test_pred)

            # MAPE（平均絶対パーセント誤差）を計算（ゼロ除算に注意）
            epsilon = np.finfo(np.float64).eps
            train_nonzero_mask = np.abs(y_train_for_metrics) > epsilon
            if np.any(train_nonzero_mask):
                train_mape = np.mean(np.abs((y_train_for_metrics[train_nonzero_mask] - y_train_pred[train_nonzero_mask]) / 
                                             y_train_for_metrics[train_nonzero_mask])) * 100
                metrics['train_mape'] = train_mape
            else:
                metrics['train_mape'] = np.nan
                
            test_nonzero_mask = np.abs(y_test_for_metrics) > epsilon
            if np.any(test_nonzero_mask):
                test_mape = np.mean(np.abs((y_test_for_metrics[test_nonzero_mask] - y_test_pred[test_nonzero_mask]) / 
                                           y_test_for_metrics[test_nonzero_mask])) * 100
                metrics['test_mape'] = test_mape
            else:
                metrics['test_mape'] = np.nan
            
        except Exception as e:
            print(f"評価指標の計算中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー時はNoneを設定
            metrics['train_rmse'] = None
            metrics['test_rmse'] = None
        
        # 統計情報を追加
        metrics['train_mean'] = np.mean(y_train)
        metrics['train_std'] = np.std(y_train)
        metrics['test_mean'] = np.mean(y_test)
        metrics['test_std'] = np.std(y_test)
        
        # 指標の表示
        self._print_metrics(metrics)
        
        # 指標をファイルに出力
        if self.output_dir:
            self._save_metrics(metrics, model_name)
        
        return metrics
    
    def _print_metrics(self, metrics):
        """
        評価指標を表示する内部メソッド
        
        Parameters:
        -----------
        metrics : dict
            評価指標の辞書
        """
        print("\n============= 回帰モデル評価指標 =============")
        print(f"【訓練データでの精度】")
        print(f"  RMSE: {metrics.get('train_rmse', 'N/A'):.4f}" if metrics.get('train_rmse') is not None else "  RMSE: 計算できませんでした")
        print(f"  MAE: {metrics.get('train_mae', 'N/A'):.4f}" if metrics.get('train_mae') is not None else "  MAE: 計算できませんでした")
        print(f"  R²: {metrics.get('train_r2', 'N/A'):.4f}" if metrics.get('train_r2') is not None else "  R²: 計算できませんでした")
        print(f"  MAPE: {metrics.get('train_mape', 'N/A'):.2f}%" if metrics.get('train_mape') is not None else "  MAPE: 計算できませんでした")
        
        print("\n【テストデータでの精度（汎化精度）】")
        print(f"  RMSE: {metrics.get('test_rmse', 'N/A'):.4f}" if metrics.get('test_rmse') is not None else "  RMSE: 計算できませんでした")
        print(f"  MAE: {metrics.get('test_mae', 'N/A'):.4f}" if metrics.get('test_mae') is not None else "  MAE: 計算できませんでした")
        print(f"  R²: {metrics.get('test_r2', 'N/A'):.4f}" if metrics.get('test_r2') is not None else "  R²: 計算できませんでした")
        print(f"  MAPE: {metrics.get('test_mape', 'N/A'):.2f}%" if metrics.get('test_mape') is not None else "  MAPE: 計算できませんでした")
        print(f"  スピアマン相関係数: {metrics.get('test_spearman_corr', 'N/A'):.4f}" if metrics.get('test_spearman_corr') is not None else "  スピアマン相関係数: 計算できませんでした")
        print("\n【データ統計量】")
        print(f"  訓練データ平均: {metrics.get('train_mean', 'N/A'):.4f}")
        print(f"  訓練データ標準偏差: {metrics.get('train_std', 'N/A'):.4f}")
        print(f"  テストデータ平均: {metrics.get('test_mean', 'N/A'):.4f}")
        print(f"  テストデータ標準偏差: {metrics.get('test_std', 'N/A'):.4f}")
        
        print("=========================================")
    
    def _save_metrics(self, metrics, model_name):
        """
        評価指標をファイルに保存する内部メソッド
        
        Parameters:
        -----------
        metrics : dict
            評価指標の辞書
        model_name : str
            モデル名（ファイル名の一部として使用）
        """
        # CSVファイルに保存
        metrics_df = pd.DataFrame([metrics])
        csv_path = os.path.join(self.output_dir, f"{model_name}_evaluation_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"指標を {csv_path} に保存しました")
        
        # テキストファイルにも詳細情報を保存
        txt_path = os.path.join(self.output_dir, f"{model_name}_evaluation_metrics.txt")
        with open(txt_path, 'w') as f:
            f.write("============= 回帰モデル評価指標 =============\n")
            f.write("【訓練データでの精度】\n")
            f.write(f"  RMSE: {metrics.get('train_rmse', 'N/A'):.4f}\n" if metrics.get('train_rmse') is not None else "  RMSE: 計算できませんでした\n")
            f.write(f"  MAE: {metrics.get('train_mae', 'N/A'):.4f}\n" if metrics.get('train_mae') is not None else "  MAE: 計算できませんでした\n")
            f.write(f"  R²: {metrics.get('train_r2', 'N/A'):.4f}\n" if metrics.get('train_r2') is not None else "  R²: 計算できませんでした\n")
            f.write(f"  MAPE: {metrics.get('train_mape', 'N/A'):.2f}%\n" if metrics.get('train_mape') is not None else "  MAPE: 計算できませんでした\n")
            
            f.write("\n【テストデータでの精度（汎化精度）】\n")
            f.write(f"  RMSE: {metrics.get('test_rmse', 'N/A'):.4f}\n" if metrics.get('test_rmse') is not None else "  RMSE: 計算できませんでした\n")
            f.write(f"  MAE: {metrics.get('test_mae', 'N/A'):.4f}\n" if metrics.get('test_mae') is not None else "  MAE: 計算できませんでした\n")
            f.write(f"  R²: {metrics.get('test_r2', 'N/A'):.4f}\n" if metrics.get('test_r2') is not None else "  R²: 計算できませんでした\n")
            f.write(f"  MAPE: {metrics.get('test_mape', 'N/A'):.2f}%\n" if metrics.get('test_mape') is not None else "  MAPE: 計算できませんでした\n")
            
            f.write("\n【データ統計量】\n")
            f.write(f"  訓練データ平均: {metrics.get('train_mean', 'N/A'):.4f}\n")
            f.write(f"  訓練データ標準偏差: {metrics.get('train_std', 'N/A'):.4f}\n")
            f.write(f"  テストデータ平均: {metrics.get('test_mean', 'N/A'):.4f}\n")
            f.write(f"  テストデータ標準偏差: {metrics.get('test_std', 'N/A'):.4f}\n")
            
            f.write("=========================================\n")
        
        print(f"詳細情報を {txt_path} に保存しました")

    def calculate_model_metrics_from_file(self, result_file, output_dir=None, model_name='model'):
        """
        保存された結果ファイルからモデル評価指標を計算する
        
        Parameters:
        -----------
        result_file : str
            結果ファイルのパス（pickleファイル）
        output_dir : str, default=None
            出力ディレクトリのパス（指定された場合は初期化時のパスを上書き）
        model_name : str, default='model'
            モデル名（ファイル名の一部として使用）
            
        Returns:
        --------
        dict
            計算された評価指標の辞書
        """
        import pickle
        
        # 出力ディレクトリの設定（指定された場合）
        if output_dir:
            self.set_output_dir(output_dir)
        
        try:
            # 結果ファイルの読み込み
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            # 評価指標の計算
            return self.calculate_metrics(results, model_name)
            
        except Exception as e:
            print(f"結果ファイルの読み込みまたは評価指標の計算中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return {}
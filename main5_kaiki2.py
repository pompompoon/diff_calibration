# mainコードです。回帰コード

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os

import matplotlib
import matplotlib.pyplot as plt
# カレントディレクトリをPythonパスに追加
import japanize_matplotlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy.stats import pearsonr
import datetime
import pickle

# 必要なインポートを追加
import lightgbm
import xgboost
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# 視覚化モジュールをインポート
from visualization.regression_visualizer import RegressionVisualizer, EyeTrackingVisualizer

from models.partial_dependence_plotter_kaiki import PartialDependencePlotter

from visualization.regression_model_evaluator import RegressionModelEvaluator
from visualization.regression_pdp_handler import create_partial_dependence_plots, organize_result_files


# 評価と部分依存プロット用のクラスをインポート
# 実際の環境に合わせてインポートパスを調整してください
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

# 部分依存プロット関連の関数
def create_partial_dependence_plots(results, output_dir=None, prefix="pdp", top_n=5):
    """
    回帰モデルの部分依存プロットを作成して保存する関数
    
    Parameters:
    -----------
    results : dict
        モデル学習結果を含む辞書
    output_dir : str, default=None
        プロットを保存するディレクトリのパス
    prefix : str, default="pdp"
        ファイル名のプレフィックス
    top_n : int, default=5
        上位何個の特徴量を表示するか
    
    Returns:
    --------
    list
        保存されたファイルパスのリスト
    """
    from sklearn.inspection import PartialDependenceDisplay
    
    if output_dir is None:
        output_dir = '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = results.get('model')
    X_train = results.get('X_train')
    feature_importance = results.get('feature_importance')
    
    if model is None or X_train is None or feature_importance is None:
        print("部分依存プロットの作成に必要なデータがありません。")
        return []
    
    # 上位n個の特徴量を取得
    top_features = feature_importance.head(top_n)['feature'].tolist()
    # 特徴量名から特徴量インデックスを取得
    top_feature_indices = [list(X_train.columns).index(feature) for feature in top_features]
    
    saved_files = []
    
    # 個別の特徴量に対する部分依存プロット
    try:
        # まとめてPDPプロットを作成
        print(f"上位{top_n}個の特徴量の部分依存プロットを作成しています...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 部分依存プロットの作成
            display = PartialDependenceDisplay.from_estimator(
                model, X_train, top_feature_indices, 
                kind='average', target=0, ax=ax,
                n_cols=3, grid_resolution=50,
                random_state=42
            )
        
        # タイトルとレイアウト調整
        fig.suptitle(f'Top {top_n} Features - Partial Dependence Plots', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # ファイル保存
        combined_path = os.path.join(output_dir, f"{prefix}_top{top_n}.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(combined_path)
        print(f"部分依存プロットを保存しました: {combined_path}")
        
        # 個別の特徴量のプロット（より詳細なプロット）
        for i, feature in enumerate(top_features):
            try:
                print(f"特徴量 {feature} の部分依存プロットを作成しています...")
                feature_idx = list(X_train.columns).index(feature)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    display = PartialDependenceDisplay.from_estimator(
                        model, X_train, [feature_idx], 
                        kind='average', target=0, ax=ax,
                        grid_resolution=100,
                        random_state=42
                    )
                
                # タイトルと軸ラベルの設定
                ax.set_title(f'Partial Dependence Plot - {feature}', fontsize=14)
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Partial Dependence', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # 部分依存の範囲に基づいてY軸を調整
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                # ファイル名の作成（特殊文字を置換）
                safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                individual_path = os.path.join(output_dir, f"{prefix}_{i+1}_{safe_feature_name}.png")
                plt.savefig(individual_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(individual_path)
                print(f"個別の部分依存プロットを保存しました: {individual_path}")
                
            except Exception as e:
                print(f"特徴量 {feature} の部分依存プロット作成中にエラーが発生しました: {e}")
    
    except Exception as e:
        print(f"部分依存プロットの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    # ICEプロットの作成（Individual Conditional Expectation）
    try:
        # ICEプロット用のサブディレクトリ作成
        ice_dir = os.path.join(output_dir, "ice_plots")
        if not os.path.exists(ice_dir):
            os.makedirs(ice_dir)
            
        print(f"ICEプロットを作成しています...")
        for i, feature in enumerate(top_features[:3]):  # 上位3つの特徴量だけICEプロットを作成
            try:
                feature_idx = list(X_train.columns).index(feature)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    display = PartialDependenceDisplay.from_estimator(
                        model, X_train, [feature_idx], 
                        kind='both', target=0, ax=ax,
                        grid_resolution=50,
                        n_jobs=-1,
                        n_ice_lines=50,  # ICEラインの数を制限
                        random_state=42
                    )
                
                # タイトルと軸ラベルの設定
                ax.set_title(f'ICE Plot - {feature}', fontsize=14)
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Partial Dependence', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # ファイル名の作成
                safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                ice_path = os.path.join(ice_dir, f"ice_{i+1}_{safe_feature_name}.png")
                plt.savefig(ice_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(ice_path)
                print(f"ICEプロットを保存しました: {ice_path}")
                
            except Exception as e:
                print(f"特徴量 {feature} のICEプロット作成中にエラーが発生しました: {e}")
        
    except Exception as e:
        print(f"ICEプロットの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    # 2変量の相互作用プロット
    try:
        if len(top_features) >= 2:
            # 相互作用プロット用のサブディレクトリ作成
            interactions_dir = os.path.join(output_dir, "interactions")
            if not os.path.exists(interactions_dir):
                os.makedirs(interactions_dir)
                
            print(f"特徴量間の相互作用プロットを作成しています...")
            
            # 上位3つの特徴量の組み合わせでループ
            for i in range(min(3, len(top_features))):
                for j in range(i+1, min(3, len(top_features))):
                    feature1 = top_features[i]
                    feature2 = top_features[j]
                    
                    try:
                        feature_idx1 = list(X_train.columns).index(feature1)
                        feature_idx2 = list(X_train.columns).index(feature2)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            display = PartialDependenceDisplay.from_estimator(
                                model, X_train, [(feature_idx1, feature_idx2)], 
                                kind='average', target=0, ax=ax,
                                grid_resolution=20,
                                random_state=42,
                                contour_kw={'cmap': 'viridis', 'alpha': 0.75}
                            )
                        
                        # タイトルと軸ラベルの設定
                        ax.set_title(f'Interaction: {feature1} vs {feature2}', fontsize=14)
                        
                        # ファイル名の作成
                        safe_feature_name1 = feature1.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        safe_feature_name2 = feature2.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        interaction_path = os.path.join(
                            interactions_dir, 
                            f"interaction_{safe_feature_name1}_vs_{safe_feature_name2}.png"
                        )
                        plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        saved_files.append(interaction_path)
                        print(f"相互作用プロットを保存しました: {interaction_path}")
                        
                    except Exception as e:
                        print(f"特徴量 {feature1} と {feature2} の相互作用プロット作成中にエラーが発生しました: {e}")
    
    except Exception as e:
        print(f"相互作用プロットの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"部分依存プロットの作成が完了しました。合計 {len(saved_files)} 個のファイルを保存しました。")
    return saved_files

def organize_result_files(data_file_name, output_dir):
    """
    現在の実行で生成されたファイルのみを新しいディレクトリに整理する関数
    部分依存プロットとその他の解析結果を含め、タイムスタンプに基づいて管理する
    
    Parameters:
    -----------
    data_file_name : str
        データファイル名
    output_dir : str
        出力ディレクトリ
    
    Returns:
    --------
    str
        新しい出力ディレクトリのパス
    """
    # データファイル名から拡張子を除去したベース名を取得
    base_name = os.path.splitext(os.path.basename(data_file_name))[0]
    
    # 現在の時刻を取得（実行開始時刻として扱う）
    current_time = datetime.datetime.now()
    # 少し前の時刻（例：30分前）を計算し、その時刻以降に生成・更新されたファイルのみをコピー
    filter_time = current_time - datetime.timedelta(minutes=30)
    
    # 新しい出力ディレクトリを作成
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    new_output_dir = f"{base_name}_{timestamp}"
    
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        print(f"\n新しい出力ディレクトリを作成しました: {new_output_dir}")
    
    file_count = 0
    
    # output_dirが存在する場合のみ処理を実行
    if os.path.exists(output_dir):
        # interactionsディレクトリの処理
        interactions_dir = os.path.join(output_dir, "interactions")
        if os.path.exists(interactions_dir) and os.path.isdir(interactions_dir):
            new_interactions_dir = os.path.join(new_output_dir, "interactions")
            if not os.path.exists(new_interactions_dir):
                os.makedirs(new_interactions_dir)
            
            # 最近更新されたファイルのみをコピー
            for filename in os.listdir(interactions_dir):
                source_path = os.path.join(interactions_dir, filename)
                if os.path.isfile(source_path):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        dest_path = os.path.join(new_interactions_dir, filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                            file_count += 1
                            print(f"コピーしました: {dest_path}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
        
        # ice_plotsディレクトリの処理
        ice_plots_dir = os.path.join(output_dir, "ice_plots")
        if os.path.exists(ice_plots_dir) and os.path.isdir(ice_plots_dir):
            new_ice_plots_dir = os.path.join(new_output_dir, "ice_plots")
            if not os.path.exists(new_ice_plots_dir):
                os.makedirs(new_ice_plots_dir)
            
            # 最近更新されたファイルのみをコピー
            for filename in os.listdir(ice_plots_dir):
                source_path = os.path.join(ice_plots_dir, filename)
                if os.path.isfile(source_path):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        dest_path = os.path.join(new_ice_plots_dir, filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                            file_count += 1
                            print(f"コピーしました: {dest_path}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
        
        # rootディレクトリ内の重要なファイルをコピー
        important_patterns = [
            'pdp_*.png',                     # 部分依存プロット
            'true_vs_predicted.png',         # 実測値vs予測値
            'residuals.png',                 # 残差プロット
            'residual_distribution.png',     # 残差分布
            'feature_importance.png',        # 特徴量重要度
            'feature_importance_*.csv',      # 特徴量重要度CSV
            'correlation_matrix.png',        # 相関行列
            'feature_target_correlations.png', # 特徴量と目的変数の相関
            '*_evaluation_metrics.txt',      # 評価指標テキスト
            '*_evaluation_metrics.csv'       # 評価指標CSV
        ]
        
        # パターンに一致する最近のファイルをコピー
        import glob
        for pattern in important_patterns:
            pattern_path = os.path.join(output_dir, pattern)
            for source_file in glob.glob(pattern_path):
                if os.path.isfile(source_file):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_file))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        filename = os.path.basename(source_file)
                        dest_file = os.path.join(new_output_dir, filename)
                        try:
                            shutil.copy2(source_file, dest_file)
                            file_count += 1
                            print(f"コピーしました: {dest_file}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_file} -> {e}")
        
        # saved_modelディレクトリの処理
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        if os.path.exists(saved_model_dir) and os.path.isdir(saved_model_dir):
            # 新しいディレクトリ内にsaved_modelディレクトリを作成
            new_saved_model_dir = os.path.join(new_output_dir, 'saved_model')
            if not os.path.exists(new_saved_model_dir):
                os.makedirs(new_saved_model_dir)
            
            # saved_modelディレクトリ内の最近のファイルをコピー
            for filename in os.listdir(saved_model_dir):
                source_path = os.path.join(saved_model_dir, filename)
                if os.path.isfile(source_path):
                    # base_nameが含まれるファイルのみを対象にする
                    if base_name in filename:
                        # ファイルの更新時刻を取得
                        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                        # フィルタ時間より新しいファイルのみコピー
                        if mod_time >= filter_time:
                            dest_path = os.path.join(new_saved_model_dir, filename)
                            try:
                                shutil.copy2(source_path, dest_path)
                                file_count += 1
                                print(f"モデルファイルをコピーしました: {dest_path}")
                            except Exception as e:
                                print(f"モデルファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
    
    else:
        print(f"警告: 出力ディレクトリ {output_dir} が見つかりません")
    
    print(f"{file_count}個のファイルを {new_output_dir} にコピーしました")
    return new_output_dir
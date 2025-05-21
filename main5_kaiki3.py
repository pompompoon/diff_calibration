# mainコードです。回帰コード

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
import shutil  # ファイルコピー用に追加

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

# 評価と部分依存プロット用のクラスをインポート
from visualization.regression_model_evaluator import RegressionModelEvaluator
from visualization.regression_pdp_handler import create_partial_dependence_plots, organize_result_files
# カスタムCatBoostRegressionModelをインポート
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from models.catboost_regression_model import CatBoostRegressionModel

# RegressionBaggingModelを回帰用に修正したクラス
class RegressionBaggingModel:
    """
    回帰問題用のバギングモデル
    
    複数のサブモデルを作成し、それぞれの予測を平均化するアプローチを実装しています。
    """
    
    def __init__(self, base_model='lightgbm', n_bags=10, sample_size=0.8, 
                 random_state=42, base_params=None, categorical_features=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        base_model : str, default='lightgbm'
            使用するベースモデル。'lightgbm', 'xgboost', 'random_forest', 'catboost'から選択
        n_bags : int, default=10
            作成するバッグ（サブモデル）の数
        sample_size : float, default=0.8
            各バッグに使用するデータサンプルの割合
        random_state : int, default=42
            再現性のための乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル変数のインデックスリスト（CatBoost用）
        """
        self.base_model = base_model
        self.n_bags = n_bags
        self.sample_size = sample_size
        self.random_state = random_state
        self.base_params = base_params or {}
        self.categorical_features = categorical_features
        self.models = []
        self.feature_importances_ = None

    def _get_lightgbm_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """LightGBMモデルを作成して学習するメソッド（回帰用）"""
        import lightgbm as lgb
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # データセット作成
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = None
        if X_eval is not None and y_eval is not None:
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        
        # モデル学習
        model = lgb.train(
            params, 
            lgb_train, 
            valid_sets=lgb_eval if lgb_eval else None,
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if lgb_eval else None,
        )
        
        return model
    
    def _get_xgboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """XGBoostモデルを作成して学習するメソッド（回帰用）"""
        import xgboost as xgb
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0,
            'seed': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # データセット作成
        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = None
        if X_eval is not None and y_eval is not None:
            deval = xgb.DMatrix(X_eval, label=y_eval)
        
        # モデル学習
        watchlist = [(dtrain, 'train')]
        if deval:
            watchlist.append((deval, 'eval'))
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=watchlist,
            early_stopping_rounds=50 if deval else None,
            verbose_eval=False
        )
        
        return model
    
    def _get_random_forest_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """RandomForestモデルを作成して学習するメソッド（回帰用）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # デフォルトパラメータ
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成と学習（回帰器に変更）
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def _get_catboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """CatBoostモデルを作成して学習するメソッド（回帰用）"""
        from catboost import CatBoostRegressor, Pool
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',  # 回帰用に変更
            'random_seed': self.random_state,
            'verbose': False
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成（回帰器に変更）
        model = CatBoostRegressor(**params)
        
        # カテゴリカル変数がある場合はPoolを使用
        if self.categorical_features is not None:
            train_pool = Pool(X_train, y_train, cat_features=self.categorical_features)
            eval_pool = None
            if X_eval is not None and y_eval is not None:
                eval_pool = Pool(X_eval, y_eval, cat_features=self.categorical_features)
            
            # モデル学習
            model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            # カテゴリカル変数がない場合は通常の学習
            eval_set = None
            if X_eval is not None and y_eval is not None:
                eval_set = (X_eval, y_eval)
            
            # モデル学習
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
        
        return model
    
    def _sample_and_train(self, X_train, y_train, X_eval=None, y_eval=None):
        """訓練データからサンプリングしてモデルを学習するメソッド"""
        # データのインデックスからランダムサンプリング
        n_samples = X_train.shape[0]
        sample_size = int(n_samples * self.sample_size)
        
        np.random.seed(np.random.randint(0, 10000))  # バッグごとに異なるシード
        indices = np.random.choice(n_samples, sample_size, replace=True)
        
        X_sampled = X_train[indices]
        y_sampled = y_train[indices]
        
        # 指定されたベースモデルに基づいてモデルを学習
        if self.base_model == 'lightgbm':
            model = self._get_lightgbm_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'xgboost':
            model = self._get_xgboost_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'random_forest':
            model = self._get_random_forest_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'catboost':
            model = self._get_catboost_model(X_sampled, y_sampled, X_eval, y_eval)
        else:
            raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
        
        return model
    
    def fit(self, X, y, X_eval=None, y_eval=None):
        """
        複数のモデルを学習するメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            学習用特徴量
        y : array-like
            学習用目的変数
        X_eval : array-like or pandas DataFrame, optional
            評価用特徴量（早期停止用）
        y_eval : array-like, optional
            評価用目的変数（早期停止用）
        
        Returns:
        --------
        self : object
            自身を返す
        """
        # 必要に応じてnumpy配列に変換
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_values = X
            self.feature_names = None
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        if X_eval is not None:
            if isinstance(X_eval, pd.DataFrame):
                X_eval_values = X_eval.values
            else:
                X_eval_values = X_eval
        else:
            X_eval_values = None
            
        if y_eval is not None:
            if isinstance(y_eval, pd.Series):
                y_eval_values = y_eval.values
            else:
                y_eval_values = y_eval
        else:
            y_eval_values = None
        
        # 複数のモデルを学習
        self.models = []
        for i in range(self.n_bags):
            print(f"バッグ {i+1}/{self.n_bags} を学習中...")
            model = self._sample_and_train(X_values, y_values, X_eval_values, y_eval_values)
            self.models.append(model)
        
        # 全モデルの平均から特徴量重要度を計算
        self._calculate_feature_importance(X_values)
        
        return self
    
    def _calculate_feature_importance(self, X):
        """全バッグの平均から特徴量重要度を計算するメソッド"""
        # 特徴量の数を取得
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        # 全モデルの特徴量重要度を合計
        for model in self.models:
            if self.base_model == 'lightgbm':
                importances = model.feature_importance(importance_type='gain')
            elif self.base_model == 'xgboost':
                importances = model.get_score(importance_type='gain')
                # XGBoostは特徴量名をキーとした辞書を返す
                for i, imp in importances.items():
                    feature_idx = int(i.replace('f', ''))
                    feature_importances[feature_idx] += imp
                continue  # XGBoostは別処理のため、平均化をスキップ
            elif self.base_model == 'random_forest':
                importances = model.feature_importances_
            elif self.base_model == 'catboost':
                importances = model.get_feature_importance()
            else:
                return None
            
            # 合計に追加
            feature_importances += importances
        
        # 特徴量重要度の平均を計算
        if self.base_model != 'xgboost':  # XGBoostは別処理
            feature_importances /= len(self.models)
        
        self.feature_importances_ = feature_importances
        return feature_importances
    
    def predict(self, X):
        """
        全バッグの予測を平均化して予測値を返すメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        
        Returns:
        --------
        predictions : array-like
            予測値
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # 予測配列の初期化
        n_samples = X_values.shape[0]
        predictions = np.zeros(n_samples)
        
        # 全モデルの予測を合計
        for model in self.models:
            if self.base_model == 'lightgbm':
                preds = model.predict(X_values)
            elif self.base_model == 'xgboost':
                import xgboost as xgb
                dtest = xgb.DMatrix(X_values)
                preds = model.predict(dtest)
            elif self.base_model == 'random_forest':
                preds = model.predict(X_values)
            elif self.base_model == 'catboost':
                from catboost import Pool
                if self.categorical_features is not None:
                    test_pool = Pool(X_values, cat_features=self.categorical_features)
                    preds = model.predict(test_pool)
                else:
                    preds = model.predict(X_values)
            else:
                raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
            
            predictions += preds
        
        # 予測の平均化
        predictions /= len(self.models)
        
        return predictions
    
    def perform_grid_search(self, X_train, y_train, X_test, y_test):
        """
        簡易的なパラメータ選択を行うメソッド
        
        Parameters:
        -----------
        X_train : array-like or pandas DataFrame
            学習用特徴量
        y_train : array-like
            学習用目的変数
        X_test : array-like or pandas DataFrame
            テスト用特徴量
        y_test : array-like
            テスト用目的変数
        
        Returns:
        --------
        best_params : dict
            最適なパラメータ
        best_model : object
            最適なパラメータで学習したモデル
        """
        # 簡易版では、バッグの数を変えてみる
        param_grid = {'n_bags': [5, 10, 15]}
        
        best_score = float('inf')  # RMSEを最小化したいので無限大から開始
        best_params = {}
        
        for n_bags in param_grid['n_bags']:
            print(f"\nバッグ数 {n_bags} でテスト中...")
            self.n_bags = n_bags
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            
            # RMSEで評価（回帰問題なので）
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"RMSE: {rmse:.4f}")
            
            if rmse < best_score:
                best_score = rmse
                best_params = {'n_bags': n_bags}
        
        # 最適パラメータでモデルを再学習
        print(f"\n最適なバッグ数: {best_params['n_bags']}")
        self.n_bags = best_params['n_bags']
        self.fit(X_train, y_train)
        
        return best_params, self
    
    def get_model(self):
        """互換性のために自身を返すメソッド"""
        return self
    
    @staticmethod
    def run_cv(X, y, base_model='lightgbm', n_bags=10, n_splits=5, random_state=42, 
              base_params=None, categorical_features=None):
        """
        交差検証を実行する静的メソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        y : array-like
            目的変数
        base_model : str, default='lightgbm'
            ベースモデルの種類
        n_bags : int, default=10
            バッグの数
        n_splits : int, default=5
            交差検証の分割数
        random_state : int, default=42
            乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル特徴量のインデックス（CatBoost用）
            
        Returns:
        --------
        oof_preds : array-like
            Out-of-foldの予測値
        scores : dict
            評価指標のスコア
        """
        from sklearn.model_selection import KFold
        
        # 必要に応じてnumpy配列に変換
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
        else:
            X_values = X
            feature_names = None
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # 交差検証の設定（回帰なのでKFoldを使用）
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Out-of-fold予測の初期化
        oof_preds = np.zeros(len(y_values))
        
        # 交差検証ループ
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_values)):
            print(f"\n===== Fold {fold+1}/{n_splits} =====")
            
            # データの分割
            X_train, X_val = X_values[train_idx], X_values[val_idx]
            y_train, y_val = y_values[train_idx], y_values[val_idx]
            
            # モデルの作成と学習
            model = RegressionBaggingModel(
                base_model=base_model,
                n_bags=n_bags,
                random_state=random_state + fold,
                base_params=base_params,
                categorical_features=categorical_features
            )
            model.fit(X_train, y_train)
            
            # バリデーションデータでの予測
            oof_preds[val_idx] = model.predict(X_val)
            
        # 評価指標の計算
        mse = mean_squared_error(y_values, oof_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_values, oof_preds)
        r2 = r2_score(y_values, oof_preds)
        
        scores = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # 結果の表示
        print("\n===== 交差検証結果 =====")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        return oof_preds, scores


    def prepare_data_for_regression(df, target_column='target'):
        """データの前処理を行う関数（回帰用）"""
        print("\nデータの形状:", df.shape)
        
        # IDカラムを除外し、特徴量として使用するカラムを選択
        id_columns = ['InspectionDateAndId', 'Id']
        drop_cols = [col for col in id_columns if col in df.columns]
        drop_cols.append(target_column)
        
        features = df.drop(drop_cols, axis=1, errors='ignore')
        
        # 目的変数はそのまま使用（回帰なので変換は不要）
        target = df[target_column].copy()
        
        # 配列文字列の処理
        for col in ['freq', 'power_spectrum']:
            if col in features.columns:
                features[col] = features[col].apply(safe_convert_array)
        
        # 欠損値の削除
        features = features.dropna()
        target = target[features.index]
        
        print(f"\n処理後のデータ数: {len(features)}")
        print("\n目的変数の統計量:")
        print(target.describe())
        
        return features, target


    def safe_convert_array(x):
        """安全に配列文字列を数値に変換する関数"""
        if not isinstance(x, str):
            return np.nan
        
        if '...' in x:
            return np.nan
        
        try:
            x = x.strip('[]')
            numbers = []
            for num in x.split():
                try:
                    numbers.append(float(num))
                except ValueError:
                    continue
            
            if not numbers:
                return np.nan
            
            return np.mean(numbers)
        except:
            return np.nan


    def mean_absolute_percentage_error(y_true, y_pred):
        """
        平均絶対パーセント誤差(MAPE)を計算する関数
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like
            予測値
        
        Returns:
        --------
        mape : float
            平均絶対パーセント誤差（%単位）
        """
        # 0による除算を防ぐため、小さな値を追加
        epsilon = np.finfo(np.float64).eps
        
        # 実測値が0または非常に小さい場合の対処
        mask = np.abs(y_true) > epsilon
        
        if np.sum(mask) == 0:
            return np.nan  # 全ての実測値が0または非常に小さい場合
        
        # マスクを適用して計算
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        # MAPEの計算（%で表示）
        mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
        
        return mape


    def evaluate_regression_results(results):
        """回帰結果の評価を行う関数"""
        y_true = results['true_values']
        y_pred = results['predictions']
        
        # 評価指標の計算
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # ピアソン相関係数
        pearson_corr, p_value = pearsonr(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Pearson相関係数': pearson_corr,
            'P値': p_value
        }
        
        # 予測値と実測値の比較データフレーム
        pred_df = pd.DataFrame({
            'True_Value': results['true_values'],
            'Predicted_Value': results['predictions'],
            'Error': results['true_values'] - results['predictions'],
            'Abs_Error': np.abs(results['true_values'] - results['predictions'])
        }, index=results['test_indices'])
        
        return pred_df, metrics


    def train_regression_model(features, target, model_class=None, use_bagging=False, 
                           use_custom_catboost=False, random_state=42, **kwargs):
        """モデルの学習を行う関数（回帰用）"""
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=random_state
        )
        
        # 特徴量のスケーリング
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=features.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=features.columns
        )
        
        # オリジナルデータを保存（可視化のため）
        X_train_orig = X_train.copy()
        
        # CatBoostRegressionModelを使用する場合
        if use_custom_catboost:
            print("CatBoostRegressionModelを使用します...")
            
            # カテゴリカル特徴量の特定（例として文字列型の列）
            categorical_features = [i for i, dtype in enumerate(X_train.dtypes) if dtype == 'object']
            
            # モデルの初期化
            model = CatBoostRegressionModel(random_state=random_state)
            
            try:
                # グリッドサーチを実施
                best_params, best_model = model.perform_grid_search(
                    X_train_scaled, y_train, 
                    X_test_scaled, y_test,
                    categorical_features=categorical_features if categorical_features else None
                )
            except Exception as e:
                print(f"グリッドサーチでエラーが発生しました: {e}")
                print("基本モデルを使用します...")
                model.fit(X_train_scaled, y_train)
                best_model = model
                best_params = {}
            
            # 特徴量重要度を設定
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
        
        # RegressionBaggingModelを使用する場合
        elif use_bagging:
            print("RegressionBaggingModelを使用します...")
            base_model = kwargs.get('base_model', 'lightgbm')
            n_bags = kwargs.get('n_bags', 10)
            model = RegressionBaggingModel(
                base_model=base_model,
                n_bags=n_bags,
                random_state=random_state
            )
            
            try:
                best_params, best_model = model.perform_grid_search(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test
                )
            except Exception as e:
                print(f"グリッドサーチでエラーが発生しました: {e}")
                print("基本モデルを使用します...")
                best_model = model
                best_model.fit(X_train_scaled, y_train)
                best_params = {'n_bags': n_bags}
            
            # 特徴量重要度を設定
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
        
        else:
            # 通常のモデルを使用する場合
            try:
                # model_classはlambda関数なので呼び出す
                model = model_class()
                
                # scikit-learnベースのシンプルなグリッドサーチ
                from sklearn.model_selection import GridSearchCV
                
                # モデルタイプごとに適切なパラメータグリッドを設定
                if isinstance(model, lightgbm.LGBMRegressor):
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                elif isinstance(model, xgboost.XGBRegressor):
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                elif isinstance(model, RandomForestRegressor):
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20]
                    }
                elif isinstance(model, CatBoostRegressor):
                    param_grid = {
                        'iterations': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                else:
                    # パラメータグリッドが定義されていない場合はデフォルトモデルをそのまま使用
                    param_grid = {}
                if param_grid:
                    print(f"グリッドサーチを実行中...")
                    grid_search = GridSearchCV(
                        model, 
                        param_grid,
                        cv=3,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    print(f"最適パラメータ: {best_params}")
                else:
                    # パラメータグリッドがない場合はそのままフィット
                    model.fit(X_train_scaled, y_train)
                    best_model = model
                    best_params = {}
            except Exception as e:
                print(f"グリッドサーチでエラーが発生しました: {e}")
                print("基本モデルを使用します...")
                model = model_class()
                model.fit(X_train_scaled, y_train)
                best_model = model
                best_params = {}
            
            # 特徴量の重要度
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
        
        # 予測
        y_pred = best_model.predict(X_test_scaled)
        
        return {
            'model': best_model,
            'predictions': y_pred,
            'true_values': y_test,
            'feature_importance': feature_importance,
            'test_indices': X_test.index,
            'best_params': best_params,
            'features': features,
            'target': target,  # 全データのターゲット
            'X_train': X_train_scaled,
            'X_train_orig': X_train_orig,  # スケーリング前のトレーニングデータ
            'y_train': y_train,  # トレーニングデータのターゲット
            'X_test': X_test_scaled,
            'X_test_orig': X_test,  # スケーリング前のテストデータ
            'scaler': scaler
        }

    

    def run_regression_analysis(df, model_class=None, use_bagging=False, use_custom_catboost=False,
                         random_state=42, output_dir=None, target_column='target', 
                         data_file_name=None, **kwargs):
        """回帰分析を実行する関数"""
        # 新しい出力ディレクトリをここで作成
        original_output_dir = output_dir  # 元の出力ディレクトリを保存
        new_output_dir = None
        
        if output_dir and data_file_name:
            base_name = os.path.splitext(os.path.basename(data_file_name))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_output_dir = f"{base_name}_{timestamp}"
            
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)
                print(f"\n新しい出力ディレクトリを作成しました: {new_output_dir}")
                
            # interactionsとice_plots、saved_modelサブディレクトリも事前に作成
            interactions_dir = os.path.join(new_output_dir, "interactions")
            ice_plots_dir = os.path.join(new_output_dir, "ice_plots")
            saved_model_dir = os.path.join(new_output_dir, "saved_model")
            for subdir in [interactions_dir, ice_plots_dir, saved_model_dir]:
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                
            # 処理結果の出力先を新しいディレクトリに設定
            output_dir = new_output_dir

        if use_custom_catboost:
            print(f"\nCatBoostRegressionModelを使用します...")
        elif use_bagging:
            base_model = kwargs.get('base_model', 'lightgbm')
            n_bags = kwargs.get('n_bags', 10)
            print(f"\nRegressionBaggingModel with {base_model} base model and {n_bags} bags を使用します...")
        elif model_class is not None:
            print(f"\n{model_class.__name__} を使用します...")
        else:
            raise ValueError("model_classまたはuse_baggingまたはuse_custom_catboostのいずれかを指定してください。")
        
        print("データの前処理を開始...")
        features, target = prepare_data_for_regression(df, target_column=target_column)
        
        # データが空の場合はエラー
        if len(features) == 0 or len(target) == 0:
            raise ValueError("前処理後のデータが空です。データを確認してください。")
        
        # 可視化のためのインスタンス作成
        visualizer = EyeTrackingVisualizer(output_dir=output_dir)
        
        # 特徴量と目的変数の関係の可視化
        print("\n特徴量と目的変数の関係を分析します...")
        visualizer.plot_feature_correlations(features, target)
        
        print("\nモデルの学習を開始...")
        results = train_regression_model(
            features, target, 
            model_class=model_class, 
            use_bagging=use_bagging,
            use_custom_catboost=use_custom_catboost,
            random_state=random_state,
            **kwargs
        )
        
        # モデル名の決定
        if use_custom_catboost:
            model_name = "custom_catboost"
        elif use_bagging:
            model_name = f"reg_bagging_{kwargs.get('base_model', 'lightgbm')}"
        else:
            try:
                model_name = results['model'].__class__.__name__.lower()
            except:
                model_name = "regression_model"
        
        print("\n結果の評価を開始...")
        predictions_df, metrics = evaluate_regression_results(results)
        
        # 訓練精度と汎化精度の計算（新機能）
        print("\n訓練精度と汎化精度を計算します...")
        evaluator = RegressionModelEvaluator(output_dir=output_dir)
        evaluation_metrics = evaluator.calculate_metrics(results, model_name=model_name)
        
        # 結果の表示
        print("\n最適なパラメータ:")
        for param, value in results['best_params'].items():
            print(f"{param}: {value}")
        
        print("\n性能指標:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\n特徴量の重要度（上位10件）:")
        print(results['feature_importance'].head(10))
        
        # 結果の可視化
        print("\nモデル評価の可視化...")
        visualizer.plot_true_vs_predicted(results['true_values'], results['predictions'])
        visualizer.plot_residuals(results['true_values'], results['predictions'])
        visualizer.plot_feature_importance(results['feature_importance'])
        visualizer.plot_residual_distribution(results['true_values'], results['predictions'])
        
        # 部分依存グラフの作成（新機能）
        print("\n部分依存プロットを作成しています...")
        create_partial_dependence_plots(results, output_dir=output_dir, prefix=f"pdp_{model_name}", top_n=5)
        
        # 部分依存グラフの作成
        try:
            if hasattr(results['model'], 'feature_importances_'):
                # 特徴量重要度でソートした上位の特徴量を選択
                top_features = results['feature_importance'].head(6)['feature'].tolist()
                
                # 特徴量の名前をインデックスに変換
                feature_indices = [list(results['X_train'].columns).index(feature) for feature in top_features]
                
                # visualizer.plot_partial_dependenceメソッドを使用
                visualizer.plot_partial_dependence(
                    results['model'], 
                    results['X_train'], 
                    feature_indices,  # インデックスのリストを渡す
                    feature_names=results['X_train'].columns
                )
        except Exception as e:
            print(f"部分依存プロットの作成中にエラーが発生しました: {e}")
            
            # バックアップ: 直接PartialDependencePlotterを使用
            try:
                from models.partial_dependence_plotter_kaiki import PartialDependencePlotter
                
                plotter = PartialDependencePlotter(
                    model=results['model'],
                    features=results['X_train']
                )
                
                # 部分依存プロットを生成
                top_features = results['feature_importance'].head(6)['feature'].tolist()
                fig = plotter.plot_multiple_features(
                    top_features, 
                    n_cols=3,
                    grid_resolution=50
                )
                
                # 保存
                if output_dir:
                    plt.savefig(f'{output_dir}/partial_dependence.png')
                plt.close(fig)
            except Exception as e2:
                print(f"バックアップ方法でも部分依存プロットの作成に失敗しました: {e2}")
        
        # モデルの保存
        if output_dir:
            # モデルの保存ディレクトリ
            saved_model_dir = os.path.join(output_dir, 'saved_model')
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            
            # モデル名の取得
            model_file_path = os.path.join(saved_model_dir, f'{model_name}_model.pkl')
            
            try:
                # モデルとスケーラーを含む辞書を保存
                with open(model_file_path, 'wb') as f:
                    pickle.dump({
                        'model': results['model'],
                        'scaler': results['scaler'],
                        'feature_names': list(features.columns),
                        'creation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f)
                print(f"\nモデルを保存しました: {model_file_path}")
            except Exception as e:
                print(f"モデルの保存中にエラーが発生しました: {e}")
        
        # 結果をまとめる
        results['predictions_df'] = predictions_df
        results['metrics'] = metrics
        results['evaluation_metrics'] = evaluation_metrics
        
        # 出力ディレクトリがある場合は結果を保存
        if output_dir:
            # 予測結果の保存
            predictions_path = os.path.join(output_dir, f'predictions_{model_name}.csv')
            try:
                predictions_df.to_csv(predictions_path)
                print(f"予測結果を保存しました: {predictions_path}")
            except Exception as e:
                print(f"予測結果の保存中にエラーが発生しました: {e}")
            
            # 特徴量の重要度の保存
            importance_path = os.path.join(output_dir, f'feature_importance_{model_name}.csv')
            try:
                results['feature_importance'].to_csv(importance_path)
                print(f"特徴量重要度を保存しました: {importance_path}")
            except Exception as e:
                print(f"特徴量重要度の保存中にエラーが発生しました: {e}")
            
            # 最適パラメータを保存
            params_path = os.path.join(output_dir, f'best_parameters_{model_name}.txt')
            try:
                with open(params_path, 'w') as f:
                    for param, value in results['best_params'].items():
                        f.write(f"{param}: {value}\n")
                print(f"最適パラメータを保存しました: {params_path}")
            except Exception as e:
                print(f"最適パラメータの保存中にエラーが発生しました: {e}")
        
        # ファイル整理オプションがある場合
        if kwargs.get('organize_files', False) and output_dir != original_output_dir and new_output_dir:
            try:
                print("\nファイルを整理します...")
                organize_result_files(data_file_name, output_dir)
            except Exception as e:
                print(f"ファイル整理中にエラーが発生しました: {e}")
        
        print(f"\n分析が完了しました。結果は {output_dir} に保存されました。")
        return results, new_output_dir

    def setup_matplotlib_japanese_font():
        """
        matplotlibで日本語を表示するための設定を行う関数
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import platform
        
        # OSの判定
        system = platform.system()
        
        # フォント設定
        if system == 'Windows':
            # Windowsの場合
            font_family = 'MS Gothic'
            matplotlib.rcParams['font.family'] = font_family
        elif system == 'Darwin':
            # macOSの場合
            font_family = 'AppleGothic'
            matplotlib.rcParams['font.family'] = font_family
        else:
            # Linux/その他の場合
            try:
                # IPAフォントがインストールされていることを前提
                font_family = 'IPAGothic'
                matplotlib.rcParams['font.family'] = font_family
            except:
                print("日本語フォントの設定に失敗しました。デフォルトフォントを使用します。")
        
        # matplotlibのグローバル設定
        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
        
        print(f"matplotlibのフォントを'{font_family}'に設定しました")
        return font_family


if __name__ == "__main__":
   
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='視線追跡データを用いた回帰モデルの実行')
    parser.add_argument('--cv', action='store_true', help='交差検証を実行する')
    parser.add_argument('--bagging', action='store_true', help='バギングモデルを使用する')
    parser.add_argument('--base-model', type=str, default='lightgbm', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'], 
                        help='バギングで使用するベースモデル')
    parser.add_argument('--n-bags', type=int, default=10, help='バギングのバッグ数')
    parser.add_argument('--splits', type=int, default=5, help='交差検証の分割数')
    parser.add_argument('--model', type=str, default='lightgbm', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'],
                        help='使用するモデル')
    parser.add_argument('--random-state', type=int, default=42, help='乱数シード')
    parser.add_argument('--data-path', type=str, 
                       default="data",
                       help='データファイルのパス')
    parser.add_argument('--data-file', type=str, 
                       default="S2023combined_15sec_windows_analysis_sens1_vel0_excl0s_k.csv",
                       help='データファイル名')
    parser.add_argument('--target-column', type=str,
                       default="target",
                       help='目的変数のカラム名')
    parser.add_argument('--output-dir', type=str, 
                       default="result",
                       help='結果出力ディレクトリ')
    parser.add_argument('--no-save', dest='save_plots', action='store_false', help='プロットをファイルに保存しない')
    parser.add_argument('--no-organize', dest='organize_files', action='store_false', help='実行結果ファイルを整理しない')
    parser.set_defaults(bagging=False, save_plots=True, organize_files=True)
    parser.add_argument('--use-custom-catboost', action='store_true', 
                    help='カスタムCatBoostRegressionModelを使用する')
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir if args.save_plots else None
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # データの読み込み
    data_file_path = os.path.join(args.data_path, args.data_file)
    print(f"データファイルを読み込みます: {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"データファイルの読み込みに失敗しました: {e}")
        print("ファイルパスと形式を確認してください。")
        exit(1)
    
    # モデルの選択
    model_class = None
    if args.use_custom_catboost:
        # カスタムCatBoostRegressionModelを使用する場合はmodel_classをNoneのままにする
        model_class = None
        model_name = "custom_catboost"
    elif not args.bagging:
        if args.model == 'lightgbm':
            import lightgbm as lgb
            model_class = lambda: lgb.LGBMRegressor(objective='regression', random_state=args.random_state)
        elif args.model == 'xgboost':
            import xgboost as xgb
            model_class = lambda: xgb.XGBRegressor(objective='reg:squarederror', random_state=args.random_state)
        elif args.model == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model_class = lambda: RandomForestRegressor(random_state=args.random_state)
        elif args.model == 'catboost':
            from catboost import CatBoostRegressor
            model_class = lambda: CatBoostRegressor(loss_function='RMSE', random_seed=args.random_state, verbose=False)
        
        model_name = args.model
    else:
        model_class = None
        model_name = f"reg_bagging_{args.base_model}"
    
    # 実行設定の表示
    print(f"実行設定:")
    if args.bagging:
        print(f"- モデル: RegressionBagging ({args.base_model})")
        print(f"- バッグ数: {args.n_bags}")
    else:
        print(f"- モデル: {args.model}")
    print(f"- 交差検証: {'あり' if args.cv else 'なし'}")
    if args.cv:
        print(f"- 交差検証分割数: {args.splits}")
    print(f"- 乱数シード: {args.random_state}")
    print(f"- データファイル: {args.data_file}")
    print(f"- 目的変数: {args.target_column}")
    
    try:
        # 分析実行
        if args.cv:
            # 交差検証の実行
            results, cv_metrics = run_regression_cv_analysis(
                df, 
                model_class=model_class if not args.bagging else None, 
                n_splits=args.splits,
                use_bagging=args.bagging,
                output_dir=output_dir,
                random_state=args.random_state,
                target_column=args.target_column,
                base_model=args.base_model,
                n_bags=args.n_bags
            )
            print("\n交差検証による分析が完了しました。")
        else:
            # 単一の学習・評価による分析
            results, new_output_dir = run_regression_analysis(
                df, 
                model_class=model_class if not args.bagging else None,
                use_bagging=args.bagging,
                random_state=args.random_state,
                output_dir=output_dir,
                target_column=args.target_column,
                data_file_name=args.data_file,
                base_model=args.base_model,
                n_bags=args.n_bags
            )
            
            # 結果をファイルに保存
            if args.bagging:
                model_name = f"reg_bagging_{args.base_model}"
            else:
                try:
                    model_name = results['model'].__class__.__name__.lower()
                except:
                    model_name = args.model
            
            # ファイル整理を選択した場合
            if args.organize_files and args.save_plots and new_output_dir != output_dir:
                print("\nファイルを整理します...")
                organize_result_files(args.data_file, output_dir)
            
            print("\n単一モデルによる分析が完了しました。")
            if args.save_plots:
                print(f"結果は {new_output_dir if new_output_dir else output_dir} ディレクトリに保存されました。")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()   
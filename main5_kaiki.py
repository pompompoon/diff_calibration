# mainコードです。回帰コード
import models.RegressionResultManager as RegressionModelEvaluator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import japanize_matplotlib
import sys
import os
import datetime
import matplotlib

import matplotlib.pyplot as plt
# import japanize_matplotlib  # これを追加
# カレントディレクトリをPythonパスに追加

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

# 必要なインポートを追加
import lightgbm
import xgboost
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# 視覚化モジュールをインポート
from visualization.regression_visualizer import RegressionVisualizer, EyeTrackingVisualizer

from models.partial_dependence_plotter_kaiki import PartialDependencePlotter


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

# メイン関数

def prepare_data_for_regression(df, target_column='target'):
    """データの前処理を行う関数（回帰用）"""
    print("\nデータの形状:", df.shape)
    
    # IDカラムを除外し、特徴量として使用するカラムを選択
    features = df.drop(['InspectionDateAndId', target_column], axis=1, errors='ignore')
    
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

def train_regression_model(features, target, model_class=None, use_bagging=False, 
                         random_state=42, **kwargs):
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
    
    # RegressionBaggingModelを使用する場合
    if use_bagging:
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
# ファイルの上部に以下の関数を追加
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
    import numpy as np
    
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
def spearman_correlation(x, y):
    """
    スピアマンの順位相関係数を計算する関数
    
    Parameters:
    -----------
    x : array-like
        第1変数のデータ
    y : array-like
        第2変数のデータ
        
    Returns:
    --------
    float
        スピアマンの順位相関係数
    """
    # NaNを含むデータを除外する
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    # データが空の場合はNaNを返す
    if len(x_clean) == 0 or len(y_clean) == 0:
        return np.nan
    
    # データを順位に変換
    # method='average'は同順位の場合に平均順位を割り当てる
    rank_x = rankdata(x_clean, method='average')
    rank_y = rankdata(y_clean, method='average')
    
    # 順位の差の二乗和を計算
    n = len(x_clean)
    d_squared_sum = np.sum((rank_x - rank_y) ** 2)
    
    # スピアマンの順位相関係数を計算
    # 公式: rho = 1 - (6 * Σd²) / (n * (n² - 1))
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    
    return rho


def prepare_data_for_regression(df, target_column='target'):
    """データの前処理を行う関数（回帰用）"""
    print("\nデータの形状:", df.shape)
    
    # 重要: ID列の取得と保存
    id_column = None
    if 'InspectionDateAndId' in df.columns:
        id_column = df['InspectionDateAndId'].copy()
    elif 'Id' in df.columns:
        id_column = df['Id'].copy()
    
    # IDカラムを除外し、特徴量として使用するカラムを選択
    drop_cols = []
    if 'InspectionDateAndId' in df.columns:
        drop_cols.append('InspectionDateAndId')
    if 'Id' in df.columns:
        drop_cols.append('Id')
    if target_column in df.columns:
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
    
    # ID列も同様にフィルタリング
    if id_column is not None:
        id_column = id_column[features.index]
    
    print(f"\n処理後のデータ数: {len(features)}")
    print("\n目的変数の統計量:")
    print(target.describe())
    
    # featuresとtargetに加えて、id_columnも返す
    return features, target, id_column

def train_regression_model(features, target, id_column=None, model_class=None, use_bagging=False, 
                         random_state=42, **kwargs):
    """モデルの学習を行う関数（回帰用）"""
    # データの分割（ID列も分割する）
    if id_column is not None:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, id_column, test_size=0.2, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=random_state
        )
        id_train, id_test = None, None
    
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
    X_test_orig = X_test.copy()
    
    # RegressionBaggingModelを使用する場合
    if use_bagging:
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
        'id_test': id_test,  # テストデータのID列を追加
        'scaler': scaler
    }

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
     # スピアマン相関係数
    spearman_corr = spearman_correlation(y_true, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Pearson相関係数': pearson_corr,
        'P値': p_value
        
    }
    # スピアマン相関係数を追加
    metrics['Spearman_Corr'] = spearman_corr
    # テストデータの元のインデックスを取得
    test_indices = results['test_indices']
    
    # 予測値と実測値の比較データフレーム
    pred_df = pd.DataFrame({
        'True_Value': results['true_values'],
        'Predicted_Value': results['predictions'],
        'Error': results['true_values'] - results['predictions'],
        'Abs_Error': np.abs(results['true_values'] - results['predictions'])
    }, index=test_indices)
    
    # ID列の取得
    id_test = results.get('id_test')
    
    # IDカラムがある場合は先頭に追加
    if id_test is not None:
        # ID列名を特定（InspectionDateAndIdかIdか）
        if isinstance(id_test, pd.Series):
            id_name = id_test.name if hasattr(id_test, 'name') and id_test.name else 'InspectionDateAndId'
        else:
            id_name = 'InspectionDateAndId'  # デフォルト名
        
        # ID列をデータフレームに追加
        id_df = pd.DataFrame({id_name: id_test}, index=test_indices)
        pred_df = id_df.join(pred_df)
        
        # インデックスをリセット
        pred_df = pred_df.reset_index(drop=True)
    
    return pred_df, metrics
def organize_regression_result_files(data_file_name, output_dir):
    """
    現在の実行で生成された回帰分析の結果ファイルのみを新しいディレクトリに整理する関数
    
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
        # 重要なパターンのファイルをコピー
        important_patterns = [
            'pdp_*.png',                    # 部分依存プロット
            'true_vs_predicted.png',        # 真値と予測値の散布図
            'residuals.png',                # 残差プロット
            'residual_distribution.png',    # 残差分布
            'feature_importance.png',       # 特徴量重要度
            'feature_importance_*.csv',     # 特徴量重要度CSV
            'correlation_heatmap.png',      # 相関ヒートマップ
            'predictions_*.csv',            # 予測結果
            'metrics_*.txt'                 # 評価指標
        ]
        
        # パターンに一致する最近のファイルをコピー
        for pattern in important_patterns:
            import glob
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
    
    print(f"{file_count}個の新しいファイルを {new_output_dir} にコピーしました")
    return new_output_dir
def save_regression_model(model, features, target, scaler, output_path, model_name):
    """
    回帰モデルと関連するデータを保存する関数
    
    Parameters:
    -----------
    model : object
        保存する学習済みモデル
    features : pandas.DataFrame
        学習に使用した特徴量
    target : pandas.Series or numpy.ndarray
        学習に使用した目的変数
    scaler : object
        特徴量のスケーリングに使用したスケーラー
    output_path : str
        モデルを保存するディレクトリのパス
    model_name : str
        保存するモデルの名前
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"モデル保存ディレクトリを作成しました: {output_path}")
    
    # saved_modelサブディレクトリの作成
    saved_model_dir = os.path.join(output_path, 'saved_model')
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    
    import pickle
    
    # モデルを保存
    model_path = os.path.join(saved_model_dir, f"{model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {model_path}")
    
    # スケーラーを保存
    scaler_path = os.path.join(saved_model_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"スケーラーを保存しました: {scaler_path}")
    
    # 特徴量の列名を保存
    feature_names = features.columns.tolist()
    feature_names_path = os.path.join(saved_model_dir, f"{model_name}_feature_names.pkl")
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"特徴量名を保存しました: {feature_names_path}")
    
    # 学習に使用したデータのサンプルを保存
    data_sample = pd.DataFrame(features.head(100))
    data_sample['target'] = target.iloc[:100] if hasattr(target, 'iloc') else target[:100]
    data_sample_path = os.path.join(saved_model_dir, f"{model_name}_data_sample.csv")
    data_sample.to_csv(data_sample_path, index=False)
    print(f"学習データのサンプルを保存しました: {data_sample_path}")
    
    # モデル情報のテキストファイル作成
    info_path = os.path.join(saved_model_dir, f"{model_name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"モデル名: {model_name}\n")
        f.write(f"モデルタイプ: {type(model).__name__}\n")
        f.write(f"特徴量数: {features.shape[1]}\n")
        f.write(f"学習データ数: {len(features)}\n")
        f.write(f"保存日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # モデルのパラメータがある場合は追加
        if hasattr(model, 'get_params'):
            f.write("\nモデルパラメータ:\n")
            for param, value in model.get_params().items():
                f.write(f"  {param}: {value}\n")
    
    print(f"モデル情報を保存しました: {info_path}")
    
    return model_path

def run_regression_analysis(df, model_class=None, use_bagging=False, 
                         random_state=42, output_dir=None, target_column='target', 
                         data_file_name=None, organize_files=True, **kwargs):
    """回帰分析を実行する関数"""
    import datetime
    import shutil
    
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
            
        # saved_modelサブディレクトリも事前に作成
        saved_model_dir = os.path.join(new_output_dir, "saved_model")
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
            
        # 処理結果の出力先を新しいディレクトリに設定
        output_dir = new_output_dir
    
    if use_bagging:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"\nRegressionBaggingModel with {base_model} base model and {n_bags} bags を使用します...")
    elif model_class is not None:
        try:
            model_name = model_class().__class__.__name__
        except:
            model_name = "回帰モデル"
        print(f"\n{model_name} を使用します...")
    else:
        raise ValueError("model_classまたはuse_baggingのいずれかを指定してください。")
    
    print("データの前処理を開始...")
    try:
        # 新しい関数シグネチャでの呼び出し
        features, target, id_column = prepare_data_for_regression(df, target_column=target_column)
    except ValueError as e:
        if "too many values to unpack" in str(e):
            # 古い関数と互換性を保つ
            features, target = prepare_data_for_regression(df, target_column=target_column)
            id_column = None
            
            # IDカラムを別途取得
            if 'InspectionDateAndId' in df.columns:
                id_column = df['InspectionDateAndId'].copy()
                # 特徴量のインデックスに合わせてフィルタリング
                id_column = id_column[features.index]
            elif 'Id' in df.columns:
                id_column = df['Id'].copy()
                # 特徴量のインデックスに合わせてフィルタリング
                id_column = id_column[features.index]
                
            print("警告: IDカラムが関数から直接取得できませんでした。別途取得しました。")
        else:
            raise
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # 可視化のためのインスタンス作成 - 新しい出力ディレクトリを使用
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)
    
    # 特徴量と目的変数の関係の可視化
    print("\n特徴量と目的変数の関係を分析します...")
    visualizer.plot_feature_correlations(features, target)
    
    print("\nモデルの学習を開始...")
    try:
        # ID列も含めて学習
        results = train_regression_model(
            features, target, id_column,
            model_class=model_class, 
            use_bagging=use_bagging,
            random_state=random_state,
            **kwargs
        )
    except TypeError as e:
        # 古い関数シグネチャとの互換性を保つ
        if "got an unexpected keyword argument" in str(e) or "takes" in str(e) and "positional argument" in str(e):
            print("警告: 古い関数シグネチャを使用します（id_columnなし）")
            results = train_regression_model(
                features, target,
                model_class=model_class, 
                use_bagging=use_bagging,
                random_state=random_state,
                **kwargs
            )
            
            # ID列を取得して追加
            if id_column is not None:
                # テストインデックスを使用してIDをフィルタリング
                test_indices = results['test_indices']
                results['id_test'] = id_column.loc[test_indices]
        else:
            raise
    
    # モデル名の決定
    if use_bagging:
        model_name = f"reg_bagging_{kwargs.get('base_model', 'lightgbm')}"
    else:
        try:
            model_name = results['model'].__class__.__name__.lower()
        except:
            model_name = "regression_model"
    
    print("\n結果の評価を開始...")
    predictions_df, metrics = evaluate_regression_results(results)
    
    # 結果の表示
    print("\n最適なパラメータ:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    
    print("\n性能指標:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\n特徴量の重要度（上位10件）:")
    print(results['feature_importance'].head(10))
    
    # 結果の可視化
    print("\nモデル評価の可視化...")
    visualizer.plot_true_vs_predicted(results['true_values'], results['predictions'])
    visualizer.plot_residuals(results['true_values'], results['predictions'])
    visualizer.plot_feature_importance(results['feature_importance'])
    visualizer.plot_residual_distribution(results['true_values'], results['predictions'])
    
    # 部分依存プロット - 直接生成
    print("\n部分依存プロットを生成します...")
    try:
        # scikit-learnから直接インポート
        from sklearn.inspection import PartialDependenceDisplay
        
        # 特徴量重要度でソートした上位の特徴量を選択
        top_features = results['feature_importance'].head(6)['feature'].tolist()
        print(f"部分依存プロット用の特徴量: {top_features}")
        
        # 特徴量のインデックスを取得
        feature_indices = [list(results['X_train'].columns).index(feature) for feature in top_features]
        
        # pdp_top6.pngファイルのみを作成
        fig, ax = plt.subplots(figsize=(15, 10))
        display = PartialDependenceDisplay.from_estimator(
            results['model'], 
            results['X_train'], 
            feature_indices, 
            kind='average',
            n_cols=3,
            grid_resolution=50,
            ax=ax
        )
        
        plt.suptitle('部分依存プロット (PDP)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存
        if output_dir:
            # タイムスタンプを含めたファイル名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pdp_path = os.path.join(output_dir, f'pdp_top6_{timestamp}.png')
            plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"部分依存プロットを保存しました: {pdp_path}")
    
    except Exception as e:
        print(f"部分依存プロットの生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # バックアップメソッド
        try:
            # 直接PartialDependencePlotterを使用
            print("代替方法で部分依存プロットを生成します...")
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
                # タイムスタンプを含めたファイル名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(output_dir, f'pdp_top6_{timestamp}.png')
                plt.savefig(backup_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"代替方法で部分依存プロットを保存しました: {backup_path}")
        except Exception as e2:
            print(f"代替方法でも部分依存プロットの作成に失敗しました: {e2}")
            traceback.print_exc()
    
    # 結果をまとめる
    results['predictions_df'] = predictions_df
    
    # 出力ディレクトリがある場合は結果を保存 - 新しい出力ディレクトリに直接保存
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
    
    # モデルの保存 - 新しい出力ディレクトリのsaved_modelサブディレクトリに保存
    if output_dir and data_file_name:
        model_base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        
        try:
            save_regression_model(
                model=results['model'],
                features=results['features'],
                target=results['target'],
                scaler=results['scaler'],
                output_path=output_dir,
                model_name=model_base_name
            )
            print(f"モデルを保存しました: {saved_model_dir}/{model_base_name}")
        except Exception as e:
            print(f"モデルの保存中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # 結果ファイルの整理（オプション）
    if organize_files and original_output_dir:
        try:
            new_dir = organize_regression_result_files(data_file_name, original_output_dir)
            print(f"\n結果ファイルを整理しました。整理された結果は {new_dir} にあります。")
        except Exception as e:
            print(f"結果ファイルの整理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n分析が完了しました。すべての結果は {output_dir} に保存されました。")
    
    # 必ず2つの値を返す
    return results, new_output_dir
def run_regression_cv_analysis(df, model_class=None, n_splits=5, use_bagging=False,
                            output_dir='result', random_state=42, target_column='target', **kwargs):
    """
    回帰問題用の交差検証を実行する関数
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bagging_str = "（バギングあり）" if use_bagging else ""
    
    print(f"\n回帰モデルでの {n_splits}分割交差検証を開始...{bagging_str}")
    
    # データの前処理
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
    
    # RegressionBaggingModelを使用する場合
    if use_bagging:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"RegressionBaggingModelを使用した交差検証を実行します（ベースモデル: {base_model}、バッグ数: {n_bags}）")
        
        # RegressionBaggingModelのクラスメソッドを使用
        oof_preds, scores = RegressionBaggingModel.run_cv(
            X=features, 
            y=target, 
            base_model=base_model,
            n_bags=n_bags,
            n_splits=n_splits,
            random_state=random_state
        )
        
        # 結果の表示
        print("\n交差検証の結果:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
            
        # 結果を保存
        model_name = f"reg_bagging_{base_model}"
        cv_results_df = pd.DataFrame({
            'true': target,
            'pred': oof_preds
        })
        
        if output_dir:
            # 予測結果の保存
            cv_results_df.to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_predictions.csv')
            
            # 評価指標の保存
            with open(f'{output_dir}/cv_{model_name}_{n_splits}fold_metrics.txt', 'w') as f:
                for metric, score in scores.items():
                    f.write(f"{metric}: {score:.4f}\n")
        
        # 予測値と実測値の散布図
        visualizer.plot_true_vs_predicted(target, oof_preds)
        
        return None, scores
    
    # 通常のモデルを使用する場合（交差検証）
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print(f"通常の回帰モデル {model_class.__name__} を使用した交差検証を実行します")
    
    # 特徴量とターゲットの準備
    X = features.values
    y = target.values
    
    # 交差検証の設定
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 結果の格納用
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    feature_importances = []
    
    # 各分割でモデルを学習・評価
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # データの分割
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデルの作成と学習
        model = model_class()
        model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        oof_predictions[test_idx] = y_pred
        
        # 評価指標の計算
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 結果の保存
        fold_metrics.append({
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        # 特徴量重要度の保存（あれば）
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_,
                'fold': fold + 1
            })
            feature_importances.append(importances)
        
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # 結果のまとめ
    metrics_df = pd.DataFrame(fold_metrics)
    
    # 平均指標の計算
    avg_metrics = {
        'mse': metrics_df['mse'].mean(),
        'rmse': metrics_df['rmse'].mean(),
        'mae': metrics_df['mae'].mean(),
        'r2': metrics_df['r2'].mean()
    }
    
    # 標準偏差の計算
    std_metrics = {
        'mse_std': metrics_df['mse'].std(),
        'rmse_std': metrics_df['rmse'].std(),
        'mae_std': metrics_df['mae'].std(),
        'r2_std': metrics_df['r2'].std()
    }
    
    # 平均特徴量重要度の計算（特徴量重要度がある場合）
    avg_importance = None
    if feature_importances:
        importances_df = pd.concat(feature_importances)
        avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # 結果の表示
    print("\n=== 交差検証の結果 ===")
    print("\n平均性能指標:")
    for metric, value in avg_metrics.items():
        std = std_metrics[f'{metric}_std']
        print(f"{metric}: {value:.4f} ± {std:.4f}")
    
    # 特徴量重要度の表示（あれば）
    if avg_importance is not None:
        print("\n特徴量の平均重要度（上位10件）:")
        print(avg_importance.head(10))
    
    # 予測値と実測値の可視化
    visualizer.plot_true_vs_predicted(y, oof_predictions)
    visualizer.plot_residuals(y, oof_predictions)
    
    # 結果の保存
    model_name = model_class.__name__.lower()
    if output_dir:
        # 予測結果の保存
        pd.DataFrame({
            'true': y,
            'pred': oof_predictions
        }).to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_predictions.csv')
        
        # 評価指標の保存
        with open(f'{output_dir}/cv_{model_name}_{n_splits}fold_metrics.txt', 'w') as f:
            for metric, value in avg_metrics.items():
                std = std_metrics[f'{metric}_std']
                f.write(f"{metric}: {value:.4f} ± {std:.4f}\n")
        
        # 特徴量重要度の保存（あれば）
        if avg_importance is not None:
            avg_importance.to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_importance.csv', index=False)
    
    # 結果のまとめ
    results = {
        'oof_predictions': oof_predictions,
        'metrics': avg_metrics,
        'metrics_std': std_metrics,
        'fold_metrics': metrics_df,
        'feature_importance': avg_importance
    }
    
    return results, avg_metrics

def create_regression_model(model_type='lightgbm', random_state=42):
    """回帰モデルを作成するヘルパー関数"""
    if model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective='regression', random_state=random_state)
    elif model_type == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_state)
    elif model_type == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function='RMSE', random_seed=random_state, verbose=False)
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")


def create_regression_bagging_model(base_model='lightgbm', n_bags=10, random_state=42):
    """RegressionBaggingModelを作成するヘルパー関数"""
    return {
        'model': RegressionBaggingModel(
            base_model=base_model,
            n_bags=n_bags,
            random_state=random_state
        ),
        'name': f'reg_bagging_{base_model}',
        'params': {
            'base_model': base_model,
            'n_bags': n_bags
        }
    }

if __name__ == "__main__":
    setup_matplotlib_japanese_font()
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
                       default="features_0428_ton56.csv",
                       help='データファイル名')
    parser.add_argument('--target-column', type=str,
                       default="target",
                       help='目的変数のカラム名')
    parser.add_argument('--output-dir', type=str, 
                       default="result",
                       help='結果出力ディレクトリ')
    parser.add_argument('--no-save', dest='save_plots', action='store_false', help='プロットをファイルに保存しない')
    parser.add_argument('--no-organize', dest='organize_files', action='store_false', 
                        help='結果ファイルを整理しない')
    parser.set_defaults(bagging=False, save_plots=True, organize_files=True)
    
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
    if not args.bagging:
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
    print(f"- 結果ファイル整理: {'あり' if args.organize_files else 'なし'}")
    
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
            
            # 結果ファイルの整理
            if args.organize_files and output_dir:
                new_dir = organize_regression_result_files(args.data_file, output_dir)
                print(f"\n交差検証による分析が完了しました。結果は {new_dir} に保存されています。")
            else:
                print("\n交差検証による分析が完了しました。")
                if output_dir:
                    print(f"結果は {output_dir} ディレクトリに保存されました。")
        else:
            # 単一の学習・評価による分析
            results, new_dir = run_regression_analysis(
                df, 
                model_class=model_class if not args.bagging else None,
                use_bagging=args.bagging,
                random_state=args.random_state,
                output_dir=output_dir,
                target_column=args.target_column,
                data_file_name=args.data_file,
                organize_files=args.organize_files,
                base_model=args.base_model,
                n_bags=args.n_bags
            )
            
            if new_dir:
                print(f"\n単一モデルによる分析が完了しました。結果は {new_dir} に保存されています。")
            else:
                print("\n単一モデルによる分析が完了しました。")
                if output_dir:
                    print(f"結果は {output_dir} ディレクトリに保存されました。")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
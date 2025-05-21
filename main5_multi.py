import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy.stats import chi2_contingency
import pickle
import datetime
import shutil

# モデルのインポート
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report

# SMOTEとundersampling用
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def safe_convert_array(value):
    """
    配列形式の文字列をNumPy配列に安全に変換する
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    if isinstance(value, str):
        try:
            # 文字列からNumPy配列に変換を試みる
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                # リスト形式の文字列を処理
                inner_val = value[1:-1].strip()
                if not inner_val:  # 空の配列の場合
                    return np.array([])
                # カンマで分割
                elements = inner_val.split(',')
                # 各要素を数値に変換
                numeric_elements = [float(e.strip()) for e in elements]
                return np.array(numeric_elements)
            else:
                # その他の形式は元の値を返す
                return value
        except:
            # 変換に失敗した場合は元の値を返す
            return value
    return value

class UndersamplingBaggingModel:
    """アンダーサンプリング+バギングによるクラス不均衡対応モデル"""
    
    def __init__(self, base_model='lightgbm', n_bags=10, random_state=42):
        self.n_bags = n_bags
        self.random_state = random_state
        self.base_model_type = base_model
        self.models = []
        self.feature_importances_ = None
        
    def _create_base_model(self):
        """ベースモデルを作成"""
        if self.base_model_type == 'lightgbm':
            return LGBMClassifier(random_state=self.random_state)
        elif self.base_model_type == 'xgboost':
            return XGBClassifier(random_state=self.random_state)
        elif self.base_model_type == 'randomforest':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.base_model_type == 'catboost':
            return CatBoostClassifier(random_state=self.random_state, verbose=0)
        else:
            raise ValueError(f"未対応のモデルタイプ: {self.base_model_type}")
    
    def fit(self, X, y):
        """アンダーサンプリング+バギングでモデルを学習"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes > 2:
            # 多値分類の場合は各クラスに対して二項分類問題として扱う
            print(f"多値分類問題 (クラス数: {n_classes}) に対応したアンダーサンプリング+バギングを適用します")
        
        # クラスごとのサンプル数をカウント
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        print(f"各クラスのサンプル数: {class_counts}")
        print(f"アンダーサンプリング後の各クラスのサンプル数: {min_class_count}")
        
        # 特徴量重要度を集計するための配列
        all_importances = []
        
        # 各バッグでモデルを学習
        self.models = []
        for i in range(self.n_bags):
            # アンダーサンプリング (各クラスを最小クラスのサイズに合わせる)
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=self.random_state+i)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            # モデルを学習
            model = self._create_base_model()
            model.fit(X_resampled, y_resampled)
            self.models.append(model)
            
            # 特徴量重要度を取得（あれば）
            if hasattr(model, 'feature_importances_'):
                all_importances.append(model.feature_importances_)
        
        # 特徴量重要度の平均を計算（あれば）
        if all_importances:
            self.feature_importances_ = np.mean(all_importances, axis=0)
        
        return self
    
    def predict(self, X):
        """全モデルの多数決で予測"""
        # 各モデルの予測を取得
        predictions = np.array([model.predict(X) for model in self.models])
        
        # 行ごとに最頻値を取得（多数決）
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
    
    def predict_proba(self, X):
        """全モデルの確率平均で予測確率を計算"""
        # 各モデルの予測確率を取得
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        # 平均を取る
        return np.mean(probas, axis=0)

def prepare_data_multi(df):
    """多値分類用のデータの前処理を行う関数"""
    print("\nデータの形状:", df.shape)
    
    # デバッグ情報: 各カラムのNaN数を表示
    print("\n各カラムのNaN数:")
    nan_columns = df.columns[df.isna().any()].tolist()
    for col in nan_columns:
        nan_count = df[col].isna().sum()
        print(f"  {col}: {nan_count}")
    
    # 目的変数のカラム名を確認
    if 'target' in df.columns:
        target_column = 'target'
    elif 'Target' in df.columns:
        target_column = 'Target'
    else:
        raise ValueError("目的変数の列 'target' または 'Target' が見つかりません")
        
    # 元のデータの値を確認
    original_values = df[target_column].unique()
    print(f"元の{target_column}の一意な値:", original_values)
        
    # InspectionDateAndId、Id、および目的変数を除外した特徴量を抽出
    drop_cols = []
    if 'InspectionDateAndId' in df.columns:
        drop_cols.append('InspectionDateAndId')
    if 'Id' in df.columns:
        drop_cols.append('Id')
    drop_cols.append(target_column)

    features = df.drop(drop_cols, axis=1)
    
    # 特徴量名のクリーニング - LightGBM用の対応
    # JSON特殊文字や日本語を含む列名を修正
    new_columns = {}
    for col in features.columns:
        # 特殊文字や日本語を含む列名をアルファベット、数字、アンダースコアのみに置換
        new_col = ''.join(c if c.isascii() and (c.isalnum() or c == '_') else '_' for c in col)
        # 空の名前や数字から始まる名前の場合はプレフィックスを追加
        if not new_col or new_col[0].isdigit():
            new_col = 'f_' + new_col
        # 同じ名前になってしまう場合は連番を付与
        if new_col in new_columns.values():
            i = 1
            while f"{new_col}_{i}" in new_columns.values():
                i += 1
            new_col = f"{new_col}_{i}"
        
        if new_col != col:
            new_columns[col] = new_col
    
    # 列名の置換が必要な場合は置換
    if new_columns:
        print("\n特殊文字を含む列名を修正します:")
        for old_col, new_col in new_columns.items():
            print(f"  {old_col} -> {new_col}")
        
        features = features.rename(columns=new_columns)
    
    # 多値分類問題用に修正: targetはそのまま使用
    target = df[target_column].copy()
    print(f"\nターゲット変数の値をそのまま使用します (多値分類)")
    print(f"ターゲット値の分布:\n{target.value_counts()}")
    
    # ターゲット変数の確認
    print("ターゲット変数のNaN数:", target.isna().sum())
    print("ターゲット変数の一意な値:", target.unique())
    
    # 特殊列の処理
    for col in ['freq', 'power_spectrum']:
        if col in features.columns:
            features[col] = features[col].apply(safe_convert_array)
    
    # 欠損値の確認
    missing_values = features.isna().sum().sum()
    if missing_values > 0:
        print(f"\n欠損値を含む行: {missing_values}個")
        
        # 欠損値の補完（削除ではなく）
        print("欠損値を中央値で補完します")
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col].fillna(median_val, inplace=True)
    
    print(f"\n処理後のデータ数: {len(features)}")
    print(f"\nクラス分布:")
    print(target.value_counts())
    
    # NaNがある場合は警告
    if target.isna().any():
        print("警告: ターゲット変数にNaNが含まれています。これらの行を削除します。")
        valid_idx = ~target.isna()
        features = features[valid_idx]
        target = target[valid_idx]
        print(f"NaN削除後のデータ数: {len(features)}")
        print("修正後のクラス分布:")
        print(target.value_counts())
    
    return features, target

def train_multi_model(features, target, model_type='lightgbm', use_smote=False, 
                     use_undersampling_bagging=False, n_bags=10, random_state=42, test_size=0.2):
    """多値分類モデルの学習を行う関数"""
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
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
    
    # 元のデータをスケーリング前の状態で保存（可視化用）
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()
    
    # クラスの分布を確認
    print("\nトレーニングデータのクラス分布:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # SMOTEを適用する場合
    if use_smote:
        print("\nSMOTEを適用してデータをバランシングします...")
        
        # クラスの数を確認
        n_classes = len(np.unique(y_train))
        if n_classes > 2:
            # 多値分類の場合はk_neighbors引数を調整する必要がある
            print(f"多値分類問題 (クラス数: {n_classes}) に対して適切なk_neighborsを設定します")
            # 各クラスのサンプル数をカウント
            class_counts = pd.Series(y_train).value_counts()
            # 最小クラスのサンプル数を確認
            min_class_count = class_counts.min()
            
            # k_neighbors は min_class_count - 1 を超えてはならない
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors < 1:
                print("警告: 最小クラスのサンプル数が少なすぎるため、SMOTEを適用できません")
                print("SMOTE適用をスキップします")
            else:
                print(f"k_neighbors = {k_neighbors} でSMOTEを適用します")
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train_scaled_np = X_train_scaled.values
                y_train_np = y_train.values
                
                try:
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_np, y_train_np)
                    X_train_scaled = pd.DataFrame(X_train_resampled, columns=features.columns)
                    y_train = pd.Series(y_train_resampled)
                    print("SMOTE適用後のクラス分布:")
                    print(pd.Series(y_train).value_counts().sort_index())
                except Exception as e:
                    print(f"SMOTE適用中にエラーが発生しました: {e}")
                    print("SMOTE適用をスキップします")
        else:
            # 2クラス分類の場合は標準のSMOTEを適用
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            X_train_scaled = pd.DataFrame(X_train_resampled, columns=features.columns)
            y_train = pd.Series(y_train_resampled)
            print("SMOTE適用後のクラス分布:")
            print(pd.Series(y_train).value_counts().sort_index())
    
    # Undersampling+Baggingモデルを使用する場合
    if use_undersampling_bagging:
        print(f"\nUndersampling+Baggingモデル（{model_type}ベース、バッグ数: {n_bags}）を使用します")
        model = UndersamplingBaggingModel(base_model=model_type, n_bags=n_bags, random_state=random_state)
    else:
        # モデルの選択と初期化
        if model_type.lower() == 'lightgbm':
            print("\nLightGBMモデルを初期化...")
            model = LGBMClassifier(objective='multiclass', random_state=random_state)
        elif model_type.lower() == 'xgboost':
            print("\nXGBoostモデルを初期化...")
            model = XGBClassifier(objective='multi:softprob', random_state=random_state)
        elif model_type.lower() == 'randomforest':
            print("\nRandomForestモデルを初期化...")
            model = RandomForestClassifier(random_state=random_state)
        elif model_type.lower() == 'catboost':
            print("\nCatBoostモデルを初期化...")
            model = CatBoostClassifier(random_state=random_state, verbose=0)
        else:
            raise ValueError(f"未対応のモデルタイプです: {model_type}")
    
    # モデルの学習
    print(f"\nモデルの学習を開始...")
    model.fit(X_train_scaled, y_train)
    
    # ここから追加: 訓練データに対する予測と評価
    y_train_pred = model.predict(X_train_scaled)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n訓練データでの精度: {training_accuracy:.4f}")
    
    # 詳細な分類レポート
    print("\n訓練データでの分類レポート:")
    print(classification_report(y_train, y_train_pred))
    
    # 訓練データでの混同行列
    print("訓練データでの混同行列:")
    train_cm = confusion_matrix(y_train, y_train_pred)
    print(train_cm)
    # ここまで追加
    
    # テストデータでの予測
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # 精度評価
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nテストデータでの精度: {accuracy:.4f}")
    
    # 分類レポート
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred))
    
    # 特徴量の重要度
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n特徴量の重要度 (上位10件):")
        print(feature_importance.head(10))
    else:
        feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
    
    # 混同行列
    print("\n混同行列:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 結果を辞書形式で返す（訓練精度情報を追加）
    return {
        'model': model,
        'predictions': y_pred,
        'prediction_proba': y_pred_proba,
        'true_values': y_test,
        'feature_importance': feature_importance,
        'test_indices': X_test.index,
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'X_test_orig': X_test_orig,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'training_accuracy': training_accuracy,  # 追加
        'training_confusion_matrix': train_cm,   # 追加
        'scaler': scaler
    }
def visualize_results(results, output_dir=None):
    """結果の可視化"""
    # 特徴量重要度の可視化
    if 'feature_importance' in results and not results['feature_importance'].empty:
        feature_importance = results['feature_importance']
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('特徴量の重要度 (上位20件)')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 混同行列の可視化
    if 'confusion_matrix' in results and 'true_values' in results:
        cm = results['confusion_matrix']
        class_names = sorted(np.unique(results['true_values']))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=class_names,
                  yticklabels=class_names)
        plt.xlabel('予測ラベル')
        plt.ylabel('真のラベル')
        plt.title('混同行列')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 予測確率の分布可視化
    if 'prediction_proba' in results and 'true_values' in results:
        y_proba = results['prediction_proba']
        y_true = results['true_values']
        
        # 各クラスの予測確率分布を可視化
        n_classes = y_proba.shape[1]
        class_names = sorted(np.unique(y_true))
        
        plt.figure(figsize=(15, 10))
        for i, class_name in enumerate(class_names):
            plt.subplot(2, (n_classes + 1) // 2, i + 1)
            
            # 現在のクラスに属するかどうかでサンプルを分割
            mask_pos = (y_true == class_name)
            mask_neg = ~mask_pos
            
            # 各グループの予測確率を抽出
            probs_pos = y_proba[mask_pos, i]
            probs_neg = y_proba[mask_neg, i]
            
            # 確率分布をプロット
            plt.hist(probs_pos, bins=20, alpha=0.5, label=f'真のクラス: {class_name}')
            plt.hist(probs_neg, bins=20, alpha=0.5, label=f'他のクラス')
            
            plt.title(f'クラス {class_name} の予測確率分布')
            plt.xlabel('予測確率')
            plt.ylabel('サンプル数')
            plt.legend()
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'prediction_probability_dist.png'), dpi=300, bbox_inches='tight')
        plt.show()

def save_model(model, features, target, scaler, output_path, model_name):
    """モデルの保存"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # モデル情報をまとめる
    model_info = {
        'model': model,
        'feature_names': features.columns.tolist(),
        'target_classes': sorted(target.unique()),
        'scaler': scaler
    }
    
    # ファイル名を生成
    model_file = os.path.join(output_path, f"{model_name}_multiclass_model.pkl")
    
    # モデルを保存
    with open(model_file, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"モデルを保存しました: {model_file}")
    
    # モデル情報のテキスト保存
    info_file = os.path.join(output_path, f"{model_name}_model_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"モデルタイプ: {model.__class__.__name__}\n")
        f.write(f"特徴量数: {len(features.columns)}\n")
        f.write(f"クラス数: {len(target.unique())}\n")
        f.write(f"クラス: {sorted(target.unique())}\n")
        f.write(f"サンプル数: {len(features)}\n")
    
    print(f"モデル情報を保存しました: {info_file}")

def main(data_file, model_type='lightgbm', use_smote=False, use_undersampling_bagging=False, 
         n_bags=10, random_state=42, output_dir=None):
    """メイン関数"""
    # 出力ディレクトリの作成
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # データの読み込み
    print(f"データファイルを読み込みます: {data_file}")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"データファイルの読み込みに失敗しました: {e}")
        print("ファイルパスと形式を確認してください。")
        return
    
    # データの前処理
    print("\nデータの前処理を開始...")
    features, target = prepare_data_multi(df)
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # ターゲットに一意な値が1つしかない場合はエラー
    if len(target.unique()) < 2:
        raise ValueError(f"ターゲット変数に一意な値が{len(target.unique())}種類しかありません。少なくとも2クラス必要です。")
    
    # モデルの学習
    results = train_multi_model(
        features, target, 
        model_type=model_type, 
        use_smote=use_smote,
        use_undersampling_bagging=use_undersampling_bagging,
        n_bags=n_bags,
        random_state=random_state
    )
    
    # 結果の可視化
    visualize_results(results, output_dir)
    
    # モデルの保存
    if output_dir:
        save_model(
            model=results['model'],
            features=features,
            target=target,
            scaler=results['scaler'],
            output_path=output_dir,
            model_name=model_type
        )
    
    print("\n処理完了")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多値分類モデルの学習と評価')
    parser.add_argument('--data-path', type=str, 
                       default="data",
                       help='データファイルのパス')
    parser.add_argument('--data-file', type=str, 
                       default="features_0428_to.csv",
                       help='データファイル名')
    parser.add_argument('--model', type=str, default='lightgbm', 
                        choices=['lightgbm', 'xgboost', 'randomforest', 'catboost'], 
                        help='使用するモデル')
    parser.add_argument('--smote', action='store_true', help='SMOTEを使用する')
    parser.add_argument('--undersampling-bagging', action='store_true', help='Undersampling+Baggingを使用する')
    parser.add_argument('--n-bags', type=int, default=10, help='Undersampling+Baggingのバッグ数')
    parser.add_argument('--random-state', type=int, default=42, help='乱数シード')
    parser.add_argument('--output-dir', type=str, default='result_multi', help='結果出力ディレクトリ')
    
    args = parser.parse_args()
    
    # データファイルパスの結合
    data_file_path = os.path.join(args.data_path, args.data_file)
    
    main(
        data_file_path, 
        model_type=args.model, 
        use_smote=args.smote, 
        use_undersampling_bagging=args.undersampling_bagging,
        n_bags=args.n_bags,
        random_state=args.random_state, 
        output_dir=args.output_dir
    )
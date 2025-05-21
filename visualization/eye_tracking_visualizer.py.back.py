"""
分類問題用の部分依存プロット視覚化モジュール（完全版）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

class PartialDependencePlotter:
    """
    部分依存グラフを生成・分析するためのクラス
    
    分類問題用に特化した部分依存プロットを生成します。
    """
    
    def __init__(self, model=None, features=None, target_names=None):
        """
        初期化関数
        
        Parameters:
        -----------
        model : 学習済みモデル, default=None
            部分依存プロットを生成するための学習済みモデル
        features : pandas.DataFrame, default=None
            特徴量データ
        target_names : list or dict, default=None
            目標変数の名前（クラス名）のリストまたは辞書
            例: ['intact', 'mci'] または {0: 'intact', 1: 'mci'}
        """
        self.model = model
        self.features = features
        self.target_names = target_names
        self.scaler = None
        
    def set_model(self, model):
        """モデルを設定"""
        self.model = model
        return self
    
    def set_features(self, features):
        """特徴量を設定"""
        self.features = features
        return self
    
    def set_target_names(self, target_names):
        """目標変数の名前を設定"""
        self.target_names = target_names
        return self
    
    def scale_features(self, features=None):
        """特徴量をスケーリング"""
        if features is None:
            features = self.features
            
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=features.columns
            )
        else:
            scaled_features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns
            )
            
        return scaled_features
    
    def plot_single_feature(self, feature_idx, target_idx=1, ax=None, centered=False, grid_resolution=100):
        """
        単一の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_idx : int or str
            プロットする特徴量のインデックスまたは名前
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
        ax : matplotlib.axes.Axes, default=None
            プロット先の軸。Noneの場合は新しい図を作成
        centered : bool, default=False
            プロットを中心化するかどうか
        grid_resolution : int, default=100
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            作成された図と軸
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
            
        # 特徴量名の取得
        if isinstance(feature_idx, str):
            feature_name = feature_idx
            if feature_name in self.features.columns:
                feature_idx = list(self.features.columns).index(feature_name)
            else:
                raise ValueError(f"特徴量 '{feature_name}' がデータフレームに存在しません")
        else:
            feature_name = self.features.columns[feature_idx]
        
        # ターゲット名の取得
        target_name = f"Class {target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 最新のscikit-learnのAPIを使用
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_idx], 
                    target=target_idx, ax=ax,
                    kind='average', centered=centered,
                    grid_resolution=grid_resolution
                )
            except TypeError:
                # scikit-learn 0.22以前のAPIの場合
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_idx], 
                    target=target_idx, ax=ax,
                    kind='average',  # centeredパラメータなし
                    grid_resolution=grid_resolution
                )
            except Exception as e:
                # 直接部分依存関数を使用して手動でプロット
                print(f"Warning: Using fallback method for PDP: {e}")
                pd_result = partial_dependence(
                    self.model, scaled_features, [feature_idx],
                    grid_resolution=grid_resolution
                )
                
                # APIバージョンによって戻り値の構造が異なる
                if isinstance(pd_result, tuple):
                    if len(pd_result) == 2:
                        pd_values, pd_grid = pd_result
                    else:
                        pd_values, pd_grid, _ = pd_result
                else:
                    pd_values = pd_result.average
                    pd_grid = pd_result.grid_values
                
                ax.plot(pd_grid[0], pd_values[0].T)
                
                # グラフスタイリング
                ax.set_xlabel(feature_name)
                ax.set_ylabel(f'Partial dependence')
        
        # グラフの装飾
        ax.set_title(f"部分依存プロット: {feature_name} → {target_name}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel(f"部分依存 ({target_name})", fontsize=10)
        
        return fig, ax
        
    def plot_multiple_features(self, feature_indices, target_idx=1, n_cols=3, figsize=None,
                              centered=False, grid_resolution=100):
        """
        複数の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_indices : list
            プロットする特徴量のインデックスまたは名前のリスト
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
        n_cols : int, default=3
            列の数
        figsize : tuple, default=None
            図のサイズ。Noneの場合は自動計算
        centered : bool, default=False
            プロットを中心化するかどうか
        grid_resolution : int, default=100
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
        
        # 特徴量インデックスの変換
        feature_idxs = []
        feature_names = []
        for idx in feature_indices:
            if isinstance(idx, str):
                if idx in self.features.columns:
                    feature_idxs.append(list(self.features.columns).index(idx))
                    feature_names.append(idx)
                else:
                    raise ValueError(f"特徴量 '{idx}' がデータフレームに存在しません")
            else:
                feature_idxs.append(idx)
                feature_names.append(self.features.columns[idx])
        
        # 図のサイズを計算
        n_features = len(feature_idxs)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        # ターゲット名の取得
        target_name = f"Class {target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # 図の作成
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # 個別の特徴量に対してプロット
        for i, (feature_idx, feature_name) in enumerate(zip(feature_idxs, feature_names)):
            # 平坦化された軸の取得
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1:
                ax = axes[i % n_cols]
            elif n_cols == 1:
                ax = axes[i // n_cols]
            else:
                ax = axes[i // n_cols, i % n_cols]
                
            try:
                # 個別の特徴量に対して部分依存プロットを作成
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # 最新のscikit-learnのAPIを使用
                        display = PartialDependenceDisplay.from_estimator(
                            self.model, scaled_features, [feature_idx], 
                            target=target_idx, ax=ax,
                            kind='average', centered=centered,
                            grid_resolution=grid_resolution
                        )
                    except TypeError:
                        # centeredパラメータがないバージョン
                        display = PartialDependenceDisplay.from_estimator(
                            self.model, scaled_features, [feature_idx], 
                            target=target_idx, ax=ax,
                            kind='average',
                            grid_resolution=grid_resolution
                        )
                    except Exception as e:
                        # 直接部分依存関数を使用して手動でプロット
                        print(f"Warning: Using fallback method for feature {feature_name}: {e}")
                        pd_result = partial_dependence(
                            self.model, scaled_features, [feature_idx],
                            grid_resolution=grid_resolution
                        )
                        
                        # APIバージョンによって戻り値の構造が異なる
                        if isinstance(pd_result, tuple):
                            if len(pd_result) == 2:
                                pd_values, pd_grid = pd_result
                            else:
                                pd_values, pd_grid, _ = pd_result
                        else:
                            pd_values = pd_result.average
                            pd_grid = pd_result.grid_values
                        
                        ax.plot(pd_grid[0], pd_values[0].T)
                
                # グラフの装飾
                ax.set_title(feature_name, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            except Exception as e:
                print(f"Error plotting feature {feature_name}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                        ha='center', va='center', transform=ax.transAxes)
        
        # 使用しない軸を非表示に
        for i in range(n_features, n_rows * n_cols):
            if n_rows == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            elif n_cols == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            else:
                row, col = i // n_cols, i % n_cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].set_visible(False)
        
        fig.suptitle(f"部分依存プロット (ターゲット: {target_name})", fontsize=14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_feature_interaction(self, feature_pair, target_idx=1, figsize=(8, 6),
                                contour_kw=None, ax=None, grid_resolution=50):
        """
        2つの特徴量間の相互作用を2Dプロットで可視化
        
        Parameters:
        -----------
        feature_pair : tuple or list
            プロットする2つの特徴量のインデックスまたは名前のペア
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
        figsize : tuple, default=(8, 6)
            図のサイズ
        contour_kw : dict, default=None
            等高線プロットのキーワード引数
        ax : matplotlib.axes.Axes, default=None
            プロット先の軸。Noneの場合は新しい図を作成
        grid_resolution : int, default=50
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            作成された図と軸
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
            
        if len(feature_pair) != 2:
            raise ValueError("feature_pairは2つの要素を持つリストかタプルである必要があります")
            
        # 特徴量インデックスの変換
        feature_idxs = []
        feature_names = []
        
        for idx in feature_pair:
            if isinstance(idx, str):
                if idx in self.features.columns:
                    feature_idxs.append(list(self.features.columns).index(idx))
                    feature_names.append(idx)
                else:
                    raise ValueError(f"特徴量 '{idx}' がデータフレームに存在しません")
            else:
                feature_idxs.append(idx)
                feature_names.append(self.features.columns[idx])
                
        # ターゲット名の取得
        target_name = f"Class {target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        # デフォルトの等高線プロットオプション
        if contour_kw is None:
            contour_kw = {
                'alpha': 0.75,
                'cmap': 'viridis',
                'levels': 50
            }
            
        # 2つの特徴量の組み合わせを計算
        feature_pair_idx = (feature_idxs[0], feature_idxs[1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 最新のscikit-learnのAPIを使用
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_pair_idx], 
                    target=target_idx, ax=ax,
                    kind='average', grid_resolution=grid_resolution,
                    contour_kw=contour_kw
                )
            except Exception as e:
                print(f"Warning: Using fallback method for interaction plot: {e}")
                # 手動で2D部分依存プロットを生成
                try:
                    pd_result = partial_dependence(
                        self.model, scaled_features, [feature_pair_idx],
                        grid_resolution=grid_resolution
                    )
                    
                    # APIバージョンによって戻り値の構造が異なる
                    if isinstance(pd_result, tuple):
                        if len(pd_result) == 2:
                            pd_values, pd_grid = pd_result
                        else:
                            pd_values, pd_grid, _ = pd_result
                    else:
                        pd_values = pd_result.average
                        pd_grid = pd_result.grid_values
                    
                    # 等高線プロットの作成
                    X, Y = np.meshgrid(pd_grid[0][0], pd_grid[0][1])
                    Z = pd_values[0].T
                    
                    CS = ax.contourf(X, Y, Z, **contour_kw)
                    fig.colorbar(CS, ax=ax)
                    
                    ax.set_xlabel(feature_names[0])
                    ax.set_ylabel(feature_names[1])
                except Exception as e2:
                    print(f"Failed to create interaction plot: {e2}")
                    ax.text(0.5, 0.5, f"Error creating plot: {str(e2)[:100]}...", 
                            ha='center', va='center', transform=ax.transAxes)
            
        ax.set_title(f"特徴量相互作用: {feature_names[0]} vs {feature_names[1]}\nターゲット: {target_name}", fontsize=12)
        
        return fig, ax
    
    def save_plots(self, feature_importances, output_dir='plots', target_idx=1, 
                   n_features=10, n_cols=3, grid_resolution=100, format='png', dpi=300):
        """
        重要度の高い特徴量の部分依存プロットを保存
        
        Parameters:
        -----------
        feature_importances : pandas.DataFrame
            'feature'と'importance'列を持つ特徴量重要度のデータフレーム
        output_dir : str, default='plots'
            プロットの保存先ディレクトリ
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
        n_features : int, default=10
            プロットする上位n個の特徴量
        n_cols : int, default=3
            列の数
        grid_resolution : int, default=100
            グリッド解像度
        format : str, default='png'
            保存する画像形式（'png', 'pdf', 'svg'など）
        dpi : int, default=300
            画像のDPI（解像度）
            
        Returns:
        --------
        list
            保存されたファイルパスのリスト
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
            
        # 出力ディレクトリを作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 重要度の高い特徴量を取得
        top_features = feature_importances.sort_values('importance', ascending=False).head(n_features)
        top_feature_names = top_features['feature'].tolist()
        
        # 存在しない特徴量を除外
        valid_features = [f for f in top_feature_names if f in self.features.columns]
        
        if len(valid_features) == 0:
            print("警告: 有効な特徴量が見つかりません。特徴量名を確認してください。")
            # 代わりにすべての特徴量を使用
            valid_features = self.features.columns.tolist()[:n_features]
            print(f"代わりに最初の{len(valid_features)}個の特徴量を使用します。")
        
        # ターゲット名の取得
        target_name = f"class_{target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
                
        target_name = target_name.lower().replace(' ', '_')
        
        saved_files = []
        
        # 1. 全ての重要な特徴量の部分依存プロットを一つの図に
        try:
            fig = self.plot_multiple_features(
                valid_features, target_idx=target_idx, n_cols=n_cols,
                grid_resolution=grid_resolution
            )
            
            filename = os.path.join(output_dir, f"pdp_top{len(valid_features)}_{target_name}.{format}")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
            print(f"保存しました: {filename}")
        except Exception as e:
            print(f"複数特徴量のプロットの保存中にエラーが発生しました: {e}")
        
        # 2. 個別の特徴量ごとにプロット
        for feature_name in valid_features:
            try:
                fig, ax = self.plot_single_feature(
                    feature_name, target_idx=target_idx,
                    grid_resolution=grid_resolution
                )
                
                # ファイル名から無効な文字を削除
                clean_name = feature_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '_-.')
                
                filename = os.path.join(output_dir, f"pdp_{clean_name}_{target_name}.{format}")
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)
                print(f"保存しました: {filename}")
            except Exception as e:
                print(f"特徴量 '{feature_name}' のプロット保存中にエラーが発生しました: {e}")
            
        # 3. 上位2つの特徴量の相互作用プロット
        if len(valid_features) >= 2:
            try:
                fig, ax = self.plot_feature_interaction(
                    valid_features[:2], target_idx=target_idx,
                    grid_resolution=min(grid_resolution, 50)  # 2Dプロットは解像度を下げる
                )
                
                clean_name1 = valid_features[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                
                clean_name2 = valid_features[1].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                
                filename = os.path.join(output_dir, f"pdp_interaction_{clean_name1}_{clean_name2}_{target_name}.{format}")
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)
                print(f"保存しました: {filename}")
            except Exception as e:
                print(f"特徴量相互作用プロットの保存中にエラーが発生しました: {e}")
            
        print(f"{len(saved_files)}個のプロットを {output_dir} ディレクトリに保存しました")
        return saved_files


class ClassificationVisualizer:
    """
    分類問題用の汎用視覚化クラス
    
    このクラスは、分類モデルの入力データ、モデルの結果、特徴量の重要度などを
    視覚化するためのメソッドを提供します。
    """
    
    def __init__(self, output_dir=None, fig_size=(12, 8), style='whitegrid', 
                 save_format='png', dpi=300):
        """
        初期化メソッド
        
        Parameters:
        -----------
        output_dir : str, default=None
            図を保存するディレクトリ。Noneの場合は保存しません。
        fig_size : tuple, default=(12, 8)
            デフォルトの図のサイズ
        style : str, default='whitegrid'
            seabornのスタイル
        save_format : str, default='png'
            保存する図の形式
        dpi : int, default=300
            保存する図の解像度
        """
        self.output_dir = output_dir
        self.fig_size = fig_size
        self.save_format = save_format
        self.dpi = dpi
        
        # 出力ディレクトリの作成
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # グラフスタイルの設定
        sns.set_style(style)
    
    def _save_figure(self, fig, filename):
        """図を保存するヘルパーメソッド"""
        if self.output_dir:
            filepath = os.path.join(self.output_dir, f"{filename}.{self.save_format}")
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存しました: {filepath}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, figsize=None, normalize=False):
        """
        混同行列の可視化
        
        Parameters:
        -----------
        y_true : array-like
            実際のクラスラベル
        y_pred : array-like
            予測されたクラスラベル
        class_names : list, default=None
            クラス名のリスト。Noneの場合は数値インデックスを使用
        figsize : tuple, default=None
            図のサイズ
        normalize : bool, default=False
            行ごとに正規化するかどうか
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        plt.figure(figsize=figsize or (10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            square=True,
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title('混同行列', fontsize=14)
        plt.ylabel('実際のクラス', fontsize=12)
        plt.xlabel('予測されたクラス', fontsize=12)
        
        # 図の保存
        self._save_figure(plt.gcf(), 'confusion_matrix')
        plt.show()
    
    def plot_classification_metrics(self, y_true, y_pred, y_proba=None, class_names=None, figsize=None):
        """
        分類メトリクスの視覚化
        
        Parameters:
        -----------
        y_true : array-like
            実際のクラスラベル
        y_pred : array-like
            予測されたクラスラベル
        y_proba : array-like, default=None
            クラスの予測確率（ROC曲線のため）
        class_names : list, default=None
            クラス名のリスト。Noneの場合は数値インデックスを使用
        figsize : tuple, default=None
            図のサイズ
        """
        # クラス名の設定
        if class_names is None:
            n_classes = len(np.unique(y_true))
            class_names = [f'Class {i}' for i in range(n_classes)]
        else:
            n_classes = len(class_names)                            
        
        # 各クラスに対するメトリクスを計算
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # サブプロットの設定
        if y_proba is not None:
            fig, axes = plt.subplots(2, 2, figsize=figsize or (16, 12))
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize or (16, 10))
            axes = axes.reshape(-1, 1)
        
        # 1. 混同行列
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            square=True,
            xticklabels=class_names, 
            yticklabels=class_names,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('混同行列', fontsize=14)
        axes[0, 0].set_ylabel('実際のクラス', fontsize=12)
        axes[0, 0].set_xlabel('予測されたクラス', fontsize=12)
        
        # 2. クラスごとのメトリクス
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }, index=class_names)
        
        metrics_df.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('クラスごとのメトリクス', fontsize=14)
        axes[0, 1].set_ylabel('スコア', fontsize=12)
        axes[0, 1].set_xlabel('クラス', fontsize=12)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend(loc='lower right')
        
        # 3. ROC曲線（確率値が提供された場合）
        if y_proba is not None:
            # 2値分類の場合
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                
                axes[1, 0].plot(fpr, tpr, label=f'ROC 曲線 (AUC = {roc_auc:.3f})')
                axes[1, 0].plot([0, 1], [0, 1], 'k--')
                axes[1, 0].set_xlabel('偽陽性率 (FPR)', fontsize=12)
                axes[1, 0].set_ylabel('真陽性率 (TPR)', fontsize=12)
                axes[1, 0].set_title('ROC曲線', fontsize=14)
                axes[1, 0].legend(loc='lower right')
                
                # 予測確率の分布
                sns.histplot(
                    x=y_proba[:, 1], 
                    hue=y_true, 
                    bins=30, 
                    stat='probability', 
                    common_norm=False,
                    alpha=0.6,
                    ax=axes[1, 1]
                )
                axes[1, 1].set_title('クラスごとの予測確率分布', fontsize=14)
                axes[1, 1].set_xlabel('陽性クラスの予測確率', fontsize=12)
                axes[1, 1].set_ylabel('確率密度', fontsize=12)
            
            # 多クラス分類の場合
            else:
                # 全体の精度を表示
                ax = axes[1, 0]
                ax.axis('off')
                ax.text(0.5, 0.5, f"全体の精度: {accuracy:.3f}", fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
                
                # クラスごとの予測確率分布
                axes[1, 1].set_visible(False)
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                # 各クラスごとに確率分布を表示
                for i in range(n_classes):
                    proba_class = y_proba[:, i]
                    class_indices = (y_true == i)
                    
                    sns.kdeplot(
                        proba_class[class_indices], 
                        label=f'{class_names[i]} (実際)',
                        linestyle='-',
                        ax=ax2
                    )
                    
                    sns.kdeplot(
                        proba_class[~class_indices], 
                        label=f'他のクラス',
                        linestyle='--',
                        ax=ax2
                    )
                
                ax2.set_title('クラスごとの予測確率分布', fontsize=14)
                ax2.set_xlabel('予測確率', fontsize=12)
                ax2.set_ylabel('確率密度', fontsize=12)
                ax2.legend()
                
                # 別の図として保存
                self._save_figure(fig2, 'prediction_probability_distribution')
                plt.close(fig2)
        else:
            # y_probaがない場合は残りの軸を非表示にする
            axes[1, 0].axis('off')
            if len(axes.shape) > 1:  # 2x2のサブプロットの場合
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(fig, 'classification_metrics')
        plt.show()

class EyeTrackingVisualizer:
    """
    視線追跡データの視覚化クラス
    """
    
    def __init__(self, output_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        output_dir : str, default=None
            出力ディレクトリのパス
        """
        self.output_dir = output_dir
        
        # 出力ディレクトリの作成
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def visualize_feature_target_relationships(self, features, target):
        """
        特徴量と目的変数の関係を可視化
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量データ
        target : pandas.Series
            目的変数データ
        """
        print("特徴量と目的変数の関係を可視化します")
        
        # 相関関係の分析
        # 連続変数の場合
        target_name = 'target'
        if hasattr(target, 'name'):
            target_name = target.name
            
        # 列数を計算
        n_cols = 3
        n_features = len(features.columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # 図を作成
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        
        # 軸を平坦化（1次元にする）
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        elif n_cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        # 各特徴量に対してプロット
        for i, feature_name in enumerate(features.columns):
            ax = axes[i]
            
            # 散布図または箱ひげ図
            try:
                if target.dtype in [np.float64, np.int64]:
                    # 連続値のターゲットの場合は散布図
                    ax.scatter(features[feature_name], target, alpha=0.5)
                    ax.set_ylabel(target_name)
                else:
                    # カテゴリカルなターゲットの場合は箱ひげ図
                    sns.boxplot(x=target, y=features[feature_name], ax=ax)
                
                ax.set_title(f"{feature_name}")
                ax.set_xlabel(feature_name)
                    
            except Exception as e:
                print(f"特徴量 '{feature_name}' の可視化中にエラーが発生しました: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                        ha='center', va='center', transform=ax.transAxes)
        
        # 未使用の軸を非表示に
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 図を保存
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, "feature_target_relationships.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_saccade_features(self, df):
        """
        サッカード特徴量を可視化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            サッカード特徴量を含むデータフレーム
        """
        print("サッカード特徴量を可視化します")
        
        # サッカード関連の列を抽出
        saccade_cols = [col for col in df.columns if 'saccade' in col.lower()]
        
        if not saccade_cols:
            print("サッカード関連の特徴量が見つかりません")
            return
        
        # ターゲット変数を特定
        target_col = 'target' if 'target' in df.columns else 'Target'
        if target_col not in df.columns:
            print(f"警告: ターゲット列 '{target_col}' が見つかりません。可視化はターゲットなしで続行します。")
            has_target = False
        else:
            has_target = True
        
        # 可視化
        for col in saccade_cols:
            plt.figure(figsize=(10, 6))
            
            if has_target:
                # ターゲットクラス別の分布
                unique_targets = df[target_col].unique()
                
                if len(unique_targets) <= 5:  # クラス数が少ない場合
                    for target_val in unique_targets:
                        subset = df[df[target_col] == target_val]
                        plt.hist(subset[col], alpha=0.6, label=f'{target_col}={target_val}', bins=20)
                    
                    plt.legend()
                else:  # クラス数が多い場合は散布図
                    plt.scatter(df[col], df[target_col], alpha=0.6)
                    plt.ylabel(target_col)
            else:
                # ターゲットなしの場合は単純なヒストグラム
                plt.hist(df[col], bins=20)
            
            plt.title(f"{col} 分布")
            plt.xlabel(col)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 図を保存
            if self.output_dir:
                clean_name = col.replace(' ', '_').replace('/', '_').replace('\\', '_')
                plt.savefig(os.path.join(self.output_dir, f"saccade_{clean_name}.png"), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def visualize_eye_movement_trajectories(self, df):
        """
        視線軌跡を可視化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            視線軌跡データを含むデータフレーム
        """
        print("視線軌跡を可視化します")
        
        # 視線角度関連の列を特定
        angle_cols = [col for col in df.columns if 'angle' in col.lower() or 'eye' in col.lower()]
        
        if not angle_cols:
            print("視線軌跡データが見つかりません")
            return
        
        # ターゲット変数を特定
        target_col = 'target' if 'target' in df.columns else 'Target'
        has_target = target_col in df.columns
        
        # サンプリングして視覚化（データポイントが多すぎる場合）
        sample_size = min(1000, len(df))
        sampled_df = df.sample(sample_size)
        
        # 2次元散布図行列
        if len(angle_cols) >= 2:
            plt.figure(figsize=(12, 10))
            sns.pairplot(sampled_df[angle_cols + ([target_col] if has_target else [])], 
                       hue=target_col if has_target else None,
                       diag_kind='kde',
                       plot_kws={'alpha': 0.6})
            
            plt.suptitle("視線軌跡特徴量のペアプロット", y=1.02)
            
            # 図を保存
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "eye_movement_pairplot.png"), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
        
        # 時系列データが含まれている場合の可視化
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols and angle_cols:
            time_col = time_cols[0]  # 最初の時間列を使用
            
            # ランダムに数人のデータを抽出
            unique_ids = []
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            
            if id_cols:
                id_col = id_cols[0]
                unique_ids = df[id_col].unique()
                n_samples = min(5, len(unique_ids))  # 最大5人分表示
                sample_ids = np.random.choice(unique_ids, n_samples, replace=False)
                
                plt.figure(figsize=(12, 8))
                
                for i, sample_id in enumerate(sample_ids):
                    sample_data = df[df[id_col] == sample_id]
                    
                    # 時間順にソート
                    if time_col in sample_data.columns:
                        sample_data = sample_data.sort_values(time_col)
                    
                    # 視線角度プロット
                    for angle_col in angle_cols[:3]:  # 最大3種類の角度を表示
                        plt.subplot(n_samples, len(angle_cols[:3]), i * len(angle_cols[:3]) + angle_cols[:3].index(angle_col) + 1)
                        plt.plot(sample_data[time_col], sample_data[angle_col])
                        plt.title(f"ID={sample_id}, {angle_col}")
                        plt.xlabel(time_col)
                        plt.ylabel(angle_col)
                
                plt.tight_layout()
                
                # 図を保存
                if self.output_dir:
                    plt.savefig(os.path.join(self.output_dir, "eye_movement_time_series.png"), 
                               dpi=300, bbox_inches='tight')
                
                plt.show()

    def plot_feature_importance(self, feature_importance):
        """
        特徴量重要度を可視化
        
        Parameters:
        -----------
        feature_importance : pandas.DataFrame
            'feature'と'importance'列を持つ特徴量重要度のデータフレーム
        """
        # 重要度でソート
        sorted_imp = feature_importance.sort_values('importance', ascending=False)
        
        # 上位15件のみ表示
        n_features = min(15, len(sorted_imp))
        top_features = sorted_imp.head(n_features)
        
        plt.figure(figsize=(10, n_features * 0.4))
        
        # 水平バープロット
        sns.barplot(x='importance', y='feature', data=top_features)
        
        plt.title('特徴量重要度', fontsize=14)
        plt.xlabel('重要度', fontsize=12)
        plt.ylabel('特徴量', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # 図を保存
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_true_vs_predicted(self, y_true, y_pred):
        """
        真値と予測値の散布図を作成
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title('真値 vs 予測値')
        plt.grid(True)
        
        # 理想的な線（y=x）を追加
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'true_vs_predicted.png'), dpi=300)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        混同行列を可視化
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('予測値')
        plt.ylabel('真値')
        plt.title('混同行列')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()

    def plot_partial_dependence(self, model, X, feature_indices, feature_names=None):
        """
        部分依存グラフを作成
        """
        # PartialDependencePlotterクラスを使用
        plotter = PartialDependencePlotter(model=model, features=X)
        
        # 複数の特徴量に対する部分依存プロットを作成
        fig = plotter.plot_multiple_features(feature_indices, n_cols=2)
        
        if self.output_dir:
            fig.savefig(os.path.join(self.output_dir, 'partial_dependence.png'), dpi=300)
        plt.show()

    def create_partial_dependence_plots(self, results):
        """
        結果オブジェクトから部分依存プロットを作成
        """
        model = results['model']
        X = results['X_test']
        feature_importance = results['feature_importance']
        
        # 上位10個の特徴量を選択
        top_features = feature_importance.head(10)['feature'].tolist()
        
        # 特徴量のインデックスを取得
        feature_indices = [list(X.columns).index(feat) for feat in top_features if feat in X.columns]
        
        if not feature_indices:
            print("警告: 有効な特徴量インデックスが見つかりません")
            return
        
        # 部分依存プロットを作成
        self.plot_partial_dependence(model, X, feature_indices, feature_names=X.columns)

def create_partial_dependence_plots(self, results):
    """
    結果オブジェクトから部分依存プロットを作成
    上位5つの特徴量の全ての組み合わせで2次元部分依存プロットも作成
    """
    model = results['model']
    X = results['X_test']
    feature_importance = results['feature_importance']
    
    # 上位10個の特徴量を選択
    top_features = feature_importance.head(10)['feature'].tolist()
    
    # 特徴量のインデックスを取得
    feature_indices = [list(X.columns).index(feat) for feat in top_features if feat in X.columns]
    
    if not feature_indices:
        print("警告: 有効な特徴量インデックスが見つかりません")
        return
    
    # 1. 個別の特徴量に対する部分依存プロット
    self.plot_partial_dependence(model, X, feature_indices, feature_names=X.columns)
    
    # 2. 上位5つの特徴量の全ての組み合わせに対する相互作用プロット
    top_5_features = [feat for feat in top_features if feat in X.columns][:5]
    
    if len(top_5_features) < 2:
        print("警告: 2次元部分依存プロットを作成するには少なくとも2つの特徴量が必要です")
        return
    
    # 全ての2つの組み合わせを生成
    from itertools import combinations
    feature_pairs = list(combinations(top_5_features, 2))
    
    print(f"\n上位5つの特徴量の組み合わせに対する2次元部分依存プロット({len(feature_pairs)}組)を作成します...")
    
    # プロットを保存するディレクトリ
    interaction_dir = os.path.join(self.output_dir, 'interactions') if self.output_dir else None
    if interaction_dir and not os.path.exists(interaction_dir):
        os.makedirs(interaction_dir)
    
    # 各組み合わせについてプロット
    for i, (feat1, feat2) in enumerate(feature_pairs):
        try:
            print(f"特徴量の組み合わせ {i+1}/{len(feature_pairs)}: {feat1} と {feat2}")
            
            # PartialDependencePlotterインスタンスを作成
            plotter = PartialDependencePlotter(model=model, features=X)
            
            # 特徴量相互作用プロット作成
            fig, ax = plotter.plot_feature_interaction(
                feature_pair=[feat1, feat2], 
                target_idx=1,  # ターゲットクラスのインデックス (通常MCIクラスは1)
                figsize=(10, 8),
                grid_resolution=50,  # 解像度調整
                contour_kw={'cmap': 'viridis', 'alpha': 0.8, 'levels': 50}
            )
            
            # タイトルにランクを追加
            rank1 = top_features.index(feat1) + 1
            rank2 = top_features.index(feat2) + 1
            ax.set_title(f"特徴量相互作用: {feat1} (ランク{rank1}) vs {feat2} (ランク{rank2})\nターゲット: MCI", fontsize=12)
            
            # プロットを保存
            if interaction_dir:
                clean_name1 = feat1.replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name2 = feat2.replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                
                filename = os.path.join(interaction_dir, f"pdp_interaction_{rank1}_{clean_name1}_vs_{rank2}_{clean_name2}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"保存しました: {filename}")
            
            plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"特徴量 '{feat1}' と '{feat2}' の相互作用プロット作成中にエラーが発生しました: {e}")
    
    print("すべての2次元部分依存プロットの作成が完了しました")
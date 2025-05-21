"""
回帰問題用の視覚化モジュール
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import japanize_matplotlib

class RegressionVisualizer:
    """
    回帰問題用の汎用視覚化クラス
    
    このクラスは、回帰モデルの入力データ、モデルの結果、特徴量の重要度などを
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
    
    def plot_correlation_matrix(self, data, target_column=None, 
                              top_n=None, method='pearson', 
                              cmap='coolwarm', figsize=None):
        """
        相関行列のヒートマップを作成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            相関を計算するデータフレーム
        target_column : str, default=None
            ターゲット列名。指定した場合、この列との相関を強調します
        top_n : int, default=None
            表示する上位の特徴量数。Noneの場合はすべて表示
        method : str, default='pearson'
            相関係数の計算方法。'pearson', 'spearman', 'kendall'から選択
        cmap : str, default='coolwarm'
            ヒートマップのカラーマップ
        figsize : tuple, default=None
            図のサイズ。Noneの場合はデフォルトサイズを使用
        """
        # 相関行列を計算
        corr_matrix = data.corr(method=method)
        
        # ターゲット変数との相関が強い上位N個の特徴量を選択
        if top_n and target_column:
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
            top_features = list(target_corr.index[:top_n])
            
            if target_column not in top_features:
                top_features.append(target_column)
            
            corr_matrix = corr_matrix.loc[top_features, top_features]
        
        # ヒートマップのプロット
        plt.figure(figsize=figsize or self.fig_size)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap=cmap,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('特徴量の相関行列', fontsize=14)
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'correlation_matrix')
        plt.show()
    
    def plot_feature_correlations(self, features, target, top_n=15, figsize=None):
        """
        目的変数との相関の強さでソートした特徴量の棒グラフを作成
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量を含むデータフレーム
        target : pandas.Series
            目的変数
        top_n : int, default=15
            表示する上位の特徴量数
        figsize : tuple, default=None
            図のサイズ
        """
        # 相関係数の計算
        correlations = []
        for col in features.columns:
            try:
                corr, _ = pearsonr(features[col].values, target.values)
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # 相関係数の絶対値でソート
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_corrs = correlations[:top_n]
        
        # 可視化
        plt.figure(figsize=figsize or self.fig_size)
        colors = ['blue' if c > 0 else 'red' for _, c in top_corrs]
        sns.barplot(x=[abs(c) for _, c in top_corrs], y=[f for f, _ in top_corrs], palette=colors)
        plt.title(f'目的変数との相関係数（上位{top_n}）', fontsize=14)
        plt.xlabel('相関係数（絶対値）', fontsize=12)
        plt.ylabel('特徴量', fontsize=12)
        
        # 正負の凡例を追加
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='正の相関'),
            Line2D([0], [0], color='red', lw=4, label='負の相関')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'feature_target_correlations')
        plt.show()
    def plot_spearman_correlations(self, features, target, top_n=15, figsize=None):
        """
        スピアマンの順位相関係数でソートした特徴量の棒グラフを作成
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量を含むデータフレーム
        target : pandas.Series
            目的変数
        top_n : int, default=15
            表示する上位の特徴量数
        figsize : tuple, default=None
            図のサイズ
        """
        # 相関係数の計算
        correlations = []
        for col in features.columns:
            try:
                corr = spearman_correlation(features[col].values, target.values)
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # 相関係数の絶対値でソート
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_corrs = correlations[:top_n]
        
        # 可視化
        plt.figure(figsize=figsize or self.fig_size)
        colors = ['blue' if c > 0 else 'red' for _, c in top_corrs]
        sns.barplot(x=[abs(c) for _, c in top_corrs], y=[f for f, _ in top_corrs], palette=colors)
        plt.title(f'目的変数とのスピアマン相関係数（上位{top_n}）', fontsize=14)
        plt.xlabel('相関係数（絶対値）', fontsize=12)
        plt.ylabel('特徴量', fontsize=12)
        
        # 正負の凡例を追加
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='正の相関'),
            Line2D([0], [0], color='red', lw=4, label='負の相関')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'feature_target_spearman_correlations')
        plt.show()
        
    def plot_feature_distributions(self, data, target_column=None, top_n=4, figsize=None):
        """
        特徴量の分布をプロット
        
        Parameters:
        -----------
        data : pandas.DataFrame
            特徴量を含むデータフレーム
        target_column : str, default=None
            ターゲット列名。指定した場合、この列は除外
        top_n : int, default=4
            表示する特徴量の数
        figsize : tuple, default=None
            図のサイズ
        """
        # ターゲット列を除外
        if target_column and target_column in data.columns:
            features = data.drop(target_column, axis=1)
        else:
            features = data
        
        # 数値列のみ選択
        num_cols = features.select_dtypes(include=['number']).columns
        
        # 表示する特徴量を制限
        if len(num_cols) > top_n:
            num_cols = num_cols[:top_n]
        
        # 分布のプロット
        n_cols = len(num_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize or (10, 3 * n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(num_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'{col} の分布')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('頻度')
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'feature_distributions')
        plt.show()
    
    def plot_feature_target_scatter(self, features, target, top_n=4, figsize=None):
        """
        特徴量と目的変数の散布図を作成
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量を含むデータフレーム
        target : pandas.Series
            目的変数
        top_n : int, default=4
            表示する上位の特徴量数
        figsize : tuple, default=None
            図のサイズ
        """
        # 相関係数でソートした上位の特徴量を選択
        correlations = []
        for col in features.columns:
            try:
                corr, _ = pearsonr(features[col].values, target.values)
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = [f for f, _ in correlations[:top_n]]
        
        if len(top_features) == 0:
            print("散布図を作成するための有効な特徴量がありません。")
            return
        
        # 散布図マトリックスの作成
        scatter_data = features[top_features].copy()
        scatter_data['target'] = target.values
        
        # pairplotの作成
        g = sns.pairplot(
            scatter_data, 
            x_vars=top_features, 
            y_vars=['target'],
            height=3, 
            aspect=1.2, 
            kind='reg'
        )
        
        g.fig.suptitle('上位特徴量と目的変数の散布図', y=1.02, fontsize=16)
        
        # 図の保存
        self._save_figure(g.fig, 'feature_target_scatter')
        plt.show()
    
    def plot_feature_importance(self, feature_importance, top_n=20, figsize=None):
        """
        特徴量重要度のプロット
        
        Parameters:
        -----------
        feature_importance : pandas.DataFrame
            'feature'と'importance'列を持つデータフレーム
        top_n : int, default=20
            表示する上位の特徴量数
        figsize : tuple, default=None
            図のサイズ
        """
        # 上位N件の特徴量を選択
        n_features = min(top_n, len(feature_importance))
        top_features = feature_importance.sort_values('importance', ascending=False).head(n_features)
        
        plt.figure(figsize=figsize or self.fig_size)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'特徴量重要度（上位{n_features}）', fontsize=14)
        plt.xlabel('重要度', fontsize=12)
        plt.ylabel('特徴量', fontsize=12)
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'feature_importance')
        plt.show()
    
    def plot_true_vs_predicted(self, y_true, y_pred, figsize=None):
        """
        実測値と予測値の散布図
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like
            予測値
        figsize : tuple, default=None
            図のサイズ
        """
        plt.figure(figsize=figsize or (10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # 完全一致の線を追加
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('実測値 vs 予測値', fontsize=14)
        plt.xlabel('実測値', fontsize=12)
        plt.ylabel('予測値', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # R²値の計算と表示
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'true_vs_predicted')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, figsize=None):
        """
        残差プロット
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like
            予測値
        figsize : tuple, default=None
            図のサイズ
        """
        residuals = y_true - y_pred
        
        # 残差のプロット
        plt.figure(figsize=figsize or (10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('残差プロット', fontsize=14)
        plt.xlabel('予測値', fontsize=12)
        plt.ylabel('残差 (実測値 - 予測値)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 残差の統計値を表示
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        plt.annotate(
            f'平均残差: {mean_residual:.4f}\n標準偏差: {std_residual:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'residuals')
        plt.show()
    
    def plot_residual_distribution(self, y_true, y_pred, figsize=None):
        """
        残差の分布
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like
            予測値
        figsize : tuple, default=None
            図のサイズ
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=figsize or (10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('残差の分布', fontsize=14)
        plt.xlabel('残差 (実測値 - 予測値)', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 残差の統計値を表示
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        plt.annotate(
            f'平均: {mean_residual:.4f}\n標準偏差: {std_residual:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(plt.gcf(), 'residual_distribution')
        plt.show()
    
    def plot_regression_evaluation(self, y_true, y_pred, figsize=None):
        """
        回帰評価のためのプロットをまとめて表示
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like
            予測値
        figsize : tuple, default=None
            図のサイズ
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 評価指標の計算
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        residuals = y_true - y_pred
        
        # サブプロットの設定
        fig, axes = plt.subplots(2, 2, figsize=figsize or (14, 10))
        
        # 1. 実測値 vs 予測値
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_title('実測値 vs 予測値', fontsize=14)
        axes[0, 0].set_xlabel('実測値', fontsize=12)
        axes[0, 0].set_ylabel('予測値', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].annotate(
            f'R² = {r2:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        # 2. 残差プロット
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('残差プロット', fontsize=14)
        axes[0, 1].set_xlabel('予測値', fontsize=12)
        axes[0, 1].set_ylabel('残差', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差の分布
        sns.histplot(residuals, kde=True, bins=30, ax=axes[1, 0])
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_title('残差の分布', fontsize=14)
        axes[1, 0].set_xlabel('残差', fontsize=12)
        axes[1, 0].set_ylabel('頻度', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 評価指標の表示
        axes[1, 1].axis('off')
        metrics_text = (
            f"回帰モデル評価指標\n\n"
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"R²: {r2:.4f}\n\n"
            f"残差統計量\n"
            f"平均: {np.mean(residuals):.4f}\n"
            f"標準偏差: {np.std(residuals):.4f}\n"
            f"最小値: {np.min(residuals):.4f}\n"
            f"最大値: {np.max(residuals):.4f}"
        )
        axes[1, 1].text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # 図の保存
        self._save_figure(fig, 'regression_evaluation')
        plt.show()
    
    def plot_partial_dependence(self, model, X, features, feature_names=None, **kwargs):
        """
        部分依存プロットを生成するメソッド
        
        Parameters:
        -----------
        model : モデル
            学習済みの予測モデル
        X : pandas.DataFrame
            特徴量データ
        features : list
            部分依存プロットを生成する特徴量のインデックスリスト
        feature_names : list, default=None
            特徴量の名前のリスト
        """
        try:
            # 提供されたPartialDependencePlotterクラスを使用
            from models.partial_dependence_plotter_kaiki import PartialDependencePlotter
            
            # 特徴量重要度データの準備
            if feature_names is None:
                feature_names = X.columns.tolist()
                
            # 部分依存プロット生成
            plotter = PartialDependencePlotter(
                model=model,
                features=X
            )
            
            # top_nの特徴量についてプロット
            top_n = min(len(features), 6)  # 最大6つまで
            fig = plotter.plot_multiple_features(
                feature_names[:top_n], 
                n_cols=3,
                grid_resolution=50
            )
            
            # 保存
            if self.output_dir:
                plt.savefig(f'{self.output_dir}/partial_dependence.png')
            plt.close(fig)
            
            print("部分依存プロットの作成に成功しました")
            return True
        
        except Exception as e:
            print(f"部分依存プロットの作成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return False


class EyeTrackingVisualizer(RegressionVisualizer):
    """
    視線追跡データの可視化に特化したクラス
    
    RegressionVisualizerを継承し、視線追跡データの分析に特化した
    追加の可視化メソッドを提供します。
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
        
        super().__init__(output_dir, fig_size, style, save_format, dpi)
    
    def visualize_eye_movement_trajectories(self, eye_data, sample_n=None, figsize=None):
        """
        視線の軌跡を可視化
        
        Parameters:
        -----------
        eye_data : pandas.DataFrame
            視線データを含むデータフレーム
        sample_n : int, default=None
            サンプリングするデータ数。Noneの場合は全データを使用
        figsize : tuple, default=None
            図のサイズ
        """
        try:
            # 視線のX, Y座標を含む列を特定
            x_cols = [col for col in eye_data.columns if 'eye' in col.lower() and 'x' in col.lower()]
            y_cols = [col for col in eye_data.columns if 'eye' in col.lower() and 'y' in col.lower()]
            
            if not x_cols or not y_cols:
                print("視線の軌跡を可視化するためのX, Y座標列が見つかりません。")
                return
            
            # 代表的なX, Y座標列を選択
            x_col = x_cols[0]
            y_col = y_cols[0]
            
            # サンプリング
            if sample_n and len(eye_data) > sample_n:
                eye_sample = eye_data.sample(sample_n)
            else:
                eye_sample = eye_data
            
            # ターゲットカラム（'target'）があれば、それによる色分け
            has_target = 'target' in eye_sample.columns
            
            plt.figure(figsize=figsize or (10, 8))
            
            if has_target:
                # ターゲットの値に基づいてスキャッタープロット
                scatter = plt.scatter(
                    eye_sample[x_col], 
                    eye_sample[y_col],
                    c=eye_sample['target'],
                    cmap='viridis',
                    alpha=0.7,
                    s=30
                )
                plt.colorbar(scatter, label='Target Value')
            else:
                # 通常のスキャッタープロット
                plt.scatter(
                    eye_sample[x_col], 
                    eye_sample[y_col],
                    alpha=0.7,
                    s=30
                )
            
            plt.title('視線の軌跡', fontsize=14)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 図の保存
            self._save_figure(plt.gcf(), 'eye_movement_trajectories')
            plt.show()
            
            # ヒートマップ表示（密度）
            plt.figure(figsize=figsize or (10, 8))
            sns.kdeplot(
                x=eye_sample[x_col],
                y=eye_sample[y_col],
                cmap="YlOrRd",
                fill=True,
                levels=20
            )
            plt.title('視線の密度ヒートマップ', fontsize=14)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            
            # 図の保存
            self._save_figure(plt.gcf(), 'eye_movement_heatmap')
            plt.show()
            
        except Exception as e:
            print(f"視線軌跡の可視化中にエラーが発生しました: {e}")
    
    def visualize_saccade_features(self, eye_data, figsize=None):
        """
        サッカード特徴量を可視化
        
        Parameters:
        -----------
        eye_data : pandas.DataFrame
            サッカード特徴量を含むデータフレーム
        figsize : tuple, default=None
            図のサイズ
        """
        try:
            # サッカード関連の列を特定
            saccade_cols = [col for col in eye_data.columns if 'saccade' in col.lower()]
            
            if not saccade_cols:
                print("サッカード特徴量が見つかりません。")
                return
            
            # ターゲットカラム（'target'）があれば、特徴量との関係をプロット
            has_target = 'target' in eye_data.columns
            
            # サッカード特徴量の分布をプロット
            n_cols = len(saccade_cols)
            fig, axes = plt.subplots(
                n_cols, 1, 
                figsize=figsize or (10, 4 * n_cols),
                sharex=False
            )
            
            if n_cols == 1:
                axes = [axes]
            
            for i, col in enumerate(saccade_cols):
                if has_target:
                    # 目的変数との散布図
                    axes[i].scatter(
                        eye_data[col], 
                        eye_data['target'],
                        alpha=0.6,
                        s=20
                    )
                    axes[i].set_title(f'{col} vs Target', fontsize=12)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Target', fontsize=10)
                    
                    # 相関係数を計算して表示
                    try:
                        corr, p = pearsonr(eye_data[col].dropna(), eye_data.loc[eye_data[col].dropna().index, 'target'])
                        axes[i].annotate(
                            f'r = {corr:.3f}, p = {p:.3e}',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=10, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                        )
                    except:
                        pass
                else:
                    # ヒストグラム
                    sns.histplot(eye_data[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'{col} の分布', fontsize=12)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('頻度', fontsize=10)
                
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 図の保存
            self._save_figure(fig, 'saccade_features')
            plt.show()
            
            # サッカード特徴量間の相関マトリックスをプロット
            if len(saccade_cols) > 1:
                plt.figure(figsize=figsize or (10, 8))
                saccade_corr = eye_data[saccade_cols].corr()
                
                # ヒートマップ
                mask = np.triu(np.ones_like(saccade_corr, dtype=bool))
                sns.heatmap(
                    saccade_corr, 
                    mask=mask,
                    cmap='coolwarm',
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                    square=True
                )
                plt.title('サッカード特徴量間の相関', fontsize=14)
                plt.tight_layout()
                
                # 図の保存
                self._save_figure(plt.gcf(), 'saccade_correlation')
                plt.show()
                
        except Exception as e:
            print(f"サッカード特徴量の可視化中にエラーが発生しました: {e}")
    
    def visualize_eye_velocity_analysis(self, eye_data, velocity_cols=None, target_column='target', figsize=None):
        """
        視線の速度に関する分析を可視化
        
        Parameters:
        -----------
        eye_data : pandas.DataFrame
            視線データを含むデータフレーム
        velocity_cols : list, default=None
            速度関連の列の名前リスト。Noneの場合は自動検出
        target_column : str, default='target'
            目的変数の列名
        figsize : tuple, default=None
            図のサイズ
        """
        try:
            # 速度関連の列を特定
            if velocity_cols is None:
                velocity_cols = [col for col in eye_data.columns if 'velocity' in col.lower() or 'speed' in col.lower()]
            
            if not velocity_cols:
                print("速度関連の特徴量が見つかりません。")
                return
            
            has_target = target_column in eye_data.columns
            
            # 速度特徴量と目的変数の関係を可視化
            if has_target:
                n_cols = len(velocity_cols)
                fig, axes = plt.subplots(
                    n_cols, 1, 
                    figsize=figsize or (10, 4 * n_cols),
                    sharex=False
                )
                
                if n_cols == 1:
                    axes = [axes]
                
                for i, col in enumerate(velocity_cols):
                    # 散布図とヒストグラム（KDE）を合わせてプロット
                    sns.regplot(
                        x=eye_data[col], 
                        y=eye_data[target_column],
                        scatter_kws={'alpha': 0.5, 's': 15},
                        line_kws={'color': 'red'},
                        ax=axes[i]
                    )
                    
                    axes[i].set_title(f'{col} vs {target_column}', fontsize=12)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel(target_column, fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    
                    # 相関係数を計算して表示
                    try:
                        corr, p = pearsonr(eye_data[col].dropna(), eye_data.loc[eye_data[col].dropna().index, target_column])
                        axes[i].annotate(
                            f'r = {corr:.3f}, p = {p:.3e}',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=10, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                        )
                    except:
                        pass
                
                plt.tight_layout()
                
                # 図の保存
                self._save_figure(fig, 'eye_velocity_analysis')
                plt.show()
            
            # 速度分布を可視化
            plt.figure(figsize=figsize or (12, 6))
            for col in velocity_cols:
                sns.kdeplot(eye_data[col].dropna(), label=col)
            
            plt.title('視線速度分布', fontsize=14)
            plt.xlabel('速度', fontsize=12)
            plt.ylabel('密度', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 図の保存
            self._save_figure(plt.gcf(), 'eye_velocity_distribution')
            plt.show()
            
        except Exception as e:
            print(f"視線速度の分析中にエラーが発生しました: {e}")
    
    def visualize_fixation_analysis(self, eye_data, fixation_cols=None, target_column='target', figsize=None):
        """
        注視に関する分析を可視化
        
        Parameters:
        -----------
        eye_data : pandas.DataFrame
            視線データを含むデータフレーム
        fixation_cols : list, default=None
            注視関連の列の名前リスト。Noneの場合は自動検出
        target_column : str, default='target'
            目的変数の列名
        figsize : tuple, default=None
            図のサイズ
        """
        try:
            # 注視関連の列を特定
            if fixation_cols is None:
                fixation_cols = [col for col in eye_data.columns if 'fixation' in col.lower()]
            
            if not fixation_cols:
                print("注視関連の特徴量が見つかりません。")
                return
            
            has_target = target_column in eye_data.columns
            
            # 注視特徴量と目的変数の関係を可視化
            if has_target:
                n_cols = len(fixation_cols)
                fig, axes = plt.subplots(
                    n_cols, 1, 
                    figsize=figsize or (10, 4 * n_cols),
                    sharex=False
                )
                
                if n_cols == 1:
                    axes = [axes]
                
                for i, col in enumerate(fixation_cols):
                    # 散布図とヒストグラム（KDE）を合わせてプロット
                    sns.regplot(
                        x=eye_data[col], 
                        y=eye_data[target_column],
                        scatter_kws={'alpha': 0.5, 's': 15},
                        line_kws={'color': 'red'},
                        ax=axes[i]
                    )
                    
                    axes[i].set_title(f'{col} vs {target_column}', fontsize=12)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel(target_column, fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    
                    # 相関係数を計算して表示
                    try:
                        corr, p = pearsonr(eye_data[col].dropna(), eye_data.loc[eye_data[col].dropna().index, target_column])
                        axes[i].annotate(
                            f'r = {corr:.3f}, p = {p:.3e}',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=10, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                        )
                    except:
                        pass
                
                plt.tight_layout()
                
                # 図の保存
                self._save_figure(fig, 'fixation_analysis')
                plt.show()
            
            # 注視時間の分布など
            if 'fixation_time' in fixation_cols or any('time' in col.lower() for col in fixation_cols):
                time_cols = [col for col in fixation_cols if 'time' in col.lower()]
                
                plt.figure(figsize=figsize or (12, 6))
                for col in time_cols:
                    sns.kdeplot(eye_data[col].dropna(), label=col)
                
                plt.title('注視時間の分布', fontsize=14)
                plt.xlabel('時間', fontsize=12)
                plt.ylabel('密度', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 図の保存
                self._save_figure(plt.gcf(), 'fixation_time_distribution')
                plt.show()
            
        except Exception as e:
            print(f"注視分析の可視化中にエラーが発生しました: {e}")
    def create_partial_dependence_plots(self, results):
        """
        部分依存プロットを生成して保存するメソッド
        
        Parameters:
        -----------
        results : dict
            モデル学習結果を含む辞書
        """
        print("\n部分依存プロットを生成します...")
        
        try:
            from sklearn.inspection import PartialDependenceDisplay
            
            # 特徴量重要度に基づいて上位の特徴量を選択
            feature_importance = results['feature_importance']
            top_features = feature_importance.head(6)['feature'].tolist()
            
            # 特徴量のインデックスを取得
            feature_indices = [list(results['X_train'].columns).index(feature) for feature in top_features]
            
            # 部分依存プロットの作成
            fig, ax = plt.subplots(figsize=(15, 10))
            PartialDependenceDisplay.from_estimator(
                results['model'], 
                results['X_train'], 
                feature_indices, 
                kind='average',
                n_cols=3,
                grid_resolution=50,
                random_state=42,
                ax=ax
            )
            
            plt.suptitle('部分依存プロット (PDP)', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # 保存
            if self.output_dir:
                pdp_path = os.path.join(self.output_dir, 'pdp_top6.png')
                plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"部分依存プロットを保存しました: {pdp_path}")
            else:
                plt.show()
            
            # 個別の特徴量のプロット
            for i, feature in enumerate(top_features):
                try:
                    feature_idx = list(results['X_train'].columns).index(feature)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    PartialDependenceDisplay.from_estimator(
                        results['model'], 
                        results['X_train'], 
                        [feature_idx], 
                        kind='average',
                        grid_resolution=100,
                        random_state=42,
                        ax=ax
                    )
                    
                    # タイトルと軸ラベルの設定
                    ax.set_title(f'部分依存プロット - {feature}', fontsize=14)
                    ax.set_xlabel(feature, fontsize=12)
                    ax.set_ylabel('部分依存', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    # 保存
                    if self.output_dir:
                        # ファイル名の作成（特殊文字を置換）
                        safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        individual_path = os.path.join(self.output_dir, f"pdp_{i+1}_{safe_feature_name}.png")
                        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"個別の部分依存プロットを保存しました: {individual_path}")
                    else:
                        plt.show()
                except Exception as e:
                    print(f"特徴量 {feature} の部分依存プロット作成中にエラーが発生しました: {e}")
                    import traceback
                    traceback.print_exc()
            
            return True
        except Exception as e:
            print(f"部分依存プロットの生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                # バックアップ: 以前のPartialDependencePlotterを使用
                from models.partial_dependence_plotter_kaiki import PartialDependencePlotter
                
                print("別の方法で部分依存プロットを生成します...")
                
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
                if self.output_dir:
                    plt.savefig(f'{self.output_dir}/partial_dependence.png', dpi=300, bbox_inches='tight')
                    print(f"バックアップ方法で部分依存プロットを保存しました: {self.output_dir}/partial_dependence.png")
                    plt.close(fig)
                else:
                    plt.show()
                    
                return True
            except Exception as e2:
                print(f"バックアップ方法でも部分依存プロットの作成に失敗しました: {e2}")
                traceback.print_exc()
                return False
    def analyze_eye_movement_patterns(self, eye_data, target_column='target', n_clusters=3, figsize=None):
        """
        視線パターンをクラスタリングして分析（K-meansを使用）
        
        Parameters:
        -----------
        eye_data : pandas.DataFrame
            視線データを含むデータフレーム
        target_column : str, default='target'
            目的変数の列名
        n_clusters : int, default=3
            クラスタ数
        figsize : tuple, default=None
            図のサイズ
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 視線関連の特徴量を選択
            eye_features = eye_data.select_dtypes(include=['number'])
            
            if target_column in eye_features.columns:
                eye_features = eye_features.drop(target_column, axis=1)
            
            # ID列などを除外
            exclude_cols = ['InspectionDateAndId', 'id', 'date', 'time']
            for col in exclude_cols:
                if col in eye_features.columns:
                    eye_features = eye_features.drop(col, axis=1)
            
            # 欠損値を含む行を除外
            eye_features = eye_features.dropna()
            
            # スケーリング
            scaler = StandardScaler()
            eye_features_scaled = scaler.fit_transform(eye_features)
            
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(eye_features_scaled)
            
            # クラスタリング結果をデータフレームに追加
            clustered_data = eye_features.copy()
            clustered_data['cluster'] = clusters
            
            # 目的変数がある場合は追加
            if target_column in eye_data.columns:
                clustered_data[target_column] = eye_data.loc[clustered_data.index, target_column]
            
            # クラスタごとの特徴量平均を計算
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_),
                columns=eye_features.columns
            )
            
            # 各クラスタのサンプル数を表示
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
            plt.title('クラスタごとのサンプル数', fontsize=14)
            plt.xlabel('クラスタ', fontsize=12)
            plt.ylabel('サンプル数', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 図の保存
            self._save_figure(plt.gcf(), 'cluster_sample_counts')
            plt.show()
            
            # クラスタごとの特徴量平均をヒートマップで表示
            plt.figure(figsize=figsize or (14, 10))
            
            # 特徴量を正規化して表示
            normalized_centers = pd.DataFrame(
                kmeans.cluster_centers_,
                columns=eye_features.columns
            )
            
            sns.heatmap(
                normalized_centers,
                cmap='coolwarm',
                annot=False,
                fmt='.2f',
                linewidths=0.5,
                yticklabels=[f'Cluster {i}' for i in range(n_clusters)]
            )
            plt.title('クラスタごとの特徴量プロファイル（正規化）', fontsize=14)
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # 図の保存
            self._save_figure(plt.gcf(), 'cluster_feature_profiles')
            plt.show()
            
            # 目的変数とクラスタの関係（目的変数がある場合）
            if target_column in clustered_data.columns:
                plt.figure(figsize=(10, 6))
                
                sns.boxplot(x='cluster', y=target_column, data=clustered_data)
                plt.title(f'クラスタごとの{target_column}分布', fontsize=14)
                plt.xlabel('クラスタ', fontsize=12)
                plt.ylabel(target_column, fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # 図の保存
                self._save_figure(plt.gcf(), 'cluster_target_relationship')
                plt.show()
                
                # 各クラスタの目的変数平均
                target_means = clustered_data.groupby('cluster')[target_column].mean().sort_index()
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=target_means.index, y=target_means.values)
                plt.title(f'クラスタごとの{target_column}平均', fontsize=14)
                plt.xlabel('クラスタ', fontsize=12)
                plt.ylabel(f'{target_column}平均', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # 図の保存
                self._save_figure(plt.gcf(), 'cluster_target_means')
                plt.show()
            
            return clustered_data
            
        except Exception as e:
            print(f"視線パターンの分析中にエラーが発生しました: {e}")
            return None
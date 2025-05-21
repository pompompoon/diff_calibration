import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix

class ThresholdEvaluator:
    """
    分類モデルの閾値評価を行うためのクラス
    
    適合率、再現率、キュー率などの指標を閾値ごとに計算し、
    最適な閾値を見つけるための機能を提供します。
    """
    
    def __init__(self, output_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        output_dir : str, default=None
            結果を保存するディレクトリ
        """
        self.output_dir = output_dir
        
    def calculate_threshold_metrics(self, y_true, y_proba, thresholds=None, n_points=100):
        """
        閾値ごとの適合率、再現率、キュー率を計算する関数
        
        Parameters:
        -----------
        y_true : array-like
            実際のクラスラベル (0 または 1)
        y_proba : array-like
            陽性クラス(1)の予測確率
        thresholds : array-like, default=None
            評価する閾値の配列。Noneの場合は0から1までのn_points個の等間隔の値を使用
        n_points : int, default=100
            閾値として使用する点の数
            
        Returns:
        --------
        pandas.DataFrame
            閾値ごとの各指標を含むデータフレーム
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, n_points)  # 0と1は除外して数値的安定性を確保
        
        results = []
        
        for threshold in thresholds:
            # 閾値を適用して予測を生成
            y_pred = (y_proba >= threshold).astype(int)
            
            # 適合率と再現率を計算
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # 混同行列を計算
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # キュー率（Queue Rate）の計算 - 閾値以上と判定されたデータの割合
            queue_rate = (tp + fp) / (tp + fp + tn + fn)
            
            # 精度（Accuracy）の計算
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            
            # F1スコアの計算
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 特異度（Specificity）の計算
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'queue_rate': queue_rate,
                'accuracy': accuracy,
                'f1': f1,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        return pd.DataFrame(results)
    
    def plot_threshold_metrics(self, metrics_df, figsize=(12, 8), save_path=None):
        """
        閾値ごとの指標をプロットする関数
        
        Parameters:
        -----------
        metrics_df : pandas.DataFrame
            calculate_threshold_metrics関数から返されるデータフレーム
        figsize : tuple, default=(12, 8)
            図のサイズ
        save_path : str, default=None
            図の保存先パス。Noneの場合は保存しない
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        plt.figure(figsize=figsize)
        
        # メインの指標をプロット
        plt.plot(metrics_df['threshold'], metrics_df['precision'], '-', color='blue', label='適合率 (Precision)')
        plt.plot(metrics_df['threshold'], metrics_df['recall'], '-', color='green', label='再現率 (Recall)')
        plt.plot(metrics_df['threshold'], metrics_df['queue_rate'], '-', color='red', label='キュー率 (Queue Rate)')
        
        # グリッドラインの追加
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 軸ラベルとタイトルの設定
        plt.xlabel('閾値 (Threshold)', fontsize=12)
        plt.ylabel('値', fontsize=12)
        plt.title('閾値による適合率・再現率・キュー率の変化', fontsize=14)
        
        # 凡例の設定
        plt.legend(loc='best', fontsize=12)
        
        # x軸とy軸の範囲を設定
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        # 図の保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, metrics_df, figsize=(12, 8), save_path=None):
        """
        適合率-再現率曲線を描画する関数
        
        Parameters:
        -----------
        metrics_df : pandas.DataFrame
            calculate_threshold_metrics関数から返されるデータフレーム
        figsize : tuple, default=(12, 8)
            図のサイズ
        save_path : str, default=None
            図の保存先パス。Noneの場合は保存しない
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        plt.figure(figsize=figsize)
        
        # 適合率と再現率のデータを並べ替え
        pr_curve_df = metrics_df.sort_values('recall')
        
        # 適合率-再現率曲線をプロット
        plt.plot(pr_curve_df['recall'], pr_curve_df['precision'], '-', color='purple', linewidth=2)
        
        # 代表的な閾値にマーカーを追加
        thresholds_to_mark = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds_to_mark:
            # 最も近い閾値のインデックスを検索
            idx = (metrics_df['threshold'] - threshold).abs().idxmin()
            row = metrics_df.loc[idx]
            plt.plot(row['recall'], row['precision'], 'o', markersize=8, 
                     label=f'閾値 = {row["threshold"]:.2f}')
        
        # グリッドラインの追加
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 軸ラベルとタイトルの設定
        plt.xlabel('再現率 (Recall)', fontsize=12)
        plt.ylabel('適合率 (Precision)', fontsize=12)
        plt.title('適合率-再現率曲線', fontsize=14)
        
        # 凡例の設定
        plt.legend(loc='best', fontsize=10)
        
        # x軸とy軸の範囲を設定
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # 図の保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def find_optimal_threshold(self, metrics_df, method='f1', queue_constraint=None):
        """
        最適な閾値を見つける関数
        
        Parameters:
        -----------
        metrics_df : pandas.DataFrame
            calculate_threshold_metrics関数から返されるデータフレーム
        method : str, default='f1'
            最適化する指標
            'f1': F1スコアを最大化
            'accuracy': 精度を最大化
            'precision_recall_balance': 適合率と再現率の差を最小化
        queue_constraint : float, default=None
            キュー率の制約値。Noneの場合は制約なし
            
        Returns:
        --------
        float
            最適な閾値
        dict
            最適な閾値での各指標
        """
        # キュー率制約でフィルタリング
        if queue_constraint is not None:
            filtered_df = metrics_df[metrics_df['queue_rate'] <= queue_constraint]
            if len(filtered_df) == 0:
                print(f"警告: キュー率 {queue_constraint} 以下のデータがありません。全データを使用します。")
                filtered_df = metrics_df
        else:
            filtered_df = metrics_df
        
        # 最適化する指標に基づいて最適な閾値を見つける
        if method == 'f1':
            # F1スコアを最大化
            best_idx = filtered_df['f1'].idxmax()
        elif method == 'accuracy':
            # 精度を最大化
            best_idx = filtered_df['accuracy'].idxmax()
        elif method == 'precision_recall_balance':
            # 適合率と再現率の差を最小化
            filtered_df['pr_diff'] = abs(filtered_df['precision'] - filtered_df['recall'])
            best_idx = filtered_df['pr_diff'].idxmin()
        else:
            raise ValueError(f"不明な方法: {method}")
        
        # 最適な閾値と指標を取得
        best_row = filtered_df.loc[best_idx]
        best_threshold = best_row['threshold']
        
        # 結果を辞書として返す
        best_metrics = {
            'threshold': best_threshold,
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'queue_rate': best_row['queue_rate'],
            'accuracy': best_row['accuracy'],
            'f1': best_row['f1'],
            'specificity': best_row['specificity']
        }
        
        return best_threshold, best_metrics
    def plot_recall_specificity_curve(self, y_true, y_proba, output_dir=None, model_name="model", smote_suffix=""):
        """
        再現率と特異度の閾値による変化を可視化するメソッド
        
        Parameters:
        -----------
        y_true : array-like
            実際のクラスラベル (0 または 1)
        y_proba : array-like
            陽性クラス(1)の予測確率
        output_dir : str, default=None
            出力ディレクトリ。Noneの場合は保存しない
        model_name : str, default="model"
            モデル名（ファイル名に使用）
        smote_suffix : str, default=""
            SMOTEを使用した場合の接尾辞
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        # 使用する出力ディレクトリを設定
        if output_dir is None:
            output_dir = self.output_dir
        
        # 閾値の範囲を設定（0.01から0.99まで）
        thresholds = np.linspace(0.01, 0.99, 100)
        
        # 各閾値での再現率と特異度を計算
        recalls = []
        specificities = []
        
        for threshold in thresholds:
            # 閾値を適用して予測を生成
            y_pred = (y_proba >= threshold).astype(int)
            
            # 混同行列の要素を計算
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            # 再現率と特異度を計算
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            recalls.append(recall)
            specificities.append(specificity)
        
        # 図の作成
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, recalls, 'b-', linewidth=2, label='再現率 (Recall)')
        plt.plot(thresholds, specificities, 'r-', linewidth=2, label='特異度 (Specificity)')
        
        # 特定の閾値にマーカーを追加
        thresholds_to_mark = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for t in thresholds_to_mark:
            idx = np.abs(thresholds - t).argmin()
            plt.plot(thresholds[idx], recalls[idx], 'bo', markersize=8)
            plt.plot(thresholds[idx], specificities[idx], 'ro', markersize=8)
            plt.text(thresholds[idx] + 0.01, recalls[idx], f'{recalls[idx]:.2f}', fontsize=9)
            plt.text(thresholds[idx] + 0.01, specificities[idx], f'{specificities[idx]:.2f}', fontsize=9)
        
        # 交点を探す
        intersection_idx = np.argmin(np.abs(np.array(recalls) - np.array(specificities)))
        intersection_threshold = thresholds[intersection_idx]
        intersection_value = (recalls[intersection_idx] + specificities[intersection_idx]) / 2
        
        plt.plot(intersection_threshold, intersection_value, 'go', markersize=10)
        plt.text(intersection_threshold + 0.01, intersection_value, 
                f'閾値 = {intersection_threshold:.2f}\n再現率 = {recalls[intersection_idx]:.2f}\n特異度 = {specificities[intersection_idx]:.2f}', 
                fontsize=10)
        
        # グリッドと軸ラベルの設定
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('閾値 (Threshold)', fontsize=12)
        plt.ylabel('値', fontsize=12)
        plt.title('再現率と特異度の閾値による変化', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        
        # x軸とy軸の範囲を設定
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        # 図の保存
        if output_dir:
            save_path = os.path.join(output_dir, f'recall_specificity_curve_{model_name}{smote_suffix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"再現率-特異度曲線を保存しました: {save_path}")
        
        return plt.gcf()
    
    def create_threshold_recommendation_report(self, y_true, y_proba, output_dir=None):
        """
        閾値推奨レポートを作成する関数
        
        Parameters:
        -----------
        y_true : array-like
            実際のクラスラベル (0 または 1)
        y_proba : array-like
            陽性クラス(1)の予測確率
        output_dir : str, default=None
            出力ディレクトリ。Noneの場合は保存しない
            
        Returns:
        --------
        dict
            各最適化方法ごとの最適閾値と指標
        pandas.DataFrame
            閾値ごとの指標を含むデータフレーム
        """
        # 使用する出力ディレクトリを設定
        if output_dir is None:
            output_dir = self.output_dir
        
        # 閾値ごとの指標を計算
        metrics_df = self.calculate_threshold_metrics(y_true, y_proba, n_points=100)
        
        # 図の保存パス
        threshold_plot_path = None
        pr_curve_path = None
        optimal_thresholds_path = None
        
        if output_dir:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            threshold_plot_path = os.path.join(output_dir, 'threshold_metrics.png')
            pr_curve_path = os.path.join(output_dir, 'precision_recall_curve.png')
            optimal_thresholds_path = os.path.join(output_dir, 'optimal_thresholds.csv')
        
        # 閾値ごとの指標をプロット
        self.plot_threshold_metrics(metrics_df, save_path=threshold_plot_path)
        
        # 適合率-再現率曲線を描画
        self.plot_precision_recall_curve(metrics_df, save_path=pr_curve_path)
        
        # 各方法での最適閾値を検索
        methods = ['f1', 'accuracy', 'precision_recall_balance']
        optimal_thresholds = {}
        
        for method in methods:
            threshold, metrics = self.find_optimal_threshold(metrics_df, method=method)
            optimal_thresholds[method] = metrics
            
        # キュー率制約ありの最適閾値も検索
        queue_constraints = [0.1, 0.2, 0.3, 0.5]
        for constraint in queue_constraints:
            method_name = f'f1_queue_{constraint}'
            threshold, metrics = self.find_optimal_threshold(
                metrics_df, method='f1', queue_constraint=constraint
            )
            optimal_thresholds[method_name] = metrics
        
        # 結果をCSVに保存
        if optimal_thresholds_path:
            results_df = pd.DataFrame(optimal_thresholds).T
            results_df.index.name = 'method'
            results_df.to_csv(optimal_thresholds_path)
            print(f"最適閾値を保存しました: {optimal_thresholds_path}")
        
        return optimal_thresholds, metrics_df
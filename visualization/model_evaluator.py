import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    """
    モデルの評価指標を計算し、結果をファイルに出力するクラス。
    訓練精度、汎化精度、テストデータのtarget比率などの評価指標を算出します。
    """
    
    def __init__(self, output_dir=None):
        """
        ModelEvaluatorクラスの初期化
        
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
        モデルの評価指標を計算する
        
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
        print("\n訓練精度、汎化精度、テストデータのターゲット比率を算出します...")
        
        # 結果辞書から必要なデータを取得
        model = results.get('model')
        X_train = results.get('X_train')
        y_train = results.get('y_train')
        X_test = results.get('X_test')
        y_test = results.get('true_values')
        
        # NaN値の処理
        X_train, y_train = self._handle_nan_values(X_train, y_train, "訓練データ")
        X_test, y_test = self._handle_nan_values(X_test, y_test, "テストデータ")
        
        # テストデータのターゲット比率
        test_target_counts = np.bincount(y_test.astype(int))
        test_target_ratio = test_target_counts / len(y_test)
        test_ratio_str = f"{test_target_ratio[0]:.4f}:{test_target_ratio[1]:.4f}" if len(test_target_ratio) > 1 else "データなし"
        
        # 訓練データのターゲット比率
        train_target_counts = np.bincount(y_train.astype(int))
        train_target_ratio = train_target_counts / len(y_train)
        train_ratio_str = f"{train_target_ratio[0]:.4f}:{train_target_ratio[1]:.4f}" if len(train_target_ratio) > 1 else "データなし"
        
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
            metrics['train_accuracy'] = accuracy_score(y_train_for_metrics, y_train_pred)
            metrics['train_precision'] = precision_score(y_train_for_metrics, y_train_pred, average='weighted', zero_division=0)
            metrics['train_recall'] = recall_score(y_train_for_metrics, y_train_pred, average='weighted', zero_division=0)
            metrics['train_f1'] = f1_score(y_train_for_metrics, y_train_pred, average='weighted', zero_division=0)
            
            # 混同行列の計算（訓練データ）
            cm_train = confusion_matrix(y_train_for_metrics, y_train_pred)
            if cm_train.shape == (2, 2):  # 二値分類の場合
                tn, fp, fn, tp = cm_train.ravel()
                metrics['train_tn'] = int(tn)
                metrics['train_fp'] = int(fp)
                metrics['train_fn'] = int(fn)
                metrics['train_tp'] = int(tp)
                
                # 特異度（Specificity）
                metrics['train_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
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
            metrics['test_accuracy'] = accuracy_score(y_test_for_metrics, y_test_pred)
            metrics['test_precision'] = precision_score(y_test_for_metrics, y_test_pred, average='weighted', zero_division=0)
            metrics['test_recall'] = recall_score(y_test_for_metrics, y_test_pred, average='weighted', zero_division=0)
            metrics['test_f1'] = f1_score(y_test_for_metrics, y_test_pred, average='weighted', zero_division=0)
            
            # 混同行列の計算（テストデータ）
            cm_test = confusion_matrix(y_test_for_metrics, y_test_pred)
            if cm_test.shape == (2, 2):  # 二値分類の場合
                tn, fp, fn, tp = cm_test.ravel()
                metrics['test_tn'] = int(tn)
                metrics['test_fp'] = int(fp)
                metrics['test_fn'] = int(fn)
                metrics['test_tp'] = int(tp)
                
                # 特異度（Specificity）
                metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        except Exception as e:
            print(f"評価指標の計算中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー時はNoneを設定
            metrics['train_accuracy'] = None
            metrics['test_accuracy'] = None
        
        # ターゲットの分布情報を追加
        metrics['train_target_ratio'] = train_ratio_str
        metrics['test_target_ratio'] = test_ratio_str
        metrics['train_class_0_count'] = int(train_target_counts[0]) if len(train_target_counts) > 0 else 0
        metrics['train_class_1_count'] = int(train_target_counts[1]) if len(train_target_counts) > 1 else 0
        metrics['test_class_0_count'] = int(test_target_counts[0]) if len(test_target_counts) > 0 else 0
        metrics['test_class_1_count'] = int(test_target_counts[1]) if len(test_target_counts) > 1 else 0
        
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
        print("\n============= モデル評価指標 =============")
        print(f"訓練精度: {metrics.get('train_accuracy', 'N/A'):.4f}" if metrics.get('train_accuracy') is not None else "訓練精度: 計算できませんでした")
        print(f"汎化精度 (テスト精度): {metrics.get('test_accuracy', 'N/A'):.4f}" if metrics.get('test_accuracy') is not None else "汎化精度: 計算できませんでした")
        print(f"訓練データのターゲット比率 (0:1): {metrics.get('train_target_ratio', 'N/A')}")
        print(f"テストデータのターゲット比率 (0:1): {metrics.get('test_target_ratio', 'N/A')}")
        print(f"訓練データのクラス分布: クラス0={metrics.get('train_class_0_count', 0)}件, クラス1={metrics.get('train_class_1_count', 0)}件")
        print(f"テストデータのクラス分布: クラス0={metrics.get('test_class_0_count', 0)}件, クラス1={metrics.get('test_class_1_count', 0)}件")
        
        # 追加の指標（あれば表示）
        if 'train_precision' in metrics:
            print("\n--- 詳細な評価指標 ---")
            print("訓練データ:")
            print(f"  - 精度 (Accuracy): {metrics.get('train_accuracy', 'N/A'):.4f}" if metrics.get('train_accuracy') is not None else "  - 精度: N/A")
            print(f"  - 適合率 (Precision): {metrics.get('train_precision', 'N/A'):.4f}" if metrics.get('train_precision') is not None else "  - 適合率: N/A")
            print(f"  - 再現率 (Recall): {metrics.get('train_recall', 'N/A'):.4f}" if metrics.get('train_recall') is not None else "  - 再現率: N/A")
            print(f"  - F1スコア: {metrics.get('train_f1', 'N/A'):.4f}" if metrics.get('train_f1') is not None else "  - F1スコア: N/A")
            
            if 'train_specificity' in metrics:
                print(f"  - 特異度 (Specificity): {metrics.get('train_specificity', 'N/A'):.4f}" if metrics.get('train_specificity') is not None else "  - 特異度: N/A")
            
            print("テストデータ:")
            print(f"  - 精度 (Accuracy): {metrics.get('test_accuracy', 'N/A'):.4f}" if metrics.get('test_accuracy') is not None else "  - 精度: N/A")
            print(f"  - 適合率 (Precision): {metrics.get('test_precision', 'N/A'):.4f}" if metrics.get('test_precision') is not None else "  - 適合率: N/A")
            print(f"  - 再現率 (Recall): {metrics.get('test_recall', 'N/A'):.4f}" if metrics.get('test_recall') is not None else "  - 再現率: N/A")
            print(f"  - F1スコア: {metrics.get('test_f1', 'N/A'):.4f}" if metrics.get('test_f1') is not None else "  - F1スコア: N/A")
            
            if 'test_specificity' in metrics:
                print(f"  - 特異度 (Specificity): {metrics.get('test_specificity', 'N/A'):.4f}" if metrics.get('test_specificity') is not None else "  - 特異度: N/A")
        
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
            f.write("============= モデル評価指標 =============\n")
            f.write(f"訓練精度: {metrics.get('train_accuracy', 'N/A'):.4f}\n" if metrics.get('train_accuracy') is not None else "訓練精度: 計算できませんでした\n")
            f.write(f"汎化精度 (テスト精度): {metrics.get('test_accuracy', 'N/A'):.4f}\n" if metrics.get('test_accuracy') is not None else "汎化精度: 計算できませんでした\n")
            f.write(f"訓練データのターゲット比率 (0:1): {metrics.get('train_target_ratio', 'N/A')}\n")
            f.write(f"テストデータのターゲット比率 (0:1): {metrics.get('test_target_ratio', 'N/A')}\n")
            f.write(f"訓練データのクラス分布: クラス0={metrics.get('train_class_0_count', 0)}件, クラス1={metrics.get('train_class_1_count', 0)}件\n")
            f.write(f"テストデータのクラス分布: クラス0={metrics.get('test_class_0_count', 0)}件, クラス1={metrics.get('test_class_1_count', 0)}件\n")
            
            # 追加の指標（あれば保存）
            if 'train_precision' in metrics:
                f.write("\n--- 詳細な評価指標 ---\n")
                f.write("訓練データ:\n")
                f.write(f"  - 精度 (Accuracy): {metrics.get('train_accuracy', 'N/A'):.4f}\n" if metrics.get('train_accuracy') is not None else "  - 精度: N/A\n")
                f.write(f"  - 適合率 (Precision): {metrics.get('train_precision', 'N/A'):.4f}\n" if metrics.get('train_precision') is not None else "  - 適合率: N/A\n")
                f.write(f"  - 再現率 (Recall): {metrics.get('train_recall', 'N/A'):.4f}\n" if metrics.get('train_recall') is not None else "  - 再現率: N/A\n")
                f.write(f"  - F1スコア: {metrics.get('train_f1', 'N/A'):.4f}\n" if metrics.get('train_f1') is not None else "  - F1スコア: N/A\n")
                
                if 'train_specificity' in metrics:
                    f.write(f"  - 特異度 (Specificity): {metrics.get('train_specificity', 'N/A'):.4f}\n" if metrics.get('train_specificity') is not None else "  - 特異度: N/A\n")
                
                if 'train_tp' in metrics:
                    f.write(f"  - 混同行列: TP={metrics.get('train_tp', 'N/A')}, FP={metrics.get('train_fp', 'N/A')}, FN={metrics.get('train_fn', 'N/A')}, TN={metrics.get('train_tn', 'N/A')}\n")
                
                f.write("\nテストデータ:\n")
                f.write(f"  - 精度 (Accuracy): {metrics.get('test_accuracy', 'N/A'):.4f}\n" if metrics.get('test_accuracy') is not None else "  - 精度: N/A\n")
                f.write(f"  - 適合率 (Precision): {metrics.get('test_precision', 'N/A'):.4f}\n" if metrics.get('test_precision') is not None else "  - 適合率: N/A\n")
                f.write(f"  - 再現率 (Recall): {metrics.get('test_recall', 'N/A'):.4f}\n" if metrics.get('test_recall') is not None else "  - 再現率: N/A\n")
                f.write(f"  - F1スコア: {metrics.get('test_f1', 'N/A'):.4f}\n" if metrics.get('test_f1') is not None else "  - F1スコア: N/A\n")
                
                if 'test_specificity' in metrics:
                    f.write(f"  - 特異度 (Specificity): {metrics.get('test_specificity', 'N/A'):.4f}\n" if metrics.get('test_specificity') is not None else "  - 特異度: N/A\n")
                
                if 'test_tp' in metrics:
                    f.write(f"  - 混同行列: TP={metrics.get('test_tp', 'N/A')}, FP={metrics.get('test_fp', 'N/A')}, FN={metrics.get('test_fn', 'N/A')}, TN={metrics.get('test_tn', 'N/A')}\n")
            
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
# visualizationフォルダ内のregression_pdp_handler.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import warnings
import datetime
import shutil
import japanize_matplotlib

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
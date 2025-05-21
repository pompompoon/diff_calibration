"""
SMOTE効果の可視化・分析モジュール

このモジュールはSMOTE適用前後の効果を部分依存プロットや
その他の可視化手法で分析するための関数を提供します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 必要に応じて、プロジェクトの他のモジュールからインポート
from models.partial_dependence_plotter_kaiki import PartialDependencePlotter


def visualize_smote_effect_with_pdp(original_model, smote_model, original_features, smote_features, 
                                   feature_name, output_dir=None):
   """
   SMOTE適用前後の部分依存プロットを比較する関数
   
   Parameters:
   -----------
   original_model : 学習済みモデル
       元データで学習したモデル
   smote_model : 学習済みモデル
       SMOTE適用データで学習したモデル
   original_features : pandas.DataFrame
       元の特徴量データ
   smote_features : pandas.DataFrame
       SMOTE適用後の特徴量データ
   feature_name : str
       比較する特徴量名
   output_dir : str, default=None
       保存先ディレクトリ
       
   Returns:
   --------
   str or None
       保存されたファイルのパス（output_dirが指定された場合）
   """
   fig, axes = plt.subplots(1, 2, figsize=(16, 6))
   
   try:
       # 元データのプロッター
       original_plotter = PartialDependencePlotter(model=original_model, features=original_features)
       original_plotter.plot_single_feature(feature_name, ax=axes[0])
       axes[0].set_title(f"元データ: {feature_name}", fontsize=12)
       
       # SMOTE適用後データのプロッター
       smote_plotter = PartialDependencePlotter(model=smote_model, features=smote_features)
       smote_plotter.plot_single_feature(feature_name, ax=axes[1])
       axes[1].set_title(f"SMOTE適用後: {feature_name}", fontsize=12)
       
       plt.suptitle(f"SMOTE効果の部分依存プロット比較", fontsize=14)
       plt.tight_layout()
       
       if output_dir:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           # ファイル名から無効な文字を削除
           safe_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
           save_path = os.path.join(output_dir, f'smote_pdp_comparison_{safe_feature_name}_{timestamp}.png')
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
           plt.close(fig)
           print(f"SMOTE効果比較プロットを保存しました: {save_path}")
           return save_path
       else:
           plt.show()
           plt.close(fig)
           return None
   except Exception as e:
       print(f"SMOTE効果比較プロットの生成中にエラーが発生しました: {e}")
       plt.close(fig)
       return None


def analyze_smote_effect_comprehensive(original_data, smote_data, feature_importances, 
                                    original_model, smote_model, output_dir=None):
   """
   SMOTE効果の包括的な分析と可視化を行う関数
   
   Parameters:
   -----------
   original_data : dict
       元データの情報 {'features': DataFrame, 'target': Series}
   smote_data : dict
       SMOTE適用後データの情報 {'features': DataFrame, 'target': Series}
   feature_importances : pandas.DataFrame
       特徴量重要度（'feature'と'importance'列を含む）
   original_model : 学習済みモデル
       元データで学習したモデル
   smote_model : 学習済みモデル
       SMOTE適用データで学習したモデル
   output_dir : str, default=None
       保存先ディレクトリ
       
   Returns:
   --------
   dict
       分析結果を含む辞書
   """
   saved_files = []
   analysis_results = {
       'summary': {},
       'saved_files': [],
       'metrics_comparison': {},
       'distribution_changes': {}
   }
   
   if output_dir:
       smote_analysis_dir = os.path.join(output_dir, 'smote_analysis')
       if not os.path.exists(smote_analysis_dir):
           os.makedirs(smote_analysis_dir)
   else:
       smote_analysis_dir = None
   
   print("\nSMOTE効果の包括的分析を開始...")
   
   # ============================================
   # 1. データ分布の比較
   # ============================================
   try:
       print("1. データ分布の比較分析...")
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
       # 1-1. ターゲット値の分布比較
       axes[0, 0].hist(original_data['target'], bins=50, alpha=0.7, label='元データ', density=True)
       axes[0, 0].hist(smote_data['target'], bins=50, alpha=0.7, label='SMOTE適用後', density=True)
       axes[0, 0].set_title('ターゲット値の分布', fontsize=12)
       axes[0, 0].set_xlabel('ターゲット値')
       axes[0, 0].set_ylabel('密度')
       axes[0, 0].legend()
       axes[0, 0].grid(True, alpha=0.3)
       
       # 統計値を計算
       original_target_stats = {
           'mean': np.mean(original_data['target']),
           'std': np.std(original_data['target']),
           'min': np.min(original_data['target']),
           'max': np.max(original_data['target'])
       }
       smote_target_stats = {
           'mean': np.mean(smote_data['target']),
           'std': np.std(smote_data['target']),
           'min': np.min(smote_data['target']),
           'max': np.max(smote_data['target'])
       }
       
       analysis_results['distribution_changes']['target'] = {
           'original': original_target_stats,
           'smote': smote_target_stats
       }
       
       # 1-2. 第1主成分の分布比較
       pca = PCA(n_components=1)
       
       original_pca = pca.fit_transform(original_data['features'])
       pca_smote = pca.transform(smote_data['features'][:len(original_data['features'])])  # 元のサンプル部分
       pca_synthetic = pca.transform(smote_data['features'][len(original_data['features']):])  # 合成部分
       
       axes[0, 1].hist(original_pca, bins=50, alpha=0.7, label='元データ', density=True)
       axes[0, 1].hist(pca_synthetic, bins=50, alpha=0.7, label='合成データ', density=True)
       axes[0, 1].set_title('第1主成分の分布', fontsize=12)
       axes[0, 1].set_xlabel('第1主成分')
       axes[0, 1].set_ylabel('密度')
       axes[0, 1].legend()
       axes[0, 1].grid(True, alpha=0.3)
       
       # 1-3. 重要特徴量の分布比較
       top_feature = feature_importances.iloc[0]['feature']
       axes[1, 0].hist(original_data['features'][top_feature], bins=50, alpha=0.7, label='元データ', density=True)
       axes[1, 0].hist(smote_data['features'][top_feature], bins=50, alpha=0.7, label='SMOTE適用後', density=True)
       axes[1, 0].set_title(f'重要特徴量の分布: {top_feature}', fontsize=12)
       axes[1, 0].set_xlabel(top_feature)
       axes[1, 0].set_ylabel('密度')
       axes[1, 0].legend()
       axes[1, 0].grid(True, alpha=0.3)
       
       # 特徴量の統計値
       original_feature_stats = {
           'mean': original_data['features'][top_feature].mean(),
           'std': original_data['features'][top_feature].std()
       }
       smote_feature_stats = {
           'mean': smote_data['features'][top_feature].mean(),
           'std': smote_data['features'][top_feature].std()
       }
       
       analysis_results['distribution_changes'][top_feature] = {
           'original': original_feature_stats,
           'smote': smote_feature_stats
       }
       
       # 1-4. データサイズの比較
       sizes = [len(original_data['features']), len(smote_data['features']) - len(original_data['features'])]
       labels = ['元データ', '合成データ']
       axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
       axes[1, 1].set_title('データ構成比', fontsize=12)
       
       # データサイズの情報を記録
       analysis_results['summary']['data_sizes'] = {
           'original': len(original_data['features']),
           'synthetic': len(smote_data['features']) - len(original_data['features']),
           'total': len(smote_data['features']),
           'increase_ratio': ((len(smote_data['features']) - len(original_data['features'])) / len(original_data['features']) * 100)
       }
       
       plt.suptitle('SMOTE効果: データ分布の変化', fontsize=16)
       plt.tight_layout()
       
       if smote_analysis_dir:
           distribution_path = os.path.join(smote_analysis_dir, 'smote_data_distribution.png')
           plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
           saved_files.append(distribution_path)
           plt.close(fig)
       else:
           plt.show()
           plt.close(fig)
           
   except Exception as e:
       print(f"データ分布比較プロットの生成中にエラーが発生しました: {e}")
       if 'fig' in locals():
           plt.close(fig)
   
   # ============================================
   # 2. 部分依存プロットの詳細比較
   # ============================================
   try:
       print("2. 部分依存プロットの比較分析...")
       
       top_features = feature_importances.head(3)['feature'].tolist()
       
       fig, axes = plt.subplots(len(top_features), 2, figsize=(16, 6 * len(top_features)))
       
       original_plotter = PartialDependencePlotter(model=original_model, features=original_data['features'])
       smote_plotter = PartialDependencePlotter(model=smote_model, features=smote_data['features'])
       
       for i, feature in enumerate(top_features):
           # 元データでの部分依存プロット
           original_plotter.plot_single_feature(feature, ax=axes[i, 0] if len(top_features) > 1 else axes[0])
           (axes[i, 0] if len(top_features) > 1 else axes[0]).set_title(f"元データ: {feature}", fontsize=12)
           
           # SMOTE適用後データでの部分依存プロット
           smote_plotter.plot_single_feature(feature, ax=axes[i, 1] if len(top_features) > 1 else axes[1])
           (axes[i, 1] if len(top_features) > 1 else axes[1]).set_title(f"SMOTE適用後: {feature}", fontsize=12)
       
       plt.suptitle('SMOTE効果: 部分依存プロットの比較', fontsize=16)
       plt.tight_layout()
       
       if smote_analysis_dir:
           pdp_comparison_path = os.path.join(smote_analysis_dir, 'smote_pdp_comparison.png')
           plt.savefig(pdp_comparison_path, dpi=300, bbox_inches='tight')
           saved_files.append(pdp_comparison_path)
           plt.close(fig)
       else:
           plt.show()
           plt.close(fig)
           
   except Exception as e:
       print(f"部分依存プロット比較の生成中にエラーが発生しました: {e}")
       if 'fig' in locals():
           plt.close(fig)
   
   # ============================================
   # 3. 特徴量重要度の変化分析
   # ============================================
   try:
       print("3. 特徴量重要度の変化分析...")
       
       # モデルから特徴量重要度を取得
       original_importance = None
       smote_importance = None
       
       if hasattr(original_model, 'feature_importances_'):
           original_importance = original_model.feature_importances_
       elif hasattr(original_model, 'get_feature_importance'):
           original_importance = original_model.get_feature_importance()
       
       if hasattr(smote_model, 'feature_importances_'):
           smote_importance = smote_model.feature_importances_
       elif hasattr(smote_model, 'get_feature_importance'):
           smote_importance = smote_model.get_feature_importance()
       
       if original_importance is not None and smote_importance is not None:
           fig, ax = plt.subplots(figsize=(12, 8))
           
           # 上位10特徴量を比較
           top_10_features = feature_importances.head(10)['feature'].tolist()
           indices = [list(original_data['features'].columns).index(f) for f in top_10_features]
           
           x = np.arange(len(top_10_features))
           width = 0.35
           
           ax.bar(x - width/2, original_importance[indices], width, label='元データ', alpha=0.8)
           ax.bar(x + width/2, smote_importance[indices], width, label='SMOTE適用後', alpha=0.8)
           
           ax.set_xlabel('特徴量')
           ax.set_ylabel('重要度')
           ax.set_title('SMOTE効果: 特徴量重要度の変化')
           ax.set_xticks(x)
           ax.set_xticklabels(top_10_features, rotation=45, ha='right')
           ax.legend()
           ax.grid(True, alpha=0.3)
           
           plt.tight_layout()
           
           # 重要度の変化を数値で記録
           importance_changes = {}
           for i, feature in enumerate(top_10_features):
               idx = indices[i]
               change = smote_importance[idx] - original_importance[idx]
               change_pct = (change / original_importance[idx]) * 100 if original_importance[idx] != 0 else 0
               importance_changes[feature] = {
                   'original': original_importance[idx],
                   'smote': smote_importance[idx],
                   'change': change,
                   'change_percent': change_pct
               }
           
           analysis_results['metrics_comparison']['feature_importance_changes'] = importance_changes
           
           if smote_analysis_dir:
               importance_path = os.path.join(smote_analysis_dir, 'smote_importance_comparison.png')
               plt.savefig(importance_path, dpi=300, bbox_inches='tight')
               saved_files.append(importance_path)
               plt.close(fig)
           else:
               plt.show()
               plt.close(fig)
               
   except Exception as e:
       print(f"特徴量重要度比較の生成中にエラーが発生しました: {e}")
       if 'fig' in locals():
           plt.close(fig)
   
   # ============================================
   # 4. モデル性能の比較（同じテストデータがある場合）
   # ============================================
   try:
       print("4. モデル性能の比較分析...")
       
       # テストデータでの性能比較（別途提供されている場合）
       performance_comparison = analyze_model_performance_comparison(
           original_model, smote_model, 
           original_data, smote_data,
           smote_analysis_dir
       )
       
       if performance_comparison:
           analysis_results['metrics_comparison']['model_performance'] = performance_comparison
           
   except Exception as e:
       print(f"モデル性能比較の生成中にエラーが発生しました: {e}")
   
   # ============================================
   # 5. 合成データの品質評価
   # ============================================
   try:
       print("5. 合成データの品質評価...")
       
       quality_metrics = evaluate_synthetic_data_quality(
           original_data['features'],
           smote_data['features'],
           smote_analysis_dir
       )
       
       analysis_results['metrics_comparison']['synthetic_quality'] = quality_metrics
       
   except Exception as e:
       print(f"合成データ品質評価中にエラーが発生しました: {e}")
   
   # ============================================
   # 6. 結果のまとめ
   # ============================================
   analysis_results['saved_files'] = saved_files
   
   # サマリーレポートの作成
   create_summary_report(analysis_results, smote_analysis_dir)
   
   print("SMOTE効果の包括的分析が完了しました。")
   print(f"保存されたファイル数: {len(saved_files)}")
   
   return analysis_results


def analyze_model_performance_comparison(original_model, smote_model, original_data, smote_data, output_dir=None):
   """
   モデル性能の比較分析を行う関数
   """
   try:
       # 各モデルでの予測を比較（可能な場合）
       # この部分は実装状況に応じて調整が必要
       
       performance_comparison = {
           'cross_validation_performed': False,
           'test_data_available': False,
           'note': 'テストデータまたは交差検証結果が必要です'
       }
       
       return performance_comparison
       
   except Exception as e:
       print(f"性能比較分析中にエラー: {e}")
       return None


def evaluate_synthetic_data_quality(original_features, smote_features, output_dir=None):
   """
   合成データの品質を評価する関数
   """
   try:
       synthetic_features = smote_features[len(original_features):]
       
       # 1. 統計的類似性
       stat_similarity = {}
       for col in original_features.columns:
           original_stats = {
               'mean': original_features[col].mean(),
               'std': original_features[col].std(),
               'skew': original_features[col].skew(),
               'kurt': original_features[col].kurtosis()
           }
           synthetic_stats = {
               'mean': synthetic_features[col].mean(),
               'std': synthetic_features[col].std(),
               'skew': synthetic_features[col].skew(),
               'kurt': synthetic_features[col].kurtosis()
           }
           
           # 類似度スコア（簡単な例）
           mean_diff = abs(original_stats['mean'] - synthetic_stats['mean']) / abs(original_stats['mean'])
           std_diff = abs(original_stats['std'] - synthetic_stats['std']) / abs(original_stats['std'])
           
           stat_similarity[col] = {
               'original': original_stats,
               'synthetic': synthetic_stats,
               'mean_diff_ratio': mean_diff,
               'std_diff_ratio': std_diff
           }
       
       # 2. 相関構造の比較
       original_corr = original_features.corr()
       synthetic_corr = synthetic_features.corr()
       
       # Frobenius norm で相関行列の差を計算
       corr_diff = np.linalg.norm(original_corr - synthetic_corr, 'fro')
       
       quality_metrics = {
           'statistical_similarity': stat_similarity,
           'correlation_difference': corr_diff,
           'n_synthetic_samples': len(synthetic_features),
           'quality_score': 1 / (1 + corr_diff)  # 簡単な品質スコア
       }
       
       # 品質評価の可視化
       if output_dir:
           create_quality_visualization(original_features, synthetic_features, quality_metrics, output_dir)
       
       return quality_metrics
       
   except Exception as e:
       print(f"合成データ品質評価中にエラー: {e}")
       return None


def create_quality_visualization(original_features, synthetic_features, quality_metrics, output_dir):
   """
   品質評価の可視化を作成
   """
   try:
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
       # 1. 相関行列の比較
       original_corr = original_features.corr()
       synthetic_corr = synthetic_features.corr()
       
       # 上位10特徴量の相関行列を表示
       top_features = original_features.columns[:10]
       
       sns.heatmap(original_corr.loc[top_features, top_features], 
                  annot=True, fmt='.2f', ax=axes[0, 0], 
                  cmap='coolwarm', center=0)
       axes[0, 0].set_title('元データの相関行列')
       
       sns.heatmap(synthetic_corr.loc[top_features, top_features], 
                  annot=True, fmt='.2f', ax=axes[0, 1], 
                  cmap='coolwarm', center=0)
       axes[0, 1].set_title('合成データの相関行列')
       
       # 2. 統計量の比較
       features_to_plot = original_features.columns[:5]
       mean_diffs = [quality_metrics['statistical_similarity'][f]['mean_diff_ratio'] for f in features_to_plot]
       
       axes[1, 0].bar(range(len(features_to_plot)), mean_diffs)
       axes[1, 0].set_xticks(range(len(features_to_plot)))
       axes[1, 0].set_xticklabels(features_to_plot, rotation=45)
       axes[1, 0].set_ylabel('平均値の差（比率）')
       axes[1, 0].set_title('特徴量の統計量比較')
       
       # 3. 品質スコアの表示
       axes[1, 1].text(0.5, 0.5, f'品質スコア: {quality_metrics["quality_score"]:.3f}\n'
                                  f'相関差: {quality_metrics["correlation_difference"]:.3f}\n'
                                  f'合成サンプル数: {quality_metrics["n_synthetic_samples"]}',
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
       axes[1, 1].axis('off')
       
       plt.tight_layout()
       
       quality_path = os.path.join(output_dir, 'synthetic_data_quality.png')
       plt.savefig(quality_path, dpi=300, bbox_inches='tight')
       plt.close(fig)
       
   except Exception as e:
       print(f"品質可視化作成中にエラー: {e}")


def create_summary_report(analysis_results, output_dir):
   """
   分析結果のサマリーレポートを作成
   """
   if not output_dir:
       return
       
   try:
       report_path = os.path.join(output_dir, 'smote_analysis_summary.txt')
       
       with open(report_path, 'w', encoding='utf-8') as f:
           f.write("=== SMOTE効果の包括的分析レポート ===\n")
           f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
           
           # データサイズの情報
           if 'data_sizes' in analysis_results['summary']:
               data_info = analysis_results['summary']['data_sizes']
               f.write("1. データサイズの変化\n")
               f.write(f"   元データ: {data_info['original']} サンプル\n")
               f.write(f"   合成データ: {data_info['synthetic']} サンプル\n")
               f.write(f"   合計: {data_info['total']} サンプル\n")
               f.write(f"   増加率: {data_info['increase_ratio']:.2f}%\n\n")
           
           # 分布の変化
           if 'target' in analysis_results['distribution_changes']:
               target_change = analysis_results['distribution_changes']['target']
               f.write("2. ターゲット値の分布変化\n")
               f.write("   元データ:\n")
               f.write(f"     平均: {target_change['original']['mean']:.4f}\n")
               f.write(f"     標準偏差: {target_change['original']['std']:.4f}\n")
               f.write("   SMOTE適用後:\n")
               f.write(f"     平均: {target_change['smote']['mean']:.4f}\n")
               f.write(f"     標準偏差: {target_change['smote']['std']:.4f}\n\n")
           
           # 特徴量重要度の変化
           if 'feature_importance_changes' in analysis_results['metrics_comparison']:
               f.write("3. 特徴量重要度の主な変化\n")
               importance_changes = analysis_results['metrics_comparison']['feature_importance_changes']
               for feature, change in list(importance_changes.items())[:5]:
                   f.write(f"   {feature}: {change['change_percent']:.2f}% 変化\n")
               f.write("\n")
           
           # 保存されたファイル
           f.write("4. 生成されたファイル\n")
           for file_path in analysis_results['saved_files']:
               f.write(f"   - {os.path.basename(file_path)}\n")
           
           f.write("\n=== レポート終了 ===\n")
       
       print(f"サマリーレポートを保存しました: {report_path}")
       
   except Exception as e:
       print(f"サマリーレポート作成中にエラー: {e}")


def visualize_smote_data_distribution(original_data, smote_data, output_dir=None):
   """
   SMOTE適用前後の基本的なデータ分布を比較する関数
   
   Parameters:
   -----------
   original_data : dict
       元データの情報 {'features': DataFrame, 'target': Series}
   smote_data : dict
       SMOTE適用後データの情報 {'features': DataFrame, 'target': Series}
   output_dir : str, default=None
       保存先ディレクトリ
       
   Returns:
   --------
   str or None
       保存されたファイルのパス（output_dirが指定された場合）
   """
   try:
       fig, axes = plt.subplots(1, 3, figsize=(18, 6))
       
       # 1. ターゲット値の散布図表示
       n_original = len(original_data['features'])
       
       # 元データ
       axes[0].scatter(range(n_original), original_data['target'], alpha=0.6, s=20, label='元データ')
       axes[0].set_title('元データのターゲット値', fontsize=12)
       axes[0].set_xlabel('サンプルインデックス')
       axes[0].set_ylabel('ターゲット値')
       axes[0].grid(True, alpha=0.3)
       
       # SMOTE適用後のデータ（元データ + 合成データ）
       axes[1].scatter(range(n_original), smote_data['target'][:n_original], alpha=0.6, s=20, label='元データ', color='blue')
       axes[1].scatter(range(n_original, len(smote_data['target'])), smote_data['target'][n_original:], 
                      alpha=0.6, s=20, label='合成データ', color='red')
       axes[1].set_title('SMOTE適用後のターゲット値', fontsize=12)
       axes[1].set_xlabel('サンプルインデックス')
       axes[1].set_ylabel('ターゲット値')
       axes[1].legend()
       axes[1].grid(True, alpha=0.3)

       # 3. ヒストグラム比較
       axes[2].hist(original_data['target'], bins=30, alpha=0.7, label='元データ', density=True)
       axes[2].hist(smote_data['target'], bins=30, alpha=0.7, label='SMOTE適用後', density=True)
       axes[2].set_title('ターゲット値の分布比較', fontsize=12)
       axes[2].set_xlabel('ターゲット値')
       axes[2].set_ylabel('密度')
       axes[2].legend()
       axes[2].grid(True, alpha=0.3)

       plt.suptitle('SMOTE効果: データ分布の概要', fontsize=16)
       plt.tight_layout()

       if output_dir:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           save_path = os.path.join(output_dir, f'smote_distribution_overview_{timestamp}.png')
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
           plt.close(fig)
           print(f"SMOTE分布概要プロットを保存しました: {save_path}")
           return save_path
       else:
           plt.show()
           plt.close(fig)
           return None
   except Exception as e:
       print(f"SMOTE分布概要プロットの生成中にエラーが発生しました: {e}")
       if 'fig' in locals():
           plt.close(fig)
       return None
   
def compare_model_predictions(original_model, smote_model, test_features, test_target=None, output_dir=None):
    """
    SMOTE適用前後のモデルの予測結果を比較する関数
    
    Parameters:
    -----------
    original_model : 学習済みモデル
        元データで学習したモデル
    smote_model : 学習済みモデル
        SMOTE適用データで学習したモデル
    test_features : pandas.DataFrame
        テスト用特徴量データ
    test_target : pandas.Series, default=None
        実際のターゲット値（提供されている場合）
    output_dir : str, default=None
        保存先ディレクトリ
        
    Returns:
    --------
    dict
        比較結果を含む辞書
    """
    try:
        # 各モデルでの予測
        original_pred = original_model.predict(test_features)
        smote_pred = smote_model.predict(test_features)
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 予測値の比較散布図
        axes[0].scatter(original_pred, smote_pred, alpha=0.6, s=20)
        axes[0].plot([min(original_pred), max(original_pred)], 
                     [min(original_pred), max(original_pred)], 'r--', alpha=0.7)
        axes[0].set_xlabel('元データモデルの予測値')
        axes[0].set_ylabel('SMOTEモデルの予測値')
        axes[0].set_title('モデル間の予測値比較')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 予測値の分布比較
        axes[1].hist(original_pred, bins=50, alpha=0.7, label='元データモデル', density=True)
        axes[1].hist(smote_pred, bins=50, alpha=0.7, label='SMOTEモデル', density=True)
        axes[1].set_xlabel('予測値')
        axes[1].set_ylabel('密度')
        axes[1].set_title('予測値の分布比較')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 実際の値がある場合の比較
        if test_target is not None:
            # 誤差の比較
            original_error = original_pred - test_target
            smote_error = smote_pred - test_target
            
            axes[2].hist(original_error, bins=50, alpha=0.7, label='元データモデル', density=True)
            axes[2].hist(smote_error, bins=50, alpha=0.7, label='SMOTEモデル', density=True)
            axes[2].set_xlabel('誤差')
            axes[2].set_ylabel('密度')
            axes[2].set_title('予測誤差の分布比較')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            # 実際の値がない場合は予測差を表示
            pred_diff = smote_pred - original_pred
            axes[2].hist(pred_diff, bins=50, alpha=0.7)
            axes[2].set_xlabel('予測値の差 (SMOTE - 元)')
            axes[2].set_ylabel('頻度')
            axes[2].set_title('予測値の差の分布')
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('SMOTE効果: モデル予測の比較', fontsize=16)
        plt.tight_layout()
        
        # 結果の保存
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f'model_predictions_comparison_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"モデル予測比較プロットを保存しました: {save_path}")
        else:
            plt.show()
            plt.close(fig)
        
        # 統計情報の計算
        comparison_stats = {
            'original_pred_mean': np.mean(original_pred),
            'original_pred_std': np.std(original_pred),
            'smote_pred_mean': np.mean(smote_pred),
            'smote_pred_std': np.std(smote_pred),
            'correlation': np.corrcoef(original_pred, smote_pred)[0, 1],
            'mean_abs_diff': np.mean(np.abs(smote_pred - original_pred))
        }
        
        if test_target is not None:
            comparison_stats.update({
                'original_mse': mean_squared_error(test_target, original_pred),
                'smote_mse': mean_squared_error(test_target, smote_pred),
                'original_mae': mean_absolute_error(test_target, original_pred),
                'smote_mae': mean_absolute_error(test_target, smote_pred),
                'original_r2': r2_score(test_target, original_pred),
                'smote_r2': r2_score(test_target, smote_pred)
            })
        
        return comparison_stats
        
    except Exception as e:
        print(f"モデル予測比較中にエラーが発生しました: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None
    
def run_smote_analysis_pipeline(original_data, smote_data, feature_importances, 
                               original_model, smote_model, analysis_config=None, output_dir=None):
        """
        SMOTE分析のパイプライン実行関数
        
        Parameters:
        -----------
        original_data : dict
            元データの情報 {'features': DataFrame, 'target': Series}
        smote_data : dict
            SMOTE適用後データの情報 {'features': DataFrame, 'target': Series}
        feature_importances : pandas.DataFrame
            特徴量重要度（'feature'と'importance'列を含む）
        original_model : 学習済みモデル
            元データで学習したモデル
        smote_model : 学習済みモデル
            SMOTE適用データで学習したモデル
        analysis_config : dict, default=None
            分析設定（どの分析を実行するかの設定）
        output_dir : str, default=None
            保存先ディレクトリ
            
        Returns:
        --------
        dict
            パイプライン実行結果
        """
        # デフォルト設定
        default_config = {
            'run_comprehensive_analysis': True,
            'run_pdp_comparison': True,
            'run_distribution_analysis': True,
            'run_model_comparison': True,
            'top_features_for_pdp': 3,
            'create_individual_pdp_plots': True
        }
        
        if analysis_config is None:
            analysis_config = default_config
        else:
            # 設定のマージ
            for key, value in default_config.items():
                if key not in analysis_config:
                    analysis_config[key] = value
        
        print("=== SMOTE分析パイプライン開始 ===")
        print(f"設定: {analysis_config}")
        
        pipeline_results = {
            'config': analysis_config,
            'results': {},
            'saved_files': [],
            'errors': []
        }
        
        try:
            # 1. 包括的分析の実行
            if analysis_config.get('run_comprehensive_analysis', True):
                print("\n1. 包括的分析を実行中...")
                comprehensive_results = analyze_smote_effect_comprehensive(
                    original_data, smote_data, feature_importances,
                    original_model, smote_model, output_dir
                )
                pipeline_results['results']['comprehensive'] = comprehensive_results
                pipeline_results['saved_files'].extend(comprehensive_results.get('saved_files', []))
        
        except Exception as e:
            error_msg = f"包括的分析中にエラー: {e}"
            print(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        try:
            # 2. 基本的な分布分析
            if analysis_config.get('run_distribution_analysis', True):
                print("\n2. 基本分布分析を実行中...")
                distribution_result = visualize_smote_data_distribution(
                    original_data, smote_data, output_dir
                )
                if distribution_result:
                    pipeline_results['saved_files'].append(distribution_result)
                    pipeline_results['results']['distribution'] = {'saved_file': distribution_result}
        
        except Exception as e:
            error_msg = f"分布分析中にエラー: {e}"
            print(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        try:
            # 3. 個別特徴量のPDP比較
            if analysis_config.get('create_individual_pdp_plots', True):
                print("\n3. 個別特徴量のPDP比較を実行中...")
                top_n = analysis_config.get('top_features_for_pdp', 3)
                top_features = feature_importances.head(top_n)['feature'].tolist()
                
                pdp_results = {}
                for feature in top_features:
                    print(f"   - {feature}の部分依存プロット作成中...")
                    pdp_result = visualize_smote_effect_with_pdp(
                        original_model, smote_model,
                        original_data['features'], smote_data['features'],
                        feature, output_dir
                    )
                    if pdp_result:
                        pdp_results[feature] = pdp_result
                        pipeline_results['saved_files'].append(pdp_result)
                
                pipeline_results['results']['individual_pdp'] = pdp_results
        
        except Exception as e:
            error_msg = f"個別PDP分析中にエラー: {e}"
            print(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        try:
            # 4. モデル予測比較（実装されている場合）
            if analysis_config.get('run_model_comparison', True):
                print("\n4. モデル予測比較を実行中...")
                if 'compare_model_predictions' in dir():
                    # テストデータがあると仮定（実際の使用では適切なテストデータを指定）
                    # この部分は実装に応じて調整が必要
                    print("   注意: モデル予測比較にはテストデータが必要です")
                    pipeline_results['results']['model_comparison'] = {
                        'note': 'テストデータが必要です',
                        'status': 'skipped'
                    }
        
        except Exception as e:
            error_msg = f"モデル比較中にエラー: {e}"
            print(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        # 5. パイプライン結果のサマリー作成
        try:
            if output_dir:
                pipeline_summary_path = os.path.join(output_dir, 'smote_pipeline_summary.txt')
                create_pipeline_summary(pipeline_results, pipeline_summary_path)
                pipeline_results['saved_files'].append(pipeline_summary_path)
        
        except Exception as e:
            error_msg = f"パイプラインサマリー作成中にエラー: {e}"
            print(error_msg)
            pipeline_results['errors'].append(error_msg)
        
        print("\n=== SMOTE分析パイプライン完了 ===")
        print(f"生成されたファイル数: {len(pipeline_results['saved_files'])}")
        if pipeline_results['errors']:
            print(f"エラー数: {len(pipeline_results['errors'])}")
            for error in pipeline_results['errors']:
                print(f"  - {error}")
        
        return pipeline_results


def create_pipeline_summary(pipeline_results, output_path):
        """
        パイプライン実行結果のサマリーを作成
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== SMOTE分析パイプライン実行サマリー ===\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 設定情報
                f.write("1. 実行設定\n")
                for key, value in pipeline_results['config'].items():
                    f.write(f"   {key}: {value}\n")
                f.write("\n")
                
                # 実行結果
                f.write("2. 実行結果\n")
                for analysis_type, result in pipeline_results['results'].items():
                    f.write(f"   - {analysis_type}: {'実行済み' if result else '未実行'}\n")
                f.write("\n")
                
                # 生成ファイル
                f.write("3. 生成されたファイル\n")
                for file_path in pipeline_results['saved_files']:
                    f.write(f"   - {os.path.basename(file_path)}\n")
                f.write("\n")
                
                # エラー情報
                if pipeline_results['errors']:
                    f.write("4. エラー情報\n")
                    for error in pipeline_results['errors']:
                        f.write(f"   - {error}\n")
                
                f.write("\n=== サマリー終了 ===\n")
            
            print(f"パイプラインサマリーを保存しました: {output_path}")
            
        except Exception as e:
            print(f"パイプラインサマリー作成中にエラー: {e}")
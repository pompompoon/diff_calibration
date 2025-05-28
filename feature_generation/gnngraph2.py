import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import japanize_matplotlib

# ファイルパスの設定
file_path = r"G:\共有ドライブ\GAP_長寿研\GAPデータ\20241009101605_1\eyetracking_1_20241009_20241119145059.csv"
# 出力ディレクトリの設定
output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\graph_data"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# アイトラッキングデータの読み込み
def load_eye_data(file_path):
    # ファイルの存在確認
    if not os.path.exists(file_path):
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None
    
    try:
        # カンマ区切りCSVファイルの読み込み
        df = pd.read_csv(file_path)
        
        # 列名を表示
        print("ファイルの列名:", df.columns.tolist())
        
        # 最初の数行を表示
        print("\nデータサンプル:")
        print(df.head())
        
        # 列が存在するか確認
        if 'LeftGazeX' in df.columns and 'LeftGazeY' in df.columns:
            # 左目のデータがすべてゼロかチェック
            if (df['LeftGazeX'] == 0).all() and (df['LeftGazeY'] == 0).all():
                print("左目のデータがすべてゼロです。右目のデータを使用します。")
                df['LeftGazeX'] = df['RightGazeX']
                df['LeftGazeY'] = df['RightGazeY']
        else:
            print("LeftGazeXまたはLeftGazeYが見つかりません。")
            return None
        
        # 目が閉じているデータを除外
        if 'CloseEye' in df.columns:
            df = df[df['CloseEye'] == 0].reset_index(drop=True)
        
        print(f"データ読み込み完了: {len(df)} レコード")
        return df
    except Exception as e:
        print(f"データ読み込みエラー: {str(e)}")
        return None

# 注視点の特定（速度閾値アルゴリズムを使用）
def identify_fixations(df, velocity_threshold=50, min_fixation_duration=100):
    print("注視点の特定を開始...")
    
    # データに必要な列があるか確認
    required_cols = ['LeftGazeX', 'LeftGazeY', 'TimeStamp']
    for col in required_cols:
        if col not in df.columns:
            print(f"エラー: 必要な列 '{col}' がデータに見つかりません。")
            return []
    
    # 時間差分の計算
    df['time_diff'] = df['TimeStamp'].diff().fillna(0)
    
    # 連続するポイント間の距離を計算
    df['distance'] = np.sqrt(
        (df['LeftGazeX'].diff().fillna(0))**2 + 
        (df['LeftGazeY'].diff().fillna(0))**2
    )
    
    # 速度を計算（距離/時間、単位はミリ秒）
    df['velocity'] = np.where(df['time_diff'] > 0, 
                             df['distance'] / (df['time_diff'] / 1000), 
                             0)
    
    # 注視点候補（低速度ポイント）の特定
    df['is_fixation'] = df['velocity'] < velocity_threshold
    
    # 連続する注視点をグループ化
    fixation_groups = []
    current_group = []
    
    for i, row in df.iterrows():
        if row['is_fixation']:
            current_group.append(i)
        else:
            if len(current_group) > 0:
                fixation_groups.append(current_group)
                current_group = []
    
    # 最後のグループを追加
    if len(current_group) > 0:
        fixation_groups.append(current_group)
    
    # 持続時間でフィルタリング
    valid_fixations = []
    
    for group in fixation_groups:
        group_df = df.loc[group]
        duration = group_df['TimeStamp'].iloc[-1] - group_df['TimeStamp'].iloc[0]
        
        if duration >= min_fixation_duration:
            # 平均位置を計算
            avg_x = group_df['LeftGazeX'].mean()
            avg_y = group_df['LeftGazeY'].mean()
            start_time = group_df['TimeStamp'].iloc[0]
            end_time = group_df['TimeStamp'].iloc[-1]
            
            valid_fixations.append({
                'x': avg_x,
                'y': avg_y,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'indices': group
            })
    
    print(f"特定された注視点: {len(valid_fixations)}")
    return valid_fixations

# 注視点からグラフを作成（方向成分を含む）
def create_gaze_graph(fixations, df):
    print("視線グラフの作成中...")
    
    if not fixations:
        print("警告: 注視点が見つかりませんでした。グラフは作成できません。")
        return nx.DiGraph()  # 空のグラフを返す
    
    # 有向グラフを作成
    G = nx.DiGraph()
    
    # ノード（注視点）の追加
    for i, fixation in enumerate(fixations):
        G.add_node(i, 
                  pos=(fixation['x'], fixation['y']),
                  x=fixation['x'],
                  y=fixation['y'],
                  duration=fixation['duration'],
                  start_time=fixation['start_time'],
                  end_time=fixation['end_time'])
    
    # エッジ（サッカード - 注視点間の速い眼球運動）の追加
    for i in range(len(fixations) - 1):
        current_fixation = fixations[i]
        next_fixation = fixations[i+1]
        
        # 開始終了インデックスの抽出
        end_index = current_fixation['indices'][-1]
        start_index = next_fixation['indices'][0]
        
        # サッカードデータの計算
        saccade_df = df.iloc[end_index:start_index+1]
        
        # X方向とY方向の距離
        dx = next_fixation['x'] - current_fixation['x']
        dy = next_fixation['y'] - current_fixation['y']
        
        # 合成距離
        saccade_distance = np.sqrt(dx**2 + dy**2)
        
        # 持続時間
        saccade_duration = next_fixation['start_time'] - current_fixation['end_time']
        
        # 注視点間に時間差がある場合のみエッジを追加
        if saccade_duration > 0:
            # X方向、Y方向、合成の速度を計算（単位/秒）
            velocity_x = dx / (saccade_duration / 1000)
            velocity_y = dy / (saccade_duration / 1000)
            velocity = saccade_distance / (saccade_duration / 1000)
            
            # 加速度の計算（十分なポイントがある場合）
            accelerations = []
            if 'velocity' in df.columns and len(saccade_df) > 2:
                velocities = saccade_df['velocity'].values
                times = saccade_df['TimeStamp'].values / 1000  # 秒に変換
                
                if len(velocities) > 3:  # スムージングに十分なポイント
                    window_size = min(5, len(velocities))
                    # 奇数であることを確認
                    window_size = window_size if window_size % 2 == 1 else window_size - 1
                    if window_size >= 3:  # 最小ウィンドウサイズは3
                        smoothed_velocities = savgol_filter(velocities, window_size, 1)
                        for j in range(1, len(times)):
                            if times[j] - times[j-1] > 0:
                                acc = (smoothed_velocities[j] - smoothed_velocities[j-1]) / (times[j] - times[j-1])
                                accelerations.append(acc)
            
            mean_acceleration = np.mean(accelerations) if accelerations else 0
            
            # エッジの追加（ノード間の接続）
            G.add_edge(i, i+1, 
                      distance=saccade_distance,
                      dx=dx,
                      dy=dy,
                      duration=saccade_duration,
                      velocity_x=velocity_x,
                      velocity_y=velocity_y,
                      velocity=velocity,
                      acceleration=mean_acceleration)
    
    print(f"グラフ作成完了: ノード {G.number_of_nodes()} 個, エッジ {G.number_of_edges()} 個")
    return G

# 視線グラフの可視化
def plot_gaze_graph(G, fixations, output_file=None):
    if output_file is None:
        output_file = os.path.join(output_dir, "gaze_graph.png")
    
    print("グラフの可視化...")
    
    if not fixations or G.number_of_nodes() == 0:
        print("警告: 可視化するグラフがありません。")
        return
    
    plt.figure(figsize=(12, 10))
    
    # ノードの位置を取得
    pos = nx.get_node_attributes(G, 'pos')
    
    # ノードを描画（サイズは注視時間に比例）
    node_sizes = [max(100, fixation['duration'] / 10) for fixation in fixations]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # ノードラベルの描画
    labels = {i: f"({fixation['x']:.1f}, {fixation['y']:.1f})" for i, fixation in enumerate(fixations)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # エッジの描画（太さは速度に比例）
    edges = G.edges()
    if edges:
        edge_velocities = [G[u][v]['velocity'] for u, v in edges]
        
        # 速度の正規化（エッジの太さ用）
        max_vel = max(edge_velocities) if edge_velocities else 1
        edge_widths = [1 + 5 * (vel / max_vel) for vel in edge_velocities]
        
        # エッジの描画
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                              edge_color='blue', arrows=True, arrowsize=15)
        
        # エッジラベルの描画
        edge_labels = {(u, v): f"vx={G[u][v]['velocity_x']:.1f}\nvy={G[u][v]['velocity_y']:.1f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("目の視線グラフ: ノード=注視点, エッジ=サッカード")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"グラフを {output_file} に保存しました")
    plt.show()

# グラフデータをCSVファイルに出力（方向成分を含む）
def export_graph_data(G, fixations, output_dir=output_dir):
    print("グラフデータのエクスポート...")
    
    if not fixations or G.number_of_nodes() == 0:
        print("警告: エクスポートするグラフデータがありません。")
        return None, None
    
    # ノードデータの準備
    node_data = []
    for i, fixation in enumerate(fixations):
        node_data.append({
            'node_id': i,
            'node_x': fixation['x'],
            'node_y': fixation['y'],
            'duration': fixation['duration'],
            'start_time': fixation['start_time'],
            'end_time': fixation['end_time']
        })
    
    # エッジデータの準備
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'edge_vx': data['velocity_x'],  # X方向の速度
            'edge_vy': data['velocity_y'],  # Y方向の速度
            'edge_v': data['velocity'],     # 合成速度
            'dx': data['dx'],               # X方向の距離
            'dy': data['dy'],               # Y方向の距離
            'distance': data['distance'],   # 合成距離
            'duration': data['duration'],   # 持続時間
            'acceleration': data['acceleration']  # 加速度
        })
    
    # DataFrameの作成
    nodes_df = pd.DataFrame(node_data)
    edges_df = pd.DataFrame(edge_data)
    
    # CSVファイルへの出力
    nodes_file = os.path.join(output_dir, 'gaze_nodes.csv')
    edges_file = os.path.join(output_dir, 'gaze_edges.csv')
    
    nodes_df.to_csv(nodes_file, index=False)
    edges_df.to_csv(edges_file, index=False)
    
    print(f"ノード {len(nodes_df)} 個とエッジ {len(edges_df)} 個のデータをCSVファイルにエクスポートしました")
    print(f"ノードデータを {nodes_file} に保存しました")
    print(f"エッジデータを {edges_file} に保存しました")
    
    return nodes_df, edges_df

# 指定形式でのグラフデータ出力
def export_graph_data_combined(G, fixations, output_dir=output_dir):
    print("グラフデータを指定形式でエクスポート...")
    
    if not fixations or G.number_of_nodes() == 0:
        print("警告: エクスポートするグラフデータがありません。")
        return None
    
    # 結合データの準備
    combined_data = []
    
    # ノードデータを追加
    for i, fixation in enumerate(fixations):
        row = {
            'ID': i+1,
            'node_x': fixation['x'],
            'node_y': fixation['y'],
            'edge_vx': None,
            'edge_vy': None,
            'edge_v': None
        }
        combined_data.append(row)
    
    # エッジデータを追加
    for u, v, data in G.edges(data=True):
        # u番目のノードの行を更新
        combined_data[u]['edge_vx'] = data['velocity_x']
        combined_data[u]['edge_vy'] = data['velocity_y']
        combined_data[u]['edge_v'] = data['velocity']
    
    # DataFrameの作成
    combined_df = pd.DataFrame(combined_data)
    
    # CSVファイルへの出力
    combined_file = os.path.join(output_dir, 'gaze_graph_combined.csv')
    combined_df.to_csv(combined_file, index=False)
    
    print(f"結合データを {combined_file} に保存しました")
    
    return combined_df

# メイン関数
def main(file_path=file_path, output_dir=output_dir):
    # データの読み込み
    df = load_eye_data(file_path)
    if df is None:
        return None, None, None, None, None
    
    # 注視点の特定
    fixations = identify_fixations(df)
    
    # グラフの作成
    G = create_gaze_graph(fixations, df)
    
    # グラフの可視化
    plot_gaze_graph(G, fixations, os.path.join(output_dir, "gaze_graph.png"))
    
    # グラフデータのエクスポート
    nodes_df, edges_df = export_graph_data(G, fixations, output_dir)
    
    # 指定形式でグラフデータをエクスポート
    combined_df = export_graph_data_combined(G, fixations, output_dir)
    
    return G, fixations, nodes_df, edges_df, combined_df

if __name__ == "__main__":
    G, fixations, nodes_df, edges_df, combined_df = main()
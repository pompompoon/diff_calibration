import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.signal import savgol_filter


# 基本ディレクトリの設定
base_dir = r"G:\共有ドライブ\GAP_長寿研\GAPデータ"
# 出力ディレクトリの設定
output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\graph_data2"

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
        print(f"ファイル '{file_path}' の列名:", df.columns.tolist())
        
        # 必要な列が存在するか確認
        required_cols = ['LeftGazeX', 'LeftGazeY', 'RightGazeX', 'RightGazeY', 'TimeStamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"エラー: 必要な列 {missing_cols} がデータに見つかりません。")
            return None
        
        # 外れ値（-100未満）を除外
        df = df[(df['LeftGazeX'] > -100) & (df['LeftGazeY'] > -100) & 
                (df['RightGazeX'] > -100) & (df['RightGazeY'] > -100)].reset_index(drop=True)
        
        # 目が閉じているデータを除外
        if 'CloseEye' in df.columns:
            df = df[df['CloseEye'] == 0].reset_index(drop=True)
        
        print(f"データ読み込み完了: {len(df)} レコード")
        return df
    except Exception as e:
        print(f"データ読み込みエラー: {str(e)}")
        return None

# 左右の目のデータを別々に処理するための関数
def process_eye_data(df, eye_side):
    if eye_side == 'Left':
        x_col, y_col = 'LeftGazeX', 'LeftGazeY'
    else:
        x_col, y_col = 'RightGazeX', 'RightGazeY'
    
    # 注視点と視線グラフを作成
    fixations = identify_fixations(df, x_col, y_col)
    G = create_gaze_graph(fixations, df, x_col, y_col)
    
    return fixations, G

# 注視点の特定（速度閾値アルゴリズムを使用）- 左右の目に対応
def identify_fixations(df, x_col, y_col, velocity_threshold=50, min_fixation_duration=100):
    print(f"{x_col}/{y_col} の注視点の特定を開始...")
    
    # 時間差分の計算
    df['time_diff'] = df['TimeStamp'].diff().fillna(0)
    
    # 連続するポイント間の距離を計算
    df['distance'] = np.sqrt(
        (df[x_col].diff().fillna(0))**2 + 
        (df[y_col].diff().fillna(0))**2
    )
    
    # 速度を計算（距離/時間、単位はミリ秒）
    df['velocity'] = np.where(df['time_diff'] > 0, 
                             df['distance'] / (df['time_diff'] / 1000), 
                             0)
    
    # X方向とY方向の速度を計算
    df['velocity_x'] = np.where(df['time_diff'] > 0,
                               (df[x_col].diff().fillna(0)) / (df['time_diff'] / 1000),
                               0)
    
    df['velocity_y'] = np.where(df['time_diff'] > 0,
                               (df[y_col].diff().fillna(0)) / (df['time_diff'] / 1000),
                               0)
    
    # 加速度を計算（速度の変化/時間）
    df['acceleration_x'] = df['velocity_x'].diff().fillna(0) / (df['time_diff'] / 1000).replace(0, np.nan).fillna(df['time_diff'].median() / 1000)
    df['acceleration_y'] = df['velocity_y'].diff().fillna(0) / (df['time_diff'] / 1000).replace(0, np.nan).fillna(df['time_diff'].median() / 1000)
    df['acceleration'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    
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
        if not group:  # 空のグループをスキップ
            continue
        
        group_df = df.loc[group]
        if len(group_df) < 2:  # 少なくとも2つのポイントが必要
            continue
            
        duration = group_df['TimeStamp'].iloc[-1] - group_df['TimeStamp'].iloc[0]
        
        if duration >= min_fixation_duration:
            # 平均位置を計算
            avg_x = group_df[x_col].mean()
            avg_y = group_df[y_col].mean()
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

# 注視点からグラフを作成（方向成分と加速度成分を含む）- 左右の目に対応
def create_gaze_graph(fixations, df, x_col, y_col):
    eye_side = "左目" if x_col == 'LeftGazeX' else "右目"
    print(f"{eye_side}の視線グラフの作成中...")
    
    if not fixations:
        print(f"警告: {eye_side}の注視点が見つかりませんでした。グラフは作成できません。")
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
            # 持続時間（秒単位）
            saccade_duration_sec = saccade_duration / 1000
            
            # X方向、Y方向、合成の速度を計算（単位/秒）
            velocity_x = dx / saccade_duration_sec
            velocity_y = dy / saccade_duration_sec
            velocity = saccade_distance / saccade_duration_sec
            
            # 速度と加速度の計算（十分なポイントがある場合）
            velocities_x = []
            velocities_y = []
            velocities = []
            accelerations_x = []
            accelerations_y = []
            accelerations = []
            
            if len(saccade_df) > 2:
                # サッカード区間の速度と加速度を収集
                if 'velocity_x' in saccade_df.columns and 'velocity_y' in saccade_df.columns:
                    velocities_x = saccade_df['velocity_x'].values
                    velocities_y = saccade_df['velocity_y'].values
                    velocities = saccade_df['velocity'].values
                
                if 'acceleration_x' in saccade_df.columns and 'acceleration_y' in saccade_df.columns:
                        # 既存のリストを拡張する
                        accelerations_x.extend(saccade_df['acceleration_x'].values)
                        accelerations_y.extend(saccade_df['acceleration_y'].values)
                        accelerations.extend(saccade_df['acceleration'].values)
                                
                # 時間データを秒に変換
                times = saccade_df['TimeStamp'].values / 1000  # 秒に変換
                
                # 平滑化（十分なポイントがある場合のみ）
                if len(velocities) > 3:  # スムージングに十分なポイント
                    window_size = min(5, len(velocities))
                    # 奇数であることを確認
                    window_size = window_size if window_size % 2 == 1 else window_size - 1
                    if window_size >= 3:  # 最小ウィンドウサイズは3
                        smoothed_velocities_x = savgol_filter(velocities_x, window_size, 1)
                        smoothed_velocities_y = savgol_filter(velocities_y, window_size, 1)
                        smoothed_velocities = savgol_filter(velocities, window_size, 1)
                        
                        # 加速度の再計算（平滑化された速度から）
                        for j in range(1, len(times)):
                            if times[j] - times[j-1] > 0:
                                dt = times[j] - times[j-1]
                                acc_x = (smoothed_velocities_x[j] - smoothed_velocities_x[j-1]) / dt
                                acc_y = (smoothed_velocities_y[j] - smoothed_velocities_y[j-1]) / dt
                                acc = (smoothed_velocities[j] - smoothed_velocities[j-1]) / dt
                                accelerations_x.append(acc_x)
                                accelerations_y.append(acc_y)
                                accelerations.append(acc)
                                
            # サッカードの平均加速度と最大加速度を計算
            mean_acceleration_x = np.mean(accelerations_x) if accelerations_x else 0
            mean_acceleration_y = np.mean(accelerations_y) if accelerations_y else 0
            mean_acceleration = np.mean(accelerations) if accelerations else 0
            
            max_acceleration_x = np.max(np.abs(accelerations_x)) if accelerations_x else 0
            max_acceleration_y = np.max(np.abs(accelerations_y)) if accelerations_y else 0
            max_acceleration = np.max(np.abs(accelerations)) if accelerations else 0
            
            # エッジの追加（ノード間の接続）- 加速度成分を追加
            G.add_edge(i, i+1, 
                      distance=saccade_distance,
                      dx=dx,
                      dy=dy,
                      duration=saccade_duration,
                      velocity_x=velocity_x,
                      velocity_y=velocity_y,
                      velocity=velocity,
                      acceleration_x=mean_acceleration_x,  # X方向の平均加速度
                      acceleration_y=mean_acceleration_y,  # Y方向の平均加速度
                      acceleration=mean_acceleration,      # 合成平均加速度
                      max_acceleration_x=max_acceleration_x,  # X方向の最大加速度
                      max_acceleration_y=max_acceleration_y,  # Y方向の最大加速度
                      max_acceleration=max_acceleration)      # 合成最大加速度
    
    print(f"{eye_side}グラフ作成完了: ノード {G.number_of_nodes()} 個, エッジ {G.number_of_edges()} 個")
    return G

# 指定形式でのグラフデータ出力 - 左右の目用に修正、加速度成分を追加
def export_graph_data_combined(left_G, left_fixations, right_G, right_fixations, folder_name):
    print(f"グラフデータを指定形式でエクスポート (フォルダ: {folder_name})...")
    
    if ((not left_fixations or left_G.number_of_nodes() == 0) and 
        (not right_fixations or right_G.number_of_nodes() == 0)):
        print("警告: エクスポートするグラフデータがありません。")
        return None
    
    # 結合データの準備
    combined_data = []
    
    # 左目のデータを処理
    if left_fixations and left_G.number_of_nodes() > 0:
        for i, fixation in enumerate(left_fixations):
            row = {
                'ID': folder_name,  # フォルダ名をIDとして使用
                'eye': 'Left',
                'node_x': fixation['x'],
                'node_y': fixation['y'],
                'edge_vx': None,
                'edge_vy': None,
                'edge_v': None,
                'edge_ax': None,  # X方向の加速度
                'edge_ay': None,  # Y方向の加速度
                'edge_a': None    # 合成加速度
            }
            combined_data.append(row)
            
        # 左目のエッジデータを追加
        left_node_count = 0
        for u, v, data in left_G.edges(data=True):
            # u番目のノードの行を更新
            combined_data[left_node_count]['edge_vx'] = data['velocity_x']
            combined_data[left_node_count]['edge_vy'] = data['velocity_y']
            combined_data[left_node_count]['edge_v'] = data['velocity']
            # 加速度成分を追加
            combined_data[left_node_count]['edge_ax'] = data['acceleration_x']
            combined_data[left_node_count]['edge_ay'] = data['acceleration_y']
            combined_data[left_node_count]['edge_a'] = data['acceleration']
            left_node_count += 1
    
    # 右目のデータを処理
    if right_fixations and right_G.number_of_nodes() > 0:
        for i, fixation in enumerate(right_fixations):
            row = {
                'ID': folder_name,  # フォルダ名をIDとして使用
                'eye': 'Right',
                'node_x': fixation['x'],
                'node_y': fixation['y'],
                'edge_vx': None,
                'edge_vy': None,
                'edge_v': None,
                'edge_ax': None,  # X方向の加速度
                'edge_ay': None,  # Y方向の加速度
                'edge_a': None    # 合成加速度
            }
            combined_data.append(row)
            
        # 右目のエッジデータを追加
        right_node_count = len(left_fixations) if left_fixations else 0
        for u, v, data in right_G.edges(data=True):
            # u番目のノードの行を更新 (右目データのインデックス計算)
            index = right_node_count + u
            if index < len(combined_data):
                combined_data[index]['edge_vx'] = data['velocity_x']
                combined_data[index]['edge_vy'] = data['velocity_y']
                combined_data[index]['edge_v'] = data['velocity']
                # 加速度成分を追加
                combined_data[index]['edge_ax'] = data['acceleration_x']
                combined_data[index]['edge_ay'] = data['acceleration_y']
                combined_data[index]['edge_a'] = data['acceleration']
    
    # DataFrameの作成
    combined_df = pd.DataFrame(combined_data)
    return combined_df

# 全フォルダの処理関数
def process_all_folders(base_dir):
    print(f"基本ディレクトリ: {base_dir} の処理を開始...")
    
    # 全結果を保存するDataFrame
    all_results = pd.DataFrame()
    
    # 基本ディレクトリ内のすべてのフォルダをチェック
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # ディレクトリのみを処理
        if not os.path.isdir(folder_path):
            continue
        
        print(f"\n{'='*50}")
        print(f"フォルダ処理中: {folder_name}")
        print(f"{'='*50}")
        
        # フォルダ内のCSVファイルを検索
        csv_found = False
        for file in os.listdir(folder_path):
            if file.startswith("eyetracking") and file.endswith(".csv"):
                csv_path = os.path.join(folder_path, file)
                print(f"アイトラッキングCSVファイルを検出: {file}")
                
                # データの処理
                df = load_eye_data(csv_path)
                if df is not None:
                    # 左目のデータ処理
                    left_fixations, left_G = process_eye_data(df, 'Left')
                    
                    # 右目のデータ処理
                    right_fixations, right_G = process_eye_data(df, 'Right')
                    
                    # フォルダ名をIDとしてデータをエクスポート（左目と右目の両方）
                    result_df = export_graph_data_combined(left_G, left_fixations, right_G, right_fixations, folder_name)
                    
                    if result_df is not None:
                        # 結果を全体のDataFrameに追加
                        all_results = pd.concat([all_results, result_df], ignore_index=True)
                        print(f"フォルダ {folder_name} のデータを結果に追加しました。")
                        csv_found = True
        
        if not csv_found:
            print(f"警告: フォルダ {folder_name} には適切なCSVファイルが見つかりませんでした。")
    
    return all_results

# メイン関数
def main():
    # すべてのフォルダを処理
    all_results = process_all_folders(base_dir)
    
    if len(all_results) > 0:
        # 結果をCSVに保存
        output_file = os.path.join(output_dir, "all_gaze_data.csv")
        all_results.to_csv(output_file, index=False)
        print(f"\n全てのデータを {output_file} に保存しました。合計 {len(all_results)} 行")
        
        # 左目と右目のデータを分けて保存
        left_data = all_results[all_results['eye'] == 'Left']
        right_data = all_results[all_results['eye'] == 'Right']
        
        left_output = os.path.join(output_dir, "left_eye_gaze_data.csv")
        right_output = os.path.join(output_dir, "right_eye_gaze_data.csv")
        
        left_data.to_csv(left_output, index=False)
        right_data.to_csv(right_output, index=False)
        
        print(f"左目データを {left_output} に保存しました。合計 {len(left_data)} 行")
        print(f"右目データを {right_output} に保存しました。合計 {len(right_data)} 行")
    else:
        print("\n警告: 処理可能なデータが見つかりませんでした。")

if __name__ == "__main__":
    main()
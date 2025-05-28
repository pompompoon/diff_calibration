import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import glob
import os
import json
import shutil
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
import matplotlib.pyplot as plt
from datetime import datetime

# 1. データの前処理とグラフ構築
class EyeMovementGraphBuilder:
    def __init__(self, window_size=50, overlap=25, edge_threshold=5.0):
        """
        眼球運動データからグラフを構築するクラス
        
        Args:
            window_size: 時間窓のサイズ（フレーム数）
            overlap: 窓の重なり（フレーム数）
            edge_threshold: エッジを作成する閾値（角度）
        """
        self.window_size = window_size
        self.overlap = overlap
        self.edge_threshold = edge_threshold
        
    def create_graph_from_trajectory(self, timestamps, x_coords, y_coords):
        """
        眼球運動の軌跡からグラフを構築
        
        Args:
            timestamps: タイムスタンプの配列
            x_coords: x座標の配列
            y_coords: y座標の配列
            
        Returns:
            Data: PyTorch Geometricのグラフオブジェクト
        """
        # 1. 固定点（fixation）とサッケード（saccade）の検出
        fixations, saccades = self._detect_fixations_and_saccades(
            timestamps, x_coords, y_coords
        )
        
        # 2. ノード特徴量の計算
        node_features = self._compute_node_features(fixations, saccades)
        
        # 3. エッジの構築
        edge_index, edge_attr = self._build_edges(fixations, saccades)
        
        # PyTorch Geometricのデータ形式に変換
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _detect_fixations_and_saccades(self, timestamps, x_coords, y_coords):
        """
        I-VT（速度閾値）アルゴリズムを使用して固定点とサッケードを検出
        注：入力が角度データの場合
        """
        # タイムスタンプを秒に変換（必要に応じて）
        timestamps = np.array(timestamps) / 1000.0  # ミリ秒から秒へ
        
        # 速度計算（角速度）
        velocities = []
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 0.001  # ゼロ除算を防ぐ
            
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            angular_velocity = np.sqrt(dx**2 + dy**2) / dt
            velocities.append(angular_velocity)
        
        # 速度閾値（視角速度30度/秒）
        velocity_threshold = 30.0
        
        fixations = []
        saccades = []
        
        # 固定点とサッケードの分離
        i = 0
        while i < len(velocities):
            if velocities[i] < velocity_threshold:
                # 固定点の開始
                start_idx = i
                while i < len(velocities) and velocities[i] < velocity_threshold:
                    i += 1
                end_idx = i
                
                # 固定点の情報を保存（最小持続時間: 100ms）
                if end_idx - start_idx >= 3 and (timestamps[end_idx] - timestamps[start_idx]) >= 0.1:
                    fixations.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': timestamps[end_idx] - timestamps[start_idx],
                        'mean_x': np.mean(x_coords[start_idx:end_idx]),
                        'mean_y': np.mean(y_coords[start_idx:end_idx]),
                        'std_x': np.std(x_coords[start_idx:end_idx]),
                        'std_y': np.std(y_coords[start_idx:end_idx])
                    })
            else:
                # サッケードの開始
                start_idx = i
                while i < len(velocities) and velocities[i] >= velocity_threshold:
                    i += 1
                end_idx = i
                
                if end_idx > start_idx:
                    # サッケードの情報を保存
                    saccades.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': timestamps[end_idx] - timestamps[start_idx],
                        'amplitude': np.sqrt(
                            (x_coords[end_idx] - x_coords[start_idx])**2 +
                            (y_coords[end_idx] - y_coords[start_idx])**2
                        ),
                        'peak_velocity': np.max(velocities[start_idx:end_idx]) if start_idx < end_idx else 0,
                        'direction': np.arctan2(
                            y_coords[end_idx] - y_coords[start_idx],
                            x_coords[end_idx] - x_coords[start_idx]
                        )
                    })
        
        return fixations, saccades
    
    def _compute_node_features(self, fixations, saccades):
        """
        各ノード（固定点）の特徴量を計算
        """
        node_features = []
        
        for i, fix in enumerate(fixations):
            features = [
                fix['mean_x'],      # 平均x座標
                fix['mean_y'],      # 平均y座標
                fix['duration'],    # 持続時間
                fix['std_x'],       # x方向の標準偏差
                fix['std_y'],       # y方向の標準偏差
            ]
            
            # 前後のサッケードの特徴
            if i > 0:
                prev_saccade = saccades[i-1] if i-1 < len(saccades) else None
                if prev_saccade:
                    features.extend([
                        prev_saccade['amplitude'],
                        prev_saccade['peak_velocity'],
                        prev_saccade['direction']
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            if i < len(fixations) - 1:
                next_saccade = saccades[i] if i < len(saccades) else None
                if next_saccade:
                    features.extend([
                        next_saccade['amplitude'],
                        next_saccade['peak_velocity'],
                        next_saccade['direction']
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            node_features.append(features)
        
        return np.array(node_features)
    
    def _build_edges(self, fixations, saccades):
        """
        グラフのエッジを構築
        """
        edge_index = []
        edge_attr = []
        
        # 1. 時系列順序に基づくエッジ
        for i in range(len(fixations) - 1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])  # 双方向
            
            # エッジ特徴量（サッケードの情報）
            if i < len(saccades):
                edge_attr.append([
                    saccades[i]['amplitude'],
                    saccades[i]['duration'],
                    saccades[i]['peak_velocity']
                ])
                edge_attr.append([
                    saccades[i]['amplitude'],
                    saccades[i]['duration'],
                    saccades[i]['peak_velocity']
                ])
            else:
                edge_attr.append([0, 0, 0])
                edge_attr.append([0, 0, 0])
        
        # 2. 空間的近接性に基づくエッジ
        positions = np.array([[f['mean_x'], f['mean_y']] for f in fixations])
        distances = squareform(pdist(positions))
        
        for i in range(len(fixations)):
            for j in range(i+1, len(fixations)):
                if distances[i, j] < self.edge_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    edge_attr.append([
                        distances[i, j],
                        abs(i - j),  # 時間的距離
                        0  # プレースホルダー
                    ])
                    edge_attr.append([
                        distances[i, j],
                        abs(i - j),
                        0
                    ])
        
        return edge_index, edge_attr

# 2. GNNモデルの定義
class EyeMovementGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, num_classes=2, dropout=0.5):
        super(EyeMovementGNN, self).__init__()
        
        # グラフ畳み込み層
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 2, hidden_dim),  # global poolingで2倍
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        # グラフ畳み込み
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        # グローバルプーリング
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 分類
        x = self.classifier(x)
        
        return x

# 3. 学習用の関数
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 訓練
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        # 検証
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 履歴を保存
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history

# 4. データ読み込みとグラフ構築
def load_eye_tracking_data(base_path, metadata_path):
    """
    眼球運動データとメタデータを読み込み、グラフデータセットを構築
    
    Args:
        base_path: データフォルダのベースパス
        metadata_path: metadata.csvのパス
    
    Returns:
        graphs: グラフのリスト
        labels: ラベルのリスト
    """
    # メタデータを読み込み（エンコーディングを試行）
    encodings = ['utf-8', 'shift-jis', 'cp932', 'utf-8-sig']
    metadata = None
    
    for encoding in encodings:
        try:
            metadata = pd.read_csv(metadata_path, encoding=encoding)
            print(f"Successfully loaded metadata with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if metadata is None:
        raise ValueError(f"Could not read metadata file with any of the encodings: {encodings}")
    
    # MoCAスコアに基づくラベル付け（例：26未満をMCIとする）
    MCI_THRESHOLD = 26
    metadata['is_mci'] = (metadata['MoCA'] < MCI_THRESHOLD).astype(int)
    
    graphs = []
    labels = []
    
    # 各被験者のデータを処理
    for idx, row in metadata.iterrows():
        gapid = row['GAPID']
        ins_path = row['ins_path']
        
        # calibration_*.csvファイルを探す
        calibration_files = glob.glob(os.path.join(ins_path, f'calibration_*{gapid}*.csv'))
        
        if not calibration_files:
            print(f"No calibration file found for GAPID: {gapid}")
            continue
        
        # 最初のファイルを使用（複数ある場合）
        calibration_file = calibration_files[0]
        
        try:
            # calibrationデータを読み込み（エンコーディングを試行）
            calib_df = None
            for encoding in encodings:
                try:
                    calib_df = pd.read_csv(calibration_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if calib_df is None:
                print(f"Could not read calibration file: {calibration_file}")
                continue
            
            # 必要な列が存在するか確認
            required_cols = ['EyeCenterAngleX', 'EyeCenterAngleY', 'TimeStamp']
            if not all(col in calib_df.columns for col in required_cols):
                print(f"Required columns missing in file: {calibration_file}")
                continue
            
            # データが空でないか確認
            if len(calib_df) < 10:  # 最低10サンプル必要
                print(f"Insufficient data in file: {calibration_file} (only {len(calib_df)} samples)")
                continue
            
            # グラフ構築
            graph_builder = EyeMovementGraphBuilder()
            graph = graph_builder.create_graph_from_trajectory(
                calib_df['TimeStamp'].values,
                calib_df['EyeCenterAngleX'].values,
                calib_df['EyeCenterAngleY'].values
            )
            
            # グラフが空でないか確認
            if graph.x.shape[0] == 0:
                print(f"No fixations detected for GAPID: {gapid}")
                continue
            
            # ラベルを追加
            graph.y = torch.tensor([row['is_mci']])
            
            # 追加のメタデータを属性として保存（オプション）
            graph.age = row['Age']
            graph.education = row['EducationYear']
            graph.moca_score = row['MoCA']
            graph.gapid = gapid
            
            graphs.append(graph)
            labels.append(row['is_mci'])
            
        except Exception as e:
            print(f"Error processing GAPID {gapid}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(graphs)} graphs")
    print(f"MCI cases: {sum(labels)}, Healthy cases: {len(labels) - sum(labels)}")
    
    return graphs, labels

# 5. メイン処理
def main():
    # パスの設定
    base_path = "G:/共有ドライブ/GAP_Analysis/Data/GAP2_ShizuokaCohort/2023"
    metadata_path = "G:/共有ドライブ/MCI/MoCA/2023/metadata.csv"
    
    # データの読み込みとグラフ構築
    print("Loading data and building graphs...")
    graphs, labels = load_eye_tracking_data(base_path, metadata_path)
    
    if len(graphs) == 0:
        print("No data loaded. Please check the file paths.")
        return
    
    # データの分割
    
    # まずtrain/testに分割
    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # さらにtrain/valに分割
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_val_graphs, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
    )

    print(f"Train set: {len(train_graphs)} samples")
    print(f"Validation set: {len(val_graphs)} samples")
    print(f"Test set: {len(test_graphs)} samples")

    # DataLoaderの作成
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # モデルの初期化
    model = EyeMovementGNN(num_node_features=11, num_classes=2)
    
    # 学習
    print("Starting training...")
    trained_model, history = train_model(model, train_loader, val_loader, epochs=100)
    
    print("Training completed!")
    
    # テストデータでの評価（必要に応じて）
    evaluate_model(trained_model, val_loader)

    def evaluate_model(model, test_loader):
        """モデルの詳細な評価"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # MCIの確率
        
        # 評価指標の計算
        print("\n=== Model Evaluation ===")
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, 
                                    target_names=['Healthy', 'MCI'],
                                    output_dict=True)
        print(classification_report(all_labels, all_preds, 
                                target_names=['Healthy', 'MCI']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        # AUC-ROCの計算
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"\nAUC-ROC Score: {auc_score:.4f}")
        # 結果を返す
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_roc': auc_score,
            'accuracy': report['accuracy'],
            'mci_precision': report['MCI']['precision'],
            'mci_recall': report['MCI']['recall'],
            'mci_f1': report['MCI']['f1-score'],
            'predictions': [{
                'true_label': int(all_labels[i]),
                'predicted_label': int(all_preds[i]),
                'mci_probability': float(all_probs[i]),
                'correct': int(all_labels[i] == all_preds[i])
            } for i in range(len(all_preds))]
        }
        
        return results

    

    # 学習
    print("Starting training...")
    trained_model, history = train_model(model, train_loader, val_loader, epochs=100)

    print("Training completed!")

    # テストデータでの評価
    print("\nEvaluating on test set...")
    test_results = evaluate_model(trained_model, test_loader)  # test_loaderを使用

    # 結果保存用のディレクトリ
    save_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\gnnresult"

    # 実験のメタデータ
    exp_metadata = {
        'model_type': 'EyeMovementGNN',
        'num_node_features': 11,
        'hidden_dim': 64,
        'num_classes': 2,
        'epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 32,
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'mci_threshold': 26,
        'base_path': base_path,
        'metadata_path': metadata_path
    }

    # 結果の保存（save_results関数がある場合）
    if 'save_results' in globals():
        save_results(save_dir, trained_model, history, test_results, exp_metadata)
    else:
        # 簡易保存
        import os
        import json
        import torch
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(save_dir, f"experiment_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # モデルの保存
        torch.save(trained_model.state_dict(), os.path.join(exp_dir, 'final_model.pth'))
        
        # 結果の保存
        with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        print(f"\nResults saved to: {exp_dir}")

    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
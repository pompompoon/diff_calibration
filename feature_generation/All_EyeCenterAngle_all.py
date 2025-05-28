import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import label
from scipy.fft import fft
import matplotlib.pyplot as plt
import os
import glob
from extractioncondition.sensitivityfilter import SensitivityFilter

class EnhancedSaccadeAnalyzer:
    def __init__(self, df):
        """
        拡張サッカード分析クラスの初期化 - MCIと健常者の分類に焦点
        
        Parameters:
        -----------
        df : pandas.DataFrame
            分析対象のデータフレーム
        """
        self.df = df
        
        # -100の値を除外（コピーを作成して警告を回避）
        self.df = self.df.loc[(self.df['EyeCenterAngleX'] != -100) & (self.df['EyeCenterAngleY'] != -100)].copy()
        
        self.df.loc[:, 'timestamp'] = pd.to_numeric(self.df['timestamp'])  # timestampカラムを数値に変換
        self.process_data()
    
    def process_data(self):
        """データの前処理とベースとなる計算を行う"""
        # 時間差分の計算
        self.dt = np.diff(self.df['timestamp']) / 1000  # msからsに変換
        
        # 位置の差分を計算
        self.dx = np.diff(self.df['EyeCenterAngleX'])
        self.dy = np.diff(self.df['EyeCenterAngleY'])
        
        # 速度の計算
        self.velocity = np.sqrt(self.dx**2 + self.dy**2) / self.dt
        
        # 移動方向の計算
        self.angles = np.arctan2(self.dy, self.dx)
        
        # 加速度の計算
        self.acceleration = np.diff(self.velocity) / self.dt[:-1]
        
        # 躍度の計算
        self.jerk = np.diff(self.acceleration) / self.dt[:-2]
    
    def calculate_basic_metrics(self):
        """基本的な統計量を計算"""
        metrics = {
            'mean_velocity': np.mean(self.velocity),
            'median_velocity': np.median(self.velocity),
            'std_velocity': np.std(self.velocity),
            'max_velocity': np.max(self.velocity),
            'mean_acceleration': np.mean(self.acceleration),
            'median_acceleration': np.median(self.acceleration),
            'std_acceleration': np.std(self.acceleration),
            'max_acceleration': np.max(self.acceleration),
            'mean_jerk': np.mean(self.jerk),
            'median_jerk': np.median(self.jerk),
            'std_jerk': np.std(self.jerk),
            'max_jerk': np.max(self.jerk)
        }
        return metrics
    
    def detect_saccades(self, velocity_threshold=30):
        """
        サッカードを検出する
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
        
        Returns:
        --------
        dict: サッカード関連の指標
        """
        # 速度条件のみでサッカードを検出
        self.saccade_mask = self.velocity > velocity_threshold
        saccade_count = np.sum(self.saccade_mask)
        
        # サッカード中の移動距離を計算
        saccade_distance = np.sum(self.velocity[self.saccade_mask] * self.dt[self.saccade_mask])
        
        # 固視時間を計算 (ms)
        fixation_time = np.sum(self.dt[~self.saccade_mask]) * 1000  # sからmsに変換
        
        # サッカードの軌跡情報を計算
        saccade_trajectories = self.calculate_saccade_trajectories(velocity_threshold)
        
        saccade_metrics = {
            'saccade_count': saccade_count,
            'saccade_rate': saccade_count / (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]) * 1000,  # サッカード頻度（1秒あたり）
            'saccade_distance': saccade_distance,
            'fixation_time': fixation_time,
            'fixation_to_saccade_ratio': fixation_time / (np.sum(self.dt[self.saccade_mask]) * 1000) if np.sum(self.saccade_mask) > 0 else 0
        }
        
        # 軌跡の統計情報を追加
        if saccade_trajectories:
            amplitudes = [t['amplitude'] for t in saccade_trajectories]
            durations = [t['duration'] for t in saccade_trajectories]
            peak_velocities = [t['peak_velocity'] for t in saccade_trajectories]
            curvature_indices = [t['curvature_index'] for t in saccade_trajectories]
            
            trajectory_metrics = {
                'saccade_mean_amplitude': np.mean(amplitudes),
                'saccade_median_amplitude': np.median(amplitudes),
                'saccade_std_amplitude': np.std(amplitudes) if len(amplitudes) > 1 else 0,
                'saccade_max_amplitude': max(amplitudes) if amplitudes else 0,
                'saccade_mean_duration': np.mean(durations),
                'saccade_median_duration': np.median(durations),
                'saccade_std_duration': np.std(durations) if len(durations) > 1 else 0,
                'saccade_mean_peak_velocity': np.mean(peak_velocities),
                'saccade_median_peak_velocity': np.median(peak_velocities),
                'saccade_std_peak_velocity': np.std(peak_velocities) if len(peak_velocities) > 1 else 0,
                'saccade_mean_curvature_index': np.mean(curvature_indices) if curvature_indices else 0,
                'saccade_median_curvature_index': np.median(curvature_indices) if curvature_indices else 0,
                'saccade_std_curvature_index': np.std(curvature_indices) if len(curvature_indices) > 1 else 0
            }
            saccade_metrics.update(trajectory_metrics)
            
            # Main sequence ratio (peak velocity/amplitude) - AD patients show abnormal main sequences
            main_sequence_ratios = [p/a if a > 0 else 0 for p, a in zip(peak_velocities, amplitudes)]
            saccade_metrics['main_sequence_ratio_mean'] = np.mean(main_sequence_ratios) if main_sequence_ratios else 0
            saccade_metrics['main_sequence_ratio_std'] = np.std(main_sequence_ratios) if len(main_sequence_ratios) > 1 else 0
            
            # 斜めサッカードの割合を計算 (Kapoula et al., 2014)
            diagonal_saccades = 0
            for saccade in saccade_trajectories:
                start_x, start_y = saccade['trajectory_x'][0], saccade['trajectory_y'][0]
                end_x, end_y = saccade['trajectory_x'][-1], saccade['trajectory_y'][-1]
                # 斜め方向のサッカードをカウント (主軸から22.5度以上離れている)
                angle = np.abs(np.arctan2(end_y - start_y, end_x - start_x))
                is_diagonal = not (np.isclose(angle, 0, atol=np.pi/8) or 
                                   np.isclose(angle, np.pi/2, atol=np.pi/8) or 
                                   np.isclose(angle, np.pi, atol=np.pi/8) or 
                                   np.isclose(angle, 3*np.pi/2, atol=np.pi/8))
                if is_diagonal:
                    diagonal_saccades += 1
            
            saccade_metrics['diagonal_saccade_ratio'] = diagonal_saccades / len(saccade_trajectories) if saccade_trajectories else 0
        
        return saccade_metrics
    
    def calculate_saccade_trajectories(self, velocity_threshold=30):
        """
        サッカードの軌跡情報を計算する
        
        Parameters:
        -----------
        velocity_threshold : float
            サッカード検出の速度閾値 (deg/s)
            
        Returns:
        --------
        list: サッカード軌跡の特徴量リスト
        """
        # 速度条件のみでサッカードを検出
        saccade_mask = self.velocity > velocity_threshold
        
        # 連続したTrueの領域（サッカード）を特定
        labeled_regions, num_regions = label(saccade_mask)
        
        saccade_trajectories = []
        
        # 各サッカード領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # サッカードの開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1  # データポイントとして次のインデックスも含める
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 開始と終了の時間・位置
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx]
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx, len(self.df) - 1)]
            
            # サッカード中の軌跡情報
            if end_idx + 1 <= len(self.df):
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
            else:
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:].values
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 継続時間（ms）
            duration = end_time - start_time
            
            # サッカード中の最大速度
            peak_velocity = np.max(self.velocity[region_indices]) if len(region_indices) > 0 else 0
            
            # 軌跡の長さを計算
            path_length = 0
            for j in range(len(trajectory_x) - 1):
                path_length += np.sqrt((trajectory_x[j+1] - trajectory_x[j])**2 + 
                                       (trajectory_y[j+1] - trajectory_y[j])**2)
            
            # 曲率インデックス（軌跡の長さ / 直線距離）
            curvature_index = path_length / amplitude if amplitude > 0 else 0
            
            saccade_info = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'amplitude': amplitude,
                'peak_velocity': peak_velocity,
                'curvature_index': curvature_index,
                'trajectory_x': trajectory_x,
                'trajectory_y': trajectory_y,
                'path_length': path_length
            }
            
            saccade_trajectories.append(saccade_info)
        
        return saccade_trajectories
    
    def detect_square_wave_jerks(self, amplitude_threshold=0.5, duration_threshold=300):
        """
        Square Wave Jerks（SWJ）を検出する - ADで増加する方形波ジャーク
        
        Parameters:
        -----------
        amplitude_threshold : float
            SWJ検出の振幅閾値 (deg)
        duration_threshold : float
            SWJ検出の最大継続時間閾値 (ms)
            
        Returns:
        --------
        dict: SWJ関連の指標
        """
        # 移動方向の急な変化を検出
        angle_diff = np.diff(self.angles)
        # -πとπの境界をまたぐ場合の補正
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi,
                              np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff))
        
        # 約180度の方向転換を検出 (Square Wave Jerksの特徴)
        swj_candidates = np.abs(angle_diff) > (np.pi * 0.8)  # ~145度以上の方向転換
        
        # 連続したSWJ候補領域を特定
        labeled_regions, num_regions = label(swj_candidates)
        
        swj_count = 0
        swj_amplitudes = []
        swj_durations = []
        
        # 各SWJ候補領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # SWJの開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df) - 1:
                end_idx = len(self.df) - 2
            
            # 開始と終了の時間・位置
            start_time = self.df['timestamp'].iloc[start_idx + 1]  # angle_diffのインデックス調整
            end_time = self.df['timestamp'].iloc[min(end_idx + 1, len(self.df) - 1)]  # angle_diffのインデックス調整
            
            # 継続時間（ms）
            duration = end_time - start_time
            
            # 位置の変化量
            delta_x = self.df['EyeCenterAngleX'].iloc[end_idx + 1] - self.df['EyeCenterAngleX'].iloc[start_idx + 1]
            delta_y = self.df['EyeCenterAngleY'].iloc[end_idx + 1] - self.df['EyeCenterAngleY'].iloc[start_idx + 1]
            amplitude = np.sqrt(delta_x**2 + delta_y**2)
            
            # SWJの条件をチェック
            if amplitude > amplitude_threshold and duration < duration_threshold:
                swj_count += 1
                swj_amplitudes.append(amplitude)
                swj_durations.append(duration)
        
        # 記録時間あたりのSWJ頻度を計算
        total_time_ms = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]
        swj_frequency = swj_count / (total_time_ms / 1000)  # SWJ/秒
        
        swj_metrics = {
            'swj_count': swj_count,
            'swj_frequency': swj_frequency,
            'swj_mean_amplitude': np.mean(swj_amplitudes) if swj_amplitudes else 0,
            'swj_std_amplitude': np.std(swj_amplitudes) if len(swj_amplitudes) > 1 else 0,
            'swj_mean_duration': np.mean(swj_durations) if swj_durations else 0,
            'swj_std_duration': np.std(swj_durations) if len(swj_durations) > 1 else 0
        }
        
        return swj_metrics
    
    def detect_microsaccades(self, velocity_threshold=15, max_amplitude=1.0):
        """
        マイクロサッカードを検出する - MCIで斜め方向の増加
        
        Parameters:
        -----------
        velocity_threshold : float
            マイクロサッカード検出の速度閾値 (deg/s)
        max_amplitude : float
            マイクロサッカードの最大振幅 (deg)
            
        Returns:
        --------
        dict: マイクロサッカード関連の指標
        """
        # 速度でマイクロサッカード候補を検出
        microsaccade_candidates = (self.velocity > velocity_threshold) & (self.velocity < 100)  # 速度範囲
        
        # 連続した領域を特定
        labeled_regions, num_regions = label(microsaccade_candidates)
        
        microsaccade_count = 0
        diagonal_microsaccade_count = 0
        microsaccade_amplitudes = []
        microsaccade_durations = []
        microsaccade_peak_velocities = []
        microsaccade_directions = []
        
        # 各マイクロサッカード候補領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:  # 少なくとも2点が必要
                continue
            
            # マイクロサッカードの開始と終了のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1
            
            # インデックスがデータ範囲内かチェック
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 開始と終了の時間・位置
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx]
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx, len(self.df) - 1)]
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 継続時間（ms）
            duration = end_time - start_time
            
            # 最大速度
            peak_velocity = np.max(self.velocity[region_indices]) if len(region_indices) > 0 else 0
            
            # マイクロサッカードの条件をチェック (振幅 < 1.0度)
            if amplitude < max_amplitude:
                microsaccade_count += 1
                microsaccade_amplitudes.append(amplitude)
                microsaccade_durations.append(duration)
                microsaccade_peak_velocities.append(peak_velocity)
                
                # 方向を計算
                direction = np.arctan2(end_y - start_y, end_x - start_x)
                microsaccade_directions.append(direction)
                
                # 斜め方向のマイクロサッカードをカウント (主軸から22.5度以上離れている)
                angle = np.abs(direction)
                is_diagonal = not (np.isclose(angle, 0, atol=np.pi/8) or 
                                   np.isclose(angle, np.pi/2, atol=np.pi/8) or 
                                   np.isclose(angle, np.pi, atol=np.pi/8) or 
                                   np.isclose(angle, 3*np.pi/2, atol=np.pi/8))
                if is_diagonal:
                    diagonal_microsaccade_count += 1
        
        # 記録時間あたりのマイクロサッカード頻度を計算
        total_time_ms = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]
        microsaccade_frequency = microsaccade_count / (total_time_ms / 1000)  # マイクロサッカード/秒
        
        microsaccade_metrics = {
            'microsaccade_count': microsaccade_count,
            'microsaccade_frequency': microsaccade_frequency,
            'microsaccade_mean_amplitude': np.mean(microsaccade_amplitudes) if microsaccade_amplitudes else 0,
            'microsaccade_std_amplitude': np.std(microsaccade_amplitudes) if len(microsaccade_amplitudes) > 1 else 0,
            'microsaccade_mean_duration': np.mean(microsaccade_durations) if microsaccade_durations else 0,
            'microsaccade_std_duration': np.std(microsaccade_durations) if len(microsaccade_durations) > 1 else 0,
            'microsaccade_mean_peak_velocity': np.mean(microsaccade_peak_velocities) if microsaccade_peak_velocities else 0,
            'diagonal_microsaccade_ratio': diagonal_microsaccade_count / microsaccade_count if microsaccade_count > 0 else 0
        }
        
        return microsaccade_metrics
    
    def calculate_fixation_stability(self):
        """
        固視安定性の指標を計算する - ADでは固視安定性が低下
        
        Returns:
        --------
        dict: 固視安定性関連の指標
        """
        # 速度が低い部分を固視として特定
        fixation_mask = self.velocity < 30
        
        # 連続した固視領域を特定
        labeled_regions, num_regions = label(fixation_mask)
        
        fixation_durations = []
        fixation_dispersion_x = []
        fixation_dispersion_y = []
        fixation_bivariate_contour_area = []
        
        # 各固視領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 5:  # 少なくとも5点が必要
                continue
            
            # 固視領域のインデックス（元のデータフレーム用）
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1
            
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 固視中の眼球位置
            fixation_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
            fixation_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
            
            # 継続時間（ms）
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            duration = end_time - start_time
            
            # 固視中の分散
            dispersion_x = np.std(fixation_x)
            dispersion_y = np.std(fixation_y)
            
            # 楕円面積の計算（BCEA: Bivariate Contour Ellipse Area）
            if len(fixation_x) > 5:  # 十分なデータポイントがある場合
                cov_matrix = np.cov(fixation_x, fixation_y)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                # 95%信頼楕円の面積
                bcea = 2.291 * np.pi * np.sqrt(np.prod(eigenvalues))
            else:
                bcea = 0
            
            fixation_durations.append(duration)
            fixation_dispersion_x.append(dispersion_x)
            fixation_dispersion_y.append(dispersion_y)
            fixation_bivariate_contour_area.append(bcea)
        
        fixation_metrics = {
            'fixation_count': len(fixation_durations),
            'fixation_mean_duration': np.mean(fixation_durations) if fixation_durations else 0,
            'fixation_std_duration': np.std(fixation_durations) if len(fixation_durations) > 1 else 0,
            'fixation_mean_dispersion_x': np.mean(fixation_dispersion_x) if fixation_dispersion_x else 0,
            'fixation_mean_dispersion_y': np.mean(fixation_dispersion_y) if fixation_dispersion_y else 0,
            'fixation_mean_bcea': np.mean(fixation_bivariate_contour_area) if fixation_bivariate_contour_area else 0,
            'fixation_max_bcea': np.max(fixation_bivariate_contour_area) if fixation_bivariate_contour_area else 0
        }
        
        return fixation_metrics
    
    def calculate_direction_changes(self, angle_threshold=np.pi/4):
        """
        方向転換の回数を計算 - ADでは方向転換が無秩序に増加
        
        Parameters:
        -----------
        angle_threshold : float
            方向転換とみなす角度閾値 (rad)
            
        Returns:
        --------
        dict: 方向転換関連の指標
        """
        # 方向の変化量
        angle_diff = np.diff(self.angles)
        
        # -πとπの境界をまたぐ場合の補正
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi,
                            np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff))
        
        # 異なる閾値での方向転換
        direction_changes_45 = np.sum(np.abs(angle_diff) > np.pi/4)  # 45度
        direction_changes_90 = np.sum(np.abs(angle_diff) > np.pi/2)  # 90度
        direction_changes_135 = np.sum(np.abs(angle_diff) > 3*np.pi/4)  # 135度
        
        # 記録時間あたりの頻度を計算
        total_time_ms = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]
        direction_change_freq_45 = direction_changes_45 / (total_time_ms / 1000)
        direction_change_freq_90 = direction_changes_90 / (total_time_ms / 1000)
        direction_change_freq_135 = direction_changes_135 / (total_time_ms / 1000)
        
        # 方向変化の無秩序性（高周波成分の測定）
        angle_diff_smooth = signal.savgol_filter(angle_diff, 5, 3) if len(angle_diff) >= 5 else angle_diff
        angular_noise = np.mean(np.abs(angle_diff - angle_diff_smooth)) if len(angle_diff) >= 5 else 0
        
        direction_metrics = {
            'direction_changes_45': direction_changes_45,
            'direction_changes_90': direction_changes_90,
            'direction_changes_135': direction_changes_135,
            'direction_change_freq_45': direction_change_freq_45,
            'direction_change_freq_90': direction_change_freq_90,
            'direction_change_freq_135': direction_change_freq_135,
            'angular_noise': angular_noise
        }
        
        return direction_metrics
    
    def calculate_fft_metrics(self):
        """
        FFT解析を行い、高周波成分の強度を計算
        
        Returns:
        --------
        dict: FFT関連の指標
        """
        # FFTの計算
        fft_result = fft(self.velocity)
        freq = np.fft.fftfreq(len(self.velocity), d=np.mean(self.dt))
        
        # 正の周波数成分のみを使用
        pos_freq_mask = freq > 0
        power_spectrum = np.abs(fft_result[pos_freq_mask])
        freq = freq[pos_freq_mask]
        
        # 周波数帯域ごとのパワーを計算
        low_freq_mask = (freq > 0) & (freq <= 5)
        mid_freq_mask = (freq > 5) & (freq <= 10)
        high_freq_mask = freq > 10
        
        low_freq_power = np.sum(power_spectrum[low_freq_mask])
        mid_freq_power = np.sum(power_spectrum[mid_freq_mask])
        high_freq_power = np.sum(power_spectrum[high_freq_mask])
        
        # 周波数帯域の比率（ADでは高周波成分が増加する傾向）
        high_to_low_ratio = high_freq_power / low_freq_power if low_freq_power > 0 else 0
        
        fft_metrics = {
            'low_freq_power': low_freq_power,
            'mid_freq_power': mid_freq_power,
            'high_freq_power': high_freq_power,
            'high_to_low_freq_ratio': high_to_low_ratio,
            'total_power': np.sum(power_spectrum)
        }
        
        return fft_metrics
    
    def calculate_smooth_pursuit_metrics(self, velocity_threshold=10):
        """
        スムースパーシュート（滑動性追跡眼球運動）の評価
        - ADでは中断が増加し、追跡時間が減少
        
        Parameters:
        -----------
        velocity_threshold : float
            スムースパーシュート検出の速度閾値 (deg/s)
            
        Returns:
        --------
        dict: スムースパーシュート関連の指標
        """
        # 速度が一定範囲内の部分をスムースパーシュートとして特定
        smooth_pursuit_mask = (self.velocity > 1) & (self.velocity < velocity_threshold)
        
        # 連続したスムースパーシュート領域を特定
        labeled_regions, num_regions = label(smooth_pursuit_mask)
        
        pursuit_durations = []
        pursuit_interruptions = 0
        
        # 各スムースパーシュート領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 5:  # 少なくとも5点が必要
                continue
            
            # 領域のインデックス
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1
            
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 継続時間（ms）
            start_time = self.df['timestamp'].iloc[start_idx]
            end_time = self.df['timestamp'].iloc[min(end_idx, len(self.df) - 1)]
            duration = end_time - start_time
            
            # 速度の変動係数（スムースさの指標）
            velocity_segment = self.velocity[region_indices]
            velocity_cv = np.std(velocity_segment) / np.mean(velocity_segment) if np.mean(velocity_segment) > 0 else 0
            
            # サッカードによる中断がないか確認
            saccade_interruptions = np.sum(self.velocity[start_idx:end_idx] > 30)
            if saccade_interruptions > 0:
                pursuit_interruptions += 1
            
            if duration > 50:  # 50ms以上のみカウント
                pursuit_durations.append(duration)
        
        total_pursuit_time = np.sum(pursuit_durations)
        total_time_ms = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]
        
        pursuit_metrics = {
            'pursuit_count': len(pursuit_durations),
            'pursuit_mean_duration': np.mean(pursuit_durations) if pursuit_durations else 0,
            'pursuit_std_duration': np.std(pursuit_durations) if len(pursuit_durations) > 1 else 0,
            'pursuit_total_time': total_pursuit_time,
            'pursuit_time_ratio': total_pursuit_time / total_time_ms if total_time_ms > 0 else 0,
            'pursuit_interruption_count': pursuit_interruptions,
            'pursuit_interruption_rate': pursuit_interruptions / len(pursuit_durations) if len(pursuit_durations) > 0 else 0
        }
        
        return pursuit_metrics
    
    def calculate_antisaccade_features(self, target_position=None):
        """
        アンチサッカード特性の評価 - ADでは抑制エラーが増加
        注: 実際のアンチサッカード課題のデータが必要
        
        Parameters:
        -----------
        target_position : tuple
            ターゲット位置の座標 (x, y)、Noneの場合は推定を試みる
            
        Returns:
        --------
        dict: アンチサッカード関連の指標
        """
        # アンチサッカード課題のデータがない場合は、眼球運動の反応特性から間接的な指標を計算
        
        # 方向変化の急峻さを計算（抑制能力の間接的指標）
        max_acceleration = np.max(np.abs(self.acceleration))
        
        # 速度プロファイルの複雑さ（実行機能の間接的指標）
        velocity_complexity = np.std(self.velocity) / np.mean(self.velocity) if np.mean(self.velocity) > 0 else 0
        
        # 高速サッカードの割合（60deg/s以上、実行制御の指標）
        fast_saccade_ratio = np.sum(self.velocity > 60) / len(self.velocity) if len(self.velocity) > 0 else 0
        
        antisaccade_metrics = {
            'max_absolute_acceleration': max_acceleration,
            'velocity_complexity': velocity_complexity,
            'fast_saccade_ratio': fast_saccade_ratio
        }
        
        return antisaccade_metrics
    
    def calculate_visual_exploration_metrics(self):
        """
        視覚探索パターンの評価 - ADでは非効率でランダムな探索パターン
        
        Returns:
        --------
        dict: 視覚探索関連の指標
        """
        # 探索範囲の計算
        x_range = np.max(self.df['EyeCenterAngleX']) - np.min(self.df['EyeCenterAngleX'])
        y_range = np.max(self.df['EyeCenterAngleY']) - np.min(self.df['EyeCenterAngleY'])
        
        # 探索の集中度を評価（中心から離れる頻度）
        center_x = np.mean(self.df['EyeCenterAngleX'])
        center_y = np.mean(self.df['EyeCenterAngleY'])
        distances_from_center = np.sqrt((self.df['EyeCenterAngleX'] - center_x)**2 + 
                                       (self.df['EyeCenterAngleY'] - center_y)**2)
        
        # 中心から周辺への移動回数
        threshold_distance = np.std(distances_from_center)
        center_to_periphery_transitions = np.sum(np.diff(distances_from_center > threshold_distance) > 0)
        
        # 探索パターンのエントロピー計算（無秩序さの指標）
        # 簡易的な2次元ヒストグラムの構築
        try:
            hist, _, _ = np.histogram2d(
                self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 
                bins=[10, 10], 
                range=[[np.min(self.df['EyeCenterAngleX']), np.max(self.df['EyeCenterAngleX'])], 
                       [np.min(self.df['EyeCenterAngleY']), np.max(self.df['EyeCenterAngleY'])]]
            )
            
            # 確率分布に変換
            hist = hist / np.sum(hist)
            
            # 0を除外してエントロピー計算
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        except:
            entropy = 0
        
        exploration_metrics = {
            'exploration_x_range': x_range,
            'exploration_y_range': y_range,
            'exploration_area': x_range * y_range,
            'mean_distance_from_center': np.mean(distances_from_center),
            'std_distance_from_center': np.std(distances_from_center),
            'center_to_periphery_transitions': center_to_periphery_transitions,
            'exploration_entropy': entropy
        }
        
        return exploration_metrics
    
    def analyze(self):
        """
        すべての分析を実行して結果を返す
        MCI/AD分類に有効な特徴量に焦点
        
        Returns:
        --------
        dict: すべての分析結果
        """
        results = {}
        
        # 基本的な統計量
        results.update(self.calculate_basic_metrics())
        
        # サッカード関連の指標（速度閾値30deg/sのみを使用）
        results.update(self.detect_saccades(velocity_threshold=30))
        
        # マイクロサッカードの検出と分析
        results.update(self.detect_microsaccades())
        
        # Square Wave Jerks（SWJ）の検出と分析
        results.update(self.detect_square_wave_jerks())
        
        # 固視安定性の分析
        results.update(self.calculate_fixation_stability())
        
        # 方向転換の分析
        results.update(self.calculate_direction_changes())
        
        # スムースパーシュートの分析
        results.update(self.calculate_smooth_pursuit_metrics())
        
        # アンチサッカード特性の間接的評価
        results.update(self.calculate_antisaccade_features())
        
        # 視覚探索パターンの評価
        results.update(self.calculate_visual_exploration_metrics())
        
        # FFT解析
        results.update(self.calculate_fft_metrics())
        
        return results
    
    def plot_saccade_trajectories(self, output_dir, base_name, velocity_threshold=30):
        """サッカードの軌跡をプロット"""
        # サッカードの軌跡を検出
        saccade_trajectories = self.calculate_saccade_trajectories(velocity_threshold)
        
        if not saccade_trajectories:
            print(f"サッカードが検出されませんでした: {base_name}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 全体の眼球運動軌跡
        ax.plot(self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 'b-', alpha=0.3, label='全体の軌跡')
        
        # 各サッカードの軌跡
        for i, saccade in enumerate(saccade_trajectories):
            ax.plot(saccade['trajectory_x'], saccade['trajectory_y'], 'r-', linewidth=2, 
                    label=f'サッカード {i+1}' if i == 0 else "")
            
            # 始点と終点をマーク
            ax.plot(saccade['trajectory_x'][0], saccade['trajectory_y'][0], 'go', label='開始点' if i == 0 else "")
            ax.plot(saccade['trajectory_x'][-1], saccade['trajectory_y'][-1], 'mo', label='終了点' if i == 0 else "")
        
        ax.set_title('サッカード軌跡の可視化')
        ax.set_xlabel('水平方向の眼球角度 (deg)')
        ax.set_ylabel('垂直方向の眼球角度 (deg)')
        ax.axis('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_saccades.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_microsaccades(self, output_dir, base_name):
        """マイクロサッカードの軌跡と方向分布をプロット"""
        # マイクロサッカードの検出
        microsaccade_candidates = (self.velocity > 15) & (self.velocity < 100)
        
        # 連続した領域を特定
        labeled_regions, num_regions = label(microsaccade_candidates)
        
        microsaccade_trajectories = []
        microsaccade_directions = []
        
        # 各マイクロサッカード候補領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:
                continue
            
            start_idx = region_indices[0]
            end_idx = region_indices[-1] + 1
            
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            start_x = self.df['EyeCenterAngleX'].iloc[start_idx]
            start_y = self.df['EyeCenterAngleY'].iloc[start_idx]
            end_x = self.df['EyeCenterAngleX'].iloc[min(end_idx, len(self.df) - 1)]
            end_y = self.df['EyeCenterAngleY'].iloc[min(end_idx, len(self.df) - 1)]
            
            # 軌跡情報
            if end_idx + 1 <= len(self.df):
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
            else:
                trajectory_x = self.df['EyeCenterAngleX'].iloc[start_idx:].values
                trajectory_y = self.df['EyeCenterAngleY'].iloc[start_idx:].values
            
            # 振幅（直線距離）
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # マイクロサッカードの条件をチェック (振幅 < 1.0度)
            if amplitude < 1.0:
                microsaccade_trajectories.append((trajectory_x, trajectory_y))
                
                # 方向を計算
                direction = np.arctan2(end_y - start_y, end_x - start_x)
                microsaccade_directions.append(direction)
        
        if not microsaccade_trajectories:
            print(f"マイクロサッカードが検出されませんでした: {base_name}")
            return
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 軌跡のプロット
        ax1.plot(self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 'b-', alpha=0.1, label='全体の軌跡')
        
        for i, (traj_x, traj_y) in enumerate(microsaccade_trajectories):
            if i < 20:  # 最初の20個のみ個別表示
                ax1.plot(traj_x, traj_y, '-', linewidth=1.5)
                ax1.plot(traj_x[0], traj_y[0], 'go', markersize=3)
                ax1.plot(traj_x[-1], traj_y[-1], 'ro', markersize=3)
        
        ax1.set_title('マイクロサッカードの軌跡')
        ax1.set_xlabel('水平方向の眼球角度 (deg)')
        ax1.set_ylabel('垂直方向の眼球角度 (deg)')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # 方向のヒストグラム
        if microsaccade_directions:
            bins = np.linspace(-np.pi, np.pi, 16)
            ax2.hist(microsaccade_directions, bins=bins)
            ax2.set_title('マイクロサッカードの方向分布')
            ax2.set_xlabel('方向 (rad)')
            ax2.set_ylabel('頻度')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_microsaccades.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_square_wave_jerks(self, output_dir, base_name):
        """Square Wave Jerksの可視化"""
        # 移動方向の急な変化を検出
        angle_diff = np.diff(self.angles)
        # -πとπの境界をまたぐ場合の補正
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi,
                              np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff))
        
        # 約180度の方向転換を検出
        swj_candidates = np.abs(angle_diff) > (np.pi * 0.8)
        
        # 連続したSWJ候補領域を特定
        labeled_regions, num_regions = label(swj_candidates)
        
        swj_segments = []
        
        # 各SWJ候補領域について分析
        for i in range(1, num_regions + 1):
            region_indices = np.where(labeled_regions == i)[0]
            
            if len(region_indices) < 2:
                continue
            
            # インデックス調整（angle_diffはdiff計算で長さが1減少）
            start_idx = region_indices[0] + 1
            end_idx = region_indices[-1] + 2  # +1(diff調整) +1(次のポイント含む)
            
            if end_idx >= len(self.df):
                end_idx = len(self.df) - 1
            
            # 位置の変化量
            delta_x = self.df['EyeCenterAngleX'].iloc[end_idx] - self.df['EyeCenterAngleX'].iloc[start_idx]
            delta_y = self.df['EyeCenterAngleY'].iloc[end_idx] - self.df['EyeCenterAngleY'].iloc[start_idx]
            amplitude = np.sqrt(delta_x**2 + delta_y**2)
            
            # 継続時間
            duration = self.df['timestamp'].iloc[end_idx] - self.df['timestamp'].iloc[start_idx]
            
            # SWJの条件をチェック (振幅 > 0.5度、継続時間 < 300ms)
            if amplitude > 0.5 and duration < 300:
                segment_x = self.df['EyeCenterAngleX'].iloc[start_idx:end_idx+1].values
                segment_y = self.df['EyeCenterAngleY'].iloc[start_idx:end_idx+1].values
                swj_segments.append((segment_x, segment_y))
        
        if not swj_segments:
            print(f"方形波ジャークが検出されませんでした: {base_name}")
            return
        
        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 全体の眼球運動軌跡
        ax.plot(self.df['EyeCenterAngleX'], self.df['EyeCenterAngleY'], 'b-', alpha=0.2, label='全体の軌跡')
        
        # 各SWJの軌跡
        for i, (segment_x, segment_y) in enumerate(swj_segments):
            if i < 30:  # 最初の30個のみ表示
                ax.plot(segment_x, segment_y, 'r-', linewidth=1.5)
                ax.plot(segment_x[0], segment_y[0], 'go', markersize=4)
                ax.plot(segment_x[-1], segment_y[-1], 'mo', markersize=4)
        
        ax.set_title('Square Wave Jerks (SWJ) の可視化')
        ax.set_xlabel('水平方向の眼球角度 (deg)')
        ax.set_ylabel('垂直方向の眼球角度 (deg)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_square_wave_jerks.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_profiles(self, output_dir, base_name):
        """速度、加速度、躍度のプロファイルをプロット"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 速度プロファイル
        ax1.plot(self.df['timestamp'][1:], self.velocity)
        ax1.set_title('Velocity Profile')
        ax1.set_ylabel('Velocity (deg/s)')
        
        # 加速度プロファイル
        ax2.plot(self.df['timestamp'][2:], self.acceleration)
        ax2.set_title('Acceleration Profile')
        ax2.set_ylabel('Acceleration (deg/s²)')
        
        # 躍度プロファイル
        ax3.plot(self.df['timestamp'][3:], self.jerk)
        ax3.set_title('Jerk Profile')
        ax3.set_ylabel('Jerk (deg/s³)')
        ax3.set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{base_name}_profiles.png')
        plt.savefig(plot_path)
        plt.close()


def preprocess_data(df):
    """
    データの前処理を行う関数
    - -100の値を除外
    - 距離の計算
    - カラム名の調整
    """
    # -100の値を除外（コピーを作成して警告を回避）
    df = df.loc[(df['EyeCenterAngleX'] != -100) & (df['EyeCenterAngleY'] != -100)].copy()
    
    # マンハッタン距離の計算
    manhattan_distances = np.abs(df['EyeCenterAngleX']) + np.abs(df['EyeCenterAngleY'])
    df.loc[:, 'manhattan_distance'] = manhattan_distances
    manhattan_stats = {
        'manhattan_distance_sum': manhattan_distances.sum(),
        'manhattan_distance_mean': manhattan_distances.mean()
    }
    
    # ユークリッド距離の計算
    euclidean_distances = np.sqrt(
        df['EyeCenterAngleX']**2 + 
        df['EyeCenterAngleY']**2
    )
    df.loc[:, 'euclidean_distance'] = euclidean_distances
    euclidean_stats = {
        'euclidean_distance_sum': euclidean_distances.sum(),
        'euclidean_distance_mean': euclidean_distances.mean()
    }
    
    # TimeStampカラムの名前を確認し、必要に応じて変更
    if 'TimeStamp' in df.columns:
        df.loc[:, 'timestamp'] = df['TimeStamp']
    elif 'timestamp' in df.columns:
        df.loc[:, 'timestamp'] = df['timestamp']
    else:
        raise ValueError("タイムスタンプのカラムが見つかりません")
    
    return df, manhattan_stats, euclidean_stats

def analyze_eye_movements(input_file, output_dir):
    """
    眼球運動データを分析する関数
    
    Parameters:
    -----------
    input_file : str
        入力ファイルのパス
    output_dir : str
        出力ディレクトリのパス
    """
    # データの読み込みと前処理
    df = pd.read_csv(input_file)
    df, manhattan_stats, euclidean_stats = preprocess_data(df)
    
    # 拡張サッカード分析の実行
    analyzer = EnhancedSaccadeAnalyzer(df)
    results = analyzer.analyze()
    
    # 距離の統計量を結果に追加
    results.update(manhattan_stats)
    results.update(euclidean_stats)
    
    # 各種プロット生成（必要に応じてコメントアウト解除）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # analyzer.plot_saccade_trajectories(output_dir, base_name, velocity_threshold=30)
    # analyzer.plot_microsaccades(output_dir, base_name)
    # analyzer.plot_square_wave_jerks(output_dir, base_name)
    # analyzer.plot_profiles(output_dir, base_name)
    
    return results

def analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold=15):
    """
    複数のキャリブレーションデータを分析する関数（感度フィルター付き）
    
    Parameters:
    -----------
    base_dir : str
        GAPデータが格納されているベースディレクトリ
    output_dir : str
        結果を出力するディレクトリ
    sensitivity_threshold : int
        LessThanSensitivityの閾値
    """
    # 感度フィルターを初期化
    sensitivity_filter = SensitivityFilter(threshold=sensitivity_threshold)
    
    # calibrationファイルを含むすべてのフォルダを取得
    folders = sorted(glob.glob(os.path.join(base_dir, "2024*")))
    
    # 結果を格納するリスト
    all_results = []
    skipped_folders = []
    
    print(f"合計{len(folders)}個のフォルダを処理します。")
    
    # 各フォルダ内のcalibrationファイルを処理
    for i, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        
        # InspectionDateAndIdをフォルダ名から抽出（例: 20241009101605_1 -> 20241009101605_1）
        inspection_date_and_id = folder_name
        
        # 感度チェック
        should_process, sensitivity_count = sensitivity_filter.check_result_file(folder)
        
        if not should_process:
            print(f"\nスキップ ({i}/{len(folders)}): {folder_name}")
            print(f"理由: LessThanSensitivity合計 = {sensitivity_count} > {sensitivity_threshold}")
            skipped_folders.append({
                'folder': folder_name,
                'sensitivity_count': sensitivity_count
            })
            continue
            
        # calibrationファイルを検索
        calib_files = glob.glob(os.path.join(folder, "calibration_*.csv"))
        
        for calib_file in calib_files:
            try:
                print(f"\n処理中 ({i}/{len(folders)}): {folder_name}")
                print(f"LessThanSensitivity合計: {sensitivity_count}")
                
                # 個別のファイルを分析
                results = analyze_eye_movements(calib_file, output_dir)
                
                # InspectionDateAndIdと感度情報を追加
                results['InspectionDateAndId'] = inspection_date_and_id
                results['less_than_sensitivity_count'] = sensitivity_count
                
                # freq と power_spectrum を文字列として保存
                if 'freq' in results:
                    results['freq'] = str(results['freq'])
                if 'power_spectrum' in results:
                    results['power_spectrum'] = str(results['power_spectrum'])
                
                # 結果をリストに追加
                all_results.append(results)
                
            except Exception as e:
                print(f"エラー - {os.path.basename(calib_file)}: {str(e)}")
    
    # 結果をデータフレームに変換
    if all_results:
        # DataFrameに変換
        combined_results = pd.DataFrame(all_results)
        
        # InspectionDateAndIdでソート
        combined_results = combined_results.sort_values('InspectionDateAndId')
        
        # 列の順序を指定（InspectionDateAndIdを最初に）
        cols = ['InspectionDateAndId', 'less_than_sensitivity_count'] + [col for col in combined_results.columns if col not in ['InspectionDateAndId', 'less_than_sensitivity_count']]
        combined_results = combined_results[cols]
        
        # CSVファイルとして保存（閾値を含むファイル名）
        output_file = os.path.join(output_dir, f'enhanced_eye_movement_analysis_{sensitivity_threshold}.csv')
        combined_results.to_csv(output_file, index=False, encoding='utf-8')
        
        # スキップされたフォルダの情報を保存（閾値を含むファイル名）
        if skipped_folders:
            skipped_df = pd.DataFrame(skipped_folders)
            skipped_file = os.path.join(output_dir, f'skipped_folders_{sensitivity_threshold}.csv')
            skipped_df.to_csv(skipped_file, index=False, encoding='utf-8')
        
        print(f"\n分析完了:")
        print(f"- 処理された結果: {len(all_results)}件")
        print(f"- スキップされたフォルダ: {len(skipped_folders)}件")
        print(f"保存先:")
        print(f"- 分析結果: {output_file}")
        if skipped_folders:
            print(f"- スキップ情報: {skipped_file}")

# 実行コード
if __name__ == "__main__":
    # ベースディレクトリ（GAPデータフォルダ）とアウトプットディレクトリを指定
    base_dir = r"G:\共有ドライブ\GAP_長寿研\GAPデータ"
    output_dir = r"G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\data"
    
    # 感度の閾値を設定
    sensitivity_threshold = 15
    
    # 複数のcalibrationデータを分析（感度フィルター付き）
    combined_results = analyze_multiple_calibrations(base_dir, output_dir, sensitivity_threshold)
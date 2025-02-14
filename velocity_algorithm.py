import pandas as pd

def velocity_based_algorithm_summary(eyetrack_df, xlabel, ylabel, threshold=0.01475):
    #リサンプリング 4ms
    target_sampling_rate = 4

    # カラム名の大文字小文字の違いを修正
    processed_df = eyetrack_df.copy()
    if 'TimeStamp' in processed_df.columns:
        processed_df['timestamp'] = processed_df['TimeStamp']
    
    gaze_x, gaze_y = __calc_resampled_gaze_with_df(target_sampling_rate, processed_df, xlabel, ylabel)
    velocities = __calc_velocities_with_resampled_gaze(target_sampling_rate, gaze_x, gaze_y)
    is_fixations = __calc_is_fixations(threshold, velocities)
    
    resampled_time_stamp = __resample(processed_df['timestamp'].values, processed_df['timestamp'].values, target_sampling_rate, lambda x0, y0, x1, y1, x: x)
    resampled_time_stamp = exclude_tail_close_eye(resampled_time_stamp, is_fixations)
    close_eye = __calc_resampled_close_eye_with_df(target_sampling_rate, processed_df)
    close_eye = exclude_tail_close_eye(close_eye, is_fixations)
    
    frame_df = pd.DataFrame({
        'elapsed_times': __calc_elapsed_times(resampled_time_stamp),
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'velocities': velocities,
        'is_fixations': is_fixations,
        'fixation_indices': __calc_fixation_indices(is_fixations),
        'beginning_to_move': __calc_beginning_to_move(is_fixations),
        'end_of_movement': __calc_end_of_movement(is_fixations),
        'close_eye': close_eye,
    })
    
    time_df = pd.DataFrame({
        'fixation_indices': __calc_fixation_indices_unique(is_fixations),
        'fixation_times': __calc_fixation_time(resampled_time_stamp, is_fixations)
    })
    
    return frame_df, time_df

# exclude_tail_close_eye関数も追加
def exclude_tail_close_eye(x, is_fixations):
    return x[:len(is_fixations)]
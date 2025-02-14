def get_cleansed_df(file_path):
    """
    眼球運動データのCSVファイルを読み込み、クレンジングを行う関数
    
    Parameters:
        file_path (str): CSVファイルのパス
    
    Returns:
        pd.DataFrame: クレンジング済みのDataFrame
    """
    # CSVファイルの読み込み
    df = pd.read_csv(file_path)
    
    # タイムスタンプのソート
    if 'TimeStamp' in df.columns:
        df = df.sort_values('TimeStamp')
    
    # 欠損値の処理
    # EyeCenterAngleX, EyeCenterAngleYの欠損値は線形補間
    if 'EyeCenterAngleX' in df.columns and 'EyeCenterAngleY' in df.columns:
        df['EyeCenterAngleX'] = df['EyeCenterAngleX'].interpolate(method='linear')
        df['EyeCenterAngleY'] = df['EyeCenterAngleY'].interpolate(method='linear')
    
    # 異常値の除去（角度の範囲チェック）
    if 'EyeCenterAngleX' in df.columns and 'EyeCenterAngleY' in df.columns:
        # 極端な値を除外（例：±90度を超える値）
        df.loc[df['EyeCenterAngleX'].abs() > 90, 'EyeCenterAngleX'] = np.nan
        df.loc[df['EyeCenterAngleY'].abs() > 90, 'EyeCenterAngleY'] = np.nan
    
    # CloseEyeカラムの追加（IsValidを基に設定）
    if 'IsValid' in df.columns and 'CloseEye' not in df.columns:
        df['CloseEye'] = df['IsValid'].apply(lambda x: 1 if x == 0 else 0)
    
    return df
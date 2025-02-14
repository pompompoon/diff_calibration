{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "何をもって俊敏さを評価するか  \n",
    "\n",
    "平均速度、最大速度[deg/ms]  \n",
    "平均加速度、最大加速度  \n",
    "平均躍度、最大躍度\n",
    "サッケード回数[回]  \n",
    "方向転換の頻度[回/ms]  \n",
    "停止時間の合計[ms]  \n",
    "速度V以上の時間の合計[ms]  \n",
    "移動距離の合計[deg]  \n",
    "サッケード中の移動距離の合計[deg]  \n",
    "FFT(高周波成分が強いほど、視線の動きが速く変化していることを示す)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('G:/共有ドライブ/MCI/v2/script/src')\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import pandas as pd\n",
    "pd.set_option('display.expand_frame_repr', False) # DataFrameの表示を折り返さない"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Added path: G:/共有ドライブ/MCI/v2/script/src\n",
      "Successfully imported modules\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "import os\n",
    "\n",
    "# パスの設定 - 画像から見えるパス構造に基づく\n",
    "src_path = 'G:/共有ドライブ/MCI/v2/script/src'\n",
    "\n",
    "# パスが存在するか確認と追加\n",
    "if os.path.exists(src_path):\n",
    "    sys.path.append(src_path)\n",
    "    print(f\"Added path: {src_path}\")\n",
    "else:\n",
    "    print(f\"Warning: Path does not exist: {src_path}\")\n",
    "\n",
    "try:\n",
    "    # .pycファイルが存在するので、モジュール名だけで importできるはず\n",
    "    import mciutil\n",
    "    import velocity_algorithm\n",
    "    print(\"Successfully imported modules\")\n",
    "except ImportError as e:\n",
    "    print(f\"Import error: {e}\")\n",
    "    print(\"Current sys.path:\")\n",
    "    for p in sys.path:\n",
    "        print(f\"  {p}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "列名一覧:\n",
      "['Index', 'CalibrationCount', 'EyeSide', 'CalibrationPointerX', 'CalibrationPointerY', 'DrawPointerX', 'DrawPointerY', 'timestamp', 'EyeCenterX', 'EyeCenterY', 'RotatedEyeCenterX', 'RotatedEyeCenterY', 'IsFixed', 'RetryCount', 'IsBadPosition', 'IsValid', 'PupilSize', 'PupilAspect', 'Phase', 'EyeCenterAngleX', 'EyeCenterAngleY']\n",
      "\n",
      "データの最初の5行:\n",
      "   Index  CalibrationCount  EyeSide  CalibrationPointerX  CalibrationPointerY  DrawPointerX  DrawPointerY  timestamp  EyeCenterX  EyeCenterY  ...  RotatedEyeCenterY  IsFixed  RetryCount  IsBadPosition  IsValid  PupilSize  PupilAspect  Phase  EyeCenterAngleX  EyeCenterAngleY\n",
      "0      0                 1        1                  NaN                  NaN           0.0           0.0         19         302         223  ...           106.2614        0         NaN              0        0         51          720      1         0.776040         1.559823\n",
      "1      1                 1        1                  NaN                  NaN           0.0           0.0         37         302         223  ...           106.2614        0         NaN              0        0         53          722      1         0.776040         1.559823\n",
      "2      2                 1        1                  NaN                  NaN           0.0           0.0         57         302         224  ...           107.2011        0         NaN              0        0         54          724      1         0.916767         1.034734\n",
      "3      3                 1        1                  NaN                  NaN           0.0           0.0         72         302         224  ...           107.2011        0         NaN              0        0         56          728      1         0.916767         1.034734\n",
      "4      4                 1        1                  NaN                  NaN           0.0           0.0         88         302         223  ...           106.2614        0         NaN              0        0         54          727      1         0.776040         1.559823\n",
      "\n",
      "[5 rows x 21 columns]\n"
     ]
    }
   ],
   "source": [
    "# CSVファイルの内容を確認\n",
    "file_path = \"G:/共有ドライブ/GAP_長寿研/user/iwamoto/20241009101605_1/calibration_1_20241009_20241119145059.csv\"\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# カラム名の表示\n",
    "print(\"列名一覧:\")\n",
    "print(df.columns.tolist())\n",
    "\n",
    "# 最初の数行を表示\n",
    "print(\"\\nデータの最初の5行:\")\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "解析結果の最初の5行:\n",
      "   timestamp    gaze_x    gaze_y  velocities  is_fixations  fixation_indices\n",
      "0         19  0.776040  1.559823    0.000000         False                 0\n",
      "1         37  0.776040  1.559823    0.135905         False                 0\n",
      "2         57  0.916767  1.034734    0.000000         False                 0\n",
      "3         72  0.916767  1.034734    0.135905         False                 0\n",
      "4         88  0.776040  1.559823    0.135905         False                 0\n"
     ]
    }
   ],
   "source": [
    "# 必要なカラム名の追加とデータの前処理\n",
    "processed_df = df.copy()\n",
    "\n",
    "# CloseEyeカラムを追加（IsValidを基に設定：IsValid=0の場合を瞬きとみなす）\n",
    "processed_df['CloseEye'] = processed_df['IsValid'].apply(lambda x: 1 if x == 0 else 0)\n",
    "\n",
    "# velocity_based_algorithm_summaryの呼び出し\n",
    "frame_df, time_df = velocity_based_algorithm_summary(\n",
    "    processed_df,  # 処理済みデータフレーム\n",
    "    'EyeCenterAngleX',  # x座標のカラム名\n",
    "    'EyeCenterAngleY',  # y座標のカラム名\n",
    "    threshold=0.0003  # 閾値\n",
    ")\n",
    "\n",
    "# 結果の確認\n",
    "print(\"\\n解析結果の最初の5行:\")\n",
    "print(frame_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 必要なカラム名の追加とデータの前処理\n",
    "processed_df = df.copy()\n",
    "\n",
    "# CloseEyeカラムを追加（IsValidを基に設定）\n",
    "processed_df['CloseEye'] = processed_df['IsValid'].apply(lambda x: 1 if x == 0 else 0)\n",
    "\n",
    "# velocity_based_algorithm_summaryの呼び出し\n",
    "frame_df, time_df = velocity_based_algorithm_summary(\n",
    "    processed_df,  # 処理済みデータフレーム\n",
    "    'EyeCenterAngleX',  # x座標のカラム名\n",
    "    'EyeCenterAngleY',  # y座標のカラム名\n",
    "    threshold=0.0003  # 閾値\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "# データの読み込みとクレンジング\n",
    "df = get_cleansed_df(\"G:/共有ドライブ/GAP_長寿研/user/iwamoto/20241009101605_1/calibration_1_20241009_20241119145059.csv\")\n",
    "\n",
    "# 速度ベースの分析実行\n",
    "frame_df, time_df = velocity_based_algorithm_summary(\n",
    "    df,\n",
    "    'EyeCenterAngleX',\n",
    "    'EyeCenterAngleY',\n",
    "    threshold=0.0003\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "データフレームのカラム名:\n",
      "['Index', 'CalibrationCount', 'EyeSide', 'CalibrationPointerX', 'CalibrationPointerY', 'DrawPointerX', 'DrawPointerY', 'timestamp', 'EyeCenterX', 'EyeCenterY', 'RotatedEyeCenterX', 'RotatedEyeCenterY', 'IsFixed', 'RetryCount', 'IsBadPosition', 'IsValid', 'PupilSize', 'PupilAspect', 'Phase', 'EyeCenterAngleX', 'EyeCenterAngleY']\n",
      "\n",
      "データの最初の5行:\n",
      "   Index  CalibrationCount  EyeSide  CalibrationPointerX  CalibrationPointerY  DrawPointerX  DrawPointerY  timestamp  EyeCenterX  EyeCenterY  ...  RotatedEyeCenterY  IsFixed  RetryCount  IsBadPosition  IsValid  PupilSize  PupilAspect  Phase  EyeCenterAngleX  EyeCenterAngleY\n",
      "0      0                 1        1                  NaN                  NaN           0.0           0.0         19         302         223  ...           106.2614        0         NaN              0        0         51          720      1         0.776040         1.559823\n",
      "1      1                 1        1                  NaN                  NaN           0.0           0.0         37         302         223  ...           106.2614        0         NaN              0        0         53          722      1         0.776040         1.559823\n",
      "2      2                 1        1                  NaN                  NaN           0.0           0.0         57         302         224  ...           107.2011        0         NaN              0        0         54          724      1         0.916767         1.034734\n",
      "3      3                 1        1                  NaN                  NaN           0.0           0.0         72         302         224  ...           107.2011        0         NaN              0        0         56          728      1         0.916767         1.034734\n",
      "4      4                 1        1                  NaN                  NaN           0.0           0.0         88         302         223  ...           106.2614        0         NaN              0        0         54          727      1         0.776040         1.559823\n",
      "\n",
      "[5 rows x 21 columns]\n"
     ]
    }
   ],
   "source": [
    "# データフレームの列名を確認\n",
    "print(\"データフレームのカラム名:\")\n",
    "print(df.columns.tolist())\n",
    "\n",
    "# 最初の数行を表示して構造を確認\n",
    "print(\"\\nデータの最初の5行:\")\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('G:/共有ドライブ/MCI/v2/script/src')\n",
    "from mciutil import get_cleansed_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'movie_pos_x'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:3805\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3804\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m-> 3805\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine\u001b[38;5;241m.\u001b[39mget_loc(casted_key)\n\u001b[0;32m   3806\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n",
      "File \u001b[1;32mindex.pyx:167\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mindex.pyx:196\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\\\_libs\\\\hashtable_class_helper.pxi:7081\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\\\_libs\\\\hashtable_class_helper.pxi:7089\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'movie_pos_x'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[80], line 10\u001b[0m\n\u001b[0;32m      7\u001b[0m df \u001b[38;5;241m=\u001b[39m get_cleansed_df(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mG:/共有ドライブ/GAP_長寿研/user/iwamoto/20241009101605_1/calibration_1_20241009_20241119145059.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      9\u001b[0m \u001b[38;5;66;03m# 速度ベースの分析実行\u001b[39;00m\n\u001b[1;32m---> 10\u001b[0m frame_df, time_df \u001b[38;5;241m=\u001b[39m velocity_based_algorithm_summary(df)\n",
      "File \u001b[1;32mg:\\共有ドライブ\\GAP_長寿研\\user\\iwamoto\\視線の動きの俊敏さ\\velocity_algorithm.py:73\u001b[0m, in \u001b[0;36mvelocity_based_algorithm_summary\u001b[1;34m(df, x_col, y_col, threshold)\u001b[0m\n\u001b[0;32m     58\u001b[0m def velocity_based_algorithm_summary(df, x_col='EyeCenterAngleX', y_col='EyeCenterAngleY', threshold=0.0003, timestamp_col='TimeStamp'):\n\u001b[0;32m     59\u001b[0m     \"\"\"\n\u001b[0;32m     60\u001b[0m     速度ベースのアルゴリズムで眼球運動を分析する\n\u001b[0;32m     61\u001b[0m     \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     71\u001b[0m         pd.DataFrame: 固視ごとの時間情報\n\u001b[0;32m     72\u001b[0m     \"\"\"\n\u001b[1;32m---> 73\u001b[0m     # 座標データの取得\n\u001b[0;32m     74\u001b[0m     x = df[x_col].values\n\u001b[0;32m     75\u001b[0m     y = df[y_col].values\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:4102\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   4100\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   4101\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 4102\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mget_loc(key)\n\u001b[0;32m   4103\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   4104\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:3812\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3807\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(casted_key, \u001b[38;5;28mslice\u001b[39m) \u001b[38;5;129;01mor\u001b[39;00m (\n\u001b[0;32m   3808\u001b[0m         \u001b[38;5;28misinstance\u001b[39m(casted_key, abc\u001b[38;5;241m.\u001b[39mIterable)\n\u001b[0;32m   3809\u001b[0m         \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28many\u001b[39m(\u001b[38;5;28misinstance\u001b[39m(x, \u001b[38;5;28mslice\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m x \u001b[38;5;129;01min\u001b[39;00m casted_key)\n\u001b[0;32m   3810\u001b[0m     ):\n\u001b[0;32m   3811\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m InvalidIndexError(key)\n\u001b[1;32m-> 3812\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m   3813\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[0;32m   3814\u001b[0m     \u001b[38;5;66;03m# If we have a listlike key, _check_indexing_error will raise\u001b[39;00m\n\u001b[0;32m   3815\u001b[0m     \u001b[38;5;66;03m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[39;00m\n\u001b[0;32m   3816\u001b[0m     \u001b[38;5;66;03m#  the TypeError.\u001b[39;00m\n\u001b[0;32m   3817\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 'movie_pos_x'"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "sys.path.append('G:/共有ドライブ/MCI/v2/script/src')\n",
    "from mciutil import get_cleansed_df\n",
    "from velocity_algorithm import velocity_based_algorithm_summary\n",
    "\n",
    "# データの読み込みとクレンジング\n",
    "df = get_cleansed_df(\"G:/共有ドライブ/GAP_長寿研/user/iwamoto/20241009101605_1/calibration_1_20241009_20241119145059.csv\")\n",
    "\n",
    "# 速度ベースの分析実行\n",
    "frame_df, time_df = velocity_based_algorithm_summary(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "# vscodeで実行\n",
    "path = 'G:/共有ドライブ/MCI/v2'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "dir = 'kyoto'\n",
    "period = '202409'\n",
    "groups = [\n",
    "    'control',\n",
    "    'mci',\n",
    "    ]\n",
    "\n",
    "import_path = f'{path}/output/v2.1/{dir}/{period}/0th_analysis'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Others', 'Unknown', 'Mixed', 'AD', nan, 'LBD', 'Vascular', 'iNPH']"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mci_subtype = pd.read_csv(r'G:\\共有ドライブ\\MCI\\v2\\master\\old\\mci_subtype_mmse.csv')\n",
    "\n",
    "mci_subtype_list = list(mci_subtype['subtype'].unique())\n",
    "mci_subtype_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FINDEXID</th>\n",
       "      <th>subtype</th>\n",
       "      <th>mmse</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2_20046</td>\n",
       "      <td>Others</td>\n",
       "      <td>27.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2_20047</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2_20049</td>\n",
       "      <td>Mixed</td>\n",
       "      <td>20.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2_20117</td>\n",
       "      <td>AD</td>\n",
       "      <td>25.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2_20124</td>\n",
       "      <td>Others</td>\n",
       "      <td>20.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  FINDEXID  subtype  mmse\n",
       "0  2_20046   Others  27.0\n",
       "1  2_20047  Unknown  30.0\n",
       "2  2_20049    Mixed  20.0\n",
       "3  2_20117       AD  25.0\n",
       "4  2_20124   Others  20.0"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mci_subtype.head(n=5)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "group name: control\n",
      "125\n",
      "group name: mci\n",
      "80\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(4510205, 5)"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import glob\n",
    "import numpy as np\n",
    "import time\n",
    "\n",
    "cleansing_parameters = pd.DataFrame()\n",
    "start_time = time.time()\n",
    "group_paths = glob.glob(import_path+'/*')\n",
    "group_paths = [x.replace(\"\\\\\",\"/\") for x in group_paths]\n",
    "group_paths = [path for path in group_paths if any(group_name in path for group_name in groups)]\n",
    "dfs = pd.DataFrame()\n",
    "\n",
    "for index, group in enumerate(group_paths):\n",
    "    group_name = group.split('/')[-1]\n",
    "    print(f'group name: {group_name}')\n",
    "\n",
    "    files = glob.glob(f'{import_path}/{group_name}/*.csv')\n",
    "    files = [x.replace(\"\\\\\",\"/\") for x in files]\n",
    "    print(len(files))\n",
    "    for inspection_dir in files:\n",
    "        ins_name = inspection_dir.split('/')[-1].split('.')[0]\n",
    "        if int(ins_name.split(\"_\")[0]) == 4:\n",
    "            id = ins_name.split(\"_\")[0] + \"_\" + ins_name.split(\"_\")[1] + \"_\" + ins_name.split(\"_\")[2]\n",
    "        else:\n",
    "            id = ins_name.split(\"_\")[0] + \"_\" + ins_name.split(\"_\")[1]\n",
    "        inspection_date = int(ins_name.split('_')[-1])\n",
    "        '''\n",
    "        現行特徴量と比較するため、サブタイプの有無は考慮しない\n",
    "        '''\n",
    "        # if group_name == 'control':\n",
    "        #     df_index = 0\n",
    "        #     subtype = np.nan\n",
    "        #     mmse = np.nan\n",
    "\n",
    "        # if group_name == 'mci':\n",
    "        #     matched_line = mci_subtype[mci_subtype['FINDEXID'] == id]\n",
    "        #     if len(matched_line) == 0:\n",
    "        #         print('a')\n",
    "        #         continue\n",
    "        #     subtype = mci_subtype[mci_subtype['FINDEXID'] == id]['subtype'].iloc[0]\n",
    "        #     mmse = mci_subtype[mci_subtype['FINDEXID'] == id]['mmse'].iloc[0]\n",
    "        #     df_index = mci_subtype_list.index(subtype) + 1\n",
    "        df = get_cleansed_df(inspection_dir)\n",
    "        df[\"id\"] = ins_name\n",
    "        df[\"group_name\"] = group_name.upper()\n",
    "        # df[\"subtype\"] = subtype\n",
    "        # df[\"mmse\"] = mmse\n",
    "        dfs = pd.concat([dfs, df])\n",
    "dfs.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'df' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[16], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m df\u001b[38;5;241m.\u001b[39mhead(n\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m5\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'df' is not defined"
     ]
    }
   ],
   "source": [
    "df.head(n=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: []\n",
       "Index: []"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfs.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'id'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[8], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m ids \u001b[38;5;241m=\u001b[39m dfs[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mid\u001b[39m\u001b[38;5;124m\"\u001b[39m]\u001b[38;5;241m.\u001b[39munique()\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28mlen\u001b[39m(ids)\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:4102\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   4100\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   4101\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 4102\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mget_loc(key)\n\u001b[0;32m   4103\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   4104\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\range.py:417\u001b[0m, in \u001b[0;36mRangeIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    415\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m    416\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Hashable):\n\u001b[1;32m--> 417\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n\u001b[0;32m    418\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n\u001b[0;32m    419\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 'id'"
     ]
    }
   ],
   "source": [
    "ids = dfs[\"id\"].unique()\n",
    "len(ids)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# -----rawdataの用意完了-----"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_velocity_info(df): # 速度の平均、最大値\n",
    "    velo_mean = df['velocities'].mean()\n",
    "    velo_max = df['velocities'].max()\n",
    "    return velo_mean, velo_max\n",
    "\n",
    "def get_acceleration_info(df): # 加速度の平均、最大値\n",
    "    acce_mean = df['velocities'].diff().mean()\n",
    "    acce_max = df['velocities'].diff().max()\n",
    "    return acce_mean, acce_max\n",
    "\n",
    "def get_jerk_info(df):  # 躍度の平均、最大値\n",
    "    dt = 4  # sec/frame\n",
    "    \n",
    "    # 時間間隔を考慮した躍度の計算\n",
    "    acceleration = df['velocities'].diff() / dt\n",
    "    jerk = acceleration.diff() / dt\n",
    "    \n",
    "    \n",
    "    # NaNを除外して統計量を計算\n",
    "    jerk_clean = jerk.dropna()\n",
    "    jerk_mean = jerk.mean()\n",
    "    jerk_max = jerk.max()\n",
    "    jerk_min = jerk_clean.min()  # 負の躍度も重要な情報\n",
    "    \n",
    "    return jerk_mean, jerk_max\n",
    "\n",
    "def get_saccade_count(df): # サッケード回数\n",
    "    count = df['fixation_indices'].max()\n",
    "    # 最後が固視で終了していたら\n",
    "    if not pd.isna(df['fixation_indices'].iloc[-1]):\n",
    "        count -= 1\n",
    "    return count\n",
    "\n",
    "def get_turn_count(df):\n",
    "    count = 0\n",
    "    return count\n",
    "\n",
    "def get_fixtaion_time_sum(df): # 止まっている判定されている時間の合計\n",
    "    dt = 4 # sec/frame\n",
    "    time_sum = len(df[df['is_fixations'] == True]['timestamp']) * dt\n",
    "    return time_sum\n",
    "\n",
    "def get_move_time_sum(df): # 一定以上の速度が出ている時間の合計\n",
    "    threshold = 0.001\n",
    "    dt = 4 # sec/frame\n",
    "    time_sum = len(df[df['velocities'] > threshold]['timestamp']) * dt\n",
    "    return time_sum\n",
    "\n",
    "def get_move_dis_sum(df): # 移動距離の合計\n",
    "    dis_sum = df['velocities'].sum()\n",
    "    return dis_sum\n",
    "\n",
    "def get_saccade_move_dis_sum(df): # 止まっていないときの移動距離の合計\n",
    "    dis_sum = df[df['is_fixations'] == False]['velocities'].sum()\n",
    "    return dis_sum\n",
    "\n",
    "def get_velo_fft(df):\n",
    "    return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def show_graph(df_selected, params, p_index):\n",
    "    fig = plt.figure(figsize=(10,5))\n",
    "    fig.suptitle(f\"{group_name} id:{id} param:{params[p_index][4]}\")\n",
    "    ax1 = fig.add_subplot(1,2,1)\n",
    "    ax1.set_title(\"velocity\")\n",
    "    ax1.plot(df_selected[\"timestamp\"], df_selected[\"velocities\"], label='velocities')\n",
    "    ax1.plot(df_selected[\"timestamp\"], df_selected[\"velocities\"].diff(), label='acceleration')\n",
    "    ax1.set_ylim(0,0.007)\n",
    "    ax1.legend()\n",
    "\n",
    "    ax2 = fig.add_subplot(1,2,2)\n",
    "    ax2.set_title(\"軌跡\")\n",
    "    ax2.scatter(df_selected['movie_pos_x'], df_selected['movie_pos_y'], s=0.5, c='gray', alpha=0.3)\n",
    "    ax2.plot(df_selected['movie_pos_x_interpolated'], df_selected['movie_pos_y_interpolated'], c='green', alpha=0.3)\n",
    "    ax2.scatter(df_selected[df_selected['is_fixations'] == True]['movie_pos_x'], df_selected[df_selected['is_fixations'] == True]['movie_pos_y'], s=0.5, c='blue', alpha=1)\n",
    "    ax2.scatter(df_selected['movie_pos_x'].dropna().iloc[0], df_selected['movie_pos_y'].dropna().iloc[0], s=50, c='black', marker='x') # スタート\n",
    "    ax2.scatter(df_selected['movie_pos_x'].dropna().iloc[-1], df_selected['movie_pos_y'].dropna().iloc[-1], s=50, c='red', marker='x') # ラスト\n",
    "    ax2.scatter(params[p_index][2], params[p_index][3], color='red') # 視標\n",
    "    ax2.set_xlim(0,1)\n",
    "    ax2.set_ylim(1,0)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'ids' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[12], line 41\u001b[0m\n\u001b[0;32m      7\u001b[0m feature_names \u001b[38;5;241m=\u001b[39m [\n\u001b[0;32m      8\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mvelo_mean\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;66;03m# controlの方が大きい？\u001b[39;00m\n\u001b[0;32m      9\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mvelo_max\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;66;03m# controlの方が大きい？\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     16\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msaccade_move_dis_sum\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m     17\u001b[0m     ]\n\u001b[0;32m     18\u001b[0m params \u001b[38;5;241m=\u001b[39m [\n\u001b[0;32m     19\u001b[0m     [\u001b[38;5;241m25000\u001b[39m, \u001b[38;5;241m27000\u001b[39m, \u001b[38;5;241m1864\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m2880\u001b[39m, \u001b[38;5;241m385\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m1620\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m4_1\u001b[39m\u001b[38;5;124m'\u001b[39m], \u001b[38;5;66;03m# コンテンツ4_1\u001b[39;00m\n\u001b[0;32m     20\u001b[0m     [\u001b[38;5;241m28000\u001b[39m, \u001b[38;5;241m30000\u001b[39m, \u001b[38;5;241m1020\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m2880\u001b[39m, \u001b[38;5;241m1234\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m1620\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m4_2\u001b[39m\u001b[38;5;124m'\u001b[39m], \u001b[38;5;66;03m# コンテンツ4_2\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     38\u001b[0m     [\u001b[38;5;241m214000\u001b[39m, \u001b[38;5;241m219000\u001b[39m, \u001b[38;5;241m0\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m2880\u001b[39m, \u001b[38;5;241m0\u001b[39m\u001b[38;5;241m/\u001b[39m\u001b[38;5;241m1620\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m14\u001b[39m\u001b[38;5;124m'\u001b[39m], \u001b[38;5;66;03m# コンテンツ14\u001b[39;00m\n\u001b[0;32m     39\u001b[0m     ]\n\u001b[1;32m---> 41\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m \u001b[38;5;28mid\u001b[39m \u001b[38;5;129;01min\u001b[39;00m ids:\n\u001b[0;32m     42\u001b[0m     df \u001b[38;5;241m=\u001b[39m dfs[dfs[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mid\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mid\u001b[39m]\n\u001b[0;32m     43\u001b[0m     df\u001b[38;5;241m.\u001b[39mrename(columns\u001b[38;5;241m=\u001b[39m{\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mtimestamp\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTimeStamp\u001b[39m\u001b[38;5;124m'\u001b[39m}, inplace\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'ids' is not defined"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "plt.rcParams['font.family'] = 'Meiryo'\n",
    "\n",
    "result_all = pd.DataFrame()\n",
    "\n",
    "feature_names = [\n",
    "    'velo_mean', # controlの方が大きい？\n",
    "    'velo_max', # controlの方が大きい？\n",
    "    'acce_mean', # controlの方が大きい？\n",
    "    'acce_max', # controlの方が大きい？\n",
    "    'saccade_count', # mciの方が大きい？\n",
    "    'fixation_time_sum', # controlの方が大きい？\n",
    "    'move_time_sum',\n",
    "    'move_dis_sum',\n",
    "    'saccade_move_dis_sum'\n",
    "    ]\n",
    "params = [\n",
    "    [25000, 27000, 1864/2880, 385/1620, '4_1'], # コンテンツ4_1\n",
    "    [28000, 30000, 1020/2880, 1234/1620, '4_2'], # コンテンツ4_2\n",
    "    [40000, 46000, 0/2880, 0/1620, '6_1'], # コンテンツ6_1\n",
    "    [47000, 53000, 0/2880, 0/1620, '6_2'], # コンテンツ6_2\n",
    "    [54000, 60000, 0/2880, 0/1620, '6_3'], # コンテンツ6_3\n",
    "    [64500, 66500, 0/2880, 0/1620, '7_1'], # コンテンツ7_1_b\n",
    "    [70500, 72500, 0/2880, 0/1620, '7_2'], # コンテンツ7_2_b\n",
    "    [83000, 85500, 0/2880, 0/1620, '7_3'], # コンテンツ7_3_b\n",
    "    [96000, 98500, 0/2880, 0/1620, '7_4'], # コンテンツ7_4_b\n",
    "    [109000, 111500, 0/2880, 0/1620, '7_5'], # コンテンツ7_5_b\n",
    "    [91000, 96000, 0/2880, 0/1620, '8'], # コンテンツ8\n",
    "    [104000, 109000, 0/2880, 0/1620, '9'], # コンテンツ9\n",
    "    [115000, 119000, 0/2880, 0/1620, '10_1'], # コンテンツ10_1\n",
    "    [120000, 124000, 0/2880, 0/1620, '10_2'], # コンテンツ10_2\n",
    "    [125000, 129000, 0/2880, 0/1620, '10_3'], # コンテンツ10_3\n",
    "    [130000, 134000, 0/2880, 0/1620, '10_4'], # コンテンツ10_4\n",
    "    [135000, 139000, 0/2880, 0/1620, '10_5'], # コンテンツ10_5\n",
    "    [142500, 147500, 0/2880, 0/1620, '11_1'], # コンテンツ11_1\n",
    "    [147500, 152500, 0/2880, 0/1620, '11_2'], # コンテンツ11_2\n",
    "    [214000, 219000, 0/2880, 0/1620, '14'], # コンテンツ14\n",
    "    ]\n",
    "\n",
    "for id in ids:\n",
    "    df = dfs[dfs[\"id\"] == id]\n",
    "    df.rename(columns={'timestamp': 'TimeStamp'}, inplace=True)\n",
    "    df['CloseEye'] = 0\n",
    "    df['TimeStamp'] = df['TimeStamp'].astype(int)\n",
    "\n",
    "    group_name = df['group_name'].unique()[0]\n",
    "\n",
    "    frame_df, time_df = velocity_based_algorithm_summary(df, 'movie_pos_x', 'movie_pos_y', threshold=0.0003) # 元はthreshold=0.01475だが止まった判定が厳しいので閾値をあげる\n",
    "    frame_df.rename(columns={'elapsed_times': 'timestamp', 'gaze_x':'movie_pos_x',  'gaze_y':'movie_pos_y', }, inplace=True)\n",
    "\n",
    "    features_all = [group_name]\n",
    "    feature_names_all = ['group_name']\n",
    "    for p_index in range(len(params)):\n",
    "        df_selected = frame_df[(frame_df['timestamp'] >= params[p_index][0]) & (frame_df['timestamp'] < params[p_index][1])]\n",
    "        df_selected['movie_pos_x_interpolated'] = df_selected['movie_pos_x'].interpolate()\n",
    "        df_selected['movie_pos_y_interpolated'] = df_selected['movie_pos_y'].interpolate()\n",
    "        # 平均速度、最大速度[deg/ms]\n",
    "        velo_mean, velo_max = get_velocity_info(df_selected)\n",
    "        # 平均加速度、最大速度[deg/ms]\n",
    "        acce_mean, acce_max = get_acceleration_info(df_selected)\n",
    "        # サッケードの頻度[回/ms]\n",
    "        saccade_count = get_saccade_count(df_selected)\n",
    "        # # 方向転換の頻度[回/ms]\n",
    "        # turn_count = get_turn_count(df_selected)\n",
    "        # 停止時間の合計[ms]\n",
    "        fixation_time_sum = get_fixtaion_time_sum(df_selected)\n",
    "        # 速度V以上の時間の合計[ms]\n",
    "        move_time_sum = get_move_time_sum(df_selected)\n",
    "        # 移動距離の合計[deg]\n",
    "        move_dis_sum = get_move_dis_sum(df_selected)\n",
    "        # サッケード中の移動距離の合計[deg]\n",
    "        saccade_move_dis_sum = get_saccade_move_dis_sum(df_selected)\n",
    "        # # FFT(高周波成分が強いほど、視線の動きが速く変化していることを示す)\n",
    "        # velo_fft = get_velo_fft(df_selected.dropna())\n",
    "        features = [velo_mean, velo_max, acce_mean, acce_max, saccade_count, fixation_time_sum, move_time_sum, move_dis_sum, saccade_move_dis_sum]\n",
    "        features_all.extend(features)\n",
    "        feature_names_all.extend([f'{feature}_{params[p_index][4]}' for feature in feature_names])\n",
    "\n",
    "        # グラフを描画\n",
    "        # show_graph(df_selected, params, p_index)\n",
    "\n",
    "    result = pd.DataFrame(data=[features_all], columns=feature_names_all, index=[id])\n",
    "    result_all = pd.concat([result_all, result])\n",
    "print(result_all)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datetime import datetime\n",
    "# result_all.to_csv(f\"G:/共有ドライブ/MCI/v2/script/v2.1/検証中/新特徴量作成/output/視線の動きの俊敏さ_{datetime.now().strftime('%Y%m%d%H%M')}.csv\")\n",
    "result_all.to_csv(f\"G:/共有ドライブ/MCI/v2/output/v2.1/kyoto/{period}/新特徴量/稲山さん/視線の動きの俊敏さ.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ここからは特徴量の評価"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.stats import ttest_ind\n",
    "from scipy.stats import mannwhitneyu\n",
    "from scipy.stats import median_test\n",
    "from scipy.stats import ranksums\n",
    "from scipy.stats import brunnermunzel\n",
    "from scipy.stats import ks_2samp\n",
    "\n",
    "def interpret_p_value(p):\n",
    "    if p < 0.05:\n",
    "        print(\"有意差があります\")\n",
    "    else:\n",
    "        print(\"有意差がありません\")\n",
    "\n",
    "def print_value(stat, p):\n",
    "    print(f'統計量: {round(stat, 3)}')\n",
    "    print(f'p値: {round(p, 5)}')\n",
    "\n",
    "import seaborn as sns\n",
    "from matplotlib.ticker import MaxNLocator\n",
    "def show_hist_graph(column, df):\n",
    "    _, ax = plt.subplots(figsize=(4,3))\n",
    "    sns.histplot(data=df, ax=ax, x=column, hue='group_name', kde=True, palette=['blue', 'red'])\n",
    "    ax.yaxis.set_major_locator(MaxNLocator(integer=True))\n",
    "    plt.show()\n",
    "\n",
    "def show_box_graph(column, df):\n",
    "    custom_palette = {'CONTROL': 'blue', 'MCI': 'red'}\n",
    "    _, ax = plt.subplots(figsize=(4, 3))\n",
    "    sns.boxplot(x='group_name', y=column, data=df, ax=ax, palette=custom_palette, showmeans=True)\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "evaluations_df = pd.DataFrame()\n",
    "for i in range(len(list(result_all.columns))):\n",
    "    labels = []\n",
    "    values = []\n",
    "\n",
    "    column = list(result_all.columns)[i]\n",
    "    if column == 'group_name':\n",
    "        continue\n",
    "    labels.append('column')\n",
    "    values.append(column)\n",
    "\n",
    "    control_df = result_all[result_all['group_name'] == 'CONTROL'][column]\n",
    "    mci_df = result_all[result_all['group_name'] == 'MCI'][column]\n",
    "    control_df = control_df.dropna()\n",
    "    mci_df = mci_df.dropna()\n",
    "\n",
    "    # スチューデントのt検定を実行\n",
    "    stat, p = ttest_ind(control_df, mci_df, equal_var=True)\n",
    "    labels.append('stat_t検定')\n",
    "    labels.append('p_t検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "    # if p < 0.001: # 有意差があるときだけグラフを描画\n",
    "    #     show_hist_graph(column, result_all)\n",
    "    #     show_box_graph(column, result_all)\n",
    "    show_hist_graph(column, result_all)\n",
    "    show_box_graph(column, result_all)\n",
    "\n",
    "    # ウェルチのt検定を実行\n",
    "    stat, p = ttest_ind(control_df, mci_df, equal_var=False)\n",
    "    labels.append('stat_Welchst検定')\n",
    "    labels.append('p_Welchst検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "\n",
    "    # マンホイットニーのU検定を実行\n",
    "    setattr, p = mannwhitneyu(control_df, mci_df, alternative='two-sided')\n",
    "    labels.append('stat_U検定')\n",
    "    labels.append('p_U検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "\n",
    "\n",
    "    # 中央値検定\n",
    "    stat, p, med, tbl = median_test(control_df, mci_df)\n",
    "    labels.append('stat_中央値検定')\n",
    "    labels.append('p_中央値検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "\n",
    "\n",
    "    # ウィルコクソンの順位和検定\n",
    "    stat, p = ranksums(control_df, mci_df)\n",
    "    labels.append('stat_ウィルコクソンの順位和検定')\n",
    "    labels.append('p_ウィルコクソンの順位和検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "\n",
    "\n",
    "    # ブルンナー=ムンツェル検定\n",
    "    stat, p = brunnermunzel(control_df, mci_df)\n",
    "    labels.append('stat_ブルンナー=ムンツェル検定')\n",
    "    labels.append('p_ブルンナー=ムンツェル検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "\n",
    "\n",
    "    # 2標本コルモゴロフ=スミルノフ検定\n",
    "    stat, p = ks_2samp(control_df, mci_df)\n",
    "    labels.append('stat_2標本コルモゴロフ=スミルノフ検定')\n",
    "    labels.append('p_2標本コルモゴロフ=スミルノフ検定')\n",
    "    values.append(stat)\n",
    "    values.append(p)\n",
    "    evaluation_df = pd.DataFrame(data=[values], columns=labels, index=[i])\n",
    "    evaluations_df = pd.concat([evaluations_df, evaluation_df])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 0)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "evaluations_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'p_t検定'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_10096\\3444863728.py\u001b[0m in \u001b[0;36m?\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mevaluations_df\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msort_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'p_t検定'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'column'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'p_t検定'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'p_Welchst検定'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'p_U検定'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m?\u001b[1;34m(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)\u001b[0m\n\u001b[0;32m   7185\u001b[0m             \u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   7186\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mby\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   7187\u001b[0m             \u001b[1;31m# len(by) == 1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   7188\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 7189\u001b[1;33m             \u001b[0mk\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_label_or_level_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mby\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   7190\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   7191\u001b[0m             \u001b[1;31m# need to rewrap column in Series to apply key function\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   7192\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m?\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1907\u001b[0m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mxs\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mother_axes\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1908\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_is_level_reference\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1909\u001b[0m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0maxes\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_level_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1910\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1911\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1912\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1913\u001b[0m         \u001b[1;31m# Check for duplicates\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1914\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'p_t検定'"
     ]
    }
   ],
   "source": [
    "evaluations_df.sort_values('p_t検定').head(20)[['column','p_t検定', 'p_Welchst検定','p_U検定']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "\"None of [Index(['p_t検定'], dtype='object')] are in the [columns]\"",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\IPython\\core\\formatters.py:406\u001b[0m, in \u001b[0;36mBaseFormatter.__call__\u001b[1;34m(self, obj)\u001b[0m\n\u001b[0;32m    404\u001b[0m     method \u001b[38;5;241m=\u001b[39m get_real_method(obj, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprint_method)\n\u001b[0;32m    405\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m method \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 406\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m method()\n\u001b[0;32m    407\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    408\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style.py:405\u001b[0m, in \u001b[0;36mStyler._repr_html_\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    400\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    401\u001b[0m \u001b[38;5;124;03mHooks into Jupyter notebook rich display system, which calls _repr_html_ by\u001b[39;00m\n\u001b[0;32m    402\u001b[0m \u001b[38;5;124;03mdefault if an object is returned at the end of a cell.\u001b[39;00m\n\u001b[0;32m    403\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    404\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m get_option(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstyler.render.repr\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhtml\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[1;32m--> 405\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mto_html()\n\u001b[0;32m    406\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style.py:1345\u001b[0m, in \u001b[0;36mStyler.to_html\u001b[1;34m(self, buf, table_uuid, table_attributes, sparse_index, sparse_columns, bold_headers, caption, max_rows, max_columns, encoding, doctype_html, exclude_styles, **kwargs)\u001b[0m\n\u001b[0;32m   1342\u001b[0m     obj\u001b[38;5;241m.\u001b[39mset_caption(caption)\n\u001b[0;32m   1344\u001b[0m \u001b[38;5;66;03m# Build HTML string..\u001b[39;00m\n\u001b[1;32m-> 1345\u001b[0m html \u001b[38;5;241m=\u001b[39m obj\u001b[38;5;241m.\u001b[39m_render_html(\n\u001b[0;32m   1346\u001b[0m     sparse_index\u001b[38;5;241m=\u001b[39msparse_index,\n\u001b[0;32m   1347\u001b[0m     sparse_columns\u001b[38;5;241m=\u001b[39msparse_columns,\n\u001b[0;32m   1348\u001b[0m     max_rows\u001b[38;5;241m=\u001b[39mmax_rows,\n\u001b[0;32m   1349\u001b[0m     max_cols\u001b[38;5;241m=\u001b[39mmax_columns,\n\u001b[0;32m   1350\u001b[0m     exclude_styles\u001b[38;5;241m=\u001b[39mexclude_styles,\n\u001b[0;32m   1351\u001b[0m     encoding\u001b[38;5;241m=\u001b[39mencoding \u001b[38;5;129;01mor\u001b[39;00m get_option(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstyler.render.encoding\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[0;32m   1352\u001b[0m     doctype_html\u001b[38;5;241m=\u001b[39mdoctype_html,\n\u001b[0;32m   1353\u001b[0m     \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs,\n\u001b[0;32m   1354\u001b[0m )\n\u001b[0;32m   1356\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m save_to_buffer(\n\u001b[0;32m   1357\u001b[0m     html, buf\u001b[38;5;241m=\u001b[39mbuf, encoding\u001b[38;5;241m=\u001b[39m(encoding \u001b[38;5;28;01mif\u001b[39;00m buf \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[0;32m   1358\u001b[0m )\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style_render.py:204\u001b[0m, in \u001b[0;36mStylerRenderer._render_html\u001b[1;34m(self, sparse_index, sparse_columns, max_rows, max_cols, **kwargs)\u001b[0m\n\u001b[0;32m    192\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_render_html\u001b[39m(\n\u001b[0;32m    193\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[0;32m    194\u001b[0m     sparse_index: \u001b[38;5;28mbool\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    198\u001b[0m     \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs,\n\u001b[0;32m    199\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mstr\u001b[39m:\n\u001b[0;32m    200\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    201\u001b[0m \u001b[38;5;124;03m    Renders the ``Styler`` including all applied styles to HTML.\u001b[39;00m\n\u001b[0;32m    202\u001b[0m \u001b[38;5;124;03m    Generates a dict with necessary kwargs passed to jinja2 template.\u001b[39;00m\n\u001b[0;32m    203\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 204\u001b[0m     d \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_render(sparse_index, sparse_columns, max_rows, max_cols, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m&nbsp;\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m    205\u001b[0m     d\u001b[38;5;241m.\u001b[39mupdate(kwargs)\n\u001b[0;32m    206\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtemplate_html\u001b[38;5;241m.\u001b[39mrender(\n\u001b[0;32m    207\u001b[0m         \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39md,\n\u001b[0;32m    208\u001b[0m         html_table_tpl\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtemplate_html_table,\n\u001b[0;32m    209\u001b[0m         html_style_tpl\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtemplate_html_style,\n\u001b[0;32m    210\u001b[0m     )\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style_render.py:161\u001b[0m, in \u001b[0;36mStylerRenderer._render\u001b[1;34m(self, sparse_index, sparse_columns, max_rows, max_cols, blank)\u001b[0m\n\u001b[0;32m    147\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_render\u001b[39m(\n\u001b[0;32m    148\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[0;32m    149\u001b[0m     sparse_index: \u001b[38;5;28mbool\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    153\u001b[0m     blank: \u001b[38;5;28mstr\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    154\u001b[0m ):\n\u001b[0;32m    155\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    156\u001b[0m \u001b[38;5;124;03m    Computes and applies styles and then generates the general render dicts.\u001b[39;00m\n\u001b[0;32m    157\u001b[0m \n\u001b[0;32m    158\u001b[0m \u001b[38;5;124;03m    Also extends the `ctx` and `ctx_index` attributes with those of concatenated\u001b[39;00m\n\u001b[0;32m    159\u001b[0m \u001b[38;5;124;03m    stylers for use within `_translate_latex`\u001b[39;00m\n\u001b[0;32m    160\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 161\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compute()\n\u001b[0;32m    162\u001b[0m     dxs \u001b[38;5;241m=\u001b[39m []\n\u001b[0;32m    163\u001b[0m     ctx_len \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mindex)\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style_render.py:256\u001b[0m, in \u001b[0;36mStylerRenderer._compute\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    254\u001b[0m r \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\n\u001b[0;32m    255\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m func, args, kwargs \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_todo:\n\u001b[1;32m--> 256\u001b[0m     r \u001b[38;5;241m=\u001b[39m func(\u001b[38;5;28mself\u001b[39m)(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    257\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m r\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\io\\formats\\style.py:2027\u001b[0m, in \u001b[0;36mStyler._map\u001b[1;34m(self, func, subset, **kwargs)\u001b[0m\n\u001b[0;32m   2025\u001b[0m     subset \u001b[38;5;241m=\u001b[39m IndexSlice[:]\n\u001b[0;32m   2026\u001b[0m subset \u001b[38;5;241m=\u001b[39m non_reducing_slice(subset)\n\u001b[1;32m-> 2027\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdata\u001b[38;5;241m.\u001b[39mloc[subset]\u001b[38;5;241m.\u001b[39mmap(func)\n\u001b[0;32m   2028\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_update_ctx(result)\n\u001b[0;32m   2029\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1184\u001b[0m, in \u001b[0;36m_LocationIndexer.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   1182\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_is_scalar_access(key):\n\u001b[0;32m   1183\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mobj\u001b[38;5;241m.\u001b[39m_get_value(\u001b[38;5;241m*\u001b[39mkey, takeable\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_takeable)\n\u001b[1;32m-> 1184\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_tuple(key)\n\u001b[0;32m   1185\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   1186\u001b[0m     \u001b[38;5;66;03m# we by definition only have the 0th axis\u001b[39;00m\n\u001b[0;32m   1187\u001b[0m     axis \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39maxis \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;241m0\u001b[39m\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1377\u001b[0m, in \u001b[0;36m_LocIndexer._getitem_tuple\u001b[1;34m(self, tup)\u001b[0m\n\u001b[0;32m   1374\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_multi_take_opportunity(tup):\n\u001b[0;32m   1375\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_multi_take(tup)\n\u001b[1;32m-> 1377\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_tuple_same_dim(tup)\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1020\u001b[0m, in \u001b[0;36m_LocationIndexer._getitem_tuple_same_dim\u001b[1;34m(self, tup)\u001b[0m\n\u001b[0;32m   1017\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m com\u001b[38;5;241m.\u001b[39mis_null_slice(key):\n\u001b[0;32m   1018\u001b[0m     \u001b[38;5;28;01mcontinue\u001b[39;00m\n\u001b[1;32m-> 1020\u001b[0m retval \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mgetattr\u001b[39m(retval, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mname)\u001b[38;5;241m.\u001b[39m_getitem_axis(key, axis\u001b[38;5;241m=\u001b[39mi)\n\u001b[0;32m   1021\u001b[0m \u001b[38;5;66;03m# We should never have retval.ndim < self.ndim, as that should\u001b[39;00m\n\u001b[0;32m   1022\u001b[0m \u001b[38;5;66;03m#  be handled by the _getitem_lowerdim call above.\u001b[39;00m\n\u001b[0;32m   1023\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m retval\u001b[38;5;241m.\u001b[39mndim \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mndim\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1420\u001b[0m, in \u001b[0;36m_LocIndexer._getitem_axis\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1417\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mhasattr\u001b[39m(key, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mndim\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;129;01mand\u001b[39;00m key\u001b[38;5;241m.\u001b[39mndim \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   1418\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCannot index with multidimensional key\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m-> 1420\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_iterable(key, axis\u001b[38;5;241m=\u001b[39maxis)\n\u001b[0;32m   1422\u001b[0m \u001b[38;5;66;03m# nested tuple slicing\u001b[39;00m\n\u001b[0;32m   1423\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_nested_tuple(key, labels):\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1360\u001b[0m, in \u001b[0;36m_LocIndexer._getitem_iterable\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1357\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_key(key, axis)\n\u001b[0;32m   1359\u001b[0m \u001b[38;5;66;03m# A collection of keys\u001b[39;00m\n\u001b[1;32m-> 1360\u001b[0m keyarr, indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_listlike_indexer(key, axis)\n\u001b[0;32m   1361\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mobj\u001b[38;5;241m.\u001b[39m_reindex_with_indexers(\n\u001b[0;32m   1362\u001b[0m     {axis: [keyarr, indexer]}, copy\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, allow_dups\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m   1363\u001b[0m )\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexing.py:1558\u001b[0m, in \u001b[0;36m_LocIndexer._get_listlike_indexer\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1555\u001b[0m ax \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mobj\u001b[38;5;241m.\u001b[39m_get_axis(axis)\n\u001b[0;32m   1556\u001b[0m axis_name \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mobj\u001b[38;5;241m.\u001b[39m_get_axis_name(axis)\n\u001b[1;32m-> 1558\u001b[0m keyarr, indexer \u001b[38;5;241m=\u001b[39m ax\u001b[38;5;241m.\u001b[39m_get_indexer_strict(key, axis_name)\n\u001b[0;32m   1560\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m keyarr, indexer\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:6200\u001b[0m, in \u001b[0;36mIndex._get_indexer_strict\u001b[1;34m(self, key, axis_name)\u001b[0m\n\u001b[0;32m   6197\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   6198\u001b[0m     keyarr, indexer, new_indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_reindex_non_unique(keyarr)\n\u001b[1;32m-> 6200\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_raise_if_missing(keyarr, indexer, axis_name)\n\u001b[0;32m   6202\u001b[0m keyarr \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtake(indexer)\n\u001b[0;32m   6203\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Index):\n\u001b[0;32m   6204\u001b[0m     \u001b[38;5;66;03m# GH 42790 - Preserve name from an Index\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:6249\u001b[0m, in \u001b[0;36mIndex._raise_if_missing\u001b[1;34m(self, key, indexer, axis_name)\u001b[0m\n\u001b[0;32m   6247\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m nmissing:\n\u001b[0;32m   6248\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m nmissing \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mlen\u001b[39m(indexer):\n\u001b[1;32m-> 6249\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNone of [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mkey\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m] are in the [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00maxis_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m]\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m   6251\u001b[0m     not_found \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(ensure_index(key)[missing_mask\u001b[38;5;241m.\u001b[39mnonzero()[\u001b[38;5;241m0\u001b[39m]]\u001b[38;5;241m.\u001b[39munique())\n\u001b[0;32m   6252\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mnot_found\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m not in index\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mKeyError\u001b[0m: \"None of [Index(['p_t検定'], dtype='object')] are in the [columns]\""
     ]
    },
    {
     "data": {
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x170a5ecb860>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def highlight_red(val):\n",
    "    color = 'red' if val < 0.05 else ''\n",
    "    return f'background-color: {color}'\n",
    "\n",
    "# スタイルを適用して表示\n",
    "styled_df = evaluations_df.style.applymap(highlight_red, subset=['p_t検定'])\n",
    "styled_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'group'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[17], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m correlation \u001b[38;5;241m=\u001b[39m result_all[result_all[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgroup\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m==\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcontrol\u001b[39m\u001b[38;5;124m'\u001b[39m][\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m6-1_move_time_sum\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mcorr(result_all[result_all[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgroup\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m==\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcontrol\u001b[39m\u001b[38;5;124m'\u001b[39m][\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m10-5_move_time_sum\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcontrol相関:\u001b[39m\u001b[38;5;124m\"\u001b[39m, correlation)\n\u001b[0;32m      3\u001b[0m correlation \u001b[38;5;241m=\u001b[39m result_all[result_all[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgroup\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m==\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmci\u001b[39m\u001b[38;5;124m'\u001b[39m][\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m6-1_move_time_sum\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mcorr(result_all[result_all[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgroup\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m==\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmci\u001b[39m\u001b[38;5;124m'\u001b[39m][\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m10-5_move_time_sum\u001b[39m\u001b[38;5;124m'\u001b[39m])\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:4102\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   4100\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   4101\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 4102\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mget_loc(key)\n\u001b[0;32m   4103\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   4104\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\range.py:417\u001b[0m, in \u001b[0;36mRangeIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    415\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m    416\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Hashable):\n\u001b[1;32m--> 417\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n\u001b[0;32m    418\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n\u001b[0;32m    419\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 'group'"
     ]
    }
   ],
   "source": [
    "correlation = result_all[result_all['group']=='control']['6-1_move_time_sum'].corr(result_all[result_all['group']=='control']['10-5_move_time_sum'])\n",
    "print(\"control相関:\", correlation)\n",
    "correlation = result_all[result_all['group']=='mci']['6-1_move_time_sum'].corr(result_all[result_all['group']=='mci']['10-5_move_time_sum'])\n",
    "print(\"mci相関:\", correlation)\n",
    "plt.scatter(result_all[result_all['group']=='control']['6-1_move_time_sum'],result_all[result_all['group']=='control']['10-5_move_time_sum'], c='blue')\n",
    "plt.scatter(result_all[result_all['group']=='mci']['6-1_move_time_sum'],result_all[result_all['group']=='mci']['10-5_move_time_sum'], c='red')\n",
    "plt.show()\n",
    "\n",
    "correlation = result_all['6-1_move_time_sum'].corr(result_all['6-2_move_dis_sum'])\n",
    "print(\"相関:\", correlation)\n",
    "plt.scatter(result_all[result_all['group']=='control']['6-1_move_time_sum'],result_all[result_all['group']=='control']['6-2_move_dis_sum'], c='blue')\n",
    "plt.scatter(result_all[result_all['group']=='mci']['6-1_move_time_sum'],result_all[result_all['group']=='mci']['6-2_move_dis_sum'], c='red')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'p_t検定'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[18], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mdatetime\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m datetime\n\u001b[1;32m----> 2\u001b[0m filtered_rows \u001b[38;5;241m=\u001b[39m evaluations_df[evaluations_df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mp_t検定\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m<\u001b[39m \u001b[38;5;241m0.05\u001b[39m]\n\u001b[0;32m      3\u001b[0m selected_row_index \u001b[38;5;241m=\u001b[39m filtered_rows\u001b[38;5;241m.\u001b[39mindex\u001b[38;5;241m.\u001b[39mtolist()\n\u001b[0;32m      4\u001b[0m selected_row_index_and_group \u001b[38;5;241m=\u001b[39m selected_row_index\u001b[38;5;241m.\u001b[39mcopy()\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:4102\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   4100\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   4101\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 4102\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mget_loc(key)\n\u001b[0;32m   4103\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   4104\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32mc:\\Users\\user\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\range.py:417\u001b[0m, in \u001b[0;36mRangeIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    415\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m    416\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Hashable):\n\u001b[1;32m--> 417\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n\u001b[0;32m    418\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n\u001b[0;32m    419\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 'p_t検定'"
     ]
    }
   ],
   "source": [
    "from datetime import datetime\n",
    "filtered_rows = evaluations_df[evaluations_df['p_t検定'] < 0.05]\n",
    "selected_row_index = filtered_rows.index.tolist()\n",
    "selected_row_index_and_group = selected_row_index.copy()\n",
    "selected_row_index_and_group.insert(0, 0)\n",
    "# result_all.iloc[:, selected_row_index_and_group].to_csv(f\"G:/共有ドライブ/MCI/v2/script/v2.1/検証中/新特徴量作成/output/視線の動きの俊敏さ_{datetime.now().strftime('%Y%m%d%H%M')}.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'selected_row_index' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[19], line 8\u001b[0m\n\u001b[0;32m      3\u001b[0m plt\u001b[38;5;241m.\u001b[39mfigure(figsize\u001b[38;5;241m=\u001b[39m(\u001b[38;5;241m10\u001b[39m, \u001b[38;5;241m10\u001b[39m))\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# plt.title('velo_mean, velo_max, acce_mean, acce_max, saccade_count, \\nfixation_time_sum, move_time_sum, move_dis_sum, saccade_move_dis_sum')\u001b[39;00m\n\u001b[0;32m      5\u001b[0m \n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# 相関係数のヒートマップを描画\u001b[39;00m\n\u001b[0;32m      7\u001b[0m sns\u001b[38;5;241m.\u001b[39mheatmap(\n\u001b[1;32m----> 8\u001b[0m     result_all\u001b[38;5;241m.\u001b[39miloc[:, selected_row_index]\u001b[38;5;241m.\u001b[39mcorr(), \u001b[38;5;66;03m# p値<0.05列対象\u001b[39;00m\n\u001b[0;32m      9\u001b[0m     \u001b[38;5;66;03m# result_all.iloc[:,1:].corr(), # 全列対象\u001b[39;00m\n\u001b[0;32m     10\u001b[0m     cmap\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcoolwarm\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m     11\u001b[0m     square\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m,\n\u001b[0;32m     12\u001b[0m     vmax\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m,\n\u001b[0;32m     13\u001b[0m     vmin\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m,)\n\u001b[0;32m     15\u001b[0m plt\u001b[38;5;241m.\u001b[39mshow()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'selected_row_index' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x1000 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "# グラフ描画領域の作成\n",
    "plt.figure(figsize=(10, 10))\n",
    "# plt.title('velo_mean, velo_max, acce_mean, acce_max, saccade_count, \\nfixation_time_sum, move_time_sum, move_dis_sum, saccade_move_dis_sum')\n",
    "\n",
    "# 相関係数のヒートマップを描画\n",
    "sns.heatmap(\n",
    "    result_all.iloc[:, selected_row_index].corr(), # p値<0.05列対象\n",
    "    # result_all.iloc[:,1:].corr(), # 全列対象\n",
    "    cmap=\"coolwarm\",\n",
    "    square=True,\n",
    "    vmax=1,\n",
    "    vmin=-1,)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### コメント\n",
    "- ヒストグラムで見てよさそうだったのは4-1のfixation_time_sum\n",
    "- saccade countはどのコンテンツ同士で相関が高い(視線の動き方は個人の傾向がある)\n",
    "- move_dis_sumとsaccade_move_dis_sumはほぼイコールの特徴量になってしまったようで高相関→どちらかは無くてもいい"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

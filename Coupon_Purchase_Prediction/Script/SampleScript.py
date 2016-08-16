__author__ = 'Maxim Kurnikov'

import pandas as pd
import numpy as np


coupon_list_train = pd.read_csv('../input/coupon_list_train.csv')
coupon_list_test = pd.read_csv('../input/coupon_list_test.csv')
user_list = pd.read_csv('../input/user_list.csv')
coupon_purchases_train = pd.read_csv("../input/coupon_detail_train.csv")

### merge to obtain (USER_ID) <-> (COUPON_ID with features) training set
### ２つのデータフレームを COUPON_ID_hase を もちいて結合
### 購入されたクーポンの特徴のみを集めたデータフレームができる
### 一つのキーに複数のデータが対応する場合にはその分だけ同じキーに対して行ができる
purchased_coupons_train = coupon_purchases_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='inner')

### filter redundant features
features = ['COUPON_ID_hash', 'USER_ID_hash',
            'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
purchased_coupons_train = purchased_coupons_train[features]

### create 'dummyuser' records in order to merge training and testing sets in one
coupon_list_test['USER_ID_hash'] = 'dummyuser'

### filter testing set consistently with training set
coupon_list_test = coupon_list_test[features]

### merge sets together
### purchased~ の方は購入毎の購入者と購入されたクーポンの特徴を有するデータフレーム，
### coupon~ はクーポンの特徴のみ，購入者の部分が存在しないので dummyuser としている
### ２つのデータフレーム行列を縦方向に結合，存在しないキーに対しては nan が入る
combined = pd.concat([purchased_coupons_train, coupon_list_test], axis=0)

### create two new features
### 各特徴量を　１　以下にした？？
combined['DISCOUNT_PRICE'] = 1 / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100) ** 2
features.extend(['DISCOUNT_PRICE', 'PRICE_RATE']) #なぜ追加したのか？？

### convert categoricals to OneHotEncoder form
### 全ての　カテゴリ-　についてのダミー変数を取得
categoricals = ['GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,
                                    dummy_na=False)

### leaving continuous features as is, obtain transformed dataset
### set はリストの要素の重複を取り除く関数
### combined の左端に連続値を取るキーを，それ移行に 0,1 を取る特徴が来るように並べ直す
continuous = list(set(features) - set(categoricals)) #二つのリストの差集合を取る（リストで返さないので，リスト化する）
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)

### remove NaN values
NAN_SUBSTITUTION_VALUE = 1
combined = combined.fillna(NAN_SUBSTITUTION_VALUE)

### split back into training and testing sets
### 全てまとめてデータを結合してきたが，ここで training と test データを分ける
### test データは購買者（USER_ID_hase）が無く，dummyuser としていたのでここで使用
train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1) #列方向に削除する際は axis の指定が必要
                                                #inplace=Trueで メソッドを呼び出した際にDataframe から削除
                                                #デフォルトは False で単に呼び出しただけでは削除さない（代入必要）

### find most appropriate coupon for every user (mean of all purchased coupons), in other words, user profile
train_dropped_coupons = train.drop('COUPON_ID_hash', axis=1)
user_profiles = train_dropped_coupons.groupby(by='USER_ID_hash').apply(np.mean) #データをユーザーごとにグルーピング，
                                                                                #各キーの値を平均

### creating weight matrix for features
### 各特徴の重みを格納した辞書型オブジェクトの生成
FEATURE_WEIGHTS = {
    'GENRE_NAME': 2,
    'DISCOUNT_PRICE': 2,
    'PRICE_RATE': 0,
    'USABLE_DATE_': 0,
    'large_area_name': 0.5,
    'ken_name': 1,
    'small_area_name': 5
}

# dict lookup helper
# 与えられた特徴の名前に対して，辞書に登録してある重みを返す
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items(): #辞書のitems()メソッドで全ての key, value をたどる
        if col in colname:  #各ダミー特徴には それぞれの元のキー名が先頭に付いているので，その文字列が含まれるかどうかで判断できる
            return weight
    raise ValueError #呼び出し元にエラーを返し処理を任せる

W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
            for colname in user_profiles.columns] # for文で代入することで n×1 行列を作ることが出来る
W = np.diag(W_values)

### find weighted dot product(modified cosine similarity) between each test coupon and user profiles
### ユーザーの過去の購買履歴の各特徴量の平均 を FEATURE_WEIGHTS で重み付け，　予測したいテストアイテムの各特徴量と掛けあわせて総和
### similarity_socores の各行は各ユーザーのそれぞれのアイテムに対する 類似度 を示す．
test_only_features = test.drop('COUPON_ID_hash', axis=1)
similarity_scores = np.dot(np.dot(user_profiles, W),
                           test_only_features.T)

### create (USED_ID)x(COUPON_ID) dataframe, similarity scores as values
### similarity_scores は単なる行列で，ラベルが付いていないので，行にユーザーID,列にCOUPON_ID を割り振ったデータフレームに直す
coupons_ids = test['COUPON_ID_hash']
index = user_profiles.index
columns = [coupons_ids[i] for i in range(0, similarity_scores.shape[1])]
result_df = pd.DataFrame(index=index, columns=columns,
                      data=similarity_scores)

### obtain string of top10 hashes according to similarity scores for every user
def get_top10_coupon_hashes_string(row):
    row.sort()
    return ' '.join(row.index[-10:][::-1].tolist()) #スーペースを区切り文字として，index（ = COUPON_ID）top 10 を結合したものをかえす
                                                    #[-10:]は最後から10個目から最後までを指定
                                                    #[::-1]は取り出した要素を逆順にする
                                                    #.tolist でリスト化

### DataFrame.apply は各行に適用， .map は各値に適用
### 各ユーザ（行）の中で類似度 top10 の COUPON_ID_hash を取り出し．
output = result_df.apply(get_top10_coupon_hashes_string, axis=1)


output_df = pd.DataFrame(data={'USER_ID_hash': output.index,
                               'PURCHASED_COUPONS': output.values})
output_df_all_users = pd.merge(user_list, output_df, how='left', on='USER_ID_hash')
output_df_all_users.to_csv('cosine_sim_python.csv', header=True,
                           index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])

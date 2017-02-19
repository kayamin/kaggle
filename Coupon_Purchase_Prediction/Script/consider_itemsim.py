## Add SEX_ID and AGE to item_test_profiles which are for calculating cossim between users and items


import pandas as pd
import numpy as np
import sys
from IPython.core.debugger import Tracer; debug_here = Tracer()


coupon_list_train = pd.read_csv('../input/coupon_list_train.csv')
coupon_list_test = pd.read_csv('../input/coupon_list_test.csv')
coupon_purchase_train = pd.read_csv('../input/coupon_detail_train.csv')
user_list = pd.read_csv('../input/user_list.csv')

purchased_coupons_train = coupon_purchase_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='inner')
################################
## ユーザーの年齢を年齢層に分ける

generation = [10,20,30,40,50,60,100]
generation_name = ['10代','20代','30代','40代','50代','60以上']

    ##データフレームに年代に基づき GENERATION を追加する関数
def conv_age2gen(user_list, generation, generation_name):   #データフレームは引数としてコピーを渡すので，値を変えたら呼び出し元でも値が変わる
                                                            # python 一般では引数は　参照渡しにはなっていないので変わらないが
    user_list['GENERATION'] = ' '
    for i in range(len(generation)-1):
        user_list['GENERATION'][(user_list['AGE']>=generation[i]) & (user_list['AGE']<generation[i+1])] = generation_name[i]

conv_age2gen(user_list, generation, generation_name)

    ##購入されたクーポンの特徴をまとめた Dataframe に購入者の 性別と年代を追加
purchased_coupons_train = pd.merge(purchased_coupons_train, user_list, on='USER_ID_hash', how='inner')
###############################

### filter redundant features
features = ['COUPON_ID_hash', 'USER_ID_hash',
            'SEX_ID','GENERATION','GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
purchased_coupons_train = purchased_coupons_train[features]

### create 'dummyuser' records in order to merge training and testing sets in one
coupon_list_test['USER_ID_hash'] = 'dummyuser'
coupon_list_test['SEX_ID'] = 'dummysex'
coupon_list_test['GENERATION'] = 'dummygeneration'

### filter testing set consistently with training set
coupon_list_test = coupon_list_test[features]

### merge set together
combined = pd.concat([purchased_coupons_train, coupon_list_test], axis=0)

### create two new features
combined['DISCOUNT_PRICE'] = 1 / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100) ** 2


### convert categoricals to OneHotEncoder form
categoricals = ['SEX_ID','GENERATION','GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,
                                    dummy_na=False)

### leaving continuous features as is, obtain transformed dataset
continuous = list(set(features) - set(categoricals))
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)

### remove NaN values
NAN_SUBSTITUTION_VALUE = 1
combined = combined.fillna(NAN_SUBSTITUTION_VALUE)

### split back into training and testing sets
train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1)


##############################
# アイテムごとに特徴をまとめ，平均を取る．その後，それらのアイテム間の cossim を計算する．
# 各testアイテムに対し cossim top 100 の train item の 特徴を平均
# そこから，SEX_ID,GENERATION に関連する部分 のみを抜き出し， test アイテムの特徴として加える（類似のアイテムを買っている人の性別，年代を示す）

### アイテムごとの特徴行列を作成
item_train_profiles = train.groupby('COUPON_ID_hash').mean()
item_test_profiles = test.groupby('COUPON_ID_hash').mean()

### アイテム間の類似度を計算
item_similarity_scores = np.dot(item_test_profiles, item_train_profiles.T)

## アイテム間の類似度をDataFrameに直す
index_itemsim = item_test_profiles.index
columns_itemsim = item_train_profiles.index
itemsim_df = pd.DataFrame(index=index_itemsim, columns=columns_itemsim, data=item_similarity_scores)

### アイテム間の類似度 top100 をリストで返す関数
def get_top100_user_hashes_string(row):
    row.sort()

    return ' '.join(row.index[-100:][::-1].tolist())

top100_simitem = itemsim_df.apply(get_top100_user_hashes_string, axis=1)　#DataFrame の行ごとに関数を適用
index_top100 = top100_simitem.values[:len(top100_simitem)]

## ダミー変数として分離された特徴名を指定（自動的に指定するようには出来ないか？？）
customer_feature = ['SEX_ID_f','SEX_ID_m','GENERATION_10代','GENERATION_20代','GENERATION_30代',
                    'GENERATION_40代','GENERATION_50代','GENERATION_60以上']

### item_test_profiles に平均値を代入

temp_df = pd.DataFrame(index=item_test_profiles.index, columns=customer_feature)
for i in range(len(index_top100)):
    temp_df.ix[i] = item_train_profiles.ix[index_top100[i]].mean()  # i番目のインデックス の test item と類似度の高い上位100 train item の特徴を平均
                                                                    # test item の DataFrame と同じインデックスを持つ temp_df に代入(temp_dfのラベル項目のみ代入される)

item_test_profiles[customer_feature] = temp_df[customer_feature]　  #index を揃えないと代入されないので注意
#############################


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
    'small_area_name': 5,
    'SEX_ID': 2,
    'GENERATION': 1
}

# dict lookup helper
# 与えられた特徴の名前に対して，辞書に登録してある重みを返す
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items(): #辞書のitems()メソッドで全ての key(col), value(weight) をたどる
        if col in colname:  #各ダミー特徴には それぞれの元のキー名が先頭に付いているので，その文字列が含まれるかどうかで判断できる
            return weight
    raise ValueError #呼び出し元にエラーを返し処理を任せる

W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
            for colname in user_profiles.columns] # for文で代入することで n×1 行列を作ることが出来る
W = np.diag(W_values)



similarity_scores = np.dot(np.dot(user_profiles, W),
                           item_test_profiles.T)

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

output = result_df.apply(get_top10_coupon_hashes_string, axis=1)

output_df = pd.DataFrame(data={'USER_ID_hash': output.index,
                               'PURCHASED_COUPONS': output.values}) # 列にラベル名を付ける →　ラベルを用いたDataframeの merge が可能に
output_df_all_users = pd.merge(user_list, output_df, how='left', on='USER_ID_hash')
output_df_all_users.to_csv('cosine_sim_python_plus_SEXID&GENERATION.csv', header=True,
                           index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])

## アイテムごとの特徴行列を作成
item_train_profiles = train.groupby('COUPON_ID_hash').mean()
item_test_profiles = test.groupby('COUPON_ID_hash').mean()

## アイテム間の類似度をDataFrameに直す
index_itemsim = index.item_test_profiles

def get_top100_user_hashes_string(row):
    row.sort()

    return ' '.join(row.index[-100:][::-1].tolist())

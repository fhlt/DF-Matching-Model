import pandas as pd 
import lightgbm as lgb 
import warnings
warnings.filterwarnings('ignore')

## read data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## lable convert one-hot

label2current_service = dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))

train['current_service'] = train['current_service'].map(current_service2label)

## generate train and test data 

y = train.pop('current_service')
train_id = train.pop('user_id')

X = train
train_col = train.columns
X_test = test[train_col]
test_id = test['user_id']

## deal abnormal data
for line in train_col:
	X[line] = X[line].replace("\\N",-1)
	X_test[i] = X_test[i].replace("\\N",-1)

## generate the train and test data 
X,y,X_test = X.values,y,X_test.values
'''
X: train data
y: train_label
X_test: train_data
'''
## build model
lgb = lgb.LGBMClassifier(
    objective='multiclass',
    num_leaves=35,
    depth=6,
    learning_rate=0.1,
    seed=2018,
    colsample_bytree=0.8,
    subsample=0.9,
    n_estimators=10,
    n_jobs=8)	 

## train model

lgb_model = lgb.fit(X,y)

## generate the result

df_result = pd.DataFrame()
df_result['id'] = list(test_id.unique())
df_result['predict'] = lgb_model.predict(X_test)
df_result['predict'] = df_result['predict'].map(label2current_service)
df_result.to_csv('./baseline.csv',index=False)
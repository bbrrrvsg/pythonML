import pandas as pd 
df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol', 'sugar','pH']] # 와인들의 속성 3개 
target = df['class'] # 1 : 화이트와인 0 : 레드와인

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(data,target,test_size=0.2,random_state=42)

# [2] 결정트리 
from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier()
dt.fit(train_input,train_target)
print(dt.score(test_input,test_target))  # 0.8576923076923076 

# [3] 교차 검증 
from sklearn.model_selection import cross_validate
# cross_validate(학습 모델 , 학습세트 , 정갑세트)
# 교차검즈은 전체 데이터를 N등분(폴드) 하여 돌아가면서 검증 한다. 기본값은 5등분 
# 즉] 데이터를 여러 조각으로 나누어 학습하는 방법 
scores = cross_validate(dt, train_input , train_target) 
print(scores)

import numpy as np
print(np.mean(scores['test_score'])) # 5등분 학습의 평균 검증 점수 # 0.858571296364848

from sklearn.model_selection import StratifiedKFold
# n_splits : N등분  # 데이터를 N등분하여 교차 검증 수행한다. 
splits = StratifiedKFold(n_splits=10 , shuffle=True , random_state=42)
scores = cross_validate(dt, train_input , train_target , cv=splits)
print(scores)
# # { 'test_score': array([0.83653846, 0.88076923, 0.85      , 0.86538462, 0.84807692,
#        0.86923077, 0.87307692, 0.85934489, 0.8477842 , 0.88439306])}
print(np.mean(scores['test_score'])) # 10등분 학습의 평균 검증 점수 #  0.861459908107307


# [4] 그리드 서치
from sklearn.model_selection import GridSearchCV
# (1) 여러개 
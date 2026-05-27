
# [1] 
import pandas as pd 
df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol', 'sugar','pH']] # 와인들의 속성 3개 
target = df['class'] # 1 : 화이트와인 0 : 레드와인

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(data,target,test_size=0.2,random_state=42)


# [2] 결정트리 (분류모델) 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_input,train_target)
print(dt.score(train_input,train_target)) # 훈련세트의 정확도
print(dt.score(test_input,test_target)) # 테스트세트의 정확도
print(dt.predict(test_input[:5])) # 테스트세트의 처음 5개 샘플에 대한 예측값 [1. 0. 1. 1. 1.] 

# [3] 결정 트리 시각화 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plot_tree(dt, max_depth = 1 , feature_names=['alcohol','sugar','pH'] , filled=True) # (트리모델 , max_depth = 가지수 )
# filled = True : 노드 색깔 채우기
plt.show() 


# [4] 특성 중요도 
# 각 특성이 트리 모델에 얼마나 중요한 역할 하는지 수치 # 합은 1 
print(dt.feature_importances_)  # [0.23707929 0.51591041 0.24701029]
print(dt.feature_importances_[0])  # alcochol , sugar , pH 

# [5] 최소한의 불순도(gini) 설정 , 최적의 파라미터 
dt = DecisionTreeClassifier(random_state=43 , min_impurity_decrease=0.0005)
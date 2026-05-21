# [1] 
import pandas as pd 
df = pd.read_csv('./day05/Fish.csv')
# 어종 7개 , Species
fish_target = df['Species'] 
# 특성 6개 , Weight,Length1,Length2,Length3,Height,Width
fish_input = df[['Weight','Length1','Length2','Length3','Height','Width']]
# 훈련/테스트 분리 
from sklearn.model_selection import train_test_split
train_input , test_input,train_target,test_target = train_test_split(fish_input,fish_target,test_size=0.25,random_state=42)
# 스케일링 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scalerd = ss.transform(train_input)
test_scalerd = ss.transform(test_input)

# [*] fit() 학습모델 에서는 정답(target) 도 같이 학습 중이다. 예측(y) 값 과 실제 정답간의 오차 측정 
# 예] 산꼭대기 에서 내려가는 방법중에 가장 최적의 경로로 내려오는 방법 = 경사 하강법(수많은 경우의수 계산하여 판단)
# (1) 전통 경사 하강법(정확도는 좋지만 학습속도가 느리다) vs (2) 확률 경사하강법 (SGD : 정확도는 낮지만 학습속도가 빠르다. : 미니배치
# [*] 로그 로스

# [2] 
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss' , random_state=42)
sc.fit(train_scalerd,train_target)
print(sc.score(test_scalerd,test_target))
print(sc.predict(test_scalerd[ :3]))
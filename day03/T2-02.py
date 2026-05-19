
# 모델 : 데이터(자료)을 학습하는 프로그램/라이브러리( 사이킷런 )
# 학습 : 데이터(자료)의 규칙 찾는 과정
# 예측 : 학습된 모델로 새로운 데이터(결과) 추론 과정
# 특성 : 학습에 입력되는 정보               # 물고기의 '길이' , '무게'
# 타깃 : 학습에 정답이 되는 정보            # 물고기의 '종류' 
# 표준화(스케일링) : 0 ~ 1 사이로 크기 맞춤
    # StandardScaler()  # .transform( )
# 과소적합          : 너무 단순한 경우 # 이웃이 너무 많아서 기준 애매한 학습
# 과대적합/과적합    : 너무 암기된 경우 # 이웃이 너무 적어서 특정 이웃만 학습 
# -------------------------------------------------------------------- # 
# K-NN모델 : 가까운 이웃 기준의 예측 
# KNeighborsClassifier()    k최근접이웃 분류
# KNeighborsRegressor()     k최근접이웃 회귀
    # 하이퍼파리미터(K) : 이웃개수(K) 직접 설정하여 최적의 K찾기
    # 학습특성의 형태는 2차원배열만 가능 , 
        # T1-01( zip활용 ) , T1-02( column_stack활용 ) , T2-01( reshape활용 )
# -------------------------------------------------------------------- # 


# [1] 숭어의 '길이' , '무게' : '길이'(특성) 에 따른 '무게'(타깃) 예측
import pandas as pd 
df = pd.read_csv('./day03/Fish.csv')
fish_data = df[df['Species'].isin('Perch')]
perch_length = fish_data['Length2'].values
perch_weight = fish_data['Weight'].values
print( perch_length , perch_weight ) # 확인 



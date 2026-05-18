
# [1] csv 불러오기 
import pandas as pd 

df = pd.read_csv('./day02/fish.csv')
df.info() 

# [2] 필요한 어종 추출 : 논리식 대신에 .isin() 특정값만 추출 , isna() 결측치만 추출 
target_fish = df[df['Species'].isin( ['Bream' , 'Smelt'] ) ]
print(target_fish)

# [3] 필요한 특성 추출 : length2 , weight
# 넘파이 # np.column_stack( (리스트1) , (리스트2) ) : 두 리스트 간 동일한 요소로 2차원 리스트 구성
# T1-01.py [6] zip 함수 대신에 2차원 리스트 구성 방법 
import numpy as np
fish_data = np.column_stack( ( target_fish['Length2'] , target_fish['Weight'] ) )
print( fish_data ) # 2차원 리스트로 구성

# [4] 모델 학습 하기 위한 정답지 , 도미 35마리 , 빙어 14마리
fish_target = np.concatenate((np.ones(35) , np.zeros(14)) ) # 1:도미 , 0:빙어
print( fish_target )
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

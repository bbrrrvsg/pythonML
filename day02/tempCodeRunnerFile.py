plt.scatter(train_input[:,0],train_input[:,1])          # 학습용
plt.scatter(25,100) # 예측값    
plt.scatter(train_input[indexs,0],train_input[indexs,1]) # 예측에 사용된 이웃들 시각화 , 문제발견
plt.show()
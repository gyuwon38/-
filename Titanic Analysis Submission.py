import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 프레임 다루기
# 1 Titanic 데이터셋 불러오기 
df_train = pd.read_csv(r'C:\Users\이규원\Downloads\titanic\train.csv')  # 훈련 데이터
df_test = pd.read_csv(r'C:\Users\이규원\Downloads\titanic\test.csv')    # 테스트 데이터
df_gender = pd.read_csv(r'C:\Users\이규원\Downloads\titanic\gender_submission.csv')  # 성별에 따른 생존 여부 파일

# 2 'Survived', 'Pclass', 'Sex', 'Age', 'Fare' 열 선택
# 분석에 필요한 열만 선택해서 새로운 데이터프레임 생성한다
df_selected = df_train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
print(df_selected)  # 선택한 열의 모든 행을 출력

# 3 나이가 20세 이상인 사람만 선택
# 나이가 20세 이상인 승객만 필터링하여 새로운 데이터프레임 생성
df_20_above = df_selected[df_selected['Age'] >= 20]
print(df_20_above)  # 필터링된 데이터프레임 모든 행을 출력

# 4 나이를 범주형으로 변환하고 'Child'와 'Adult'로 구분하는 열 추가
# 'Age' 열을 기준으로 나이가 18세 미만이면 'Child', 18세 이상이면 'Adult'로 새로운 'AgeGroup' 열을 추가한다
df_train['AgeGroup'] = df_train['Age'].apply(lambda x: 'Child' if x < 18 else 'Adult') #나이를 범주형 데이터로 변환한다
print(df_train[['Age', 'AgeGroup']])  # 나이, 나이 그룹 포함한 모든 행을 출력

#데이터 클리닝
# 5 결측치 처리 (결측치를 평균값으로 채우고 'Embarked'는 결측치 있는 행 제거)
# 결측치를 확인하고 'Age'의 결측치는 평균값으로 대체하며, 'Embarked' 열에서 결측치가 있는 행은 제거함
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)  # 'Age' 열의 결측치를 평균값으로 채움
df_train.dropna(subset=['Embarked'], inplace=True)  # 'Embarked' 열에서 결측치가 있는 행을 삭제
print(df_train.isnull().sum())  # 결측치가 남아 있는지 확인

#데이터 분석과 시각화
# 6-1 나이 히스토그램 
plt.figure(figsize=(10,5))
sns.histplot(df_train['Age'], bins=20, kde=True)  # 나이 분포 히스토그램 생성 (20개 구간, kde: 커널 밀도 추정)
plt.title('Age Distribution')  # 그래프 제목 설정
plt.xlabel('Age')  # x축 "Age"로 설정
plt.ylabel('Frequency')  # y축 "Frequency"로 설정
plt.show()

# 6-2 요금 히스토그램
plt.figure(figsize=(10,5))
sns.histplot(df_train['Fare'], bins=20, kde=True)  # 요금 분포 히스토그램 생성 (20개 구간, kde: 커널 밀도 추정)
plt.title('Fare Distribution')  # 그래프 제목
plt.xlabel('Fare')  # x축 "Fare"로 설정
plt.ylabel('Frequency')  # y축 라벨 "Frequency"로 설정
plt.show()

# 7 성별에 따른 생존율 산점도
plt.figure(figsize=(8, 6)) #그림 크기 가로 8인치 세로 6인치 설정
sns.scatterplot(x='Sex', y='Survived', data=df_train)  # 산점도 표시
plt.title('Survival Rate by Gender') #그래프 제목 설정
plt.xlabel('Gender') # x축 "Gender"로 설정
plt.ylabel('Survival Rate') #y축 "Survival Rate"로 설정
plt.yticks([0, 1], ['Not Survived', 'Survived'])  # y축 눈금 라벨 설정
plt.show() 


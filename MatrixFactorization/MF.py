import numpy as np

class MF():
    def __init__(self, R, K, learning_rate, regular, iterations): #평점 행렬 초기화
        self.R = R #Rating Matrix
        self.num_users, self.num_items = R.shape #각각 user latent, item latent 행렬
        self.K = K #feature vector dimension
        self.learning_rate = learning_rate #Learning rate
        self.regular = regular #Regularization parameter
        self.iterations = iterations #max_iterations
        self.sample_r, self.sample_c = self.R.nonzero() #평점 행렬 중 데이터가 있는 것의 위치를 저장하기 위함
        self.n_samples = len(self.sample_r)# 평점 행렬의 크기에 맞게 하위 행렬을 업데이트 하기위한 변수

    def train(self): #Matrix Factorization의 모든 과정을 시행하는 함수
        # 평점 행렬로 부터 분해된 하위 행렬을 정의, 처음에는 P, Q 행렬은 랜덤한 값이 된다.
        self.P = np.random.rand(self.num_users, self.K) #P 행렬은 user latent 행렬이 된다
        self.Q = np.random.rand(self.num_items, self.K) #Q 행렬은 item latent 행렬이 된다

        # 최대 반복 횟수 만큼 행렬 분해(SGD 방법)를 적용한다
        iteration_result = [] #화면에 반복 횟수와 동시에 Error Value를 출력하기 위한 리스트 생성
        for i in range(self.iterations): #최대 반복 횟수만큼 행렬 분해 SGD를 계산
            self.sgd() #SGD 연산 시작
            error = self.error() #SGD 연산 후 만든 예측 행렬의 값으로 Error Value를 계산
            iteration_result.append((i, error)) #반복 횟수와 Error Value를 출력할 리스트에 저장
            if (i + 1) % 10 == 0: #반복 횟수가 10단위가 되었을 때 Error Value를 출력한다
                print("Iteration: %d ; error = %.4f" % (i + 1, error))
        return iteration_result

    def error(self): #모든 error 값의 총합
        xs, ys = self.R.nonzero() #평점 행렬 중 0이 아닌 데이터들의 행과 열을 저장
        predicted = self.full_matrix() #평점 행렬과 비교할 예측 행렬을 가져온다
        error = 0 #error 값 선언, 0으로 초기화
        for xaxis, yaxis in zip(xs, ys): #0이 아니었던 데이터들의 위치만 비교한다
            error += pow(self.R[xaxis, yaxis] - predicted[xaxis, yaxis], 2) #0이 아닌 위치의 평점 행렬에서 예측 행렬의 값을 뺀다, 또한 제곱을 해준다
        return np.sqrt(error) #이전에 제곱을 해서 더해줬던 error 값에 제곱근을 적용한다, 제곱을 하고 제곱근을 적용시켜주는 이유는 error 값이 음수가 나오는 것을 방지하기 위해서이다

    def sgd(self): #행렬 분해 SGD의 연산 과정
        # 평점 행렬을 베이스로 하고 평점 행렬의 위치, 값을 랜덤하게 적용시킨 후의 행렬을 이용하여 행렬 분해를 한다
        train_indices = np.arange(self.n_samples)
        np.random.shuffle(train_indices)

        for i in train_indices:
            users, items = self.sample_r[i], self.sample_c[i] #
            e = self.R[users, items] - self.P[users, :].dot(self.Q[items, :].T)

            #하위 행렬의 값을 갱신
            self.P[users, :] += self.learning_rate * (e * self.Q[items, :] - self.regular * self.P[users, :])
            self.Q[items, :] += self.learning_rate * (e * self.P[users, :] - self.regular * self.Q[items, :])


    def full_matrix(self): #행렬 분해의 계산이 끝난 후, 위치를 지정하지 않고 모든 위치의 예측 행렬을 return
        return np.dot(self.P, self.Q.T)

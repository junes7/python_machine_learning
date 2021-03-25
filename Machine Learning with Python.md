# Machine Learning with Python







## 1. 소개



### 1.7 첫 번째 애플리케이션: 붓꽃의 품종 분류

#### 1.7.1 데이터 적재

* 사용할 데이터셋은 머신러닝과 통계 분야에서 오래전부터 사용해온 붓꽃(iris) 데이터셋입니다. 이 데이터는 scikit-learn의 datasets 모듈에 포함되어 있습니다. `load_iris` 함수를 사용해서 데이터를 적재하겠습니다.

```python
# 데이터 적재
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```

* load_iris가 반환한 iris 객체는 파이썬의 딕셔너리 Dictionary와 유사한 Bunch 클래스의 객체입니다.  즉 키와 값으로 구성되어 있습니다.

```python
# iris_dataset의 키 출력
print("iris_dataset의 키: \n", iris_dataset.keys())
# 실행 결과
iris_dataset의 키: 
 dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

* DESCR 키에는 데이터셋에 대한 간략한 설명이 들어 있습니다. 앞부분만 조금 살펴보겠습니다.

```python
print(iris_dataset['DESCR'][:193] + "\n...")

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, pre
...
```





```python

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
```



https://wikidocs.net/21047

https://zephyrus1111.tistory.com/59

https://rfriend.tistory.com/416





### 2.2 일반화(Generalization), 과대적합(Overfitting), 과소적합(Underfitting)

* 모델이 처음 보는 데이터에 대해 정확하게 예측할 수 있으면 이를 훈련 세트에서 테스트 세트로 일반화 되었다고 합니다.

* 과대적합(Overfitting): 가진 정보를 모두 사용해서 너무 복잡한 모델을 만드는 것. 과대적합은 모델이 훈련 세트의 각 샘플에 너무 가깝게 맞춰져서 새로운 데이터에 일반화되기 어려울 때 일어납니다.
* 과소적합(Underfitting): 너무 간단한 모델이 선택되는 것. 이런 경우에는 데이터의 면면과 다양성을 잡아내지 못할 것이고 훈련 세트에도 잘 맞지 않을 것입니다.



### 2.3 지도 학습 알고리즘(Supervised Learning Algorithm)

* 가장 중요한 매개변수와 옵션의 의미도 설명하겠습니다. 분류와 회귀 모델을 모두 가지고 있는 알고리즘도 많은데, 이런 경우 둘 다 살펴보겠습니다.



#### 2.3.1 예제에 사용할 데이터셋(Make dataset which use in example)

* 어떤 데이터셋은 작고 인위적으로 만든 것이며 , 알고리즘의 특징을 부각하기 위해 만든 것도 있습니다. 실제 샘플로 만든 큰 데이터셋도 있습니다.
* 두 개의 특성일 가진 forge 데이터셋은  인위적으로 만든 이진 분류 데이터셋입니다. 







##### 분류용 가상 데이터 생성

* Scikit-Learn 패키지는 분류(classification) 모형의 테스트를 위해 여러가지 가상 데이터를 생성하는 함수를 제공한다.

* make_classification 함수는 설정에 따른 분류용 가상 데이터를 생성하는 명령이다. 이 함수의 인수와 반환값은 다음과 같다.
* 인수:

* n_samples: 표본 데이터의 수, 디폴트 100
* n_features: 독립 변수의 수, 디폴트 20
* n_informative: 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
* n_redundant: 독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트2
* n_repeated:  독립 변수 중 단순 중복된 성분의 수, 디폴트0
* n_classes: 종속 변수의 클래스 수, 디폴트2
* n_clusters_per_class: 클래스 당 클러스터의 수, 디폴트2
* weights: 각 클래스에 할당된 표본 수
* random_state: 난수 발생 시드
* 반환값:
* x: [n_samples, n_features] 크기의 배열 o 독립 변수
* y: [n_samples] 크기의 배열 o 종속 변수

다음 코드는 1개의 독립변수를 가지고 2개의 클래스를 가지는 데이터를 생성한 예이다.

```python
from sklearn.datasets import make_classification
plt.title("1개의 독립변수를 가진 가상 데이터")
X, y = make_classification(n_features=1, n_informative)


```







### make_blobs

* `make_blobs함수는` 등방성 가우시안 정규분포를 이용해 가상 데이터를 생성한다. 이 때 등방성이라는 말은 모든 방향으로 같은 성질을 가진다는 뜻이다. 다음 데이터 생성 코드의 결과를 보면 `make_classification` 함수로 만든 가상데이터와 모양이 다른 것을 확인할 수 있다. `make_blobs`는 보통 클러스링 용 가상데이터를 생성하는데 사용한다. `make_blobs`함수의 인수와 반환값은 다음과 같다.
* 인수:
* n_samples: 표본 데이터의 수, 디폴트 100
* n_features: 독립 변수의 수, 디폴트 20
* centers: 생성할 클러스터의 수 혹은 중심, [n_centers, n_features] 크기의 배열, 디폴트3
* cluster_std: 클러스터의 표준 편차, 디폴트 1.0
* center_box: 생성할 클러스터의 바운딩 박스(bounding box), 디폴트(-10.0, 10.0)
* 반환값:
* x: [n_samples, n_features] 크기의 배열 
  * 독립 변수
* y: [n_samples] 크기의 배열
  * 종속 변수

```python
from sklearn.datasets import make_blobs
plt.title("세개의 클러스터를 가진 가상 데이터")
X, y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=100, edgecolor="k", linewidth=2)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()
```





```python
# 라이브러리 임포트
from IPython.display import display
from sklearn.datasets import make_blobs
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import mglearn

import matplotlib as mpl
import matplotlib.pylab as plt
# 한글깨짐 해결
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (14,4)

# 마이너스 깨짐 해결
mpl.rcParams['axes.unicode_minus'] = False
# 캐쉬 경로 검색
matplotlib.get_cachedir()
```



https://jinyes-tistory.tistory.com/70

https://m.blog.naver.com/PostView.nhn?blogId=zeta0807&logNo=221513189108&proxyReferer=https:%2F%2Fwww.google.com%2F

https://subinium.github.io/MLwithPython-2-3-1/

https://blog.naver.com/bo53621mi/222213928626

https://datascienceschool.net/03%20machine%20learning/09.02%20%EB%B6%84%EB%A5%98%EC%9A%A9%20%EA%B0%80%EC%83%81%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%83%9D%EC%84%B1.html

https://itchipmunk.tistory.com/148?category=646518





https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/

https://woolulu.tistory.com/category/python%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20--%20%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5?page=0

https://halsu-itda.tistory.com/3



### jupyter notebook 단축키

https://technical-support.tistory.com/58

https://iludaslab.tistory.com/43

### 네이버 블로그

https://blog.naver.com/prkim99/222102299395

https://blog.naver.com/slykid/221850725829

https://blog.naver.com/spacedau/222277907320

https://m.blog.naver.com/ssdyka/221228868101

https://blog.naver.com/hiuejiwon/222279971316

#### n_neighbors값이 각기 다른 최근접 이웃 모델이 만든 결정 경계

(Decision boundaries created by recent neighbor models with different n_neighbors values)



n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도 

(Training accuracy and test accuracy with n_neighbors change)



### 선형 모델(Linear Model)

* 선형 모델은 100여년 전에 개발되었고, 지난 몇 십년 동안 폭넓게 연구되고 현재도 널리 쓰입니다. 곧 보겠지만 선형 모델은 입력 특성에 대한 선형 함수를 만들어 예측을 수행합니다.

https://datastory1.blogspot.com/2017/11/r-markdown_2.html

https://shjchad78.tistory.com/18

https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:TeX_%EB%AC%B8%EB%B2%95

### 회귀의 선형 모델

* 회귀의 경우 선형 모델을 위한 일반화된 예측 함수는 다음과 같습니다.

$$
\hat{y} = w[0] \times x[0] + w[1] \times x[1] + ... + w[p] \times x[p] + b
$$

* 이 식에서 *x*[0] 에서부터 *x*[*p*]까지는 하나의 데이터 포인트에 대한 특성을 나타내며 (특성의 개수는 *p* + 1), w와 b는 모델이 학습할 파라미터입니다. 그리고 y은 모델이 만들어낸 예측값입니다. 특성이 하나인 데이터셋이라면 이 식은 다음과 같습니다.

$$
\hat{y} = w[0] \times x[0] + b
$$



### 분류용 선형 모델(Linear Model for classification)


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





https://itchipmunk.tistory.com/148?category=646518
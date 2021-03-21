# Machine Learning with Python







## 1. 소개



### <u>1.7 첫 번째 애플리케이션: 붓꽃의 품종 분류</u>

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
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
Y = boston.target

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)
# print(X_train, X_test, Y_train, Y_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
yt = le.fit_transform(Y)
print(yt)

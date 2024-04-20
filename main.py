import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import *

from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

"""# 1. Відкрити та зчитати наданий файл з даними."""

df = pd.read_csv('WQ-R.csv', sep=';')

"""# 2. Визначити та вивести кількість записів."""



num_of_rows = len(df)
num_of_columns = len(df.columns)
print('Кількість записів:', num_of_rows)

"""# 3. Вивести атрибути набору даних."""

for i, c in enumerate(df.columns):
  print(f'{i+1}) {c}')

"""# 4. Отримати десять варіантів перемішування набору даних та розділення його на навчальну (тренувальну) та тестову вибірки, використовуючи функцію ShuffleSplit. Сформувати начальну та тестові вибірки на основі восьмого варіанту. З’ясувати збалансованість набору даних."""

spliter = ShuffleSplit(n_splits=10, train_size=0.8, random_state=1)
train_indexes, test_indexes = list(spliter.split(df))[7]


df_train, df_test = df.loc[train_indexes], df.loc[test_indexes]

df_train['quality'].value_counts()

df_test['quality'].value_counts()

"""# 5. Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, збудувати класифікаційну модель на основі методу k найближчих сусідів (значення всіх параметрів залишити за замовчуванням) та навчити її на тренувальній вибірці, вважаючи, що цільова характеристика визначається стовпчиком quality, а всі інші виступають в ролі вихідних аргументів."""

x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

model_k = KNeighborsClassifier()
model_k.fit(x_train, y_train)

"""# 6. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. Представити результати роботи моделі на тестовій вибірці графічно."""

def metrics(model, x, y):
  predictions = model.predict(x)

  accuracy = accuracy_score(y, predictions)
  precision = precision_score(y, predictions, average='weighted')
  recall = recall_score(y, predictions, average='weighted')
  f_score = f1_score(y, predictions, average='weighted')
  mcc = matthews_corrcoef(y, predictions)
  balanced_acc = balanced_accuracy_score(y, predictions)
  return {
      'Accuracy': accuracy,
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'MCC': mcc,
      'Balanced Accuracy': balanced_acc,
  }

test_k = metrics(model_k, x_test, y_test)

train_k = metrics(model_k, x_train, y_train)

test_k

plt.bar(test_k.keys(), test_k.values())
plt.title('Test knn metrics values')
plt.show()

"""# 7. З’ясувати вплив кількості сусідів (від 1 до 20) на результати класифікації. Результати представити графічно."""

xs, ys_test, ys_train = [], [], []
for i in range(1, 21):
  model = KNeighborsClassifier(i)
  model.fit(x_train, y_train)
  xs.append(i)
  ys_train.append(balanced_accuracy_score(y_train, model.predict(x_train)))
  ys_test.append(balanced_accuracy_score(y_test, model.predict(x_test)))

plt.plot(xs, ys_test, label='test')
plt.plot(xs, ys_train, label='train')
plt.legend()
plt.title('Bплив кількості сусідів на результати класифікації.')
plt.xticks(range(1, 21))
plt.show()


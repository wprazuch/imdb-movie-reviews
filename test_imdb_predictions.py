import tensorflow_datasets as tfds

_, test_data = tfds.load(
    name="imdb_reviews",
    split=('train', 'test'), 
    as_supervised=True)

x_test, y_test = [], []

for data, label in test_data.as_numpy_iterator():
    x_test.append(data)
    y_test.append(label)

ex = x_test[0:1]
ex = [item.decode('utf-8') for item in ex]

import requests
formData = {
    'instances': ex
}
res = requests.post('http://localhost:8080/v1/models/svm:predict', json=formData)
print(res)
print(res.text)
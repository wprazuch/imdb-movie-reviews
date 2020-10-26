import kfserving
import joblib
import numpy as np
import os
from typing import Dict
import pickle

MODEL_BASENAME = 'random_forest'
MODEL_EXTENSIONS = ['.joblib', '.pkl', '.pickle']

CLASS_MAPPING = os.path.join('static', 'class_mapping.pkl')


class SKLearnModel(kfserving.KFModel):

    def __init__(self, name: str, model_dir: str):

        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False
        with open(CLASS_MAPPING, 'rb') as handle:
            self.class_mapping = pickle.load(handle)

    def load(self) -> bool:
        model_path = kfserving.Storage.download(self.model_dir)
        paths = [
            os.path.join(model_path, MODEL_BASENAME + model_extension)
            for model_extension in MODEL_EXTENSIONS]

        print(paths)

        for path in paths:
            if os.path.exists(path):
                print("Path exists")
                print(path)
                self._model = joblib.load(path)
                print(type(self._model))
                self.ready = True
                break

        return self.ready

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        try:
            inputs = np.array(instances)
        except Exception as e:
            raise Exception('Failed in initialize NumPy array from inputs: %s, %s' % (e, instances))
        try:
            result = self._model.predict(inputs).tolist()[0]
            return {'predictions': self.class_mapping[result]}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

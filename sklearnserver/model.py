import kfserving
import joblib
import numpy as np
import os
from typing import Dict

MODEL_BASENAME = 'rf'
MODEL_EXTENSIONS = ['.joblib', '.pkl', '.pickle']


class SKLearnModel(kfserving.KFModel):

    def __init__(self, name: str, model_dir: str):

        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False

    def load(self) -> bool:
        model_path = kfserving.Storage.download(self.model_dir)
        paths = [
            os.path.join(model_path, MODEL_BASENAME + model_extension)
            for model_extension in MODEL_EXTENSIONS]

        for path in paths:
            if os.path.exists(path):
                self._model = joblib.load(path)
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
            result = self._model.predict(inputs).tolist()
            return {'predictions': result}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

import os
from sklearnserver.kfmodel_repository import KFModelRepository, MODEL_MOUNT_DIRS
from sklearnserver import SKLearnModel


class SKLearnModelRepository(KFModelRepository):

    def __init__(self, model_dir: str = MODEL_MOUNT_DIRS):
        super().__init__(model_dir)

    async def load(self, name: str) -> bool:
        model = SKLearnModel(name, os.path.join(self.models_dir, name))
        if model.load():
            self.update(model)
        return model.ready

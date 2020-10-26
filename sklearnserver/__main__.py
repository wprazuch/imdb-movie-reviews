import argparse
import logging
import sys

import kfserving
from sklearnserver import SKLearnModel, SKLearnModelRepository


DEFAULT_MODEL_NAME = 'model'
DEFAULT_LOCAL_MODEL_DIR = '/tmp/model'


parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_dir', required=True, help='A URI pointer to the model binary')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is server under')

args, _ = parser.parse_known_args()


if __name__ == '__main__':
    model = SKLearnModel(args.model_name, args.model_dir)
    print("Starting...")
    try:
        model.load()
    except Exception as e:
        ex_type, ex_value, _ = sys.exc_info()
        logging.error(f"fail to load model {args.model_name} from dir {args.model_dir}. "
                      f"exception type {ex_type}, exception msg: {ex_value}")
        model.ready = False
        print("Exception")

    print(model)
    kf_server = kfserving.KFServer()
    kf_server.register_model(model)

    kf_server.start([model])


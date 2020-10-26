import numpy as np
import pandas as pd

import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def assess_model(clf, name, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    logging.info(f"{name} scored {score*100} % accuracy.")
    return score

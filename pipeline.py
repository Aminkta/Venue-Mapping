import pandas as pd
import numpy as np

from preprocessors import Pipeline
import config

pipeline = Pipeline(variables = config.VARIABLES,
                    regional_eps = config.REGIONAL_EPS,
                    regional_min_samples = config.REGIONAL_MIN_SAMPLES,
                    regional_metric = config.REGIONAL_METRIC,
                    local_eps = config.LOCAL_EPS,
                    local_min_samples = config.LOCAL_MIN_SAMPLES,
                    local_metric = config.LOCAL_METRIC)


if __name__ == '__main__':

    # load data set
    data = pd.read_csv(config.PATH_TO_DATASET, sep=';')

    pipeline.fit(data)
    home_list, work_list = pipeline.predict()
    print('Likely home locations : {}'.format(home_list))
    print('Likely work locations : {}'.format(work_list))


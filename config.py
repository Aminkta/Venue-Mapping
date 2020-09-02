PATH_TO_DATASET = "person1.csv"

# columns
VARIABLES = ['longitude', 'latitude']

# regional DBSCAN Model
REGIONAL_EPS = 2
REGIONAL_MIN_SAMPLES = 12
REGIONAL_METRIC = 'euclidean'

# local DBSCAN Model
LOCAL_EPS = 0.0005
LOCAL_MIN_SAMPLES = 10
LOCAL_METRIC = 'euclidean'
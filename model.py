import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd

OUTDIR = 'output/taxi_trained'
#key,fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count
CSV_COLS = ['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude',
               'dropoff_latitude','passenger_count', 'dayofweek', 'hourofday']
DEFAULTS = [['nokey'],[0.0], ['2015-04-19 22:21:12 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0],['Mon'],[1]]
FEATURES = CSV_COLS[1:len(CSV_COLS) - 1]
LABEL = CSV_COLS[0]


def convert_day(text):
    try:
        dt = datetime.strptime(text.decode('ascii'), '%Y-%m-%d %H:%M:%S %Z')
        dayofweek = DAYS[dt.weekday()]
        hr_day = dt.hour
    except:
        print('zxcv error date parsing')
        dayofweek = tf.constant('Mon')
        hr_day = tf.constant(10)
    return dayofweek, hr_day


# Returns tensorflow dataset
# https://stackoverflow.com/a/45829855/1535090
# quick tutorial  https://github.com/tensorflow/tensorflow/tree/r1.2/tensorflow/contrib/data
def read_dataset(file, mode, batch_size = 512):
    def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults= DEFAULTS, )
        features = dict(zip(CSV_COLS, columns))
        features = add_engineered_features(features)
        #features.pop('key')
        #features.pop('pickup_datetime')
        label = features.pop('fare_amount')
        # Now using preprocessed CSV
        # day, hour = tf.py_func(convert_day, [features['pickup_datetime']], [tf.string, tf.int64])
        # features['dayofweek'] = day
        # features['hourofday'] = hour
        return features, label

    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(file)
    # Read lines from text files
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    # Parse text lines as comma-separated values (CSV)
    dataset = textlines_dataset.map(decode_csv, num_parallel_calls=8)
    dataset.prefetch(10)

    # Note:
    # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
    # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None  # loop indefinitely
        dataset = dataset.shuffle(buffer_size=10 * batch_size)
    else:
        num_epochs = 1  # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset


DAYS = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
def add_date_features(features):
    dt = datetime.strptime(features['pickup_datetime'], '%Y-%m-%d %H:%M:%S %Z')
    dayofweek = DAYS[dt.weekday()]
    hr_day = dt.hour
    features['dayofweek'] = dayofweek
    features['hourofday'] = hr_day
    return features


def add_engineered_features(features):
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)

    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist


    # try:
    #     features = add_date_features(features)
    # except:
    #     print('zxcv datetime parse error  ')
    #     features['dayofweek'] = 'Mon'
    #     features['hourofday'] = 1

    return features


def get_train_input_fn():
  return read_dataset('./data/all/sample/train_sample.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid_input_fn():
  return read_dataset('./data/all/sample/dev_sample.csv', mode = tf.estimator.ModeKeys.EVAL)


def get_feature_columns(nbuckets):
    input_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list('dayofweek',
                                                                  vocabulary_list=['Mon', 'Tues', 'Wed', 'Thu',
                                                                                   'Fri', 'Sat', 'Sun']),
        tf.feature_column.categorical_column_with_identity('hourofday', num_buckets=24),

        # Numeric columns
        tf.feature_column.numeric_column('pickup_latitude'),
        tf.feature_column.numeric_column('pickup_longitude'),
        tf.feature_column.numeric_column('dropoff_latitude'),
        tf.feature_column.numeric_column('dropoff_longitude'),
        tf.feature_column.numeric_column('passenger_count'),

        # Engineered features that are created in the input_fn
        tf.feature_column.numeric_column('latdiff'),
        tf.feature_column.numeric_column('londiff'),
        tf.feature_column.numeric_column('euclidean')
    ]

    # Input columns
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = input_columns

    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)

    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,
        day_hr,

        # Sparse columns
        dayofweek, hourofday,

        # Anything with a linear relationship
        pcount
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),

        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    return wide_columns, deep_columns


# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.
def serving_input_fn():
    json_feature_placeholders = {
        'pickup_longitude' : tf.placeholder(tf.float32, [None]),
        'pickup_latitude' : tf.placeholder(tf.float32, [None]),
        'dropoff_longitude' : tf.placeholder(tf.float32, [None]),
        'dropoff_latitude' : tf.placeholder(tf.float32, [None]),
        'passenger_count' : tf.placeholder(tf.float32, [None]),
    }
    # You can transforma data here from the input format to the format expected by your model.
    features = json_feature_placeholders # no transformation needed
    return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)


def get_model():
    myopt = tf.train.FtrlOptimizer(learning_rate=.05)  # note the learning rate
    model = tf.estimator.LinearRegressor(
        feature_columns=get_feature_columns(16), model_dir=OUTDIR, optimizer=myopt)
    return model


def get_dnn_regressor():
    #myopt = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
    myopt = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.01,
        l2_regularization_strength=0.01,
    l1_regularization_strength = 0.01
    )
    model = tf.estimator.DNNRegressor(hidden_units=[64,64,64,8],
                                      feature_columns=get_feature_columns(), model_dir=OUTDIR, optimizer=myopt)
    return model


def get_combined_regressor():
    wide, deep = get_feature_columns(16)
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=OUTDIR,
        linear_feature_columns=wide,
        dnn_feature_columns=deep,
        dnn_hidden_units=[64,64,64,8],
        )
    return estimator

def print_rmse(model, name):
    metrics = model.evaluate(input_fn=lambda : read_dataset('./data/all/sample/test_sample.csv', mode = tf.estimator.ModeKeys.EVAL))
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))


def train_and_evaluate(max_steps):
    model = get_combined_regressor()

    # Add rmse evaluation metric
    def rmse(labels, predictions):
        pred_values = tf.cast(predictions['predictions'], tf.float32)
        return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

    model = tf.contrib.estimator.add_metrics(model, rmse)

    train_spec = tf.estimator.TrainSpec(
        input_fn=get_train_input_fn,
        )

    #exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid_input_fn, steps=None,
                                      start_delay_secs=60, throttle_secs=60)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print_rmse(model, 'validation')


# Run training
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(max_steps=500000)

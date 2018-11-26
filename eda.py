import csv
from utility import *
import pandas as pd
import time
import random
from datetime import datetime
import numpy as np

# CSV processing
# https://medium.com/district-data-labs/simple-csv-data-wrangling-with-python-3496aa5d0a5e
# parallel process https://stackoverflow.com/questions/8424771/parallel-processing-of-a-large-csv-file-in-python

DAYS = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def count_panda():
    data = pd.read_csv('./data/all/train5m.csv')
    print(data.shape)

@profile
def count_csv_rows(path):
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        row_count = sum(1 for row in reader)
        print(row_count)
        return row_count


# rows = 55423857
@profile
def random_sample_index(total, samplepercentage=100, test=20, dev = 10):
    sample = random.sample(range(0,total), int(total*samplepercentage/100))
    sample = np.array(sample)
    dev_size = int(len(sample)*dev/100)
    mask = np.random.rand(len(sample)) >= test/100
    train_i = sample[mask]
    test_i = sample[~mask]
    mask = np.random.rand(len(train_i)) >= dev_size/len(train_i)
    dev_i = train_i[~mask]
    train_i = train_i[mask]
    return train_i, test_i, dev_i


def test_random_sample_index():
    random.seed(1)
    np.random.seed(1)
    train, test, dev = random_sample_index(10000, samplepercentage=10)
    print(len(train))
    print(len(test))
    print(len(dev))
    print_prof_data()



def preprocessrow(row):
    d_txt = row[2]
    dt = datetime.strptime(d_txt, '%Y-%m-%d %H:%M:%S %Z')
    row.append(DAYS[dt.weekday()])
    row.append(dt.hour)
    return row



@profile
def random_sample_data(path, samplepercentage=100, test=20, dev = 10):
    total_records = count_csv_rows(path)
    train_i, test_i, dev_i = random_sample_index(total_records, samplepercentage, test, dev)
    train_i, test_i, dev_i = create_set(train_i, test_i, dev_i)

    print_prof_data()
    print(len(train_i))
    print(len(test_i))
    print(len(dev_i))

    with open(path, 'rU') as data, open('./data/all/sample100/train_sample80.csv', 'w') as train, open('./data/all/sample100/test_sample20.csv', 'w') as test, open('./data/all/sample100/dev_sample10.csv', 'w') as dev:
        reader = csv.reader(data, delimiter=",")
        test_writer = csv.writer(test)
        train_write = csv.writer(train)
        dev_writer = csv.writer(dev)
        i = 0
        t = time.time()
        for row in reader:
            try:
                row = preprocessrow(row)
            except Exception as ex:
                print("Error in dateparsinng : ", row, "  ", ex)
                i = i+1
                continue
            if i == 0:   #header
                # test_writer.writerow(row)
                # dev_writer.writerow(row)
                # train_write.writerow(row)
                print(0)
            elif i in train_i:
                train_write.writerow(row)
            elif i in test_i:
                test_writer.writerow(row)
            elif i in dev_i:
                dev_writer.writerow(row)

            if i % 50000 == 0:
                print("i = ",i,"    time = ",time.time() - t)
                t = time.time()
            i = i + 1


    # with takes care of closing file handle
    # data.close()
    # train.close()
    # test.close()
    # dev.close()


@profile
def create_set(train, test, dev):
    train = set(train)
    dev = set(dev)
    test = set(test)
    return train, test, dev


random_sample_data('./data/all/train.csv')
print(count_csv_rows('./data/all/sample/train_sample.csv'))
print(count_csv_rows('./data/all/sample/test_sample.csv'))
print(count_csv_rows('./data/all/sample/dev_sample.csv'))
print_prof_data()





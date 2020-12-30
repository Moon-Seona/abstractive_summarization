import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def load_json_asDataFrame(input_path):
    result = pd.DataFrame()
    with open(input_path, 'r') as json_file: 
        json_list = list(json_file)
    for json_str in json_list:
        ddata = json.loads(json_str)
        df = pd.DataFrame(data=[list(ddata.values())],columns=list(ddata.keys()))
        result = result.append(df)
    return result[['article_original', 'abstractive']]

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def encoding(df, enc, ordcol, max_value):
    if enc is None:
        enc, ordcol = get_ordinalencoder(df)

    X = enc.transform(df[ordcol].values)

    categories_size = 0

    if max_value is None:
        max_value = X.max(axis = 0)
        max_value = max_value + 1
        categories_size = int(np.sum(max_value))
        max_value = np.cumsum(max_value)
        max_value = np.concatenate(([0], max_value), axis=0)[:-1]
    else:
        X += max_value

    return X, enc, ordcol, max_value, categories_size

def get_ordinalencoder(df: pd.DataFrame) -> OrdinalEncoder:
    ordcol = set(df.columns)
    ordcol = list(ordcol)
    ordcol.sort()

    enc = OrdinalEncoder()
    enc.fit(df[ordcol].values)

    return enc, ordcol

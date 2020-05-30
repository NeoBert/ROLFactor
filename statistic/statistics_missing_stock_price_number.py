import plotly.express as px
import pandas as pd
import os
from os.path import join
from tqdm import tqdm

dir_path = '/Applications/python/quant/data/china/stock'
lens = []
for filename in tqdm(os.listdir(dir_path)):
    filepath = join(dir_path, filename)
    csv = pd.read_csv(filepath)
    csv = csv.dropna(axis=0, how='any')
    lens.append(len(csv))

print(max(lens))
px.histogram(x=lens).show()

import plotly_express as px
px.pie()
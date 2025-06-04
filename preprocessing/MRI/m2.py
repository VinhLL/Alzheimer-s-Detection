import os
import pandas as pd 
import random


df = pd.read_csv('data/train.csv')
df1 = pd.DataFrame(columns=df.columns)
random.seed(46) 

for index, row in df.iterrows():
    if row['diagnosis'] == 3:
        if random.random() < 0.2:
            continue
    else:
        continue

    num = random.randint(1, 3)
    for i in range(1, num + 1):
        dti_link = f'{row["dti_link"]}/{i}'
        mri_link = f'{row["mri_link"]}/{i}'
        df1.loc[len(df1)] = [row['diagnosis'], row['ptdobyy'], row['ptgender'], mri_link, dti_link]

df1.to_csv('data/train_augmented.csv', index=False)
print("Done")
    
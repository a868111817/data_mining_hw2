import pandas as pd
import numpy as np


def generate(df):

    for i in range(0, 10000):
        if (df.iloc[i][0] == 1
            and 180 <= df.iloc[i][1] <= 190
            # and 83 <= df.iloc[i][2] <= 95 #height
            # and 17 <= df.iloc[i][3] <= 24 #weight
            and df.iloc[i][5] == 1
            and df.iloc[i][6] == 3
                and df.iloc[i][7] == 2):

            df.iloc[i][20] = 1

    return df


if __name__ == '__main__':

    #data_Reiner = [(1,185,95,17,1,1,3,2,1,1,3,9,8,9,3,8,7,7,4,5)]
    data = {'gender': np.random.randint(1, 3, size=10000),
            'height': np.random.randint(135, 200, size=10000),
            'weight': np.random.randint(40, 130, size=10000),
            'age': np.random.randint(10, 60, size=10000),

            'color': np.random.randint(1, 4, size=10000),
            'occupation': np.random.randint(1, 4, size=10000),
            'race': np.random.randint(1, 4, size=10000),
            'hair': np.random.randint(1, 4, size=10000),
            'titan': np.random.randint(1, 3, size=10000),
            'loveHistoria': np.random.randint(1, 3, size=10000),
            'eyes': np.random.randint(1, 4, size=10000),

            'loyalty': np.random.randint(0, 11, size=10000),
            'humor': np.random.randint(0, 11, size=10000),
            'fighting': np.random.randint(0, 11, size=10000),
            'intelligence': np.random.randint(0, 11, size=10000),
            'willpower': np.random.randint(0, 11, size=10000),
            'execution': np.random.randint(0, 11, size=10000),
            'passion': np.random.randint(0, 11, size=10000),
            'coordination': np.random.randint(0, 11, size=10000),
            'rationality': np.random.randint(0, 11, size=10000),
            'isReiner': 0
            }
    df = pd.DataFrame(data)
    # df.append(data_Reiner)

    df_new = generate(df)
    df_new.to_csv('train.csv', index=False)

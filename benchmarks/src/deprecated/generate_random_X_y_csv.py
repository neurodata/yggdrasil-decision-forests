import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

random_state = 42

save_path = "data/random_csvs/"

N = [128, 256, 512, 1024, 2048, 4096]
D = [128, 256, 512, 1024, 2048, 4096]#, 8192]


for n in N:
    print("Working on n =", n)
    for d in D:
        X = np.random.randn(n, d)
        y = np.random.randint(2, size=n)

        data = pd.DataFrame(X)
        data['Target'] = y

        data.to_csv(save_path+"random_n="+str(n)+"_d="+str(d)+".csv", index=False)


        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.3, random_state=random_state, stratify=y
        # )


        # pd.DataFrame(X_train).to_csv(save_path+"X_train_n="+str(n)+"_d="+str(d)+".csv", index=False)
        # pd.DataFrame(X_test).to_csv(save_path+"X_test_n="+str(n)+"_d="+str(d)+".csv", index=False)
        # pd.DataFrame(y_train).to_csv(save_path+"y_train_n="+str(n)+"_d="+str(d)+".csv", index=False)
        # pd.DataFrame(y_test).to_csv(save_path+"y_test_n="+str(n)+"_d="+str(d)+".csv", index=False)
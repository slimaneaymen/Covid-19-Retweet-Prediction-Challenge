import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

train_data = pd.read_csv(r"C:\Users\slimane\Desktop\Machine and deep learning\Chalenge\data_chalange_v3\Data-Challenge-master\preprocessed_train70.csv", encoding ='latin1',keep_default_na=False, index_col=0)

k=['time','verif','stat','folow','friend','ment_c','ment_f','url_c','url_f','#_c','#_f','v1','v2','v3','v4','v5']
scaled_data=preprocessing.scale(train_data.T)
pca = PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)
per_var=np.round(pca.explained_variance_ratio_,decimals=1)
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
# plt.bar(x=range(1,len(per_var)+1), height=per_var,tick_label=labels)
# plt.ylabel('percentage of explaine variance')
# plt.xlabel('principle components')
# plt.title('Screen plot')
# plt.show()
pca_df=pd.DataFrame(pca_data/(train_data.shape[1]),index=[*k], columns=labels)

plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('PCA Graph')
plt.ylabel('PC2-{0}%'.format(per_var[1]))
plt.xlabel('PC1-{0}%'.format(per_var[0]))

for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
plt.show()
#print(pca.explained_variance_/(train_data.shape[0]-1))
print(pca_df)
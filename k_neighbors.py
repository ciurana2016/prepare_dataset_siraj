import preprocess, predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Preprocess our data
bro = 'data/Pokemon.csv'
d = preprocess.dat_data(bro) # <<< See what I did here ;D

# Scale our data
sc = StandardScaler()
sc.fit(d['train_data'])
train_data_std = sc.transform(d['train_data'])
test_data_std  = sc.transform(d['test_data'])

# Define the classifier
knn = KNeighborsClassifier(n_neighbors=19, p=2, metric='minkowski')

# Train the classifier
knn.fit(train_data_std, d['train_labels'])

# Result
predict.print_performance(knn, test_data_std, d['test_labels'])
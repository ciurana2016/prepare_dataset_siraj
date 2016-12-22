import preprocess, predict
from sklearn.svm import SVC
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
svm = SVC(kernel='linear', C=7.0, random_state=0)

# Train the classifier
svm.fit(train_data_std, d['train_labels'])

# Result
predict.print_performance(svm, test_data_std, d['test_labels'])
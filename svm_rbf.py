import preprocess, predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


# Preprocess our data
bro = 'data/Pokemon.csv'
d = preprocess.dat_data(bro) # <<< See what I did here ;D

# Scale our data
mms = MinMaxScaler()
train_data_std = mms.fit_transform(d['train_data'])
test_data_std  = mms.transform(d['test_data'])

# Define the classifier
svm = SVC(kernel='rbf', C=200.0, gamma=0.20, random_state=0)

# Train the classifier
svm.fit(train_data_std, d['train_labels'])

# Result
predict.print_performance(svm, test_data_std, d['test_labels'])
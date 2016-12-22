import preprocess, predict
from sklearn.linear_model import LogisticRegression
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
lr = LogisticRegression(C=1000.0, random_state=0) 

# Train the classifier
lr.fit(train_data_std, d['train_labels'])

# Result
predict.print_performance(lr, test_data_std, d['test_labels'])
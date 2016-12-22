import preprocess, predict
from sklearn.ensemble import RandomForestClassifier


# Preprocess our data
bro = 'data/Pokemon.csv'
d = preprocess.dat_data(bro) # <<< See what I did here ;D

# Define the classifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=20, random_state=1, n_jobs=2) 

# Train the classifier
forest.fit(d['train_data'], d['train_labels'])

# Result
predict.print_performance(forest, d['test_data'].values, d['test_labels'])
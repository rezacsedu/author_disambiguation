import numpy as np  
import pandas as pd  

# For plotting 
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar for python and CLI
from tqdm import tqdm

## Importing sklearn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def prepareData(data, labels, persons):
	# data.json: List of 5086 various contributions, described by several attributes (features), e.g. names, information about the workplace of the author, its geolocation, and focus areas (key topics covered in contribution)
	df = pd.read_json(data)

	# ground_truth.json: "Ground truth" - actual group s of contributions from the data file, where each contribution is assigned to a person
	df_ground_truth = pd.read_json(labels)

	# persons.json: The list of unique people in the dataset
	df_person = pd.read_json(persons)  # do i need this to map back?

	### Labelling the dataframe with ground_truth for easier work
	df = df.join(df_ground_truth.set_index('contributionId'), on='contribution_id')

	df['str_focus_areas'] = [','.join(map(str, l)) for l in df['focus_areas']]
	df['str_gpes'] = [','.join(map(str, l)) for l in df['focus_areas']]
	df['str_orgs'] = [' '.join(map(str, l)) for l in df['focus_areas']]

	# We create full name for each contributor to be used as a feature (well, maybe not a good idea, but let's do it anyway)
	df['cm_full_name'] = df[["first_name","middle_name","last_name"]].agg(' '.join, axis=1)
	df["features"] = df[["cm_full_name","workplace","str_focus_areas","str_gpes","str_orgs"]].agg(' '.join, axis=1)

	author_mapping_dict = {label:idx for idx, label in enumerate(df["personId"])}
	id_author_mapping_dict = dict((v, k) for k, v in author_mapping_dict.items())
	d_labels = df["personId"].map(author_mapping_dict)
	
	return df, df_ground_truth, df_person, d_labels, author_mapping_dict, id_author_mapping_dict
    
def initModel(X_train, y_train):
	# Step-2: model creation and training: create a RF estomator 
	rf = RandomForestClassifier()

	## Now let's define some params before creating a random param grid. 
	n_estimators = [16, 32, 64, 128]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]

	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}

	rf_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

	# Training the model: I know it's exhaistive as the it will search across 100 different combinations, using all available CPU cores things can be made faster.
	#cv_model = rf_model.fit(X_train, y_train)
	
	return rf_model 

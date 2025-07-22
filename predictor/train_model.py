import pandas as pd
data=pd.read_csv("predictor/adult_3.csv")
print(data.shape)

# print(data.occupation.value_counts())

# in the occupations column replacing '?' with 'Others'
data.occupation.replace({'?':'Others'},inplace=True)
data.workclass.replace({'?':'Others'},inplace=True)
data['native-country'].replace({'?':'Others'},inplace=True)

# print(data['workclass'].value_counts())
# here employee which are 'Never-worked' and 'Without-pay' category
# are not going to contribute in salary prediction

data=data[data['workclass'] != 'Without-pay']
data=data[data['workclass'] != 'Never-worked']

# print(data['education'].value_counts())
#let's suppose those who have completed only their primary schooling
# are not going to earn

data=data[data['education'] != '5th-6th']
data=data[data['education'] != '1st-4th']
data=data[data['education'] != 'Preschool']

# remove redundancy (education and educational-num representing same thing)
data.drop(columns=['education'],inplace=True)

#remove outliers on numerical columns
def remove_outliers_iqr(column_name):
    global data  # referring to the global DataFrame

    q1 = data[column_name].quantile(0.25)
    q3 = data[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Keep only non-outliers
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]

remove_outliers_iqr('age')
remove_outliers_iqr('educational-num')
remove_outliers_iqr('hours-per-week')

#Feature engineering for capital gain/capital loss
#In most of the cases these are 0 and sometimes they are more than 0 so making it binary 
# wether they exists or is 0
data['has_capital_gain'] = (data['capital-gain'] > 0).astype(int)
data['has_capital_loss'] = (data['capital-loss'] > 0).astype(int)
data.drop(columns=['capital-gain', 'capital-loss'], inplace=True)

#dropping final-weight because it is a census data used by census authorities not in salary prediction
# final weight means that how much similar kind of people exists
data=data.drop(columns=['fnlwgt'])

#picking the input and output features seprately
x=data.drop(columns=['income'])
y=data['income']

# doing one hot and label encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
preprocessor=ColumnTransformer(
    transformers=[
        ('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False),categorical_features)
    ],
    remainder='passthrough'
)
# inside that single 'onehot' transformer, there is just one OneHotEncoder — not one per column.
# But, That one OneHotEncoder handles all the columns in categorical_features.

# handle_unknown -> sometimes oneHotEncoder learns some categories during traning 
# if it encounters some new categories in prediction then it throws an error
# but after 'ignore' encoder will output all '0's for unseen category

#applying the transformation
x = preprocessor.fit_transform(x)

# Label encoder for target variable
le_income = LabelEncoder()
y = le_income.fit_transform(y)



#doing normalization 
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
x = scaler.fit_transform(x)

#now splitting data for training and testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)
#stratify Ensures the class distribution in y is preserved in both training and testing sets.
# random_state ensures Getting the same results every time we run the code.

#Algorithm for prediction
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced',random_state=42)
model.fit(xtrain, ytrain)
# in RandomForestClassifier(Random Forest Classifier is an ensemble learning algorithm)
# A forest of decision trees, each trained on slightly different data and they vote together to decide the final prediction.
# classweight = balanced ,
# class_weight = {
#     class_0: total_samples / (2 × count_class_0),
#     class_1: total_samples / (2 × count_class_1)
# }
# It balances how much each class influences training, Give more importance to the minority class during training.
# random_state=42:
# the data is sampled for each tree
# the splits are made inside trees
# is reproducible
# So, if we runs the code again, the model structure will be the same.


#Evaluating the Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ypred = model.predict(xtest)

print("Accuracy:", accuracy_score(ytest, ypred))
print("\nClassification Report:\n", classification_report(ytest, ypred))
print("\nConfusion Matrix:\n", confusion_matrix(ytest, ypred))

#Saving the Model & Scaler for Django
import joblib

# Save model, encoder, and scaler
joblib.dump(model, 'predictor/salary_model.pkl')
joblib.dump(scaler, 'predictor/scaler.pkl')
joblib.dump(preprocessor, 'predictor/preprocessor.pkl')  
joblib.dump(le_income, 'predictor/le_income.pkl')


#Model Interpretability with SHAP

# we are creating a special kind of explainer object
# specifically designed to work efficiently with tree-based models.

# explainer variable is capable of calculating SHAP values for new
# data points

# explainer is capable of 
#   calculating SHAP values for a sample of training data -> used for summary visualization
#   calculating SHAP values for a single prediction

from django.shortcuts import render
from .forms import SalaryForm
import joblib
import numpy as np
import pandas as pd
import shap

# Load ML components
model = joblib.load('predictor/salary_model.pkl')
scaler = joblib.load('predictor/scaler.pkl')
preprocessor = joblib.load('predictor/preprocessor.pkl')
le_income = joblib.load('predictor/le_income.pkl')

# Extract proper feature names from preprocessor
categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
onehot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)
# extracting the feature names after one-hot encoding of the following features
feature_names = np.concatenate([onehot_feature_names, [
    'age', 'educational-num', 'has_capital_gain', 'has_capital_loss', 'hours-per-week'
]])
#concating those encoded features with numerical features

def home(request):
    form = SalaryForm()
    return render(request, 'predictor/input.html', {'form': form})

def predict_salary(request):
    if request.method == 'POST':
        form = SalaryForm(request.POST)
        if form.is_valid():
            try:
                data = form.cleaned_data
                # form.cleaned_data is a dictionary that contains validated and converted data from the form

                # Creating a DataFrame from the form data to maintain column structure
                # This ensures the data is in the correct format for the preprocessor
                input_df=pd.DataFrame({
                    'age':[data['age']],
                    'workclass':[data['workclass']],
                    'educational-num':[data['educational_num']],
                    'marital-status':[data['marital_status']],
                    'occupation':[data['occupation']],
                    'relationship':[data['relationship']],
                    'race':[data['race']],
                    'gender':[data['gender']],
                    # updating with new binary features
                    'has_capital_gain':[(data['capital_gain']>0)],
                    'has_capital_loss':[(data['capital_loss']>0)],
                    'hours-per-week':[data['hours_per_week']],
                    'native-country':[data['native_country']],
                })
                #Each value is wrapped in a list [ ... ] to make it a single-row DataFrame.
                # This is needed because pandas expects a list or array for each column.

                # ensuring the order
                input_df = input_df[['age','workclass','educational-num','marital-status', 
                                    'occupation','relationship','race','gender', 
                                    'has_capital_gain','has_capital_loss','hours-per-week', 
                                    'native-country']]


                #Applying one-hot encoding for categorical features
                x_transformed=preprocessor.transform(input_df)

                #applying scaler
                x_scaled=scaler.transform(x_transformed)
                #Sometimes, preprocessor.transform() or scaler.transform() returns a sparse matrix.
                # so SHAP needs dense arrays
                # Convert sparse to dense for SHAP
                if hasattr(x_scaled,"toarray"):
                    x_scaled = x_scaled.toarray()
    
                # Predict income class 0 or 1 according to given user input then take that value from array output
                prediction = model.predict(x_scaled)[0]
                # Manually decode label example-> ">50k" for 1
                prediction_text = le_income.inverse_transform([prediction])[0]


                import shap 
                explainer = shap.TreeExplainer(model)


                shap_values = explainer.shap_values(x_scaled)

                if isinstance(shap_values, list): # [[],[]]
                    shap_array = shap_values[1][0]  # class >50K
                else:                             # [ , ]
                    shap_array = shap_values[0]     # already a 1D array

                # Flatten just in case
                shap_array = shap_array.flatten()  # [ , ]
                
                # Get input row as flat array
                input_row = x_scaled[0] 

                # Identify indices of active features (non-zero)
                active_indices=[i for i, val in enumerate(input_row) if val != 0]

                # Filter active features and their SHAP values
                active_shap_pairs = [(feature_names[i], shap_array[i]) for i in active_indices]

                # Sort by absolute SHAP value, descending
                top_features = sorted(active_shap_pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

                # Build readable explanation
                explanation = "The model predicted the employee's salary class based on the following factors:\n\n"

                for feature, contribution in top_features:
                    direction = "increased" if contribution > 0 else "decreased"
                    strength = abs(contribution)
                    
                    # Clean one-hot or long names
                    clean_feature = (
                        feature.replace("workclass_", "")
                            .replace("education_", "")
                            .replace("marital_status_", "")
                            .replace("occupation_", "")
                            .replace("relationship_", "")
                            .replace("race_", "")
                            .replace("gender_", "")
                            .replace("native_country_", "")
                            .replace("_", " ")
                            .strip()
                    )
                    
                    explanation += f"- **{clean_feature}**: {direction} the likelihood of earning >50K (impact score: {strength:.4f})\n"

                explanation += "\n*Note: These are the most influential features for this specific prediction.*"
                # Render to result page
                return render(request, 'predictor/result.html', {'result': prediction_text, 'explanation': explanation})
            except Exception as e:
                # Log error 
                print(f"An error occurred: {e}")
                return render(request, 'predictor/result.html', {
                    'result': None,
                    'error': 'An error occurred during prediction. Please try again.'
                })
    else:
        form = SalaryForm()

    return render(request, 'predictor/input.html', {'form': form})



import pandas as pd
import plotly.express as px

# A helper function to load and prepare data
def load_and_prepare_data():
    try:
        # Load the data and strip any leading/trailing spaces from column names
        df = pd.read_csv('predictor/adult_3.csv')
        df.columns = df.columns.str.strip()

        # Update the target variable to 'income'
        df['income_class'] = df['income'].str.strip()

        # Prepare data for Age vs. Salary plot
        bins = [15,25,35,45,55,65,100]
        labels = ['15-24','25-34','35-44','45-54','55-64','65+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        return df

    except FileNotFoundError:
        return None


def eda_dashboard_view(request):
    df = load_and_prepare_data()

    if df is None:
        return render(request, 'predictor/error.html', {'message': 'Data file not found.'})


    # 1. Overall Income Distribution Pie Chart
    fig_income_dist = px.pie(
        df, names='income_class', title='Overall Income Distribution'
    )
    plot_income_dist = fig_income_dist.to_html(full_html=False, include_plotlyjs='cdn')

    # Using the same data and color map
    SALARY_COLOR_MAP = {
    '>50K': '#FF0000',  
    '<=50K': '#800080' 
}
    # 2. Education vs. Income Stacked Bar Chart
    fig_education = px.bar(
        df.groupby(['educational-num', 'income_class']).size().reset_index(name='count'),
        x='educational-num', y='count', color='income_class', barmode='group',
        title='Income Distribution by Educational Number', text_auto=True,
        color_discrete_map=SALARY_COLOR_MAP
    )
    plot_education = fig_education.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. Age vs. Income Stacked Bar Chart
    fig_age = px.bar(
        df.groupby(['age_group', 'income_class']).size().reset_index(name='count'),
        x='age_group', y='count', color='income_class', barmode='group',
        title='Income Distribution by Age Group'
    )
    plot_age = fig_age.to_html(full_html=False, include_plotlyjs='cdn')


    # 4. Occupation Vs. Income stacked bar char
    fig_stacked_bar = px.bar(   
        df, 
        x='occupation', 
        color='income_class', 
        title='Income Distribution by Occupation (Stacked)',
        color_discrete_map=SALARY_COLOR_MAP
    )
    plot_occupation = fig_stacked_bar.to_html(full_html=False, include_plotlyjs='cdn')

    # 5. Capital Gain vs. Income Box Plot
    fig_capital_gain = px.box(
        df, x='income_class', y='capital-gain', color='income_class',
        title='Capital Gain Distribution by Income Class'
    )
    plot_capital_gain = fig_capital_gain.to_html(full_html=False, include_plotlyjs='cdn')

    # Package all plots into a context dictionary
    context = {
        'plot_income_dist': plot_income_dist,
        'plot_education': plot_education,
        'plot_age': plot_age,
        'plot_occupation': plot_occupation,
        'plot_capital_gain': plot_capital_gain,
    }

   
    return render(request, 'predictor/dashboard.html', context)
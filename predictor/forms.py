from django import forms


WORKCLASS_CHOICES = [
    ('Private', 'Private'),
    ('Self-emp-not-inc', 'Self-emp-not-inc'),
    ('Self-emp-inc', 'Self-emp-inc'),
    ('Federal-gov', 'Federal-gov'),
    ('Local-gov', 'Local-gov'),
    ('State-gov', 'State-gov'),
    ('Without-pay', 'Without-pay'),
    ('Never-worked', 'Never-worked'),
    ('Others', 'Others'),
]

MARITAL_STATUS_CHOICES = [
    ('Never-married', 'Never-married'),
    ('Married-civ-spouse', 'Married-civ-spouse'),
    ('Divorced', 'Divorced'),
    ('Separated', 'Separated'),
    ('Widowed', 'Widowed'),
    ('Married-spouse-absent', 'Married-spouse-absent'),
]

OCCUPATION_CHOICES = [
    ('Tech-support', 'Tech-support'),
    ('Craft-repair', 'Craft-repair'),
    ('Other-service', 'Other-service'),
    ('Sales', 'Sales'),
    ('Exec-managerial', 'Exec-managerial'),
    ('Prof-specialty', 'Prof-specialty'),
    ('Handlers-cleaners', 'Handlers-cleaners'),
    ('Machine-op-inspct', 'Machine-op-inspct'),
    ('Adm-clerical', 'Adm-clerical'),
    ('Farming-fishing', 'Farming-fishing'),
    ('Transport-moving', 'Transport-moving'),
    ('Priv-house-serv', 'Priv-house-serv'),
    ('Protective-serv', 'Protective-serv'),
    ('Armed-Forces', 'Armed-Forces'),
    ('Others', 'Others'),
]

RELATIONSHIP_CHOICES = [
    ('Wife', 'Wife'),
    ('Own-child', 'Own-child'),
    ('Husband', 'Husband'),
    ('Not-in-family', 'Not-in-family'),
    ('Other-relative', 'Other-relative'),
    ('Unmarried', 'Unmarried'),
]

RACE_CHOICES = [
    ('White', 'White'),
    ('Asian-Pac-Islander', 'Asian-Pac-Islander'),
    ('Amer-Indian-Eskimo', 'Amer-Indian-Eskimo'),
    ('Other', 'Other'),
    ('Black', 'Black'),
]

GENDER_CHOICES = [
    ('Male', 'Male'),
    ('Female', 'Female'),
]

COUNTRY_CHOICES = [
    ('United-States', 'United-States'),
    ('India', 'India'),
    ('Mexico', 'Mexico'),
    ('Philippines', 'Philippines'),
    ('Germany', 'Germany'),
    ('Canada', 'Canada'),
    ('Others', 'Others'),
]

class SalaryForm(forms.Form):
    age = forms.IntegerField(min_value=18,max_value=60)
    workclass = forms.ChoiceField(choices=WORKCLASS_CHOICES)
    educational_num = forms.IntegerField(min_value=1,max_value=16)
    marital_status = forms.ChoiceField(choices=MARITAL_STATUS_CHOICES)
    occupation = forms.ChoiceField(choices=OCCUPATION_CHOICES)
    relationship = forms.ChoiceField(choices=RELATIONSHIP_CHOICES)
    race = forms.ChoiceField(choices=RACE_CHOICES)
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    capital_gain = forms.IntegerField(min_value=0,max_value=100000)
    capital_loss = forms.IntegerField(min_value=0,max_value=5000)
    hours_per_week = forms.IntegerField(min_value=1,max_value=100)
    native_country = forms.ChoiceField(choices=COUNTRY_CHOICES)

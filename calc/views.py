from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm
from django.shortcuts import redirect
from django.urls import reverse
import matplotlib.pyplot as plt
import csv
from django.shortcuts import render
from django.conf import settings
import pandas as pd
import os
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import difflib  # For finding close matches


def home(request):
    return render(request, 'home.html', {'name': 'User'})

def add(request):
    val1 = str(request.POST['sym1'])
    val2 = str(request.POST['sym2'])
    val3 = str(request.POST['sym3'])
    predicted_disease, medications = predict_disease([val1, val2, val3])
    return render(request, 'result.html', {'result': predicted_disease, 'medications': medications})

def dashboard(request):
    return render(request, 'dashboard.html')

def features(request):
    return render(request, 'features.html')

import os
import difflib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from django.conf import settings
from django.shortcuts import render

# Initialize global variables
nn_model = None
encoder = None
symptoms_list = None
medications_df=None

# Train the Neural Network model
def train_models():
    global nn_model, encoder, symptoms_list,medications_df
    
    # Load the dataset
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'dataset.csv'))
    df.fillna('None', inplace=True)
    
    # Standardize symptom text format and remove duplicate rows
    for col in df.columns[1:]:  # Assuming the first column is 'Disease'
        df[col] = df[col].str.lower().str.strip()
    df.drop_duplicates(inplace=True)

    # Convert symptoms into binary features
    df = pd.get_dummies(df.set_index('Disease').stack()).groupby(level=0).max().reset_index()
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Encode target labels (Diseases)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Set the global symptoms list based on the dummy columns
    symptoms_list = list(X.columns)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build the neural network model
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(encoder.classes_), activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    nn_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Ensure models are trained at server start
train_models()

# Function for predicting disease with neural network
def predict_disease_nn(symptom1, symptom2, symptom3):
    global symptoms_list

    # Initialize an input vector for symptoms (all zeros initially)
    input_vector = np.zeros(len(symptoms_list))
    symptoms = [symptom1, symptom2, symptom3]

    # Process each symptom and set corresponding index to 1 if present in symptoms list
    for symptom in symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1
        else:
            # Handle unrecognized symptoms with closest match suggestion
            close_matches = difflib.get_close_matches(symptom, symptoms_list, n=1, cutoff=0.8)
            if close_matches:
                index = symptoms_list.index(close_matches[0])
                input_vector[index] = 1
                print(f"Auto-corrected '{symptom}' to '{close_matches[0]}'")
            else:
                print(f"Warning: Symptom '{symptom}' not recognized")

    input_vector = input_vector.reshape(1, -1)  # Reshape for prediction
    nn_prediction = nn_model.predict(input_vector)  # Predict with the neural network
    predicted_index = np.argmax(nn_prediction)  # Get predicted disease index
    predicted_disease_nn = encoder.inverse_transform([predicted_index])[0]  # Map index to disease name
    return predicted_disease_nn

# Django view function for predicting disease and showing available symptoms
def add(request):
    if request.method == 'POST':
        # Get symptoms from the form and standardize the format
        val1 = str(request.POST['sym1']).lower().strip()
        val2 = str(request.POST['sym2']).lower().strip()
        val3 = str(request.POST['sym3']).lower().strip()
        
        # Predict the disease using the neural network model
        predicted_disease = predict_disease_nn(val1, val2, val3)
        
        # Render the result page with the prediction
        return render(request, 'result.html', {'result': predicted_disease})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('home')) 
        else:
            return render(request, 'login.html', {'error': 'Invalid username or password'})
    return render(request, 'login.html')

def home_view(request):
    return render(request, 'home.html')

def rep_page(request):
    return render(request, 'rep.html')

def rep(request, disease):
    # Load the disease data from the CSV file
    data = pd.read_csv('graph1_data.csv')

    # Filter the data based on the selected disease
    filtered_data = data[data['Disease'] == predict_disease]

    # Group the data by year and calculate the sum of deaths
    deaths_per_year = filtered_data.groupby('Year')['Deaths'].sum()


def rep_view(request):
    disease_name = request.GET.get('disease')
    disease_data = None

    # Path to the CSV file
    csv_path = os.path.join(settings.BASE_DIR, 'disease.csv')

    # Read the CSV file
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Disease'] == disease_name:
                disease_data = row
                break

    context = {'disease': disease_name, 'disease_data': disease_data}
    return render(request, 'rep.html', context)


def result(request):
    return render(request,'result.html')

def find_hospitals(request):
    return render(request, 'find_hospitals.html')

def find_doctors(request):
    return render(request, 'find_doctors.html')

def best_countries(request):
    return render(request, 'best_countries.html')

def registerPage(request):
    form = UserCreationForm()
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
        else:
            messages.error(request, "Password does not meet the requirements. Please try again.")
    context = {'form': form}
    return render(request, 'register.html', context)

def registerPage(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
        else:
            messages.error(request, "Password does not meet the requirements. Please try again.")
    context = {'form': form}
    return render(request, 'register.html', context)

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            phone_n=form.cleaned_data.get('phone_number')
            user = authenticate(username=username, password=raw_password,number=phone_n)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

@login_required(login_url='login')
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def logoutPage(request):
    logout(request)
    return redirect('login')

def appointments(request):
    return render(request, 'appointments.html')

def result_view(request):
    # Example: Assuming you're getting a selected disease's name from a form in result.html
    disease = request.POST.get('disease')
    
    # Load your CSV data into a dictionary
    disease_data = {
        "COVID-19": {
            "mortality": [("2013", 10), ("2014", 12), ...],  # Replace with actual data
            "immunization": [("2013", 50), ("2014", 60), ...],
            "age_group": {"0-18": 10, "19-35": 30, "36-50": 25, "51+": 35}
        },
        # Add more diseases and their respective data...
    }

    # Pass the disease data to the template
    return render(request, 'rep.html', {
        'disease': disease,
        'mortality_data': disease_data[disease]['mortality'],
        'immunization_data': disease_data[disease]['immunization'],
        'age_group_data': disease_data[disease]['age_group'],
    })

def dashboard_view(request):
    user = request.user
    context = {
        'exercise_progress': 70,  # This could be dynamic
        'water_progress': 50,     # This could be dynamic
        'exercise_minutes': 30,   # Exercise minutes for the day
        'water_count': 2.5,       # Liters of water consumed
        'profile_picture_url': user.profile.picture.url,  # Assuming you have a profile picture
    }
    return render(request, 'dashboard.html', context)

def rep_view(request, disease_name):
    # Load your CSV data
    df = pd.read_csv('path/to/your/disease.csv')

    # Filter data for the selected disease
    disease_data = df[df['Disease'] == disease_name]

    # Convert the filtered data to a dictionary format
    data_dict = disease_data.to_dict('list')
    
    context = {
        'disease_name': disease_name,
        'data': data_dict
    }
    return render(request, 'rep.html', context)

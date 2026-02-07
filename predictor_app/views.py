import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')   # Headless backend (Fix thread/Tkinter issues)

import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render

from .forms import SalaryForm
from .models import Prediction


# ---------------------------------------------------
# Load Trained Model (.pkl)
# ---------------------------------------------------

model_path = os.path.join(
    settings.BASE_DIR,
    'predictor_app',
    'model',
    'salary_model.pkl'
)

model = pickle.load(open(model_path, 'rb'))


# ---------------------------------------------------
# Dataset Path (for scatter visualization)
# ---------------------------------------------------

data_path = os.path.join(
    settings.BASE_DIR,
    'predictor_app',
    'model',
    'Salary_dataset.csv'
)


# ---------------------------------------------------
# HOME VIEW
# ---------------------------------------------------

def home(request):

    form = SalaryForm()

    return render(
        request,
        'home.html',
        {'form': form}
    )


# ---------------------------------------------------
# PREDICTION VIEW
# ---------------------------------------------------

def predict_salary(request):

    if request.method == 'POST':

        form = SalaryForm(request.POST)

        if form.is_valid():

            # -------------------------------
            # Get Input
            # -------------------------------
            exp = form.cleaned_data['experience']

            exp_array = np.array(exp).reshape(-1, 1)

            # -------------------------------
            # Prediction
            # -------------------------------
            salary = model.predict(exp_array)[0][0]

            # -------------------------------
            # Save to Database
            # -------------------------------
            Prediction.objects.create(
                experience=exp,
                predicted_salary=salary
            )

            # -------------------------------
            # Load Dataset
            # -------------------------------
            df = pd.read_csv(data_path)

            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)

            X = df['YearsExperience'].values.reshape(-1,1)
            y = df['Salary'].values.reshape(-1,1)

            # Train-Test Split
            xtrain, xtest, ytrain, ytest = train_test_split(
                X, y,
                test_size=0.3,
                random_state=45
            )

            # -------------------------------
            # Regression Line
            # -------------------------------
            x_range = np.linspace(
                X.min(),
                X.max(),
                100
            )

            y_pred_line = model.predict(
                x_range.reshape(-1,1)
            )

            # -------------------------------
            # Plot Creation
            # -------------------------------
            plt.figure(figsize=(10,7))

            # Training Scatter
            plt.scatter(
                xtrain,
                ytrain,
                color='blue',
                label='Training Data'
            )

            # Testing Scatter
            plt.scatter(
                xtest,
                ytest,
                color='green',
                label='Testing Data'
            )

            # Regression Line
            plt.plot(
                x_range,
                y_pred_line,
                color='red',
                label='Regression Line'
            )

            # Predicted Point
            plt.scatter(
                exp,
                salary,
                color='purple',
                s=200,
                marker='X',
                label='Predicted Salary'
            )

            # Labels
            plt.title("Experience vs Salary Prediction")
            plt.xlabel("Years of Experience")
            plt.ylabel("Salary")
            plt.legend()
            plt.grid(True)

            # -------------------------------
            # Save Plot
            # -------------------------------
            plot_path = os.path.join(
                settings.MEDIA_ROOT,
                'salary_plot.png'
            )

            plt.savefig(plot_path)
            plt.close()

            # -------------------------------
            # Context Data
            # -------------------------------
            context = {
                'salary': round(salary, 2),
                'experience': exp,
                'plot_url': settings.MEDIA_URL + 'salary_plot.png'
            }

            return render(
                request,
                'result.html',
                context
            )

    # Fallback
    return render(
        request,
        'home.html',
        {'form': SalaryForm()}
    )


# ---------------------------------------------------
# HISTORY VIEW
# ---------------------------------------------------

def history(request):

    records = Prediction.objects.all().order_by('-created_at')

    return render(
        request,
        'history.html',
        {'records': records}
    )
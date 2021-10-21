from flask import Flask
# from app.models.streptococcus_regression
from flask_cors import CORS

from app.models.lactobacillus_regression import LactobacillusRegression
from app.models.neuron_classification import NeuronClassification
from app.models.streptococcus_regression import StreptococcusRegression

NeuronClassification(epochs_number=1000,
                     input_shape_val=9,
                     output_shape_val=3)
StreptococcusRegression(
    path="data/beta_dataset.csv")
LactobacillusRegression(
        path="data/beta_dataset.csv")

app = Flask(__name__)
CORS(app, support_credentials=True)

# import features
from app import routes
from app.errors import errors

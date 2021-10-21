"""
:author Arturo Negreiros
:date 10 August
    [all routes]:
        /bacteria_growth
        /strep
        /lact
        /list_strep
        /list_lact
        /classification_single
        /classification_multiple

"""

from app import app
from flask import (
    jsonify,
    request
)
import pandas as pd
from app.models.streptococcus_regression import (
    list_strep_predictions,
    single_strep_predictions
)
from app.models.lactobacillus_regression import (
    single_lact_predictions,
    list_lact_predictions
)
from app.models.neuron_classification import (
    measure_single_predictions,
    measure_list_predictions
)

import unittest
import numpy as np
from pprint import pprint


@app.cli.command()
def tst():
    tests = unittest.TestLoader().discover("tests")
    unittest.TextTestRunner().run(tests)


"""
@app.route("/")
def index():
    response = make_response(redirect("/bacteria_growth"))
    return response


@app.route("/bacteria_growth", methods=["GET"])
def bacteria_growth():
    data = pd.read_csv("data/growth_curve.csv")
    elements = [{"time": time_, "bacteria": bacteria, "id": index_ + 1} for index_, (time_, bacteria) in
                enumerate(zip(data["time"].to_list(), data["growth_log"].to_list()))]
    return jsonify({
        "data": elements
    })
"""


@app.route("/charting", methods=["GET"])
def charting():
    mae_lact = [{"x": int(inx + 1), "y": value} for inx, value in
                enumerate(single_lact_predictions()["mean_absolute_error"])]
    range_vals = [x for x in range(1, 81)]
    if request.method == "GET":
        return jsonify({
            "message": "request successfully",
            "maestrep": single_strep_predictions()["mean_absolute_error"],
            "maelact": mae_lact,
            "range_vals": range_vals,
            "classification": measure_single_predictions()["values"]
        })


@app.route("/strep", methods=["POST"])
def strep_pred():
    try:
        # if request.method == "POST":
        pprint(request.json)
        list_features = [float(request.json["minProteins"]),
                         float(request.json["tritatableAcid"]),
                         float(request.json["phSour"]),
                         float(request.json["fatMilk"])]
        strep_model = single_strep_predictions(
            values_list=list_features, target_data=float(request.json["targetBacterian"]))
        return jsonify({
            "data": {
                "message": "request successfully",
                "prediction": [strep_model],
            }
        })
    except Exception as e:
        print(e)
        print("error: ", str(e))
        return jsonify({
            "message": "{}".format(str(e))
        })


@app.route("/lact", methods=["POST"])
def lact_pred():
    try:
        pprint(request.json)
        list_features = [float(request.json["minProteins"]),
                         float(request.json["tritatableAcid"]),
                         float(request.json["phSour"]),
                         float(request.json["fatMilk"])]
        lact_model = single_lact_predictions(
            values_list=list_features, target_data=float(request.json["targetBacterian"]))
        print(lact_model)
        return jsonify({
            "data": {
                "message": "request successfully",
                "prediction": [lact_model],
            }
        })
    except Exception as e:
        return jsonify({
            "message": "Something error has ocurred",
            "error": "Error in {}".format(str(e))
        })


@app.route("/list_strep", methods=["POST"])
def list_strep_pred():
    if request.method == "POST":
        pprint(request.json)
        list_pred = request.json["strep_values"]
        list_target = request.json["strep_target"]
        values_predicted = list_strep_predictions(test_values=list_pred, target_values=list_target)
        return jsonify({
            "data": values_predicted
        })


@app.route("/list_lact", methods=["POST"])
def list_lact_pred():
    if request.method == "POST":
        pprint(request.json)
        list_predict = request.json["lact_values"]
        list_target = request.json["lact_target"]
        values_predicted = list_lact_predictions(test_values=list_predict, target_values=list_target)
        return jsonify({
            "data": values_predicted
        })


@app.route("/classification_single", methods=["POST"])
def classification_single():
    """In this endpoint the goal is sent a single values"""

    elements_target = [float(request.json["streptococcusStrainInicial"]), float(request.json["lactobacillusStrainInicial"]), 
                        float(request.json["idealTemperature"]),float(request.json["minProteins"]), 
                        float(request.json["tritatableAcid"]), float(request.json["phSour"]),
                        float(request.json["fatMilk"]), float(request.json["lactobacillusStrainFinal"]), 
                        float(request.json["streptococcusStrainFinal"])]
    pprint(request.json)
    features_data = np.array(elements_target).reshape(1, -1)
    model_class = measure_single_predictions(features_data)
    return jsonify({
            "response":[{
                "message": "successfully",
                "index":1 ,
                "status_code": 200,
                "predictions": model_class
                }]
        })


@app.route("/classification_multiple", methods=["POST"])
def classification_multiple():
    if request.method == "POST":
        pprint(request.json)
        features_data = request.json["classification_data"]
        target_data = request.json["predictions"]

        model_class = measure_list_predictions(features_value=features_data, targets_value=target_data)
        return jsonify({
            "response":[{"message": "successfully",
            "status_code": 200,
            "predictions": model_class}]
        })

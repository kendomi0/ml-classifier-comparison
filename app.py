from flask import Flask, render_template, request
from data import datasets_dict
import uuid
from classifiers import run_classifier, get_valid_number_of_combos, display_selected_combos

app = Flask(__name__)

results_store = {}

@app.route("/")
def start():
    return render_template("start_page.html")

@app.route("/dataset")
def select_dataset():
    return render_template("dataset_selection.html")

@app.route("/evaluation", methods=["POST"])
def select_evaluation_method():
    dataset = request.form["dataset"]
    return render_template("evaluation_selection.html", dataset=dataset)

@app.route("/classifier", methods=["POST"])
def select_classifier():
    dataset = request.form["dataset"]
    evaluation = request.form["evaluation"]
    return render_template("classifier_selection.html", dataset=dataset, evaluation=evaluation)

@app.route("/normalization", methods=["POST"])
def select_normalization_method():
    dataset = request.form["dataset"]
    evaluation = request.form["evaluation"]
    classifier = request.form["classifier"]
    return render_template("normalization_selection.html", dataset=dataset, evaluation=evaluation, classifier=classifier)

@app.route("/process", methods=["POST"])
def process_data():
    dataset = request.form["dataset"]
    X, y = datasets_dict[dataset]
    evaluation = request.form["evaluation"]
    classifier = request.form["classifier"]
    normalization = request.form["normalization"]
    results = run_classifier(X, y, dataset, classifier, normalization, evaluation)
    key = str(uuid.uuid4())
    results_store[key] = results
    results_length = len(results)
    return render_template("process_data.html", results_key=key, results_length = results_length)

@app.route("/display", methods=["POST"])
def display_results():
    key = request.form["results_key"]
    results = results_store.get(key)
    number_of_combos = request.form["number-of-combos"]
    if number_of_combos != "all":
        number_of_combos = int(number_of_combos)
    else:
        number_of_combos = len(results)
    combinations = display_selected_combos(results, number_of_combos)
    results_store.clear()
    return render_template("display_results.html", combinations=combinations)

from flask import Flask, render_template, request
from flask import send_file
from data import datasets_dict
import uuid
from plotting import create_and_save_plot
from classifiers import run_classifier, get_valid_number_of_combos, display_selected_combos

app = Flask(__name__)

results_store = {}
plot_data = {}

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

    results_key = str(uuid.uuid4())
    results_store[results_key] = results

    plot_key = str(uuid.uuid4())
    plot_data[plot_key] = {"X": X, "y": y, "dataset": dataset}

    results_length = len(results)
    return render_template("process_data.html", results_key=results_key, plot_key=plot_key, results_length = results_length, dataset=dataset)

@app.route("/display", methods=["POST"])
def display_results():
    dataset = request.form["dataset"]
    results_key = request.form["results_key"]
    plot_key = request.form["plot_key"]
    results = results_store.get(results_key)
    number_of_combos = request.form["number-of-combos"]
    if number_of_combos != "all":
        number_of_combos = int(number_of_combos)
    else:
        number_of_combos = len(results)
    combinations = display_selected_combos(results, number_of_combos)
    results_store.clear()
    return render_template("display_results.html", combinations=combinations, dataset=dataset, plot_key=plot_key)

@app.route('/plot/scatter')
def display_scatter_plot():
    plot_key = request.args.get("key")
    data = plot_data.get(plot_key)
    buf = create_and_save_plot(data['X'], data['y'], data['dataset'])
    plot_data.pop(plot_key, None)
    return send_file(buf, mimetype='image/png')
from flask import Flask, render_template, request
from flask import send_file
from data import datasets_dict
from classifiers import evaluation_methods, classifier_map, normalization_methods
import uuid
from plotting import create_and_save_plot
from classifiers import run_classifier, display_selected_combos

app = Flask(__name__)

results_store = {}
plot_data = {}
hidden_inputs = {}

@app.route("/")
def start():
    return render_template("start_page.html")

@app.route("/dataset")
def select_dataset():
    return render_template(
        "selection.html", 
        choices=datasets_dict.keys(),
        group="dataset",
        form_action_page="/evaluation",
        hidden_inputs = hidden_inputs
    )

@app.route("/evaluation", methods=["POST"])
def select_evaluation_method():
    dataset = request.form["dataset"]
    hidden_inputs_key = str(uuid.uuid4())
    hidden_inputs[hidden_inputs_key] = {
        "dataset": dataset
    }
    return render_template(
        "selection.html", 
        choices = evaluation_methods.keys(),
        group = "evaluation",
        form_action_page = "/classifier",
        hidden_inputs = hidden_inputs[hidden_inputs_key],
        hidden_inputs_key = hidden_inputs_key
        )

@app.route("/classifier", methods=["POST"])
def select_classifier():
    evaluation = request.form["evaluation"]
    disabled_choices = set()
    if evaluation == "leave-one-out":
        disabled_choices.add("artificial neural networks")
    hidden_inputs_key = request.form["hidden_inputs_key"]
    hidden_inputs[hidden_inputs_key]["evaluation"] = evaluation
    return render_template(
        "selection.html", 
        choices = classifier_map.keys(),
        group = "classifier",
        form_action_page = "/normalization",
        hidden_inputs = hidden_inputs[hidden_inputs_key],
        hidden_inputs_key = hidden_inputs_key,
        disabled_choices = disabled_choices
        )

@app.route("/normalization", methods=["POST"])
def select_normalization_method():
    classifier = request.form["classifier"]
    hidden_inputs_key = request.form["hidden_inputs_key"]
    hidden_inputs[hidden_inputs_key]["classifier"] = classifier
    return render_template(
        "selection.html", 
        choices = normalization_methods.keys(),
        group = "normalization",
        form_action_page = "/process",
        hidden_inputs = hidden_inputs[hidden_inputs_key],
        hidden_inputs_key = hidden_inputs_key
        )

@app.route("/process", methods=["POST"])
def process_data():
    dataset = request.form["dataset"]
    X, y = datasets_dict[dataset]
    evaluation = request.form["evaluation"]
    classifier = request.form["classifier"]
    normalization = request.form["normalization"]
    results = run_classifier(X, y, dataset, classifier, normalization, evaluation)

    hidden_inputs_key = request.form["hidden_inputs_key"]

    results_key = str(uuid.uuid4())
    results_store[results_key] = results

    plot_key = str(uuid.uuid4())
    plot_data[plot_key] = {"X": X, "y": y, "dataset": dataset}

    results_length = len(results)
    return render_template("process_data.html", results_key=results_key, plot_key=plot_key, results_length = results_length, dataset=dataset, hidden_inputs_key=hidden_inputs_key)

@app.route("/display", methods=["POST"])
def display_results():
    dataset = request.form["dataset"]
    results_key = request.form["results_key"]
    plot_key = request.form["plot_key"]
    hidden_inputs_key = request.form["hidden_inputs_key"]
    results = results_store.get(results_key)
    number_of_combos = request.form["number-of-combos"]
    if number_of_combos != "all":
        number_of_combos = int(number_of_combos)
    else:
        number_of_combos = len(results)
    heading, combinations = display_selected_combos(results, number_of_combos)
    results_store.pop(results_key, None)
    hidden_inputs.pop(hidden_inputs_key, None)
    return render_template("display_results.html", combinations=combinations, dataset=dataset, heading=heading, plot_key=plot_key)

@app.route('/plot/scatter')
def display_scatter_plot():
    plot_key = request.args.get("key")
    data = plot_data.get(plot_key)
    buf = create_and_save_plot(data['X'], data['y'], data['dataset'])
    plot_data.pop(plot_key, None)
    return send_file(buf, mimetype='image/png')
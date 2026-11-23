from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request
from ml_comparison.data import datasets_dict
from ml_comparison.classifiers import (
    evaluation_methods,
    classifier_map,
    normalization_methods,
)
import uuid
from ml_comparison.classifiers import run_classifier
from ml_comparison.ranking_formatting import (
    display_combinations_heading,
    format_results,
    get_table_headers,
    show_method_as_table_header,
)

app = Flask(__name__)

results_store = {}

WEBSITE_NAME = "Machine Learning Classifier Comparison"

classifier_choices = list(classifier_map.keys())
classifier_choices.append("all")

normalization_choices = list(normalization_methods.keys())
normalization_choices.append("all")

selection = {
    "dataset": {
        "legend": "1. Dataset",
        "choices": datasets_dict.keys(),
    },
    "evaluation": {
        "legend": "2. Evaluation Method",
        "choices": evaluation_methods.keys(),
    },
    "classifier": {
        "legend": "3. Classifier",
        "choices": classifier_choices,
    },
    "normalization": {
        "legend": "4. Normalization Method",
        "choices": normalization_choices,
    }
}


@app.route("/")
def start():
    return render_template("start_page.html", WEBSITE_NAME=WEBSITE_NAME)


@app.route("/selection")
def select_options():
    disabled_choices = ""
    return render_template(
        "selection.html",
        disabled_choices=disabled_choices,
        selection=selection,
        WEBSITE_NAME=WEBSITE_NAME)


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

    results_length = len(results)
    return render_template(
        "process_data.html",
        results_key=results_key,
        results_length=results_length,
        dataset=dataset,
        WEBSITE_NAME=WEBSITE_NAME,
    )


@app.route("/display", methods=["POST"])
def display_results():
    dataset = request.form["dataset"]
    results_key = request.form["results_key"]
    results = results_store.get(results_key)
    number_of_combos = request.form["number-of-combos"]
    if number_of_combos != "all":
        number_of_combos = int(number_of_combos)
    else:
        number_of_combos = len(results)
    show_normalization_table_header = show_method_as_table_header(
        results, "normalization_method", number_of_combos
    )
    show_classifier_header = show_method_as_table_header(
        results, "classifier_name", number_of_combos
    )
    heading = display_combinations_heading(results, number_of_combos)
    combinations = format_results(results, number_of_combos)
    results_store.pop(results_key, None)
    table_headers = get_table_headers(
        number_of_combinations=number_of_combos,
        show_classifier=show_classifier_header,
        show_normalization=show_normalization_table_header,
    )
    return render_template(
        "display_results.html",
        combinations=combinations,
        number_of_combinations=number_of_combos,
        show_normalization_header=show_normalization_table_header,
        show_classifier_header=show_classifier_header,
        dataset=dataset,
        heading=heading,
        table_headers=table_headers,
        WEBSITE_NAME=WEBSITE_NAME,
    )

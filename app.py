from flask import Flask, render_template, request

app = Flask(__name__)

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

@app.route("/results", methods=["POST"])
def show_results():
    return "Results to be added."
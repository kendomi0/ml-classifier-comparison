import { LoadingScreen } from "./loading-screen.js";

const choiceGroup = document.getElementById("choice-group").value;
const container = document.querySelector("body");
const pageTitle = document.querySelector("title");
const loadingTitle = "Running classifiers..."
let selectionLoadingScreen = new LoadingScreen(container, pageTitle, loadingTitle);

if (choiceGroup == "normalization") {
    selectionLoadingScreen.form.addEventListener("submit", selectionLoadingScreen.showLoadingScreen.bind(selectionLoadingScreen));
}
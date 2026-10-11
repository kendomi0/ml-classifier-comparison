import { LoadingScreen } from "./loading-screen.js";

const choiceGroup = document.getElementById("choice-group").value;
const container = document.querySelector("body");
let selectionLoadingScreen = new LoadingScreen(container);

if (choiceGroup == "normalization") {
    selectionLoadingScreen.form.addEventListener("submit", selectionLoadingScreen.showLoadingScreen.bind(selectionLoadingScreen));
}
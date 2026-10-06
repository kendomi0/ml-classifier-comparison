import { LoadingScreen } from "./loading-screen.js";

const choiceGroup = document.querySelector(".choice-group");
const container = document.querySelector("body");
let selectionLoadingScreen = new LoadingScreen(container);

if (choiceGroup.textContent == "normalization") {
    selectionLoadingScreen.form.addEventListener("submit", selectionLoadingScreen.showLoadingScreen.bind(selectionLoadingScreen));
}
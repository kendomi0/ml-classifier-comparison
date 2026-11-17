import { LoadingScreen } from "./loading-screen.js";

const choiceGroup = document.getElementById("choice-group").value;

const container = document.querySelector("body");
const pageTitle = document.querySelector("title");
const loadingTitle = "Classifying..."
let selectionLoadingScreen = new LoadingScreen(container, pageTitle, loadingTitle);

const form = document.querySelector("form");
const submitBtn = form.querySelector('button[type="submit"]');
const nextArrow = document.querySelector(".next-arrow");
const submitMsg = document.querySelector(".submit-msg")

if (choiceGroup == "normalization") {
    selectionLoadingScreen.form.addEventListener("submit", selectionLoadingScreen.showLoadingScreen.bind(selectionLoadingScreen));
}

const classesForValidSubmitBtn = "border-accent-border bg-accent-main text-main-button-text shadow-lg shadow-button-shadow duration-120 ease-in hover:bg-accent-5"

const classesToRemove = "border-base-300 transition-none animation-none"

form.addEventListener('change', () => {
    submitBtn.disabled = false;
    submitMsg.textContent = "Next";
    nextArrow.classList.remove("hidden");
    submitBtn.classList.add(...classesForValidSubmitBtn.split(' '));
    submitBtn.classList.remove(...classesToRemove.split(' '));
});

import { LoadingScreen } from "./loading-screen.js";

const choiceGroup = document.getElementById("choice-group").value;

const container = document.querySelector("body");
const pageTitle = document.querySelector("title");
const loadingTitle = "Classifying..."
let selectionLoadingScreen = new LoadingScreen(container, pageTitle, loadingTitle);

const form = document.querySelector("form");
const submitBtn = form.querySelector('button[type="submit"]');
const nextArrow = document.querySelector(".next-arrow");
const submitMsg = document.querySelector(".submit-msg");

let scatterPlots = document.querySelectorAll(".scatter-plot");

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


// Scatter plot modals

const plotDir = "./static/images/plots"

const datasets = ["noisy circles", "noisy moons", "blobs", "anisotropic", "varied"]

const userPreference = window.matchMedia('(prefers-color-scheme: dark)');

const themes = {
    "black": "dark", 
    "wireframe": "light"
}

datasets.forEach((dataset) => {
    const datasetHyphens = dataset.replace(" ", "-");
    const datasetUnderscores = dataset.replace(" ", "_");

    const modal = document.getElementById(`${datasetHyphens}-modal`);
    const modalIcon = document.getElementById(`${datasetHyphens}-icon`);
    const modalClose = modal.querySelector('.modal-close');
    const plot = modal.querySelector('.plot');

    modalIcon.addEventListener('click', (event) => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const theme = themes[currentTheme]
        event.stopPropagation();
        event.preventDefault();
        modal.showModal();
        plot.src =`${plotDir}/${datasetUnderscores}_plot_${theme}.png`
    })

    modalClose.addEventListener('click', () => {
        modal.close();
    })

    userPreference.addEventListener('change', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const theme = themes[currentTheme]
        plot.src =`${plotDir}/${datasetUnderscores}_plot_${theme}.png`
    })
})
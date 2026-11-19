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


// Scatter plot tooltip

const plotDir = "./static/images/plots"

const plotImages = {
    "noisy-circles-plot": {
        "light": "noisy_circles_plot_light.png",
        "dark": "noisy_circles_plot_dark.png"
    },
    "noisy-moons-plot": {
        "light": "noisy_moons_plot_light.png",
        "dark": "noisy_moons_plot_dark.png"
    },
    "blobs-plot": {
        "light": "blobs_plot_light.png",
        "dark": "blobs_plot_dark.png"
    },
    "anisotropic-plot": {
        "light": "anisotropic_plot_light.png",
        "dark": "anisotropic_plot_dark.png"
    },
    "varied-plot": {
        "light": "varied_plot_light.png",
        "dark": "varied_plot_dark.png"
    }
}

if (scatterPlots) {
    const storedTheme = localStorage.getItem('theme');
    const themes = {
        "black": "dark", 
        "wireframe": "light"
    }
    const theme = themes[storedTheme]
    scatterPlots.forEach((plot) => {
        plot.src = `${plotDir}/${plotImages[plot.id][theme]}`
    })

}
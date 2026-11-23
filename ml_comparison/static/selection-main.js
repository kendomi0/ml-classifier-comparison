import { LoadingScreen } from "./loading-screen.js";

const form = document.querySelector('form');

// Loading screen

const container = document.querySelector("body");
const pageTitle = document.querySelector("title");

const loadingTitle = "Classifying..."

let selectionLoadingScreen = new LoadingScreen(container, pageTitle, loadingTitle);

selectionLoadingScreen.form.addEventListener("submit", selectionLoadingScreen.showLoadingScreen.bind(selectionLoadingScreen));

// Enabling/disabling submit button

const submitBtn = form.querySelector('button[type="submit"]');
const nextArrow = document.querySelector(".next-arrow");
const submitMsgShort = document.querySelector(".submit-msg-short");

const fieldsets = Array.from(document.querySelectorAll('fieldset'))

const enabledSubmitMsg = "Next";
const disabledSubmitMsg = "Select all options"
const spanMsg = "to continue"
const submitMsgMore = document.querySelector('.submit-msg-more');

function checkIfFieldsetsAreChecked() {
    return fieldsets.every(fieldset =>
        fieldset.querySelector('input:checked') !== null
    )
}

function enableSubmitButton() {
    submitBtn.disabled = false;
    submitMsgShort.textContent = enabledSubmitMsg;
    nextArrow.classList.remove("hidden");
}

function disableSubmitButton() {
    submitBtn.disabled = true;
    submitMsgShort.textContent = disabledSubmitMsg;
    submitMsgMore.textContent = spanMsg;
    nextArrow.classList.add("hidden");
}

form.addEventListener('change', () => {
    const allChecked = checkIfFieldsetsAreChecked()
    if (allChecked) {
        enableSubmitButton();
    }
    else {
        disableSubmitButton();
    }
});

disableSubmitButton();

// Unselecting choices when 'all' is selected

const classifierFieldset = document.getElementById("classifier-fieldset")
const normalizationFieldset = document.getElementById("normalization-fieldset")

const checkboxFieldsets = [classifierFieldset, normalizationFieldset]

checkboxFieldsets.forEach((fieldset) => {
    const allOption = fieldset.querySelector("input[value='all']")
    const possibleOptions = fieldset.querySelectorAll("input")
    const otherOptions = (Array.from(possibleOptions)).filter(option => option != allOption);
    fieldset.addEventListener('change', (event) => {
        if ((event.target == allOption) && (allOption.checked)) {
                otherOptions.forEach((option) => {
                    option.checked = false;
                })
        }
        else {
            allOption.checked = false;
        }
    })
})

// Disabling choices and showing disclaimers

const looInput = document.querySelector("input[value='leave-one-out']")

const disabledLooChoices = ["artificial neural networks", "support vector machine"]

const evaluationFieldset = document.getElementById("evaluation-fieldset");

const classifierDisclaimer = document.getElementById("classifier-disclaimer");

function disableChoices(choices, disclaimer) {
    choices.forEach((choice) => {
        const disallowedChoice = document.querySelector(`input[value='${choice}']`)
        disallowedChoice.checked = false;
        disallowedChoice.disabled = true;
    })
    disclaimer.classList.remove('hidden');
}

function enableChoices(choices, disclaimer) {
    choices.forEach((choice) => {
        const allowedChoice = document.querySelector(`input[value='${choice}']`)
        allowedChoice.disabled = false;
    })
    disclaimer.classList.add('hidden');
}

evaluationFieldset.addEventListener('change', () => {
    if (looInput.checked) {
        disableChoices(disabledLooChoices, classifierDisclaimer);
    }
    else {
        enableChoices(disabledLooChoices, classifierDisclaimer);
    }
})


// Scatter plot modals

const plotDir = "./static/images/plots"

const datasets = ["noisy circles", "noisy moons", "blobs", "anisotropic", "varied"]

const modalIcons = document.querySelectorAll('.modal-icons');

const userPreference = window.matchMedia('(prefers-color-scheme: dark)');

const themes = {
    "black": "dark", 
    "wireframe": "light"
};

if (modalIcons.length) {
    datasets.forEach((dataset) => {
        const datasetHyphens = dataset.replaceAll(" ", "-");
        const datasetUnderscores = dataset.replaceAll(" ", "_");

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

        modal.addEventListener('click', (event) => {
            event.stopPropagation();
            event.preventDefault();
        });

        modalClose.addEventListener('click', () => {
            modal.close();
        })

        userPreference.addEventListener('change', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const theme = themes[currentTheme]
            plot.src =`${plotDir}/${datasetUnderscores}_plot_${theme}.png`
        })
    })
}

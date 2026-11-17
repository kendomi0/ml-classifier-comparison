import { LoadingScreen } from "./loading-screen.js";

let numericCombosInput = document.getElementById("numeric-combos-input");
let submitCombosBtn = document.getElementById("submit-combos-btn");
let totalCombinations = document.getElementById("total-combinations");
let toast = document.querySelector('.toast');
let validIcon = document.querySelector('.valid-icon');
let invalidIcon = document.querySelector(".invalid-icon");

let form = document.querySelector("form");
let container = document.querySelector("body");
let pageTitle = document.querySelector("title");
let loadingTitle = "Generating combination..."

if (totalCombinations.value == 1) {
    let processDataLoadingScreen = new LoadingScreen(container, pageTitle, loadingTitle);
    processDataLoadingScreen.showLoadingScreen();
    window.addEventListener("load", () => {
        numericCombosInput.value = "1";
        form.submit();
    })
}

submitCombosBtn.addEventListener("click", (event) => {
    if (!numericCombosInput.checkValidity())
        event.preventDefault();
        toast.classList.remove('hidden');
})

const classesForValidBtn = "border-accent-border bg-accent-main text-main-button-text shadow-lg shadow-button-shadow duration-120 ease-in hover:bg-accent-5"

function setInputToValid() {
    submitCombosBtn.disabled = false;
    submitCombosBtn.classList.add(...classesForValidBtn.split(' '));
    validIcon.classList.remove('hidden');
    invalidIcon.classList.add('hidden');
}

function setInputToInvalid() {
    submitCombosBtn.disabled = true;
    submitCombosBtn.classList.remove(...classesForValidBtn.split(' '));
    validIcon.classList.add('hidden');
    invalidIcon.classList.remove('hidden');
}

function showToast() {
  toast.classList.remove('opacity-0', 'pointer-events-none');
}

function hideToast() {
  toast.classList.add('opacity-0', 'pointer-events-none');
}

numericCombosInput.addEventListener("input", (event) => {
    if (!Number.isNaN(event.target.valueAsNumber)) {
        if (numericCombosInput.validity.valid) {
            setInputToValid();
            hideToast();
        }
        else {
            setInputToInvalid();
            showToast();
        }
  }
  else {
    if (submitCombosBtn.disabled == false) {
        setInputToInvalid();
    }
    else {
        hideToast();
    }
  }
})
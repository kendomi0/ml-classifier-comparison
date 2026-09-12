import { LoadingScreen } from "./loading-screen.js";

let numericCombosBtn = document.getElementById("numeric-combos-btn");
let numericCombosInput = document.getElementById("numeric-combos-input");
let allCombosBtn = document.getElementById("all-combos-btn");
let submitCombosBtn = document.getElementById("submit-combos-btn");
let totalCombinations = document.getElementById("total-combinations");
let form = document.querySelector("form");
let container = document.querySelector("body");

numericCombosBtn.addEventListener("click", () => {
    numericCombosInput.classList.remove("hidden");
    submitCombosBtn.classList.remove("hidden");
    },
    { once: true}
)

allCombosBtn.addEventListener("click", () => {
    numericCombosInput.disabled = true;
})

if (totalCombinations.value == 1) {
    let processDataLoadingScreen = new LoadingScreen(container);
    processDataLoadingScreen.showLoadingScreen();
    window.addEventListener("load", () => {
        numericCombosInput.value = "1";
        form.submit();
    })
}

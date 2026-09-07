let form = document.querySelector("form");
let submitButton = document.querySelector("button[type='submit']")
let loadingScreen = document.querySelector(".loading-screen");
let choiceGroup = document.querySelector(".choice-group");

function showLoadingScreen() {
    loadingScreen.classList.remove("hidden");
}

if (choiceGroup.textContent == "normalization") {
    form.addEventListener("submit", showLoadingScreen);
}
let form = document.querySelector("form");
let submitButton = document.querySelector("button[type='submit']")
let loadingScreen = document.querySelector(".loading-screen");
let loadingScreenTagline = document.querySelector(".loading-screen-tagline");
let choiceGroup = document.querySelector(".choice-group");

function showLoadingScreen() {
    loadingScreen.classList.remove("hidden");
    loadingScreen.classList.add("flex");
    loadingScreen.classList.add("flex-col");
    loadingScreen.classList.add("justify-center");
    loadingScreenTagline.classList.add("self-center");
    form.classList.add("hidden");
}

if (choiceGroup.textContent == "normalization") {
    form.addEventListener("submit", showLoadingScreen);
}
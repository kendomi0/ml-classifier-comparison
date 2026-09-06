let form = document.querySelector("form");
let submitButton = document.querySelector("button[type='submit']")
let loadingScreen = document.querySelector(".loading-screen");

function showLoadingScreen() {
    loadingScreen.classList.remove("hidden");
}

form.addEventListener("submit", showLoadingScreen);
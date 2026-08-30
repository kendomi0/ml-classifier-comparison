let submitButton = document.querySelector("button[type='submit']")
let loadingScreen = document.querySelector(".loading-screen");

function showLoadingScreen() {
    loadingScreen.classList.remove("display-none");
}

submitButton.addEventListener("click", showLoadingScreen);
let numericCombosBtn = document.getElementById("numeric-combos-btn");
let numericCombosInput = document.getElementById("numeric-combos-input");
let allCombosBtn = document.getElementById("all-combos-btn");
let submitCombosBtn = document.getElementById("submit-combos-btn");
let totalCombinations = document.getElementById("total-combinations");
let form = document.querySelector("form");
let loadingScreen = document.querySelector(".loading-screen");

numericCombosBtn.addEventListener("click", () => {
    numericCombosInput.classList.remove("display-none");
    submitCombosBtn.classList.remove("display-none");
    },
    { once: true}
)

allCombosBtn.addEventListener("click", () => {
    numericCombosInput.disabled = true;
})

if (totalCombinations.value == 1) {
    loadingScreen.classList.remove("display-none");
    console.log("combination is 1");
    window.addEventListener("load", () => {
        numericCombosInput.value = "1";
        form.submit();
    })
}

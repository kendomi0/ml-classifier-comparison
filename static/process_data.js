let numericCombosBtn = document.getElementById("numeric-combos-btn");
let numericCombosInput = document.getElementById("numeric-combos-input");
let allCombosBtn = document.getElementById("all-combos-btn");
let submitCombosBtn = document.getElementById("submit-combos-btn");

numericCombosBtn.addEventListener("click", () => {
    numericCombosInput.classList.remove("display-none");
    submitCombosBtn.classList.remove("display-none");
    },
    { once: true}
)

allCombosBtn.addEventListener("click", () => {
    numericCombosInput.disabled = true;
})
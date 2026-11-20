class SiteHeader extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <div class="navbar bg-base-100 border-b border-base-300 sticky top-0 z-50">
        <div class="flex-1">
          <a class="btn btn-ghost text-xl text-accent-7 hover:bg-base-300 rounded-md border-none" href="/">
          <span class="home-text hidden sm:inline">ML Classifier Comparison</span>
          <span class="home-icon sm:hidden" aria-label="Home button">
            <svg xmlns="http://www.w3.org/2000/svg" width="1.5em" height="1.5em" viewBox="0 0 24 24">
              <path d="M0 0h24v24H0z" fill="none" />
              <g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2">
                <path d="M21 19v-6.733a4 4 0 0 0-1.245-2.9L13.378 3.31a2 2 0 0 0-2.755 0L4.245 9.367A4 4 0 0 0 3 12.267V19a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2" />
                <path d="M9 15a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v6H9z" />
              </g>
            </svg>
          </span>
          </a>
        </div>
        <div class="flex">
          <label class="swap swap-rotate mr-1 hover:bg-base-300 btn btn-ghost btn-lg btn-square rounded-md border-none">
            <input type="checkbox" id="theme-toggle" />

            <!-- sun icon -->
            <svg class="swap-on text-accent-main" xmlns="http://www.w3.org/2000/svg" aria-label="Display light theme" width="2em" height="2em" viewBox="0 0 24 24">
              <path d="M0 0h24v24H0z" fill="none" />
              <path fill="currentColor" d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5s5-2.24 5-5s-2.24-5-5-5m-1-6v4h2V1zm0 18v4h2v-4zm12-8h-4v2h4zM5 11H1v2h4zm11.24 6.66l2.47 2.47l1.41-1.41l-2.47-2.47zM3.87 5.28l2.47 2.47l1.41-1.41l-2.47-2.47zm2.47 10.96l-2.47 2.47l1.41 1.41l2.47-2.47zM18.72 3.87l-2.47 2.47l1.41 1.41l2.47-2.47z" />
            </svg>

            <!-- moon icon -->
            <svg class="swap-off text-accent-main" xmlns="http://www.w3.org/2000/svg" aria-label="Display dark theme" width="2em" height="2em" viewBox="0 0 24 24">
              <path d="M0 0h24v24H0z" fill="none" />
              <path fill="currentColor" d="M12 21q-3.75 0-6.375-2.625T3 12t2.625-6.375T12 3q.35 0 .688.025t.662.075q-1.025.725-1.638 1.888T11.1 7.5q0 2.25 1.575 3.825T16.5 12.9q1.375 0 2.525-.613T20.9 10.65q.05.325.075.662T21 12q0 3.75-2.625 6.375T12 21" />
            </svg>
          </label>
        </div>


          <a class="btn btn-ghost btn-lg btn-square rounded-md border-none hover:bg-base-300" aria-label="Project's Github repository" href="https://github.com/kendomi0/ml-classifier-comparison" target="_blank" rel="noopener noreferrer">
              <svg class="text-accent-main" xmlns="http://www.w3.org/2000/svg" width="2em" height="2em" viewBox="0 0 32 32">
                  <path d="M0 0h32v32H0z" fill="none" />
                  <path fill="currentColor" fill-rule="evenodd" d="M16 2a14 14 0 0 0-4.43 27.28c.7.13 1-.3 1-.67v-2.38c-3.89.84-4.71-1.88-4.71-1.88a3.7 3.7 0 0 0-1.62-2.05c-1.27-.86.1-.85.1-.85a2.94 2.94 0 0 1 2.14 1.45a3 3 0 0 0 4.08 1.16a2.93 2.93 0 0 1 .88-1.87c-3.1-.36-6.37-1.56-6.37-6.92a5.4 5.4 0 0 1 1.44-3.76a5 5 0 0 1 .14-3.7s1.17-.38 3.85 1.43a13.3 13.3 0 0 1 7 0c2.67-1.81 3.84-1.43 3.84-1.43a5 5 0 0 1 .14 3.7a5.4 5.4 0 0 1 1.44 3.76c0 5.38-3.27 6.56-6.39 6.91a3.33 3.33 0 0 1 .95 2.59v3.84c0 .46.25.81 1 .67A14 14 0 0 0 16 2" />
              </svg>
          </a>
        </div>
      `;
  }
}
customElements.define('site-header', SiteHeader);

const toggle = document.querySelector('#theme-toggle');
const root = document.documentElement;

function getPreferredTheme() {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const preferredTheme = prefersDark ? "black" : "wireframe";
  return preferredTheme
}

function getCurrentTheme() {
  return root.getAttribute('data-theme');
}

function setToggleState() {
  const currentTheme = getCurrentTheme()
  if (currentTheme == 'black') {
    toggle.checked = true;
  }
}

function setInitialTheme() {
  let theme = getPreferredTheme();
  root.setAttribute('data-theme', theme);
}

function setThemeUponLoad() {
  if (localStorage.getItem('theme') == null) {
    setInitialTheme()
  }
  else {
    root.setAttribute('data-theme', localStorage.getItem('theme'))
  }
  setToggleState()
}

function toggleTheme() {
  const oppositeTheme = root.getAttribute('data-theme') === "black" ? "wireframe" : "black"
  root.setAttribute('data-theme', oppositeTheme);
}

function storeTheme() {
  const currentTheme = getCurrentTheme();
  localStorage.setItem('theme', currentTheme);
}

setThemeUponLoad()

toggle.addEventListener('change', () => {
  toggleTheme();
  storeTheme();
})
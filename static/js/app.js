"use strict";

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-scan-form]").forEach((form) => {
    form.addEventListener("submit", () => {
      const progress = form.querySelector("[data-scan-progress]");
      const button = form.querySelector("[data-scan-button]");
      if (progress) progress.classList.add("visible");
      if (button) {
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        const label = button.querySelector(".button-label");
        if (label) label.textContent = "Analysing…";
      }
    });
  });
});

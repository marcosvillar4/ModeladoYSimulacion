(function () {
    "use strict";

    function normalizeLatex(text) {
        var input = String(text || "").trim();
        if (input.length >= 2 && input[0] === "$" && input[input.length - 1] === "$") {
            return input.slice(1, -1).trim();
        }
        return input;
    }

    function setEmpty(preview) {
        preview.innerHTML = '<span class="math-empty">Escriba una formula para verla aqui.</span>';
    }

    function setMathSource(preview, latex) {
        preview.textContent = "\\(" + latex + "\\)";
    }

    function typeset(preview) {
        if (!window.MathJax || !window.MathJax.typesetPromise) {
            return Promise.resolve();
        }
        return window.MathJax.typesetPromise([preview]).catch(function () {
            preview.innerHTML = '<span class="math-empty">LaTeX invalido.</span>';
        });
    }

    function debounce(callback, waitMs) {
        var timer = null;
        return function () {
            if (timer) {
                clearTimeout(timer);
            }
            timer = setTimeout(callback, waitMs);
        };
    }

    function bindPreview(input) {
        var preview = document.querySelector('[data-math-preview-for="' + input.id + '"]');
        if (!preview) {
            return;
        }

        var sync = debounce(function () {
            var latex = normalizeLatex(input.value);
            if (!latex) {
                setEmpty(preview);
                return;
            }

            setMathSource(preview, latex);

            if (window.__mathjaxReadyPromise) {
                window.__mathjaxReadyPromise.then(function () {
                    return typeset(preview);
                });
                return;
            }

            typeset(preview);
        }, 90);

        sync();
        input.addEventListener("input", sync);
    }

    document.addEventListener("DOMContentLoaded", function () {
        document.querySelectorAll("[data-math-input]").forEach(function (input) {
            if (input.id) bindPreview(input);
        });
    });
})();



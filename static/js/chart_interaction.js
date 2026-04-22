(function () {
    var MIN_SCALE = 1;
    var MAX_SCALE = 10;
    var ZOOM_STEP = 1.15;

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function applyTransform(state) {
        state.image.style.transform = "translate(" + state.tx + "px, " + state.ty + "px) scale(" + state.scale + ")";
    }

    function zoomToPoint(state, clientX, clientY, nextScale) {
        var rect = state.image.getBoundingClientRect();
        var pointerX = clientX - rect.left;
        var pointerY = clientY - rect.top;
        var contentX = (pointerX - state.tx) / state.scale;
        var contentY = (pointerY - state.ty) / state.scale;

        state.scale = nextScale;
        state.tx = pointerX - contentX * state.scale;
        state.ty = pointerY - contentY * state.scale;
        applyTransform(state);
    }

    function setupInteractiveChart(image) {
        if (image.dataset.panzoomReady === "true") {
            return;
        }

        var frame = document.createElement("div");
        frame.className = "chart-panzoom-frame";
        image.parentNode.insertBefore(frame, image);
        frame.appendChild(image);

        var state = {
            image: image,
            scale: 1,
            tx: 0,
            ty: 0,
            isDragging: false,
            pointerId: null,
            lastX: 0,
            lastY: 0,
        };

        image.dataset.panzoomReady = "true";
        image.setAttribute("title", "Rueda: zoom, arrastrar: mover, doble click: reset");

        image.addEventListener("wheel", function (event) {
            event.preventDefault();
            var factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
            var nextScale = clamp(state.scale * factor, MIN_SCALE, MAX_SCALE);
            zoomToPoint(state, event.clientX, event.clientY, nextScale);
        }, { passive: false });

        image.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) {
                return;
            }
            state.isDragging = true;
            state.pointerId = event.pointerId;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            image.classList.add("is-dragging");
            image.setPointerCapture(event.pointerId);
        });

        image.addEventListener("pointermove", function (event) {
            if (!state.isDragging || state.pointerId !== event.pointerId) {
                return;
            }
            var dx = event.clientX - state.lastX;
            var dy = event.clientY - state.lastY;
            state.tx += dx;
            state.ty += dy;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            applyTransform(state);
        });

        function stopDragging(event) {
            if (state.pointerId !== event.pointerId) {
                return;
            }
            state.isDragging = false;
            state.pointerId = null;
            image.classList.remove("is-dragging");
        }

        image.addEventListener("pointerup", stopDragging);
        image.addEventListener("pointercancel", stopDragging);

        image.addEventListener("dblclick", function () {
            state.scale = 1;
            state.tx = 0;
            state.ty = 0;
            applyTransform(state);
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        var charts = document.querySelectorAll("img.chart");
        charts.forEach(setupInteractiveChart);
    });
})();


// BaySpec app.js — minimal utilities; HTMX handles most interactivity.

// ── Auto-dismiss alerts ──────────────────────────────────────────────────────
document.addEventListener('htmx:afterSwap', () => attachDismiss());
document.addEventListener('DOMContentLoaded', () => attachDismiss());

function attachDismiss() {
  document.querySelectorAll('[data-autohide]').forEach(el => {
    if (el._bspHide) return;
    el._bspHide = true;
    const ms = parseInt(el.dataset.autohide, 10) || 5000;
    setTimeout(() => el.remove(), ms);
  });
}

// ── Plotly resize on sidebar resize ─────────────────────────────────────────
// Re-flow all Plotly charts when the viewport resizes (e.g. window resize).
let _resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      if (window.Plotly) Plotly.Plots.resize(el);
    });
  }, 200);
});

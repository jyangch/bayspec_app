// Minimal JS utilities — HTMX handles most interactivity

// Auto-dismiss alerts after 5 s
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

<script>
  document.addEventListener('DOMContentLoaded', function () {
    // 1) Toasts from server-rendered messages
    const errorEl = document.querySelector('.error-message');
    const errorText = errorEl && errorEl.textContent.trim();
    if (errorText) {
      Toastify({
        text: errorText,
        duration: 4000,
        gravity: "top",
        position: "right",
        close: true,
        stopOnFocus: true,
        style: { background: "linear-gradient(90deg,#ff5c7a,#ff8aa0)" }
      }).showToast();
    }

    // If you add success_message server-side, expose it as a data attr for JS (optional):
    // <div id="form-div" data-success="{{ success_message|default('') }}">
    const successText = (document.getElementById('form-div')?.dataset?.success || "").trim();
    if (successText) {
      Toastify({
        text: successText,
        duration: 3500,
        gravity: "top",
        position: "right",
        close: true,
        stopOnFocus: true,
        style: { background: "linear-gradient(90deg,#31d0aa,#00e0c7)" }
      }).showToast();
    }

    // 2) Password show/hide toggle
    const toggleBtn = document.querySelector('.toggle-pass');
    const passInput = document.getElementById('password');
    if (toggleBtn && passInput) {
      toggleBtn.addEventListener('click', () => {
        const isPwd = passInput.type === 'password';
        passInput.type = isPwd ? 'text' : 'password';
        toggleBtn.setAttribute('aria-label', isPwd ? 'Hide password' : 'Show password');
        toggleBtn.textContent = isPwd ? '🙈' : '👁️';
        passInput.focus();
      });
    }

    // 3) Submit UX
    const form = document.getElementById('form');
    const submitBtn = form?.querySelector('button[type="submit"]');
    form?.addEventListener('submit', () => {
      if (submitBtn) {
        submitBtn.disabled = true;
        const original = submitBtn.textContent;
        submitBtn.dataset.original = original;
        submitBtn.textContent = 'Signing in…';
      }
    });
  });
</script>

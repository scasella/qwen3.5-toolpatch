// Colour theme: auto (follows the OS) -> light -> dark, remembered per browser in localStorage (every access guarded: private
// windows and blocked storage simply fall back to auto). Loaded synchronously in <head> so the stored choice applies before paint.
(function () {
  var KEY = 'toolpatch-theme', root = document.documentElement;
  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function apply(t) { if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t); else root.removeAttribute('data-theme'); }
  apply(stored());
  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle'); if (!btn) return;
    function label() { var t = stored(); var name = t === 'light' ? 'light' : t === 'dark' ? 'dark' : 'auto (follows your system)'; btn.title = 'Theme: ' + name + '. Click to change.'; btn.setAttribute('aria-label', btn.title); }
    label();
    btn.addEventListener('click', function () {
      var t = stored(), next = t === 'light' ? 'dark' : t === 'dark' ? null : 'light';
      try { if (next) localStorage.setItem(KEY, next); else localStorage.removeItem(KEY); } catch (e) {}
      apply(next); label();
    });
  });
})();

document.getElementById('btnCheck').addEventListener('click', () => {
  const statusEl = document.getElementById('status');
  statusEl.textContent = `Status: Popup JS works (${new Date().toLocaleString()})`;
});

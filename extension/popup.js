document.getElementById("btnCheck").addEventListener("click", () => {
  const el = document.getElementById("status");
  const now = new Date().toLocaleString();
  el.textContent = `Status: Popup JS works (${now})`;
});

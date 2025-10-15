const btn = document.getElementById("btn");
const fileInput = document.getElementById("fileInput");
const out = document.getElementById("out");

btn.onclick = async () => {
  if (!fileInput.files.length) { alert("Seleccioná una imagen"); return; }
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  const res = await fetch("http://localhost:8000/predict", { method: "POST", body: fd });
  const data = await res.json();
  out.textContent = JSON.stringify(data, null, 2);
};

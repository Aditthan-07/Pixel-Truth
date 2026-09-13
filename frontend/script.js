const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const loader = document.getElementById('loader');
const errorBox = document.getElementById('errorBox');
const errorMsg = document.getElementById('errorMsg');
const verdictBadge = document.getElementById('verdictBadge');
const confValue = document.getElementById('confValue');
const confBar = document.getElementById('confBar');
const originalImg = document.getElementById('originalImg');
const gradcamImg = document.getElementById('gradcamImg');
const resetBtn = document.getElementById('resetBtn');
const metadataBanner = document.getElementById('metadataBanner');
const metadataText = document.getElementById('metadataText');
const exportBtn = document.getElementById('exportBtn');
let currentPrediction = null;

dropZone.addEventListener('click', (e) => {
  if (e.target.tagName === 'LABEL' || e.target.closest('label')) return;
  fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

resetBtn.addEventListener('click', () => {
  resultSection.style.display = 'none';
  errorBox.style.display = 'none';
  if (metadataBanner) metadataBanner.style.display = 'none';
  dropZone.parentElement.style.display = 'block';
  fileInput.value = '';
  resultCard.className = 'result-card';
  confBar.style.width = '0%';
  currentPrediction = null;
});

exportBtn.addEventListener('click', () => {
  if (!currentPrediction) return;
  const blob = new Blob([JSON.stringify(currentPrediction, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pixeltruth_forensic_report.json';
  a.click();
  URL.revokeObjectURL(url);
});

function showError(msg) {
  loader.style.display = 'none';
  errorBox.style.display = 'flex';
  errorMsg.textContent = msg;
}

// Clipboard paste support
window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) handleFile(file);
      break;
    }
  }
});

// Escape key to reset
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && resultSection.style.display !== 'none') {
    resetBtn.click();
  }
});

async function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
  if (!allowed.includes(file.type)) {
    showError('Unsupported file type. Please upload a PNG, JPG, WEBP, or GIF image.');
    return;
  }

  errorBox.style.display = 'none';
  resultSection.style.display = 'none';
  dropZone.parentElement.style.display = 'none';
  loader.style.display = 'flex';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const endpoint = window.location.origin.startsWith('http') ? '/predict' : 'http://localhost:5000/predict';
    const res = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    loader.style.display = 'none';

    const isReal = data.label === 'Real Image';
    const cls = isReal ? 'real' : 'fake';

    resultCard.className = `result-card ${cls}`;
    verdictBadge.className = `verdict ${cls}`;
    verdictBadge.textContent = data.label;

    confValue.textContent = `${data.confidence}%`;
    confBar.className = `conf-bar ${cls}`;

    originalImg.src = `data:image/png;base64,${data.original_image}`;
    gradcamImg.src = `data:image/png;base64,${data.gradcam_image}`;

    currentPrediction = data;
    if (metadataBanner && metadataText) {
      if (data.metadata_forensics?.ai_metadata_detected) {
        metadataBanner.style.display = 'block';
        metadataText.innerHTML = `⚠️ <strong>Generative AI Signature:</strong> Found tag <code>${data.metadata_forensics.ai_software_tag || 'AI Software'}</code>`;
      } else if (data.metadata_forensics?.has_camera_data) {
        metadataBanner.style.display = 'block';
        metadataText.innerHTML = `📷 <strong>Hardware Camera Info:</strong> ${data.metadata_forensics.camera_make || ''} ${data.metadata_forensics.camera_model || ''}`;
      } else {
        metadataBanner.style.display = 'none';
      }
    }

    resultSection.style.display = 'block';

    setTimeout(() => {
      confBar.style.width = `${data.confidence}%`;
    }, 100);

  } catch {
    showError('Could not connect to the server. Make sure the backend is running on port 5000.');
  }
}

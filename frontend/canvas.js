const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const topChar = document.getElementById('topChar');
const topConf = document.getElementById('topConf');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const chartPlaceholder = document.getElementById('chartPlaceholder');
const barChartCanvas = document.getElementById('barChart');

const BG_COLOR = '#000000';
const BRUSH_COLOR = '#ffffff';

let isDrawing = false;
let barChart = null;

function fillBackground() {
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  ctx.beginPath();
  const p = getPos(e);
  ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const p = getPos(e);
  ctx.lineWidth = 22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = BRUSH_COLOR;
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => { isDrawing = false; });
canvas.addEventListener('mouseleave', () => { isDrawing = false; });

canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  isDrawing = true;
  ctx.beginPath();
  const p = getPos(e);
  ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!isDrawing) return;
  const p = getPos(e);
  ctx.lineWidth = 22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = BRUSH_COLOR;
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});

canvas.addEventListener('touchend', () => { isDrawing = false; });

function getProcessedImageData() {
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  const offCtx = offscreen.getContext('2d');

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imageData.data;

  let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const idx = (y * canvas.width + x) * 4;
      const r = pixels[idx], g = pixels[idx+1], b = pixels[idx+2];
      if (r > 20 || g > 20 || b > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX <= minX || maxY <= minY) {
    return offscreen.toDataURL('image/png');
  }

  const pad = 30;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(canvas.width, maxX + pad);
  maxY = Math.min(canvas.height, maxY + pad);

  const w = maxX - minX;
  const h = maxY - minY;
  const size = Math.max(w, h);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  offCtx.fillStyle = BG_COLOR;
  offCtx.fillRect(0, 0, 28, 28);
  offCtx.drawImage(canvas, cx - size/2, cy - size/2, size, size, 0, 0, 28, 28);

  return offscreen.toDataURL('image/png');
}

clearBtn.addEventListener('click', () => {
  fillBackground();
  topChar.innerHTML = '<span style="opacity:0.2; font-size:2rem;">?</span>';
  topConf.textContent = '';
  errorMsg.style.display = 'none';
  if (barChart) { barChart.destroy(); barChart = null; }
  barChartCanvas.style.display = 'none';
  chartPlaceholder.style.display = 'flex';
});

predictBtn.addEventListener('click', async () => {
  const imageData = getProcessedImageData();

  loading.style.display = 'block';
  errorMsg.style.display = 'none';
  predictBtn.disabled = true;

  try {
    const API_URL = window.location.hostname === '127.0.0.1' 
  ? 'http://127.0.0.1:5000' 
  : '';

const res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });

    if (!res.ok) throw new Error('Server error');

    const data = await res.json();
    const preds = data.predictions;

    topChar.textContent = preds[0].label;
    topConf.textContent = preds[0].confidence + '% confidence';

    chartPlaceholder.style.display = 'none';
    barChartCanvas.style.display = 'block';

    if (barChart) barChart.destroy();

    barChart = new Chart(barChartCanvas, {
      type: 'bar',
      data: {
        labels: preds.map(p => p.label),
        datasets: [{
          data: preds.map(p => p.confidence),
          backgroundColor: ['rgba(124,106,255,0.8)', 'rgba(124,106,255,0.45)', 'rgba(124,106,255,0.25)'],
          borderColor: ['#7c6aff', '#7c6aff', '#7c6aff'],
          borderWidth: 1,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => c.parsed.y.toFixed(2) + '%' } }
        },
        scales: {
          x: {
            ticks: { color: '#e8e8f0', font: { family: 'Space Mono', size: 13, weight: 'bold' } },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            min: 0,
            max: 100,
            ticks: { color: '#6b6b80', font: { family: 'Space Mono', size: 10 }, callback: (v) => v + '%' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });

  } catch (err) {
    errorMsg.textContent = '⚠ Could not connect to server. Is Flask running?';
    errorMsg.style.display = 'block';
  } finally {
    loading.style.display = 'none';
    predictBtn.disabled = false;
  }
});

fillBackground();

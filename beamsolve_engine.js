// =============================================================================
// BeamSolve Engine v3 — State, Helpers, LU Solver, FEM Solver
// =============================================================================

const state = {
  beam: { length: 5000, E: 210000, I: 83560000, A: 2850 },
  objects: [],
  result: null,
  tool: null,
  distStart: null,
  profileMode: 'custom',
  selectedProfile: null,
  isPro: false,
  editingTrapIdx: -1,
};

let dragging = null;
let hovering = -1;
let hoverPart = null;
let _id_counter = 0;
function newId() { return '_' + (++_id_counter); }

// Set today's date
try { document.getElementById('proj-date').valueAsDate = new Date(); } catch(e) {}

// ===== CANVAS REFS =====
const beamCanvas = document.getElementById('beam-canvas');
const shearCanvas = document.getElementById('shear-canvas');
const momentCanvas = document.getElementById('moment-canvas');
const normalCanvas = document.getElementById('normal-canvas');
const deflCanvas = document.getElementById('deflection-canvas');
const csCanvas = document.getElementById('cs-canvas');

const beamCtx = beamCanvas.getContext('2d');
const shearCtx = shearCanvas.getContext('2d');
const momentCtx = momentCanvas.getContext('2d');
const normalCtx = normalCanvas.getContext('2d');
const deflCtx = deflCanvas.getContext('2d');
const csCtx = csCanvas ? csCanvas.getContext('2d') : null;

const ML = 60, MR = 30, BYR = 0.48, BH = 7, HR = 16, HH = 12;
const _f = "'Inter','Segoe UI',sans-serif";

// ===== SNAP =====
function snapOn() { return document.getElementById('snap-check').checked; }
function snapSz() { return parseInt(document.getElementById('snap-size').value) || 100; }
function snap(v) { if (!snapOn()) return v; const s = snapSz(); return Math.round(v / s) * s; }

document.getElementById('snap-check').addEventListener('change', () => {
  document.getElementById('snap-status').textContent = snapOn() ? 'Snap: ' + snapSz() + 'mm' : 'Snap: Off';
});
document.getElementById('snap-size').addEventListener('change', () => {
  document.getElementById('snap-status').textContent = 'Snap: ' + snapSz() + 'mm';
});

// ===== COORDINATE HELPERS =====
function bpr(c) { const d = window.devicePixelRatio || 1; return { l: ML, r: c.width / d - MR }; }
function b2px(x, c) { const { l, r } = bpr(c); return l + (x / state.beam.length) * (r - l); }
function px2b(px, c) { const { l, r } = bpr(c); return Math.max(0, Math.min(state.beam.length, ((px - l) / (r - l)) * state.beam.length)); }
function bY(c) { const d = window.devicePixelRatio || 1; return (c.height / d) * BYR; }

// ===== FORMAT HELPERS =====
function fmtV(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(1) + 'G';
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.001) return v.toFixed(3);
  return v.toExponential(1);
}
function fmtF(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MN';
  if (v >= 1e3) return (v / 1e3).toFixed(2) + ' kN';
  return v.toFixed(0) + ' N';
}
function rr(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

// =============================================================================
// LU SOLVER
// =============================================================================
function luSolve(A, b) {
  const n = A.length;
  const Ab = A.map(row => Float64Array.from(row));
  const bb = Float64Array.from(b);
  for (let k = 0; k < n; k++) {
    let mv = 0, mi = k;
    for (let i = k; i < n; i++) { const v = Math.abs(Ab[i][k]); if (v > mv) { mv = v; mi = i; } }
    if (mv < 1e-30) continue;
    if (mi !== k) { [Ab[k], Ab[mi]] = [Ab[mi], Ab[k]]; [bb[k], bb[mi]] = [bb[mi], bb[k]]; }
    for (let i = k + 1; i < n; i++) {
      const f = Ab[i][k] / Ab[k][k]; Ab[i][k] = f;
      for (let j = k + 1; j < n; j++) Ab[i][j] -= f * Ab[k][j];
      bb[i] -= f * bb[k];
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = bb[i]; for (let j = i + 1; j < n; j++) s -= Ab[i][j] * x[j];
    x[i] = Math.abs(Ab[i][i]) > 1e-30 ? s / Ab[i][i] : 0;
  }
  return x;
}

// =============================================================================
// FEM SOLVER — Direct Stiffness Method (3 DOF/node: u, w, theta)
// =============================================================================
function solveBeam() {
  const sups = state.objects.filter(o => o.type === 'support');
  const pls = state.objects.filter(o => o.type === 'load');
  const mls = state.objects.filter(o => o.type === 'moment');
  const dls = state.objects.filter(o => o.type === 'distributed');
  const tls = state.objects.filter(o => o.type === 'trapezoid');

  if (sups.length < 1) { state.result = null; return; }

  const L = state.beam.length, EI = state.beam.E * state.beam.I, EA = state.beam.E * state.beam.A;
  const nE = 80, nN = nE + 1, nD = 3 * nN, Le = L / nE;

  const K = Array.from({ length: nD }, () => new Float64Array(nD));
  const F = new Float64Array(nD);
  const cB = EI / (Le * Le * Le), L2 = Le * Le, cA = EA / Le;

  for (let e = 0; e < nE; e++) {
    const d = [3 * e, 3 * e + 1, 3 * e + 2, 3 * e + 3, 3 * e + 4, 3 * e + 5];
    const ke = [
      [cA, 0, 0, -cA, 0, 0],
      [0, 12 * cB, 6 * Le * cB, 0, -12 * cB, 6 * Le * cB],
      [0, 6 * Le * cB, 4 * L2 * cB, 0, -6 * Le * cB, 2 * L2 * cB],
      [-cA, 0, 0, cA, 0, 0],
      [0, -12 * cB, -6 * Le * cB, 0, 12 * cB, -6 * Le * cB],
      [0, 6 * Le * cB, 2 * L2 * cB, 0, -6 * Le * cB, 4 * L2 * cB],
    ];
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) K[d[r]][d[c]] += ke[r][c];
  }

  // Point loads with angle
  for (const pl of pls) {
    const ang = (pl.angle || 0) * Math.PI / 180;
    const Fv = pl.value * Math.cos(ang), Fh = pl.value * Math.sin(ang);
    const ei = Math.min(Math.floor(pl.x / Le), nE - 1);
    const a = pl.x - ei * Le, b = Le - a;
    const d = [3 * ei + 1, 3 * ei + 2, 3 * ei + 4, 3 * ei + 5];
    const fq = [Fv * b * b * (3 * a + b) / (Le * Le * Le), Fv * a * b * b / (Le * Le),
                Fv * a * a * (a + 3 * b) / (Le * Le * Le), -Fv * a * a * b / (Le * Le)];
    for (let r = 0; r < 4; r++) F[d[r]] += fq[r];
    if (Math.abs(Fh) > 1e-10) { const xi = a / Le; F[3 * ei] += Fh * (1 - xi); F[3 * ei + 3] += Fh * xi; }
  }

  // Moment loads
  for (const ml of mls) {
    const ei = Math.min(Math.floor(ml.x / Le), nE - 1);
    const a = ml.x - ei * Le, b = Le - a, M = ml.value;
    const d = [3 * ei + 1, 3 * ei + 2, 3 * ei + 4, 3 * ei + 5];
    const fq = [-6 * M * a * b / (Le * Le * Le), M * b * (2 * a - b) / (Le * Le),
                6 * M * a * b / (Le * Le * Le), M * a * (2 * b - a) / (Le * Le)];
    for (let r = 0; r < 4; r++) F[d[r]] += fq[r];
  }

  // Distributed and trapezoid loads
  for (const dl of dls) applyDL(F, dl.x, dl.x2, dl.value, dl.value, Le, nE);
  for (const tl of tls) applyDL(F, tl.x, tl.x2, tl.value, tl.value2, Le, nE);

  // Boundary conditions
  const con = new Set(), snm = {};
  for (const s of sups) {
    const cn = Math.max(0, Math.min(nN - 1, Math.round(s.x / Le)));
    const uD = 3 * cn, wD = 3 * cn + 1, tD = 3 * cn + 2;
    con.add(wD);
    if (!snm[s._id]) snm[s._id] = [];
    snm[s._id].push(wD);
    if (s.subtype === 'pinned') { con.add(uD); snm[s._id].push(uD); }
    else if (s.subtype === 'fixed') { con.add(uD); con.add(tD); snm[s._id].push(uD); snm[s._id].push(tD); }
  }

  const free = [];
  for (let i = 0; i < nD; i++) if (!con.has(i)) free.push(i);
  if (free.length === 0) { state.result = null; return; }

  const nf = free.length;
  const Kf = Array.from({ length: nf }, () => new Float64Array(nf));
  const Ff = new Float64Array(nf);
  for (let i = 0; i < nf; i++) { Ff[i] = F[free[i]]; for (let j = 0; j < nf; j++) Kf[i][j] = K[free[i]][free[j]]; }

  const uf = luSolve(Kf, Ff);
  const u = new Float64Array(nD);
  for (let i = 0; i < nf; i++) u[free[i]] = uf[i];

  // Reactions
  const reactions = {};
  const Ku = new Float64Array(nD);
  for (let i = 0; i < nD; i++) { let s = 0; for (let j = 0; j < nD; j++) s += K[i][j] * u[j]; Ku[i] = s; }
  for (const s of sups) {
    if (snm[s._id]) {
      let rv = 0, rh = 0;
      for (const dof of snm[s._id]) { const r = Ku[dof] - F[dof]; if (dof % 3 === 1) rv += r; else if (dof % 3 === 0) rh += r; }
      reactions[s._id] = { v: rv, h: rh };
    }
  }

  // Nodal results
  const xA = new Float64Array(nN), wA = new Float64Array(nN), tA = new Float64Array(nN), uA = new Float64Array(nN);
  for (let i = 0; i < nN; i++) { xA[i] = i * Le; uA[i] = u[3 * i]; wA[i] = u[3 * i + 1]; tA[i] = u[3 * i + 2]; }

  // Internal forces
  const N = new Float64Array(nN), V = new Float64Array(nN), M = new Float64Array(nN);
  for (let e = 0; e < nE; e++) {
    const d = [3 * e, 3 * e + 1, 3 * e + 2, 3 * e + 3, 3 * e + 4, 3 * e + 5];
    const ue = d.map(i => u[i]);
    const ke = [[cA, 0, 0, -cA, 0, 0], [0, 12 * cB, 6 * Le * cB, 0, -12 * cB, 6 * Le * cB],
      [0, 6 * Le * cB, 4 * L2 * cB, 0, -6 * Le * cB, 2 * L2 * cB], [-cA, 0, 0, cA, 0, 0],
      [0, -12 * cB, -6 * Le * cB, 0, 12 * cB, -6 * Le * cB], [0, 6 * Le * cB, 2 * L2 * cB, 0, -6 * Le * cB, 4 * L2 * cB]];
    const fe = [0, 0, 0, 0, 0, 0];
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) fe[r] += ke[r][c] * ue[c];
    if (e === 0) { N[0] = fe[0]; V[0] = fe[1]; M[0] = -fe[2]; }
    N[e + 1] = -fe[3]; V[e + 1] = -fe[4]; M[e + 1] = fe[5];
  }

  state.result = {
    x: xA, uAxial: uA, w: wA, theta: tA, N, V, M, reactions,
    maxN: Math.max(...Array.from(N).map(Math.abs)),
    maxV: Math.max(...Array.from(V).map(Math.abs)),
    maxM: Math.max(...Array.from(M).map(Math.abs)),
    maxW: Math.max(...Array.from(wA).map(Math.abs)),
  };
}

function applyDL(F, xS, xE, qS, qE, Le, nE) {
  const sE = Math.max(0, Math.floor(xS / Le)), eE = Math.min(nE - 1, Math.ceil(xE / Le) - 1);
  const tL = xE - xS;
  for (let e = sE; e <= eE; e++) {
    const xL = e * Le, xR = (e + 1) * Le, xs = Math.max(xL, xS), xe = Math.min(xR, xE);
    if (xe <= xs) continue;
    const d = [3 * e + 1, 3 * e + 2, 3 * e + 4, 3 * e + 5];
    const gp = [-0.861136, -0.339981, 0.339981, 0.861136], gw = [0.347855, 0.652145, 0.652145, 0.347855];
    const mid = (xs + xe) / 2 - xL, hl = (xe - xs) / 2;
    for (let g = 0; g < 4; g++) {
      const xl = mid + hl * gp[g], xG = xL + xl, xi = xl / Le;
      const t = tL > 1e-10 ? (xG - xS) / tL : 0;
      const q = qS + (qE - qS) * t;
      const Ns = [1 - 3 * xi * xi + 2 * xi * xi * xi, Le * xi * (1 - xi) * (1 - xi), 3 * xi * xi - 2 * xi * xi * xi, Le * xi * xi * (xi - 1)];
      for (let r = 0; r < 4; r++) F[d[r]] += q * Ns[r] * hl * gw[g];
    }
  }
}

// =============================================================================
// PROFILE LIBRARY
// =============================================================================
function initProfiles() {
  const ts = document.getElementById('prof-type');
  const ms = document.getElementById('prof-mat');
  if (!ts || !ms) return;
  ts.innerHTML = '';
  for (const k of getProfileTypes()) {
    const o = document.createElement('option'); o.value = k;
    o.textContent = PROFILES[k].label + ' (' + PROFILES[k].standard + ')';
    ts.appendChild(o);
  }
  ms.innerHTML = '';
  for (const k of Object.keys(MATERIALS)) {
    const o = document.createElement('option'); o.value = k; o.textContent = MATERIALS[k].name;
    ms.appendChild(o);
  }
  onProfTypeChange();
}

function onProfTypeChange() {
  const type = document.getElementById('prof-type').value;
  const ns = document.getElementById('prof-name');
  const profs = getProfilesForType(type);
  ns.innerHTML = '';
  for (const p of profs) {
    const o = document.createElement('option'); o.value = p.name;
    o.textContent = p.name + ' (' + p.mass.toFixed(1) + ' kg/m)';
    ns.appendChild(o);
  }
  document.getElementById('prof-mat').value = PROFILES[type].material;
  onProfSelect();
}

function onProfSelect() {
  const type = document.getElementById('prof-type').value;
  const name = document.getElementById('prof-name').value;
  const matK = document.getElementById('prof-mat').value;
  const prof = getProfile(type, name);
  if (!prof) return;
  const mat = MATERIALS[matK];
  const E = mat ? mat.E : getProfileE(type);
  state.selectedProfile = { ...prof, E };
  document.getElementById('inp-E').value = E;
  document.getElementById('inp-I').value = prof.Ix;
  document.getElementById('inp-A').value = prof.A;
  const info = document.getElementById('prof-info');
  if (info) {
    info.innerHTML = '<b>' + prof.name + '</b> · h=' + prof.h + ' b=' + prof.b +
      ' tw=' + prof.tw + ' tf=' + prof.tf + ' mm<br>Ix=' + fmtV(prof.Ix) +
      ' mm⁴ · Wx=' + fmtV(prof.Wx) + ' mm³ · A=' + prof.A + ' mm² · ' + prof.mass + ' kg/m';
  }
  drawCS(prof);
  applyBeamProps();
}

function setProfileMode(mode) {
  if (mode === 'profile' && !state.isPro) { showPricing(); return; }
  state.profileMode = mode;
  document.getElementById('tog-custom').classList.toggle('active', mode === 'custom');
  document.getElementById('tog-profile').classList.toggle('active', mode === 'profile');
  document.getElementById('profile-sel').style.display = mode === 'profile' ? 'block' : 'none';
  document.querySelectorAll('#beam-fields input[type="number"]').forEach(f => {
    if (f.id === 'inp-length') return;
    f.readOnly = (mode === 'profile');
    f.style.opacity = mode === 'profile' ? '0.5' : '1';
  });
  if (mode === 'profile') onProfSelect();
}

// ===== CROSS SECTION DRAWING =====
function drawCS(p) {
  if (!p || !csCtx) return;
  const c = csCanvas, dpr = window.devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
  const ctx = csCtx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = c.clientWidth, h = c.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2;
  const sc = Math.min((w - 40) / p.b, (h - 20) / p.h) * 0.75;
  const sh = p.h * sc, sb = p.b * sc;
  const stw = Math.max(p.tw * sc, 2), stf = Math.max(p.tf * sc, 2);
  ctx.save(); ctx.translate(cx, cy);
  ctx.fillStyle = '#e2e8f0'; ctx.strokeStyle = '#475569'; ctx.lineWidth = 1.5;
  ctx.fillRect(-sb / 2, -sh / 2, sb, stf); ctx.strokeRect(-sb / 2, -sh / 2, sb, stf);
  ctx.fillRect(-sb / 2, sh / 2 - stf, sb, stf); ctx.strokeRect(-sb / 2, sh / 2 - stf, sb, stf);
  ctx.fillRect(-stw / 2, -sh / 2 + stf, stw, sh - 2 * stf); ctx.strokeRect(-stw / 2, -sh / 2 + stf, stw, sh - 2 * stf);
  // Dimensions
  ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 0.7; ctx.fillStyle = '#64748b'; ctx.font = '8px ' + _f;
  const dx = sb / 2 + 10;
  ctx.beginPath(); ctx.moveTo(dx, -sh / 2); ctx.lineTo(dx, sh / 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(dx - 2, -sh / 2); ctx.lineTo(dx + 2, -sh / 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(dx - 2, sh / 2); ctx.lineTo(dx + 2, sh / 2); ctx.stroke();
  ctx.textAlign = 'left'; ctx.fillText(p.h + '', dx + 4, 3);
  const dy = sh / 2 + 10;
  ctx.beginPath(); ctx.moveTo(-sb / 2, dy); ctx.lineTo(sb / 2, dy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-sb / 2, dy - 2); ctx.lineTo(-sb / 2, dy + 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sb / 2, dy - 2); ctx.lineTo(sb / 2, dy + 2); ctx.stroke();
  ctx.textAlign = 'center'; ctx.fillText(p.b + '', 0, dy + 10);
  ctx.restore();
}

// =============================================================================
// RENDERING
// =============================================================================
function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const bc = document.getElementById('beam-card');
  const bcW = bc.clientWidth, bcH = 210;
  beamCanvas.width = bcW * dpr; beamCanvas.height = bcH * dpr;
  beamCanvas.style.height = bcH + 'px';
  beamCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  document.querySelectorAll('.dg-card').forEach(card => {
    const cv = card.querySelector('canvas');
    if (!cv) return;
    const hdr = card.querySelector('.dg-hdr');
    const cw = card.clientWidth, ch = card.clientHeight - (hdr ? hdr.offsetHeight : 0);
    cv.width = cw * dpr; cv.height = Math.max(ch, 40) * dpr;
    cv.style.height = Math.max(ch, 40) + 'px';
    cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  });

  if (csCanvas && csCanvas.clientWidth > 0) {
    csCanvas.width = csCanvas.clientWidth * dpr;
    csCanvas.height = csCanvas.clientHeight * dpr;
    csCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (state.selectedProfile) drawCS(state.selectedProfile);
  }
}

function renderAll() {
  renderBeam();
  const r = state.result;
  // Convert to engineering units for diagrams
  const vKN = r?.V ? r.V.map(v => v / 1000) : null;
  const mKNm = r?.M ? r.M.map(v => v / 1e6) : null;
  const nKN = r?.N ? r.N.map(v => v / 1000) : null;
  const wFlip = r?.w ? r.w.map(v => -v) : null; // flip: positive load → curve goes DOWN
  renderDiag(shearCtx, shearCanvas, vKN, r?.x, '#2563eb', 'rgba(37,99,235,0.10)', 'kN');
  renderDiag(momentCtx, momentCanvas, mKNm, r?.x, '#dc2626', 'rgba(220,38,38,0.10)', 'kNm');
  renderDiag(normalCtx, normalCanvas, nKN, r?.x, '#7c3aed', 'rgba(124,58,237,0.10)', 'kN');
  renderDiag(deflCtx, deflCanvas, wFlip, r?.x, '#d97706', 'rgba(217,119,6,0.10)', 'mm');
  updateStats(); updateObjLists();
  document.getElementById('beam-info').textContent =
    'L=' + state.beam.length + 'mm · E=' + state.beam.E + 'MPa · I=' + fmtV(state.beam.I) + 'mm⁴';
}

function renderBeam() {
  const c = beamCanvas, ctx = beamCtx, dpr = window.devicePixelRatio || 1;
  const w = c.width / dpr, h = c.height / dpr;
  ctx.clearRect(0, 0, w, h);
  const { l, r } = bpr(c); const by = bY(c);

  // Grid
  ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 0.5;
  for (let gx = 0; gx < w; gx += 50) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke(); }
  for (let gy = 0; gy < h; gy += 50) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke(); }

  if (snapOn()) {
    ctx.strokeStyle = 'rgba(37,99,235,0.06)'; ctx.lineWidth = 1;
    for (let mm = 0; mm <= state.beam.length; mm += snapSz()) {
      const px = b2px(mm, c); ctx.beginPath(); ctx.moveTo(px, by - 25); ctx.lineTo(px, by + 25); ctx.stroke();
    }
  }

  // Axis
  ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 0.5; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(l - 10, by); ctx.lineTo(r + 10, by); ctx.stroke(); ctx.setLineDash([]);

  // Deflection
  if (state.result?.w) {
    const { x: xA, w: wA, maxW } = state.result;
    if (maxW > 1e-15) {
      const sc = 40 / maxW; ctx.beginPath();
      for (let i = 0; i < xA.length; i++) { const px = b2px(xA[i], c), py = by + wA[i] * sc; i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    }
  }

  // Beam body
  const gr = ctx.createLinearGradient(0, by - BH / 2, 0, by + BH / 2);
  gr.addColorStop(0, '#94a3b8'); gr.addColorStop(1, '#64748b');
  ctx.fillStyle = gr; rr(ctx, l, by - BH / 2, r - l, BH, 3); ctx.fill();

  // Dimension line
  const dimY = by + 60;
  ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(l, dimY); ctx.lineTo(r, dimY);
  ctx.moveTo(l, dimY - 3); ctx.lineTo(l, dimY + 3);
  ctx.moveTo(r, dimY - 3); ctx.lineTo(r, dimY + 3); ctx.stroke();
  ctx.fillStyle = '#94a3b8'; ctx.font = '10px ' + _f; ctx.textAlign = 'center';
  ctx.fillText(state.beam.length + ' mm', (l + r) / 2, dimY + 13);

  // Objects
  for (let i = 0; i < state.objects.length; i++) {
    const o = state.objects[i];
    if (o.type === 'distributed') drawDL(ctx, c, o, by, i);
    if (o.type === 'trapezoid') drawTL(ctx, c, o, by, i);
  }
  for (let i = 0; i < state.objects.length; i++) if (state.objects[i].type === 'support') drawSup(ctx, c, state.objects[i], by, i === hovering);
  for (let i = 0; i < state.objects.length; i++) {
    const o = state.objects[i];
    if (o.type === 'load') drawPL(ctx, c, o, by, i === hovering);
    if (o.type === 'moment') drawML(ctx, c, o, by, i === hovering);
  }

  // Reactions
  if (state.result?.reactions) {
    ctx.font = 'bold 8px ' + _f; ctx.textAlign = 'center';
    for (const s of state.objects.filter(o => o.type === 'support')) {
      const rx = state.result.reactions[s._id];
      if (rx) {
        ctx.fillStyle = 'rgba(37,99,235,0.6)';
        let lb = 'Rv=' + fmtF(Math.abs(rx.v));
        if (Math.abs(rx.h) > 0.01) lb += ' Rh=' + fmtF(Math.abs(rx.h));
        ctx.fillText(lb, b2px(s.x, c), by + 48);
      }
    }
  }

  // Tool hint
  if (state.tool) {
    ctx.fillStyle = 'rgba(37,99,235,0.06)'; rr(ctx, l, by - 45, r - l, 90, 6); ctx.fill();
    ctx.fillStyle = '#3b82f6'; ctx.font = '11px ' + _f; ctx.textAlign = 'center';
    const ht = { support: 'Click to place support', load: 'Click to place load', moment: 'Click to place moment',
      distributed: state.distStart === null ? 'Click START' : 'Click END', trapezoid: state.distStart === null ? 'Click START' : 'Click END' };
    ctx.fillText(ht[state.tool] || '', (l + r) / 2, by - 50 + 6);
  }
}

// ===== DRAW SUPPORT =====
function drawSup(ctx, c, s, by, isH) {
  const px = b2px(s.x, c), sz = isH ? 20 : 16, col = isH ? '#3b82f6' : '#2563eb';
  ctx.save(); ctx.translate(px, by);
  if (s.subtype === 'fixed') {
    ctx.strokeStyle = col; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(0, -16); ctx.lineTo(0, 16); ctx.stroke();
    ctx.lineWidth = 1.2;
    for (let i = 0; i < 5; i++) { const y = -16 + i * 8; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(-7, y + 5); ctx.stroke(); }
  } else {
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-sz / 2, sz * 0.85); ctx.lineTo(sz / 2, sz * 0.85); ctx.closePath();
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.fillStyle = isH ? 'rgba(37,99,235,0.1)' : 'rgba(37,99,235,0.04)'; ctx.fill(); ctx.stroke();
    let bsy = sz * 0.85 + 3;
    if (s.subtype === 'roller') { ctx.beginPath(); ctx.arc(0, sz * 0.85 + 5, 3.5, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); bsy = sz * 0.85 + 10; }
    ctx.strokeStyle = col; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(-sz / 2 - 3, bsy); ctx.lineTo(sz / 2 + 3, bsy); ctx.stroke();
    ctx.lineWidth = 0.8;
    for (let i = 0; i < 5; i++) { const xh = -sz / 2 - 3 + i * (sz + 6) / 4; ctx.beginPath(); ctx.moveTo(xh, bsy); ctx.lineTo(xh - 4, bsy + 5); ctx.stroke(); }
  }
  ctx.restore();
}

// ===== DRAW POINT LOAD =====
function drawPL(ctx, c, ld, by, isH) {
  const px = b2px(ld.x, c), ang = (ld.angle || 0) * Math.PI / 180, aL = 55;
  const col = isH ? '#f87171' : '#dc2626';
  ctx.save(); ctx.translate(px, by);
  if (Math.abs(ld.angle || 0) < 0.1) {
    const d = ld.value > 0 ? 1 : -1;
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, -d * aL); ctx.lineTo(0, 0); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-3.5, -d * 7); ctx.lineTo(3.5, -d * 7); ctx.closePath(); ctx.fill();
  } else {
    const d = ld.value > 0 ? 1 : -1;
    const sx = -aL * Math.sin(ang), sy = -d * aL * Math.cos(ang);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(0, 0); ctx.stroke();
    const dx = -sx, dy = -sy, ln = Math.sqrt(dx * dx + dy * dy), ux = dx / ln, uy = dy / ln;
    ctx.fillStyle = col; ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(-ux * 7 - uy * 3.5, -uy * 7 + ux * 3.5); ctx.lineTo(-ux * 7 + uy * 3.5, -uy * 7 - ux * 3.5); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = 'rgba(220,38,38,0.3)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); const bA = d > 0 ? -Math.PI / 2 : Math.PI / 2;
    ctx.arc(0, 0, 16, bA, bA + ang, ang < 0); ctx.stroke();
    ctx.fillStyle = 'rgba(220,38,38,0.5)'; ctx.font = '8px ' + _f; ctx.textAlign = 'left';
    ctx.fillText((ld.angle || 0).toFixed(0) + '°', 18, -4);
  }
  ctx.fillStyle = col; ctx.font = 'bold 10px ' + _f; ctx.textAlign = 'center';
  const d = ld.value > 0 ? 1 : -1;
  ctx.fillText(fmtF(Math.abs(ld.value)), 0, -d * aL - 7 * d);
  ctx.restore();
}

// ===== DRAW MOMENT LOAD =====
function drawML(ctx, c, ml, by, isH) {
  const px = b2px(ml.x, c), col = isH ? '#a78bfa' : '#7c3aed', rd = 16, d = ml.value > 0 ? 1 : -1;
  ctx.save(); ctx.translate(px, by);
  ctx.strokeStyle = col; ctx.lineWidth = 2;
  const sA = -Math.PI * 0.8, eA = Math.PI * 0.5;
  ctx.beginPath(); ctx.arc(0, 0, rd, sA, eA, d < 0); ctx.stroke();
  const tA = d > 0 ? eA : sA, tx = rd * Math.cos(tA), ty = rd * Math.sin(tA);
  const ta = tA + (d > 0 ? Math.PI / 2 : -Math.PI / 2), hs = 6;
  ctx.fillStyle = col; ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - hs * Math.cos(ta) - hs / 2 * Math.sin(ta), ty - hs * Math.sin(ta) + hs / 2 * Math.cos(ta));
  ctx.lineTo(tx - hs * Math.cos(ta) + hs / 2 * Math.sin(ta), ty - hs * Math.sin(ta) - hs / 2 * Math.cos(ta));
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = col; ctx.font = 'bold 9px ' + _f; ctx.textAlign = 'center';
  const av = Math.abs(ml.value);
  let lb; if (av >= 1e6) lb = (av / 1e6).toFixed(1) + ' kN·m'; else if (av >= 1e3) lb = (av / 1e3).toFixed(1) + ' N·m'; else lb = av.toFixed(0) + ' N·mm';
  ctx.fillText(lb, 0, -rd - 7);
  ctx.restore();
}

// ===== DRAW DISTRIBUTED LOAD =====
function drawDL(ctx, c, dl, by, idx) {
  const pS = b2px(dl.x, c), pE = b2px(dl.x2, c), d = dl.value > 0 ? 1 : -1, aL = 42;
  const col = '#dc2626', isH = idx === hovering;
  ctx.save(); ctx.translate(0, by);
  ctx.fillStyle = isH ? 'rgba(220,38,38,0.08)' : 'rgba(220,38,38,0.04)';
  ctx.fillRect(pS, 0, pE - pS, -d * aL);
  ctx.strokeStyle = col; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(pS, -d * aL); ctx.lineTo(pE, -d * aL); ctx.stroke();
  const nA = Math.max(3, Math.round((pE - pS) / 28));
  for (let i = 0; i <= nA; i++) {
    const ax = pS + i * (pE - pS) / nA;
    ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ax, -d * aL); ctx.lineTo(ax, 0); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(ax, 0); ctx.lineTo(ax - 2.5, -d * 5); ctx.lineTo(ax + 2.5, -d * 5); ctx.closePath(); ctx.fill();
  }
  if (isH) { ctx.fillStyle = '#f87171'; ctx.beginPath(); ctx.arc(pS, -d * aL / 2, 4, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.arc(pE, -d * aL / 2, 4, 0, Math.PI * 2); ctx.fill(); }
  ctx.fillStyle = col; ctx.font = 'bold 9px ' + _f; ctx.textAlign = 'center';
  ctx.fillText(Math.abs(dl.value).toFixed(1) + ' N/mm', (pS + pE) / 2, -d * aL - 7 * d);
  ctx.restore();
}

// ===== DRAW TRAPEZOID LOAD =====
function drawTL(ctx, c, tl, by, idx) {
  const pS = b2px(tl.x, c), pE = b2px(tl.x2, c);
  const mM = Math.max(Math.abs(tl.value), Math.abs(tl.value2), 0.01), aM = 48;
  const hS = (Math.abs(tl.value) / mM) * aM, hE = (Math.abs(tl.value2) / mM) * aM;
  const dS = tl.value >= 0 ? 1 : -1, dE = tl.value2 >= 0 ? 1 : -1;
  const col = '#dc2626', isH = idx === hovering, nP = 20;
  ctx.save(); ctx.translate(0, by);

  // Fill
  ctx.beginPath(); ctx.moveTo(pS, 0); ctx.lineTo(pS, -dS * hS);
  for (let i = 1; i <= nP; i++) { const t = i / nP; ctx.lineTo(pS + t * (pE - pS), -(dS + t * (dE - dS)) * (hS + t * (hE - hS))); }
  ctx.lineTo(pE, 0); ctx.closePath();
  ctx.fillStyle = isH ? 'rgba(220,38,38,0.08)' : 'rgba(220,38,38,0.04)'; ctx.fill();

  // Hatching
  ctx.strokeStyle = 'rgba(220,38,38,0.15)'; ctx.lineWidth = 0.6;
  const tPx = pE - pS;
  for (let dd = 7; dd < tPx + aM; dd += 7) {
    ctx.save();
    ctx.beginPath(); ctx.moveTo(pS, 0); ctx.lineTo(pS, -dS * hS);
    for (let i = 1; i <= nP; i++) { const t = i / nP; ctx.lineTo(pS + t * (pE - pS), -(dS + t * (dE - dS)) * (hS + t * (hE - hS))); }
    ctx.lineTo(pE, 0); ctx.closePath(); ctx.clip();
    ctx.beginPath(); ctx.moveTo(pS + dd, 0); ctx.lineTo(pS + dd - aM * 0.7, -aM * 0.7); ctx.stroke();
    ctx.restore();
  }

  // Outline
  ctx.strokeStyle = col; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(pS, 0); ctx.lineTo(pS, -dS * hS);
  for (let i = 1; i <= nP; i++) { const t = i / nP; ctx.lineTo(pS + t * (pE - pS), -(dS + t * (dE - dS)) * (hS + t * (hE - hS))); }
  ctx.lineTo(pE, 0); ctx.stroke();

  // Arrows
  const nA = Math.max(3, Math.round((pE - pS) / 30));
  for (let i = 0; i <= nA; i++) {
    const t = i / nA, ax = pS + t * (pE - pS), ah = hS + t * (hE - hS), ad = dS + t * (dE - dS);
    if (ah < 2) continue;
    ctx.strokeStyle = col; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(ax, -ad * ah); ctx.lineTo(ax, 0); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(ax, 0); ctx.lineTo(ax - 2, -ad * 4); ctx.lineTo(ax + 2, -ad * 4); ctx.closePath(); ctx.fill();
  }

  if (isH) { ctx.fillStyle = '#f87171'; ctx.beginPath(); ctx.arc(pS, -dS * hS / 2, 4, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.arc(pE, -dE * hE / 2, 4, 0, Math.PI * 2); ctx.fill(); }
  ctx.fillStyle = col; ctx.font = 'bold 8px ' + _f;
  ctx.textAlign = 'left'; ctx.fillText(Math.abs(tl.value).toFixed(1), pS + 3, -dS * hS - 5);
  ctx.textAlign = 'right'; ctx.fillText(Math.abs(tl.value2).toFixed(1) + ' N/mm', pE - 3, -dE * hE - 5);
  ctx.restore();
}

// ===== DIAGRAM RENDERER =====
function fmtDiag(v) {
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toFixed(1) + 'k';
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(2);
  if (a < 1e-12) return '0';
  return v.toFixed(4);
}

function renderDiag(ctx, canvas, yD, xD, lc, fc, unit) {
  const dpr = window.devicePixelRatio || 1, w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  const { l, r } = bpr(canvas);
  const pT = 14, pB = h - 8, pH = pB - pT, zY = pT + pH / 2;
  // Subtle zero line only (no grid)
  ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(l, zY); ctx.lineTo(r, zY); ctx.stroke();
  if (!yD || !xD || xD.length < 2) return;
  const mA = Math.max(...Array.from(yD).map(Math.abs), 1e-15);
  const sc = (pH / 2) * 0.8 / mA;
  // Area fill (soft, opacity 0.1)
  ctx.beginPath(); ctx.moveTo(b2px(xD[0], canvas), zY);
  for (let i = 0; i < xD.length; i++) ctx.lineTo(b2px(xD[i], canvas), zY - yD[i] * sc);
  ctx.lineTo(b2px(xD[xD.length - 1], canvas), zY); ctx.closePath();
  ctx.fillStyle = fc; ctx.fill();
  // Curve line
  ctx.beginPath();
  for (let i = 0; i < xD.length; i++) { const px = b2px(xD[i], canvas), py = zY - yD[i] * sc; i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
  ctx.strokeStyle = lc; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.stroke();
  // Max annotation
  let mi = 0, mv = 0;
  for (let i = 0; i < yD.length; i++) if (Math.abs(yD[i]) > Math.abs(mv)) { mv = yD[i]; mi = i; }
  if (Math.abs(mv) > 1e-15) {
    const mpx = b2px(xD[mi], canvas), mpy = zY - yD[mi] * sc;
    ctx.fillStyle = lc; ctx.beginPath(); ctx.arc(mpx, mpy, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 9px ' + _f; ctx.textAlign = mpx > (l + r) / 2 ? 'right' : 'left';
    ctx.fillText(fmtDiag(mv) + ' ' + unit, mpx + (mpx > (l + r) / 2 ? -6 : 6), mpy - 5);
  }
  // Y-axis labels (2 decimal max)
  ctx.fillStyle = '#94a3b8'; ctx.font = '8px ' + _f; ctx.textAlign = 'right';
  for (const fr of [-1, 0, 1]) ctx.fillText(fmtDiag(fr * mA), l - 4, zY - fr * mA * sc + 3);
}

// =============================================================================
// INTERACTION
// =============================================================================
function hitTest(mx, my, c) {
  const by = bY(c);
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const o = state.objects[i];
    if (o.type === 'distributed' || o.type === 'trapezoid') {
      const pS = b2px(o.x, c), pE = b2px(o.x2, c), mY = by - 20;
      if (Math.abs(mx - pS) < HH && Math.abs(my - mY) < HH + 20) return { index: i, part: 'start' };
      if (Math.abs(mx - pE) < HH && Math.abs(my - mY) < HH + 20) return { index: i, part: 'end' };
      if (mx >= pS - 5 && mx <= pE + 5 && Math.abs(my - mY) < 30) return { index: i, part: 'body' };
    }
  }
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const o = state.objects[i];
    if (o.type === 'distributed' || o.type === 'trapezoid') continue;
    if (Math.abs(mx - b2px(o.x, c)) < HR && Math.abs(my - bY(c)) < HR + 40) return { index: i, part: 'body' };
  }
  return null;
}

beamCanvas.addEventListener('mousedown', (e) => {
  const rc = beamCanvas.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
  if (e.button === 2) { const h = hitTest(mx, my, beamCanvas); if (h) { state.objects.splice(h.index, 1); solveAndRender(); } return; }
  if (state.tool) {
    const bx = snap(px2b(mx, beamCanvas));
    if (state.tool === 'support') { state.objects.push({ type: 'support', x: bx, subtype: 'pinned', _id: newId() }); setTool(null); solveAndRender(); }
    else if (state.tool === 'load') { state.objects.push({ type: 'load', x: bx, value: 10000, angle: 0, _id: newId() }); setTool(null); solveAndRender(); }
    else if (state.tool === 'moment') { state.objects.push({ type: 'moment', x: bx, value: 1000000, _id: newId() }); setTool(null); solveAndRender(); }
    else if (state.tool === 'distributed' || state.tool === 'trapezoid') {
      if (state.distStart === null) { state.distStart = bx; renderAll(); }
      else {
        const x1 = Math.min(state.distStart, bx), x2 = Math.max(state.distStart, bx);
        if (x2 - x1 > 10) {
          if (state.tool === 'distributed') state.objects.push({ type: 'distributed', x: x1, x2, value: 5, _id: newId() });
          else state.objects.push({ type: 'trapezoid', x: x1, x2, value: 8, value2: 0, _id: newId() });
        }
        state.distStart = null; setTool(null); solveAndRender();
      }
    }
    return;
  }
  const h = hitTest(mx, my, beamCanvas);
  if (h) {
    const o = state.objects[h.index];
    if (h.part === 'body' && (o.type === 'distributed' || o.type === 'trapezoid'))
      dragging = { index: h.index, part: 'body', offsetX: px2b(mx, beamCanvas) - o.x, origLen: o.x2 - o.x };
    else if (h.part === 'start' || h.part === 'end') dragging = { index: h.index, part: h.part, offsetX: 0 };
    else dragging = { index: h.index, part: 'body', offsetX: mx - b2px(o.x, beamCanvas) };
    beamCanvas.style.cursor = 'grabbing';
  }
});

beamCanvas.addEventListener('mousemove', (e) => {
  const rc = beamCanvas.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
  if (dragging) {
    const o = state.objects[dragging.index]; if (!o) { dragging = null; return; }
    if (o.type === 'distributed' || o.type === 'trapezoid') {
      if (dragging.part === 'body') { let nx = snap(px2b(mx, beamCanvas) - dragging.offsetX); nx = Math.max(0, Math.min(state.beam.length - dragging.origLen, nx)); o.x = nx; o.x2 = nx + dragging.origLen; }
      else if (dragging.part === 'start') { o.x = Math.max(0, Math.min(o.x2 - 10, snap(px2b(mx, beamCanvas)))); }
      else { o.x2 = Math.max(o.x + 10, Math.min(state.beam.length, snap(px2b(mx, beamCanvas)))); }
    } else { o.x = snap(px2b(mx - dragging.offsetX, beamCanvas)); }
    solveAndRender(); return;
  }
  const h = hitTest(mx, my, beamCanvas);
  const nH = h ? h.index : -1, nP = h ? h.part : null;
  if (nH !== hovering || nP !== hoverPart) {
    hovering = nH; hoverPart = nP;
    beamCanvas.style.cursor = h ? ((h.part === 'start' || h.part === 'end') ? 'ew-resize' : 'grab') : (state.tool ? 'crosshair' : 'default');
    renderAll();
  }
});

beamCanvas.addEventListener('mouseup', () => { if (dragging) { dragging = null; beamCanvas.style.cursor = hovering >= 0 ? 'grab' : 'default'; } });
beamCanvas.addEventListener('mouseleave', () => { dragging = null; hovering = -1; hoverPart = null; beamCanvas.style.cursor = 'default'; renderAll(); });
beamCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

// ===== INLINE EDITING + TRAPEZOID MODAL =====
beamCanvas.addEventListener('dblclick', (e) => {
  const rc = beamCanvas.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
  const h = hitTest(mx, my, beamCanvas); if (!h) return;
  const o = state.objects[h.index];
  // Trapezoid: open dedicated modal
  if (o.type === 'trapezoid') { openTrapModal(h.index); return; }
  // Distributed: open dedicated modal too (reuse trap modal)
  if (o.type === 'distributed') { openDistModal(h.index); return; }
  if (o.type !== 'load' && o.type !== 'moment') return;
  const ov = document.getElementById('inline-edit'), inp = document.getElementById('inline-input');
  ov.style.display = 'block'; ov.style.left = (e.clientX - 42) + 'px'; ov.style.top = (e.clientY - 14) + 'px';
  inp.value = o.value; inp.focus(); inp.select();
  const commit = () => { const v = parseFloat(inp.value); if (!isNaN(v)) { o.value = v; solveAndRender(); } ov.style.display = 'none'; inp.removeEventListener('blur', commit); inp.removeEventListener('keydown', onK); };
  const onK = (ev) => { if (ev.key === 'Enter') commit(); if (ev.key === 'Escape') { ov.style.display = 'none'; inp.removeEventListener('blur', commit); inp.removeEventListener('keydown', onK); } };
  inp.addEventListener('blur', commit); inp.addEventListener('keydown', onK);
});

// ===== TRAPEZOID EDIT MODAL =====
function openTrapModal(idx) {
  const o = state.objects[idx]; if (!o) return;
  state.editingTrapIdx = idx;
  document.getElementById('trap-modal-title').textContent = 'Edit Trapezoid Load';
  document.getElementById('trap-q2-row').style.display = 'flex';
  document.getElementById('trap-q1').value = o.value;
  document.getElementById('trap-q2').value = o.value2;
  document.getElementById('trap-xs').value = o.x.toFixed(0);
  document.getElementById('trap-xe').value = o.x2.toFixed(0);
  document.getElementById('trap-modal').classList.add('show');
}
function openDistModal(idx) {
  const o = state.objects[idx]; if (!o) return;
  state.editingTrapIdx = idx;
  document.getElementById('trap-modal-title').textContent = 'Edit Distributed Load (UDL)';
  document.getElementById('trap-q2-row').style.display = 'none';
  document.getElementById('trap-q1').value = o.value;
  document.getElementById('trap-q2').value = o.value;
  document.getElementById('trap-xs').value = o.x.toFixed(0);
  document.getElementById('trap-xe').value = o.x2.toFixed(0);
  document.getElementById('trap-modal').classList.add('show');
}
function closeTrapModal() {
  document.getElementById('trap-modal').classList.remove('show');
  state.editingTrapIdx = -1;
}
function saveTrapModal() {
  const idx = state.editingTrapIdx;
  if (idx < 0 || !state.objects[idx]) { closeTrapModal(); return; }
  const o = state.objects[idx];
  const q1 = parseFloat(document.getElementById('trap-q1').value);
  const q2 = parseFloat(document.getElementById('trap-q2').value);
  const xs = parseFloat(document.getElementById('trap-xs').value);
  const xe = parseFloat(document.getElementById('trap-xe').value);
  if (!isNaN(q1)) o.value = q1;
  if (!isNaN(q2) && o.type === 'trapezoid') o.value2 = q2;
  if (!isNaN(q2) && o.type === 'distributed') o.value = q2;
  if (!isNaN(xs)) o.x = Math.max(0, xs);
  if (!isNaN(xe)) o.x2 = Math.min(state.beam.length, xe);
  closeTrapModal();
  solveAndRender();
}

// =============================================================================
// UI HELPERS
// =============================================================================
function setTool(t) {
  state.tool = t; state.distStart = null;
  ['btn-add-support', 'btn-add-load', 'btn-add-moment', 'btn-add-dist', 'btn-add-trap'].forEach(id => {
    const el = document.getElementById(id); if (el) el.classList.remove('active');
  });
  const m = { support: 'btn-add-support', load: 'btn-add-load', moment: 'btn-add-moment', distributed: 'btn-add-dist', trapezoid: 'btn-add-trap' };
  if (t && m[t]) document.getElementById(m[t]).classList.add('active');
  beamCanvas.style.cursor = t ? 'crosshair' : 'default';
  renderAll();
}

function solveAndRender() { solveBeam(); renderAll(); }

function updateStats() {
  const r = state.result;
  document.getElementById('stat-v').textContent = r ? (r.maxV / 1000).toFixed(2) + ' kN' : '—';
  document.getElementById('stat-m').textContent = r ? (r.maxM / 1e6).toFixed(2) + ' kNm' : '—';
  document.getElementById('stat-n').textContent = r ? (r.maxN / 1000).toFixed(2) + ' kN' : '—';
  document.getElementById('stat-w').textContent = r ? r.maxW.toFixed(2) + ' mm' : '—';
}

function updateObjLists() {
  const sl = document.getElementById('support-list'), ll = document.getElementById('load-list');
  sl.innerHTML = ''; ll.innerHTML = '';
  let sc = 0, lc = 0;
  state.objects.forEach((o, i) => {
    const d = document.createElement('div'); d.className = 'oi';
    if (o.type === 'support') {
      sc++;
      d.innerHTML = '<div class="oi-icon" style="background:var(--support-c)"></div>' +
        '<span class="oi-text">' + o.subtype + ' @ ' + o.x.toFixed(0) + '</span>' +
        '<select style="width:55px" onchange="chSub(' + i + ',this.value)"><option value="pinned"' + (o.subtype === 'pinned' ? ' selected' : '') + '>Pin</option><option value="roller"' + (o.subtype === 'roller' ? ' selected' : '') + '>Roller</option><option value="fixed"' + (o.subtype === 'fixed' ? ' selected' : '') + '>Fixed</option></select>' +
        '<input type="number" value="' + o.x.toFixed(0) + '" style="width:48px" onchange="chPos(' + i + ',this.value)">' +
        '<span class="oi-del" onclick="rmObj(' + i + ')">×</span>';
      sl.appendChild(d);
    } else if (o.type === 'load') {
      lc++;
      d.innerHTML = '<div class="oi-icon" style="background:var(--load-c)"></div>' +
        '<span class="oi-text">' + fmtF(Math.abs(o.value)) + '</span>' +
        '<input type="number" value="' + o.value + '" style="width:52px" onchange="chLdV(' + i + ',this.value)">' +
        '<span class="oi-del" onclick="rmObj(' + i + ')">×</span>' +
        '<div class="oi-r2"><label>Pos</label><input type="number" value="' + o.x.toFixed(0) + '" style="width:44px" onchange="chPos(' + i + ',this.value)"><label>Ang</label><input type="number" value="' + (o.angle || 0) + '" step="5" style="width:38px" onchange="chAng(' + i + ',this.value)"><span style="font-size:9px;color:var(--text-muted)">°</span></div>';
      ll.appendChild(d);
    } else if (o.type === 'moment') {
      lc++;
      d.innerHTML = '<div class="oi-icon" style="background:var(--moment-lc)"></div>' +
        '<span class="oi-text">Moment</span>' +
        '<input type="number" value="' + o.value + '" style="width:60px" onchange="chMmV(' + i + ',this.value)">' +
        '<span class="oi-del" onclick="rmObj(' + i + ')">×</span>' +
        '<div class="oi-r2"><label>Pos</label><input type="number" value="' + o.x.toFixed(0) + '" style="width:48px" onchange="chPos(' + i + ',this.value)"></div>';
      ll.appendChild(d);
    } else if (o.type === 'distributed') {
      lc++;
      d.innerHTML = '<div class="oi-icon" style="background:var(--load-c)"></div>' +
        '<span class="oi-text">UDL ' + o.value.toFixed(1) + '</span>' +
        '<input type="number" value="' + o.value + '" step="0.5" style="width:48px" onchange="chDV(' + i + ',this.value)">' +
        '<span class="oi-del" onclick="rmObj(' + i + ')">×</span>' +
        '<div class="oi-r2"><label>S</label><input type="number" value="' + o.x.toFixed(0) + '" style="width:44px" onchange="chDS(' + i + ',this.value)"><label>E</label><input type="number" value="' + o.x2.toFixed(0) + '" style="width:44px" onchange="chDE(' + i + ',this.value)"></div>';
      ll.appendChild(d);
    } else if (o.type === 'trapezoid') {
      lc++;
      d.innerHTML = '<div class="oi-icon" style="background:var(--load-c)"></div>' +
        '<span class="oi-text">Trap ' + o.value.toFixed(1) + '→' + o.value2.toFixed(1) + '</span>' +
        '<span class="oi-del" onclick="rmObj(' + i + ')">×</span>' +
        '<div class="oi-r2"><label>q₁</label><input type="number" value="' + o.value + '" step="0.5" style="width:40px" onchange="chTQ1(' + i + ',this.value)"><label>q₂</label><input type="number" value="' + o.value2 + '" step="0.5" style="width:40px" onchange="chTQ2(' + i + ',this.value)"></div>' +
        '<div class="oi-r2"><label>S</label><input type="number" value="' + o.x.toFixed(0) + '" style="width:40px" onchange="chDS(' + i + ',this.value)"><label>E</label><input type="number" value="' + o.x2.toFixed(0) + '" style="width:40px" onchange="chDE(' + i + ',this.value)"></div>';
      ll.appendChild(d);
    }
  });
  document.getElementById('sup-count').textContent = sc;
  document.getElementById('load-count').textContent = lc;
}

function rmObj(i) { state.objects.splice(i, 1); solveAndRender(); }
function chSub(i, v) { state.objects[i].subtype = v; solveAndRender(); }
function chLdV(i, v) { state.objects[i].value = parseFloat(v) || 0; solveAndRender(); }
function chMmV(i, v) { state.objects[i].value = parseFloat(v) || 0; solveAndRender(); }
function chAng(i, v) { state.objects[i].angle = parseFloat(v) || 0; solveAndRender(); }
function chDV(i, v) { state.objects[i].value = parseFloat(v) || 0; solveAndRender(); }
function chDS(i, v) { state.objects[i].x = Math.max(0, parseFloat(v) || 0); solveAndRender(); }
function chDE(i, v) { state.objects[i].x2 = Math.min(state.beam.length, parseFloat(v) || 0); solveAndRender(); }
function chTQ1(i, v) { state.objects[i].value = parseFloat(v) || 0; solveAndRender(); }
function chTQ2(i, v) { state.objects[i].value2 = parseFloat(v) || 0; solveAndRender(); }
function chPos(i, v) { state.objects[i].x = Math.max(0, Math.min(state.beam.length, parseFloat(v) || 0)); solveAndRender(); }

function applyBeamProps() {
  const sidebar = document.getElementById('sidebar');
  const scrollPos = sidebar ? sidebar.scrollTop : 0;
  state.beam.length = parseFloat(document.getElementById('inp-length').value) || 5000;
  state.beam.E = parseFloat(document.getElementById('inp-E').value) || 210000;
  state.beam.I = parseFloat(document.getElementById('inp-I').value) || 83560000;
  state.beam.A = parseFloat(document.getElementById('inp-A').value) || 2850;
  solveAndRender();
  if (sidebar) requestAnimationFrame(() => { sidebar.scrollTop = scrollPos; });
}

function clearAll() {
  state.objects = []; state.result = null; setTool(null); solveAndRender();
  document.getElementById('status-text').textContent = 'Cleared';
}

// =============================================================================
// PAYWALL LOGIC
// =============================================================================
function showPricing() { document.getElementById('pricing-modal').classList.add('show'); }
function closePricing() { document.getElementById('pricing-modal').classList.remove('show'); }
function selectPlan(plan) {
  closePricing();
  alert('Thank you for choosing ' + (plan === 'lifetime' ? 'Lifetime ($149)' : 'Monthly ($19/mo)') + '!\n\nPayment integration coming soon. For now, enjoy Pro features!');
  state.isPro = true;
  document.getElementById('status-text').textContent = 'PRO activated!';
}
function requirePro(featureName) {
  if (state.isPro) return true;
  showPricing();
  return false;
}

// =============================================================================
// PDF EXPORT — Professional Beam Analysis Report (html2pdf.js)
// =============================================================================
function canvasToDataURL(canvas) {
  try { return canvas.toDataURL('image/png'); } catch(e) { return null; }
}

function exportPDF() {
  if (!requirePro('PDF Export')) return;
  document.getElementById('status-text').textContent = 'Generating PDF report...';

  const proj = document.getElementById('proj-name').value || 'Untitled Project';
  const comp = document.getElementById('proj-company').value || '—';
  const ref = document.getElementById('proj-ref').value || '—';
  const date = document.getElementById('proj-date').value || new Date().toISOString().split('T')[0];
  const r = state.result;
  const now = new Date().toLocaleString();

  // Capture canvases as high-res images
  const beamImg = canvasToDataURL(beamCanvas);
  const shearImg = canvasToDataURL(shearCanvas);
  const momentImg = canvasToDataURL(momentCanvas);
  const normalImg = canvasToDataURL(normalCanvas);
  const deflImg = canvasToDataURL(deflCanvas);

  // Build loads/supports table rows
  let inputRows = '';
  let rowIdx = 0;
  state.objects.forEach(o => {
    const bg = rowIdx % 2 === 0 ? '#ffffff' : '#f8fafc';
    rowIdx++;
    const td = 'style="padding:6px 10px;border-bottom:1px solid #e2e8f0;font-size:10px;background:' + bg + '"';
    if (o.type === 'support') inputRows += '<tr><td ' + td + '>Support</td><td ' + td + '>' + o.subtype.charAt(0).toUpperCase() + o.subtype.slice(1) + '</td><td ' + td + '>' + o.x.toFixed(0) + ' mm</td><td ' + td + '>—</td></tr>';
    else if (o.type === 'load') inputRows += '<tr><td ' + td + '>Point Load</td><td ' + td + '>' + fmtF(Math.abs(o.value)) + '</td><td ' + td + '>' + o.x.toFixed(0) + ' mm</td><td ' + td + '>' + (o.angle || 0) + '°</td></tr>';
    else if (o.type === 'moment') inputRows += '<tr><td ' + td + '>Moment</td><td ' + td + '>' + fmtV(Math.abs(o.value)) + ' N·mm</td><td ' + td + '>' + o.x.toFixed(0) + ' mm</td><td ' + td + '>—</td></tr>';
    else if (o.type === 'distributed') inputRows += '<tr><td ' + td + '>UDL</td><td ' + td + '>' + o.value.toFixed(2) + ' N/mm</td><td ' + td + '>' + o.x.toFixed(0) + ' – ' + o.x2.toFixed(0) + ' mm</td><td ' + td + '>—</td></tr>';
    else if (o.type === 'trapezoid') inputRows += '<tr><td ' + td + '>Trapezoid</td><td ' + td + '>' + o.value.toFixed(2) + ' → ' + o.value2.toFixed(2) + ' N/mm</td><td ' + td + '>' + o.x.toFixed(0) + ' – ' + o.x2.toFixed(0) + ' mm</td><td ' + td + '>—</td></tr>';
  });

  // Reactions table
  let reactRows = '';
  if (r && r.reactions) {
    state.objects.filter(o => o.type === 'support').forEach(s => {
      const rx = r.reactions[s._id];
      if (rx) {
        reactRows += '<tr><td style="padding:5px 10px;border-bottom:1px solid #e2e8f0;font-size:10px">' + s.subtype + ' @ ' + s.x.toFixed(0) + ' mm</td>';
        reactRows += '<td style="padding:5px 10px;border-bottom:1px solid #e2e8f0;font-size:10px;text-align:right">' + fmtF(Math.abs(rx.v)) + '</td>';
        reactRows += '<td style="padding:5px 10px;border-bottom:1px solid #e2e8f0;font-size:10px;text-align:right">' + fmtF(Math.abs(rx.h)) + '</td></tr>';
      }
    });
  }

  const el = document.createElement('div');
  el.style.cssText = 'position:absolute;left:-9999px;top:0;width:780px;font-family:Inter,Segoe UI,sans-serif;color:#1e293b;background:#fff;';

  const imgStyle = 'width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin:4px 0;';
  const thStyle = 'padding:7px 10px;text-align:left;font-size:9px;font-weight:700;color:#fff;text-transform:uppercase;letter-spacing:.4px';
  const secH = 'font-size:13px;font-weight:700;color:#1e40af;margin:18px 0 8px;padding-bottom:4px;border-bottom:2px solid #eff6ff';

  el.innerHTML = `
    <div style="padding:28px 32px">
      <!-- HEADER -->
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;padding-bottom:14px;border-bottom:3px solid #1e40af">
        <div>
          <div style="font-size:24px;font-weight:800;color:#1e40af;letter-spacing:-0.5px">⬡ BeamSolve</div>
          <div style="font-size:16px;font-weight:700;color:#334155;margin-top:2px">Beam Analysis Report</div>
        </div>
        <div style="text-align:right;font-size:10px;color:#64748b;line-height:1.7">
          <div><b style="color:#334155">Date:</b> ${date}</div>
          <div><b style="color:#334155">Reference:</b> ${ref}</div>
          <div><b style="color:#334155">Generated:</b> ${now}</div>
        </div>
      </div>

      <!-- PROJECT INFO -->
      <table style="width:100%;border-collapse:collapse;margin-bottom:14px">
        <tr>
          <td style="padding:6px 10px;background:#f1f5f9;font-weight:600;font-size:10px;width:100px;border:1px solid #e2e8f0">Project</td>
          <td style="padding:6px 10px;font-size:11px;border:1px solid #e2e8f0">${proj}</td>
          <td style="padding:6px 10px;background:#f1f5f9;font-weight:600;font-size:10px;width:100px;border:1px solid #e2e8f0">Company</td>
          <td style="padding:6px 10px;font-size:11px;border:1px solid #e2e8f0">${comp}</td>
        </tr>
      </table>

      <!-- BEAM PROPERTIES -->
      <div style="${secH}">1. Beam Properties</div>
      <table style="width:100%;border-collapse:collapse;margin-bottom:10px">
        <tr>
          <td style="padding:5px 10px;background:#f1f5f9;font-weight:600;font-size:10px;border:1px solid #e2e8f0;width:25%">Length</td>
          <td style="padding:5px 10px;font-size:11px;border:1px solid #e2e8f0;width:25%">${state.beam.length.toLocaleString()} mm</td>
          <td style="padding:5px 10px;background:#f1f5f9;font-weight:600;font-size:10px;border:1px solid #e2e8f0;width:25%">E-modulus</td>
          <td style="padding:5px 10px;font-size:11px;border:1px solid #e2e8f0;width:25%">${state.beam.E.toLocaleString()} MPa</td>
        </tr>
        <tr>
          <td style="padding:5px 10px;background:#f1f5f9;font-weight:600;font-size:10px;border:1px solid #e2e8f0">Moment of Inertia (I)</td>
          <td style="padding:5px 10px;font-size:11px;border:1px solid #e2e8f0">${fmtV(state.beam.I)} mm⁴</td>
          <td style="padding:5px 10px;background:#f1f5f9;font-weight:600;font-size:10px;border:1px solid #e2e8f0">Cross-section Area (A)</td>
          <td style="padding:5px 10px;font-size:11px;border:1px solid #e2e8f0">${state.beam.A.toLocaleString()} mm²</td>
        </tr>
      </table>

      <!-- LOADS & SUPPORTS INPUT TABLE -->
      <div style="${secH}">2. Loads & Supports</div>
      <table style="width:100%;border-collapse:collapse;margin-bottom:10px">
        <tr style="background:#1e40af"><th style="${thStyle}">Type</th><th style="${thStyle}">Value</th><th style="${thStyle}">Position</th><th style="${thStyle}">Angle / Extra</th></tr>
        ${inputRows}
      </table>

      <!-- BEAM MODEL DIAGRAM -->
      <div style="${secH}">3. Beam Model</div>
      ${beamImg ? '<img src="' + beamImg + '" style="' + imgStyle + '">' : '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:11px;border:1px solid #e2e8f0;border-radius:6px">Beam diagram not available</div>'}

      <!-- RESULTS SUMMARY -->
      <div style="${secH}">4. Maximum Results</div>
      <table style="width:100%;border-collapse:collapse;margin-bottom:10px">
        <tr style="background:#1e40af"><th style="${thStyle}">Parameter</th><th style="${thStyle};text-align:right">Value</th><th style="${thStyle};text-align:right">Unit</th></tr>
        <tr><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;font-weight:600">Maximum Shear Force (V<sub>max</sub>)</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:700;color:#1e40af;text-align:right">${r ? fmtV(r.maxV) : '—'}</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;text-align:right;color:#64748b">N</td></tr>
        <tr style="background:#f8fafc"><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;font-weight:600">Maximum Bending Moment (M<sub>max</sub>)</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:700;color:#b91c1c;text-align:right">${r ? fmtV(r.maxM) : '—'}</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;text-align:right;color:#64748b">N·mm</td></tr>
        <tr><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;font-weight:600">Maximum Normal Force (N<sub>max</sub>)</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:700;color:#6d28d9;text-align:right">${r ? fmtV(r.maxN) : '—'}</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;text-align:right;color:#64748b">N</td></tr>
        <tr style="background:#f8fafc"><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;font-weight:600">Maximum Deflection (w<sub>max</sub>)</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:700;color:#b45309;text-align:right">${r ? r.maxW.toFixed(4) : '—'}</td><td style="padding:7px 10px;border-bottom:1px solid #e2e8f0;font-size:11px;text-align:right;color:#64748b">mm</td></tr>
      </table>

      <!-- REACTIONS -->
      ${reactRows ? '<div style="' + secH + '">5. Support Reactions</div><table style="width:100%;border-collapse:collapse;margin-bottom:10px"><tr style="background:#1e40af"><th style="' + thStyle + '">Support</th><th style="' + thStyle + ';text-align:right">Vertical (R<sub>v</sub>)</th><th style="' + thStyle + ';text-align:right">Horizontal (R<sub>h</sub>)</th></tr>' + reactRows + '</table>' : ''}

      <!-- ANALYSIS DIAGRAMS -->
      <div style="${secH}">${reactRows ? '6' : '5'}. Analysis Diagrams</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
        <div><div style="font-size:9px;font-weight:700;color:#2563eb;text-transform:uppercase;margin-bottom:2px">Shear Force (V)</div>${shearImg ? '<img src="' + shearImg + '" style="' + imgStyle + '">' : ''}</div>
        <div><div style="font-size:9px;font-weight:700;color:#dc2626;text-transform:uppercase;margin-bottom:2px">Bending Moment (M)</div>${momentImg ? '<img src="' + momentImg + '" style="' + imgStyle + '">' : ''}</div>
        <div><div style="font-size:9px;font-weight:700;color:#7c3aed;text-transform:uppercase;margin-bottom:2px">Normal Force (N)</div>${normalImg ? '<img src="' + normalImg + '" style="' + imgStyle + '">' : ''}</div>
        <div><div style="font-size:9px;font-weight:700;color:#d97706;text-transform:uppercase;margin-bottom:2px">Deflection (w)</div>${deflImg ? '<img src="' + deflImg + '" style="' + imgStyle + '">' : ''}</div>
      </div>

      <!-- FOOTER -->
      <div style="margin-top:24px;padding-top:10px;border-top:2px solid #1e40af">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div style="font-size:9px;color:#94a3b8">Generated by <b style="color:#1e40af">BeamSolve</b> · beamsolve.app · Direct Stiffness Method (FEM) · 80 Elements</div>
          <div style="font-size:9px;color:#94a3b8">${now}</div>
        </div>
        <div style="margin-top:8px;padding:8px 10px;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;font-size:8px;color:#92400e;line-height:1.5">
          <b>Disclaimer:</b> Calculations are based on Euler-Bernoulli beam theory using the Direct Stiffness Method (FEM). Results are for preliminary design purposes only. The user is responsible for final verification and compliance with applicable codes and standards. BeamSolve assumes no liability for design decisions based on these results.
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(el);
  const opt = {
    margin: [8, 8, 12, 8],
    filename: (proj.replace(/\s+/g, '_') || 'BeamSolve') + '_Report.pdf',
    image: { type: 'png', quality: 1 },
    html2canvas: { scale: 2, useCORS: true, logging: false },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
  };
  html2pdf().set(opt).from(el).save().then(() => {
    document.body.removeChild(el);
    document.getElementById('status-text').textContent = 'PDF report exported!';
  }).catch((err) => {
    console.error('PDF export error:', err);
    document.body.removeChild(el);
    document.getElementById('status-text').textContent = 'PDF export failed';
  });
}

// =============================================================================
// EXCEL EXPORT — 2-Sheet Structured Report (SheetJS)
// =============================================================================
function exportExcel() {
  if (!requirePro('Excel Export')) return;
  if (typeof XLSX === 'undefined') { alert('SheetJS library not loaded. Please check your internet connection.'); return; }
  const wb = XLSX.utils.book_new();
  const proj = document.getElementById('proj-name').value || 'Untitled';
  const comp = document.getElementById('proj-company').value || '';
  const date = document.getElementById('proj-date').value || '';
  const ref = document.getElementById('proj-ref').value || '';

  // ===== SHEET 1: Summary (beam props + all loads & supports) =====
  const s1 = [
    ['BEAMSOLVE — BEAM ANALYSIS REPORT'],
    [],
    ['PROJECT INFORMATION'],
    ['Project Name', proj],
    ['Company', comp],
    ['Date', date],
    ['Reference', ref],
    [],
    ['BEAM PROPERTIES'],
    ['Property', 'Value', 'Unit'],
    ['Length', state.beam.length, 'mm'],
    ['E-modulus', state.beam.E, 'MPa'],
    ['Moment of Inertia (I)', state.beam.I, 'mm⁴'],
    ['Cross-section Area (A)', state.beam.A, 'mm²'],
    [],
    ['SUPPORTS'],
    ['#', 'Type', 'Position (mm)', 'Constraints'],
  ];
  let si = 1;
  state.objects.filter(o => o.type === 'support').forEach(o => {
    const con = o.subtype === 'fixed' ? 'u, w, θ' : o.subtype === 'pinned' ? 'u, w' : 'w';
    s1.push([si++, o.subtype.charAt(0).toUpperCase() + o.subtype.slice(1), o.x, con]);
  });
  s1.push([], ['LOADS'], ['#', 'Type', 'Value', 'Position (mm)', 'Extra']);
  let li = 1;
  state.objects.forEach(o => {
    if (o.type === 'load') s1.push([li++, 'Point Load', o.value + ' N', o.x.toFixed(0), 'angle=' + (o.angle || 0) + '°']);
    else if (o.type === 'moment') s1.push([li++, 'Moment', o.value + ' N·mm', o.x.toFixed(0), '']);
    else if (o.type === 'distributed') s1.push([li++, 'UDL', o.value.toFixed(2) + ' N/mm', o.x.toFixed(0) + ' – ' + o.x2.toFixed(0), 'uniform']);
    else if (o.type === 'trapezoid') s1.push([li++, 'Trapezoid', o.value.toFixed(2) + ' → ' + o.value2.toFixed(2) + ' N/mm', o.x.toFixed(0) + ' – ' + o.x2.toFixed(0), 'linear varying']);
  });
  s1.push([], ['MAXIMUM RESULTS'], ['Parameter', 'Value', 'Unit']);
  s1.push(['Max Shear Force (Vmax)', state.result ? +(state.result.maxV / 1000).toFixed(3) : '', 'kN']);
  s1.push(['Max Bending Moment (Mmax)', state.result ? +(state.result.maxM / 1e6).toFixed(3) : '', 'kNm']);
  s1.push(['Max Deflection (wmax)', state.result ? +state.result.maxW.toFixed(4) : '', 'mm']);
  const ws1 = XLSX.utils.aoa_to_sheet(s1);
  ws1['!cols'] = [{ wch: 30 }, { wch: 22 }, { wch: 14 }, { wch: 14 }, { wch: 16 }];
  if (ws1['A1']) ws1['A1'].s = { font: { bold: true, sz: 14 } };
  XLSX.utils.book_append_sheet(wb, ws1, 'Summary');

  // ===== SHEET 2: Analysis (100-point high-density sampling) =====
  const s2 = [['ANALYSIS DATA — 100-POINT SAMPLING'], []];
  s2.push(['Position (mm)', 'Shear (kN)', 'Moment (kNm)', 'Deflection (mm)']);
  if (state.result && state.result.x.length > 0) {
    const n = state.result.x.length;
    const sampleCount = 100;
    for (let s = 0; s < sampleCount; s++) {
      const idx = Math.min(Math.round(s * (n - 1) / (sampleCount - 1)), n - 1);
      s2.push([
        +state.result.x[idx].toFixed(1),
        +(state.result.V[idx] / 1000).toFixed(4),
        +(state.result.M[idx] / 1e6).toFixed(6),
        +state.result.w[idx].toFixed(6),
      ]);
    }
  }
  const ws2 = XLSX.utils.aoa_to_sheet(s2);
  ws2['!cols'] = [{ wch: 16 }, { wch: 16 }, { wch: 18 }, { wch: 18 }];
  if (ws2['A1']) ws2['A1'].s = { font: { bold: true, sz: 12 } };
  ['A3', 'B3', 'C3', 'D3'].forEach(c => { if (ws2[c]) ws2[c].s = { font: { bold: true } }; });
  XLSX.utils.book_append_sheet(wb, ws2, 'Analysis');

  XLSX.writeFile(wb, (proj.replace(/\s+/g, '_') || 'BeamSolve') + '_Report.xlsx');
  document.getElementById('status-text').textContent = 'Excel exported (2 sheets, 100-pt)!';
}

function copyShareLink() {
  const inp = document.getElementById('share-link');
  inp.select(); navigator.clipboard.writeText(inp.value).then(() => {
    document.getElementById('status-text').textContent = 'Link copied!';
  });
}

// ===== PRESETS =====
function loadPreset(name) {
  state.objects = []; state.result = null; setTool(null);
  const B = { length: 5000, E: 210000, I: 83560000, A: 2850 };
  if (name === 'simply_supported') {
    state.beam = { ...B };
    state.objects = [
      { type: 'support', x: 0, subtype: 'pinned', _id: newId() },
      { type: 'support', x: 5000, subtype: 'roller', _id: newId() },
      { type: 'load', x: 2500, value: 10000, angle: 0, _id: newId() },
    ];
  } else if (name === 'cantilever') {
    state.beam = { ...B, length: 3000 };
    state.objects = [
      { type: 'support', x: 0, subtype: 'fixed', _id: newId() },
      { type: 'load', x: 3000, value: 5000, angle: 0, _id: newId() },
    ];
  } else if (name === 'continuous') {
    state.beam = { ...B, length: 9000 };
    state.objects = [
      { type: 'support', x: 0, subtype: 'pinned', _id: newId() },
      { type: 'support', x: 3000, subtype: 'roller', _id: newId() },
      { type: 'support', x: 6000, subtype: 'roller', _id: newId() },
      { type: 'support', x: 9000, subtype: 'roller', _id: newId() },
      { type: 'trapezoid', x: 0, x2: 9000, value: 8, value2: 2, _id: newId() },
    ];
  } else if (name === 'angled') {
    state.beam = { ...B, length: 6000 };
    state.objects = [
      { type: 'support', x: 0, subtype: 'pinned', _id: newId() },
      { type: 'support', x: 6000, subtype: 'roller', _id: newId() },
      { type: 'load', x: 2000, value: 8000, angle: 30, _id: newId() },
      { type: 'load', x: 4000, value: 6000, angle: -45, _id: newId() },
      { type: 'moment', x: 3000, value: 2000000, _id: newId() },
    ];
  }
  document.getElementById('inp-length').value = state.beam.length;
  document.getElementById('inp-E').value = state.beam.E;
  document.getElementById('inp-I').value = state.beam.I;
  document.getElementById('inp-A').value = state.beam.A;
  solveAndRender();
  document.getElementById('status-text').textContent = 'Loaded: ' + name.replace(/_/g, ' ');
}

// ===== INIT =====
function init() {
  resizeCanvases();
  initProfiles();
  loadPreset('simply_supported');
}

window.addEventListener('resize', () => { resizeCanvases(); renderAll(); });
['inp-length', 'inp-E', 'inp-I', 'inp-A'].forEach(id => {
  document.getElementById(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') applyBeamProps(); });
});

init();

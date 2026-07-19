(() => {
  const shell = document.querySelector('.review-shell');
  if (!shell) return;
  const image = document.querySelector('#review-image');
  const imagePanel = document.querySelector('#image-panel');
  const imageLabel = document.querySelector('#image-label');
  const rasterMode = document.querySelector('#raster-mode');
  const rasterModes = document.querySelectorAll('#raster-mode [data-mode]');
  const ansiStage = document.querySelector('#ansi-stage');
  const ansiView = document.querySelector('#ansi-view');
  const ansiViewport = document.querySelector('#ansi-viewport');
  const ansiTerminal = document.querySelector('#ansi-terminal');
  const ansiLoading = document.querySelector('#ansi-loading');
  const ansiSlider = document.querySelector('#ansi-width');
  const ansiDimensions = document.querySelector('#ansi-dimensions');
  const ansiSource = document.querySelector('#ansi-source');
  const ansiPlay = document.querySelector('#ansi-play');
  const ansiModes = document.querySelectorAll('#ansi-mode [data-mode]');
  const rejectPanel = document.querySelector('#reject-panel');
  const causeStage = document.querySelector('#cause-stage');
  const commitReject = document.querySelector('#commit-reject');
  const status = document.querySelector('#save-status');
  let selectedIssue = null;
  let ansiRequest = null;
  let scrubTimer = null;
  let playing = false;
  let ansiLevel = null;
  let ansiCachePyramid = ansiStage?.dataset.pyramid || null;
  const ansiLevelCache = new Map();
  let displayMode = 'fit';
  const cellWidth = 8;
  const cellHeight = 16;
  const ansiFont = '"ANSI Symbols", "DejaVu Sans Mono", monospace';
  const sextants = [...' 🬀🬁🬂🬃🬄🬅🬆🬇🬈🬉🬊🬋🬌🬍🬎🬏🬐🬑🬒🬓▌🬔🬕🬖🬗🬘🬙🬚🬛🬜🬝🬞🬟🬠🬡🬢🬣🬤🬥🬦🬧▐🬨🬩🬪🬫🬬🬭🬮🬯🬰🬱🬲🬳🬴🬵🬶🬷🬸🬹🬺🬻█'];
  const wedgePoints = {
    ul: [0, 0], ur: [1, 0], ll: [0, 1], lr: [1, 1],
    uml: [0, 1 / 3], lml: [0, 2 / 3], umr: [1, 1 / 3], lmr: [1, 2 / 3],
    uc: [0.5, 0], lc: [0.5, 1],
  };
  const wedgeSpecs = [
    ['ll','lml','lc'], ['ll','lml','lr'], ['ll','uml','lc'], ['ll','uml','lr'], ['ll','ul','lc'],
    ['lr','uml','uc'], ['lr','uml','ur'], ['lr','lml','uc'], ['lr','lml','ur'], ['lr','ll','uc'],
    ['lr','lml','umr'], ['lr','lc','lmr'], ['lr','ll','lmr'], ['lr','lc','umr'], ['lr','ll','umr'],
    ['lr','lc','ur'], ['ll','uc','umr'], ['ll','ul','umr'], ['ll','uc','lmr'], ['ll','ul','lmr'],
    ['ll','uc','lr'], ['ll','uml','lmr'], ['ur','lml','lc'], ['ur','lml','lr'], ['ur','uml','lc'],
    ['ur','uml','lr'], ['ur','ul','lc'], ['ul','uml','uc'], ['ul','uml','ur'], ['ul','lml','uc'],
    ['ul','lml','ur'], ['ul','ll','uc'], ['ul','lml','umr'], ['ul','lc','lmr'], ['ul','ll','lmr'],
    ['ul','lc','umr'], ['ul','ll','umr'], ['ul','lc','ur'], ['ur','uc','umr'], ['ur','ul','umr'],
    ['ur','uc','lmr'], ['ur','ul','lmr'], ['ur','uc','lr'], ['ur','uml','lmr'],
  ];

  function stopPlayback() {
    playing = false;
    if (ansiPlay) ansiPlay.firstChild.textContent = '▶ ';
  }

  function sizeTerminal() {
    if (!ansiLevel || !ansiTerminal || !ansiViewport) return;
    const naturalWidth = ansiLevel.width * cellWidth;
    const naturalHeight = ansiLevel.rows * cellHeight;
    let scale = 1;
    if (displayMode === 'fit') {
      scale = Math.min(
        Math.max(1, ansiViewport.clientWidth - 28) / naturalWidth,
        Math.max(1, ansiViewport.clientHeight - 28) / naturalHeight,
      );
    }
    ansiTerminal.style.width = `${naturalWidth * scale}px`;
    ansiTerminal.style.height = `${naturalHeight * scale}px`;
  }

  function setDisplayMode(mode) {
    if (!ansiView) return;
    displayMode = mode === 'actual' ? 'actual' : 'fit';
    ansiView.classList.toggle('fit-mode', displayMode === 'fit');
    ansiView.classList.toggle('actual-mode', displayMode === 'actual');
    ansiModes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === displayMode)));
    localStorage.setItem('ansi-review-display-mode', displayMode);
    requestAnimationFrame(sizeTerminal);
  }

  function setRasterMode(mode) {
    const normalized = mode !== 'native';
    imagePanel?.classList.toggle('normalized-raster', normalized);
    rasterModes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === (normalized ? 'normalized' : 'native'))));
    localStorage.setItem('ansi-review-raster-mode', normalized ? 'normalized' : 'native');
  }

  function paletteColour(palette, index, fallback = '#000000') {
    if (index === null || index === undefined) return fallback;
    const colour = palette[index];
    return `rgb(${colour[0]} ${colour[1]} ${colour[2]})`;
  }

  function fillCellRect(context, x, y, x0, y0, x1, y1) {
    const left = Math.floor(x0 * cellWidth);
    const top = Math.floor(y0 * cellHeight);
    const right = Math.ceil(x1 * cellWidth);
    const bottom = Math.ceil(y1 * cellHeight);
    context.fillRect(x + left, y + top, right - left, bottom - top);
  }

  function drawBlockElement(context, code, x, y) {
    if (code === 0x2580) fillCellRect(context, x, y, 0, 0, 1, 0.5);
    else if (code >= 0x2581 && code <= 0x2587) fillCellRect(context, x, y, 0, 1 - (code - 0x2580) / 8, 1, 1);
    else if (code === 0x2588) fillCellRect(context, x, y, 0, 0, 1, 1);
    else if (code >= 0x2589 && code <= 0x258f) fillCellRect(context, x, y, 0, 0, (0x2590 - code) / 8, 1);
    else if (code === 0x2590) fillCellRect(context, x, y, 0.5, 0, 1, 1);
    else if (code >= 0x2591 && code <= 0x2593) {
      const level = code - 0x2590;
      for (let py = 0; py < cellHeight; py += 1) {
        for (let px = 0; px < cellWidth; px += 1) {
          if (((px * 3 + py * 5) & 3) < level) context.fillRect(x + px, y + py, 1, 1);
        }
      }
    }
    else if (code === 0x2594) fillCellRect(context, x, y, 0, 0, 1, 1 / 8);
    else if (code === 0x2595) fillCellRect(context, x, y, 7 / 8, 0, 1, 1);
    else if (code >= 0x2596 && code <= 0x259f) {
      const masks = [0b0100,0b1000,0b0001,0b1101,0b1001,0b1011,0b0011,0b0010,0b0110,0b0111];
      const mask = masks[code - 0x2596];
      if (mask & 1) fillCellRect(context, x, y, 0, 0, 0.5, 0.5);
      if (mask & 2) fillCellRect(context, x, y, 0.5, 0, 1, 0.5);
      if (mask & 4) fillCellRect(context, x, y, 0, 0.5, 0.5, 1);
      if (mask & 8) fillCellRect(context, x, y, 0.5, 0.5, 1, 1);
    }
    else return false;
    return true;
  }

  function drawSextant(context, character, x, y) {
    const mask = sextants.indexOf(character);
    if (mask < 0) return false;
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 2; column += 1) {
        if (mask & (1 << (row * 2 + column))) {
          const left = Math.floor(column * cellWidth / 2);
          const top = Math.floor(row * cellHeight / 3);
          const right = Math.floor((column + 1) * cellWidth / 2);
          const bottom = Math.floor((row + 1) * cellHeight / 3);
          context.fillRect(x + left, y + top, right - left, bottom - top);
        }
      }
    }
    return true;
  }

  function drawWedge(context, code, x, y) {
    const spec = wedgeSpecs[code - 0x1fb3c];
    if (!spec) return false;
    const [corner, start, end] = spec.map(name => wedgePoints[name]);
    const side = point => (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]);
    const keep = Math.sign(side(corner));
    for (let py = 0; py < cellHeight; py += 1) {
      for (let px = 0; px < cellWidth; px += 1) {
        const value = side([(px + 0.5) / cellWidth, (py + 0.5) / cellHeight]);
        if (value === 0 || Math.sign(value) === keep) context.fillRect(x + px, y + py, 1, 1);
      }
    }
    return true;
  }

  function insideQuarterTriangle(direction, px, py) {
    if (direction === 'left') return px <= 0.5 - Math.abs(py - 0.5);
    if (direction === 'right') return px >= 0.5 + Math.abs(py - 0.5);
    if (direction === 'upper') return py <= 0.5 - Math.abs(px - 0.5);
    return py >= 0.5 + Math.abs(px - 0.5);
  }

  function drawLegacyTriangle(context, code, x, y) {
    const directions = ['left', 'upper', 'right', 'lower'];
    const quarter = code >= 0x1fb6c && code <= 0x1fb6f;
    const threeQuarters = code >= 0x1fb68 && code <= 0x1fb6b;
    if (!quarter && !threeQuarters) return false;
    const direction = directions[code - (quarter ? 0x1fb6c : 0x1fb68)];
    for (let py = 0; py < cellHeight; py += 1) {
      for (let px = 0; px < cellWidth; px += 1) {
        const inside = insideQuarterTriangle(direction, (px + 0.5) / cellWidth, (py + 0.5) / cellHeight);
        if (quarter ? inside : !inside) context.fillRect(x + px, y + py, 1, 1);
      }
    }
    return true;
  }

  function drawLegacyBar(context, code, x, y) {
    if (code >= 0x1fb70 && code <= 0x1fb75) {
      const eighth = code - 0x1fb6f;
      fillCellRect(context, x, y, eighth / 8, 0, (eighth + 1) / 8, 1);
    } else if (code >= 0x1fb76 && code <= 0x1fb7b) {
      const eighth = code - 0x1fb75;
      fillCellRect(context, x, y, 0, eighth / 8, 1, (eighth + 1) / 8);
    } else if (code >= 0x1fb7c && code <= 0x1fb80) {
      const edges = [['left','lower'], ['left','upper'], ['right','upper'], ['right','lower'], ['upper','lower']][code - 0x1fb7c];
      for (const edge of edges) {
        if (edge === 'left') fillCellRect(context, x, y, 0, 0, 1 / 8, 1);
        if (edge === 'right') fillCellRect(context, x, y, 7 / 8, 0, 1, 1);
        if (edge === 'upper') fillCellRect(context, x, y, 0, 0, 1, 1 / 8);
        if (edge === 'lower') fillCellRect(context, x, y, 0, 7 / 8, 1, 1);
      }
    } else if (code === 0x1fb81) {
      for (const eighth of [0, 2, 4, 7]) fillCellRect(context, x, y, 0, eighth / 8, 1, (eighth + 1) / 8);
    } else if (code >= 0x1fb82 && code <= 0x1fb86) {
      const eighths = [2, 3, 5, 6, 7][code - 0x1fb82];
      fillCellRect(context, x, y, 0, 0, 1, eighths / 8);
    } else if (code >= 0x1fb87 && code <= 0x1fb8b) {
      const eighths = [2, 3, 5, 6, 7][code - 0x1fb87];
      fillCellRect(context, x, y, 1 - eighths / 8, 0, 1, 1);
    } else return false;
    return true;
  }

  function drawBraille(context, code, x, y) {
    const bits = code & 0xff;
    const columns = [Math.floor(cellWidth / 4), cellWidth - Math.floor(cellWidth / 4) - 1];
    const rows = [Math.floor(cellHeight / 8), Math.floor(cellHeight * 3 / 8), Math.floor(cellHeight * 5 / 8), Math.floor(cellHeight * 7 / 8)];
    const dotBits = [[0, 1, 2, 6], [3, 4, 5, 7]];
    for (let column = 0; column < 2; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        if (bits & (1 << dotBits[column][row])) context.fillRect(x + columns[column] - 1, y + rows[row] - 1, 3, 3);
      }
    }
    return true;
  }

  function drawPixelLine(context, x, y, from, to) {
    const x0 = Math.round(from[0] * (cellWidth - 1));
    const y0 = Math.round(from[1] * (cellHeight - 1));
    const x1 = Math.round(to[0] * (cellWidth - 1));
    const y1 = Math.round(to[1] * (cellHeight - 1));
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let step = 0; step <= steps; step += 1) {
      const px = Math.round((x0 * (steps - step) + x1 * step) / steps);
      const py = Math.round((y0 * (steps - step) + y1 * step) / steps);
      context.fillRect(x + px, y + py, 1, 1);
    }
  }

  function drawLaterLegacyGlyph(context, code, x, y) {
    if (code === 0x1fb9a) {
      for (let py = 0; py < cellHeight; py += 1) {
        for (let px = 0; px < cellWidth; px += 1) {
          const nx = (px + 0.5) / cellWidth;
          const ny = (py + 0.5) / cellHeight;
          if (insideQuarterTriangle('upper', nx, ny) || insideQuarterTriangle('lower', nx, ny)) {
            context.fillRect(x + px, y + py, 1, 1);
          }
        }
      }
      return true;
    }
    if (code === 0x1fba1) {
      drawPixelLine(context, x, y, [0.5, 0], [1, 0.5]);
      return true;
    }
    if (code === 0x1fbaa) {
      const points = [[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]];
      for (let index = 1; index < points.length; index += 1) drawPixelLine(context, x, y, points[index - 1], points[index]);
      return true;
    }
    return false;
  }

  function drawTerminalGlyph(context, character, x, y) {
    const code = character.codePointAt(0);
    if (code >= 0x2580 && code <= 0x259f) return drawBlockElement(context, code, x, y);
    if (code >= 0x2800 && code <= 0x28ff) return drawBraille(context, code, x, y);
    if (code >= 0x1fb00 && code <= 0x1fb3b) return drawSextant(context, character, x, y);
    if (code >= 0x1fb3c && code <= 0x1fb67) return drawWedge(context, code, x, y);
    if (code >= 0x1fb68 && code <= 0x1fb6f) return drawLegacyTriangle(context, code, x, y);
    if (code >= 0x1fb70 && code <= 0x1fb8b) return drawLegacyBar(context, code, x, y);
    if (code >= 0x1fb9a && code <= 0x1fbaa) return drawLaterLegacyGlyph(context, code, x, y);
    return false;
  }

  async function drawAnsi(level, request) {
    await document.fonts.load(`${cellHeight}px "ANSI Symbols"`, '█🮈');
    if (request !== ansiRequest || request.signal.aborted) return false;
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const naturalWidth = level.width * cellWidth;
    const naturalHeight = level.rows * cellHeight;
    ansiTerminal.width = Math.ceil(naturalWidth * ratio);
    ansiTerminal.height = Math.ceil(naturalHeight * ratio);
    const context = ansiTerminal.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, naturalWidth, naturalHeight);
    context.font = `${cellHeight}px ${ansiFont}`;
    context.textAlign = 'center';
    context.textBaseline = 'alphabetic';
    const block = context.measureText('█');
    const blockWidth = Math.max(1, block.actualBoundingBoxLeft + block.actualBoundingBoxRight);
    const blockHeight = Math.max(1, block.actualBoundingBoxAscent + block.actualBoundingBoxDescent);
    const glyphScaleX = cellWidth / blockWidth;
    const glyphScaleY = cellHeight / blockHeight;
    const baseline = (block.actualBoundingBoxAscent - block.actualBoundingBoxDescent) / 2;
    let column = 0;
    let row = 0;

    for (const [text, foreground, background] of level.runs) {
      for (const character of text) {
        if (character === '\n') {
          column = 0;
          row += 1;
          continue;
        }
        const x = column * cellWidth;
        const y = row * cellHeight;
        if (background !== null) {
          context.fillStyle = paletteColour(level.palette, background);
          context.fillRect(x, y, cellWidth, cellHeight);
        }
        if (character !== ' ') {
          context.fillStyle = paletteColour(level.palette, foreground);
          if (drawTerminalGlyph(context, character, x, y)) {
            column += 1;
            continue;
          }
          context.save();
          context.beginPath();
          context.rect(x, y, cellWidth, cellHeight);
          context.clip();
          context.translate(x + cellWidth / 2, y + cellHeight / 2);
          context.scale(glyphScaleX, glyphScaleY);
          context.fillText(character, 0, baseline);
          context.restore();
        }
        column += 1;
      }
    }
    ansiLevel = level;
    sizeTerminal();
    return true;
  }

  async function loadAnsi(width, {stop = true} = {}) {
    if (!ansiStage || !ansiView) return false;
    if (stop) stopPlayback();
    const pyramidId = ansiStage.dataset.pyramid;
    if (pyramidId !== ansiCachePyramid) {
      ansiLevelCache.clear();
      ansiCachePyramid = pyramidId;
    }
    const bounded = Math.max(Number(ansiView.dataset.minWidth), Math.min(Number(ansiView.dataset.maxWidth), Number(width)));
    ansiSlider.value = bounded;
    localStorage.setItem('ansi-review-width', String(bounded));
    ansiRequest?.abort();
    const request = new AbortController();
    ansiRequest = request;
    ansiLoading.textContent = 'loading ANSI…';
    ansiLoading.classList.remove('hidden');
    try {
      let level = ansiLevelCache.get(bounded);
      if (!level) {
        const response = await fetch(`/api/pyramids/${pyramidId}/levels/${bounded}`, {signal: request.signal});
        if (!response.ok) throw new Error((await response.json()).detail || 'Could not load ANSI level');
        level = await response.json();
        ansiLevelCache.set(bounded, level);
      }
      if (!await drawAnsi(level, request)) return false;
      ansiDimensions.textContent = `${level.width} × ${level.rows}`;
      ansiSource.textContent = level.source_lod.replace('-', ' ').toUpperCase();
      imageLabel.textContent = `ANSI · width ${level.width}`;
      ansiLoading.classList.add('hidden');
      return true;
    } catch (error) {
      if (error.name === 'AbortError') return false;
      ansiLoading.textContent = error.message;
      return false;
    }
  }

  function selectAnsi() {
    if (!ansiView) return;
    image?.classList.add('hidden');
    ansiView.classList.remove('hidden');
    rasterMode?.classList.add('hidden');
    document.querySelectorAll('.stage-rail button').forEach(item => item.classList.remove('selected'));
    ansiStage.classList.add('selected');
    loadAnsi(ansiSlider.value, {stop: false});
  }

  function selectImage(button) {
    if (!image || !button?.dataset.image) return;
    stopPlayback();
    ansiRequest?.abort();
    image.src = button.dataset.image;
    image.classList.remove('hidden');
    ansiView?.classList.add('hidden');
    rasterMode?.classList.remove('hidden');
    imageLabel.textContent = button.dataset.label;
    document.querySelectorAll('.stage-rail button').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
  }

  async function submit(outcome, extras = {}) {
    if (status.dataset.busy === 'yes') return;
    status.dataset.busy = 'yes';
    status.textContent = 'saving…';
    try {
      const response = await fetch('/api/reviews', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          sample_id: shell.dataset.sampleId,
          snapshot_id: shell.dataset.snapshotId,
          outcome,
          notes: document.querySelector('#review-notes')?.value || '',
          ...extras,
        }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || 'Review failed');
      const result = await response.json();
      status.textContent = 'saved';
      window.location.href = result.next_sample_id ? `/review?sample=${result.next_sample_id}` : '/review';
    } catch (error) {
      status.textContent = error.message;
      status.dataset.busy = 'no';
    }
  }

  document.querySelectorAll('.stage-rail button').forEach(button => button.addEventListener('click', () => selectImage(button)));
  ansiStage?.addEventListener('click', selectAnsi);
  ansiSlider?.addEventListener('input', () => {
    stopPlayback();
    clearTimeout(scrubTimer);
    scrubTimer = setTimeout(() => loadAnsi(ansiSlider.value), 45);
  });
  ansiModes.forEach(button => button.addEventListener('click', () => setDisplayMode(button.dataset.mode)));
  rasterModes.forEach(button => button.addEventListener('click', () => setRasterMode(button.dataset.mode)));
  ansiPlay?.addEventListener('click', async () => {
    if (playing) {
      stopPlayback();
      return;
    }
    playing = true;
    ansiPlay.firstChild.textContent = '❚❚ ';
    let width = Number(ansiSlider.value);
    const maximum = Number(ansiView.dataset.maxWidth);
    if (width >= maximum) width = Number(ansiView.dataset.minWidth) - 1;
    while (playing && width < maximum) {
      width += 1;
      if (!await loadAnsi(width, {stop: false})) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    stopPlayback();
  });
  document.querySelector('#accept')?.addEventListener('click', () => submit('accept'));
  document.querySelector('#review')?.addEventListener('click', () => submit('review'));
  document.querySelector('#reject')?.addEventListener('click', () => {
    rejectPanel.classList.toggle('hidden');
    if (!rejectPanel.classList.contains('hidden')) rejectPanel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  });
  document.querySelectorAll('.issue').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.issue').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
    selectedIssue = button.dataset.issue;
    causeStage.value = button.dataset.defaultStage;
    commitReject.disabled = false;
  }));
  commitReject?.addEventListener('click', () => submit('reject', {issue_code: selectedIssue, introduced_by: causeStage.value}));
  document.querySelector('#undo')?.addEventListener('click', async () => {
    const response = await fetch('/api/reviews/undo', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({event_id: shell.dataset.currentEvent}),
    });
    if (response.ok) window.location.reload();
  });

  document.addEventListener('keydown', event => {
    if (event.target.matches('input, select, textarea')) return;
    const key = event.key.toLowerCase();
    if (key === 'a') submit('accept');
    else if (key === 'x') document.querySelector('#reject')?.click();
    else if (key === '?' || key === '/') submit('review');
    else if (key === 'b') {
      const generated = document.querySelector('.stage-rail button[data-label="Generated raster"]');
      const cutout = document.querySelector('.stage-rail button[data-label="rembg cutout"]');
      selectImage(generated?.classList.contains('selected') ? cutout : generated);
    }
    else if (['0', '1', '2', '3'].includes(key)) selectImage(document.querySelector(`.stage-rail button[data-lod="lod-${key}"]`));
    else if (key === 'p' && ansiView && !ansiView.classList.contains('hidden')) ansiPlay?.click();
    else if (key === '[' && ansiView && !ansiView.classList.contains('hidden')) loadAnsi(Number(ansiSlider.value) - 1);
    else if (key === ']' && ansiView && !ansiView.classList.contains('hidden')) loadAnsi(Number(ansiSlider.value) + 1);
    else if (key === 'z') document.querySelector('#undo')?.click();
    else if (event.key === 'Enter' && selectedIssue) commitReject?.click();
    else if (event.key === 'ArrowRight' && shell.dataset.nextSample) window.location.href = `/review?sample=${shell.dataset.nextSample}`;
    else if (event.key === 'ArrowLeft' && shell.dataset.previousSample) window.location.href = `/review?sample=${shell.dataset.previousSample}`;
    else if (event.key === 'Escape') rejectPanel.classList.add('hidden');
  });

  setRasterMode(localStorage.getItem('ansi-review-raster-mode') || 'normalized');
  if (ansiView) {
    const remembered = Number(localStorage.getItem('ansi-review-width') || 40);
    ansiSlider.value = Math.max(Number(ansiView.dataset.minWidth), Math.min(Number(ansiView.dataset.maxWidth), remembered));
    setDisplayMode(localStorage.getItem('ansi-review-display-mode') || 'fit');
    loadAnsi(ansiSlider.value, {stop: false});
    new ResizeObserver(sizeTerminal).observe(ansiViewport);
  }
})();

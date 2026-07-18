(() => {
  const shell = document.querySelector('.review-shell');
  if (!shell) return;
  const image = document.querySelector('#review-image');
  const imageLabel = document.querySelector('#image-label');
  const rejectPanel = document.querySelector('#reject-panel');
  const causeStage = document.querySelector('#cause-stage');
  const commitReject = document.querySelector('#commit-reject');
  const status = document.querySelector('#save-status');
  let selectedIssue = null;

  function selectImage(button) {
    if (!image || !button?.dataset.image) return;
    image.src = button.dataset.image;
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
    else if (['1', '2', '3'].includes(key)) selectImage(document.querySelectorAll('.stage-rail button[data-label^="LOD"]')[Number(key) - 1]);
    else if (key === 'z') document.querySelector('#undo')?.click();
    else if (event.key === 'Enter' && selectedIssue) commitReject?.click();
    else if (event.key === 'ArrowRight' && shell.dataset.nextSample) window.location.href = `/review?sample=${shell.dataset.nextSample}`;
    else if (event.key === 'ArrowLeft' && shell.dataset.previousSample) window.location.href = `/review?sample=${shell.dataset.previousSample}`;
    else if (event.key === 'Escape') rejectPanel.classList.add('hidden');
  });
})();

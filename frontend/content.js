const B = (typeof browser !== 'undefined') ? browser : chrome;

function setTab(name) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}
document.getElementById('tab-resume').addEventListener('click', () => setTab('resume'));
document.getElementById('tab-analyze').addEventListener('click', () => setTab('analyze'));
document.getElementById('tab-help').addEventListener('click', () => setTab('help'))

async function getStoredResume() {
  const o = await B.storage.local.get('resume_text');
  return o && o.resume_text ? o.resume_text : null;
}
async function storeResume(text) {
  await B.storage.local.set({ resume_text: text });
}

(async () => {
  try {
    const t = await getStoredResume();
    if (t) {
      document.getElementById('resume-text').value = t;
      document.getElementById('resume-status').innerText = 'Resume loaded from storage.';
    } else {
      document.getElementById('resume-status').innerText = 'No resume in storage.';
    }
  } catch (e) {
    console.error(e);
    document.getElementById('resume-status').innerText = 'Error loading resume.';
  }
})();

document.getElementById('resume-upload-button').addEventListener('click', () => {
  const input = document.getElementById('resume-file-input');
  if (!input || !input.files || input.files.length === 0) {
    document.getElementById('resume-status').innerText = 'No file selected.';
    return;
  }
  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      await storeResume(e.target.result);
      document.getElementById('resume-text').value = e.target.result;
      document.getElementById('resume-status').innerText = 'Resume stored in extension storage.';
    } catch (err) {
      console.error(err);
      document.getElementById('resume-status').innerText = 'Error storing resume.';
    }
  };
  reader.onerror = (e) => {
    console.error(e);
    document.getElementById('resume-status').innerText = 'File read error.';
  };
  reader.readAsText(input.files[0], 'utf-8');
});

document.getElementById('resume-save-button').addEventListener('click', async () => {
  try {
    await storeResume(document.getElementById('resume-text').value || '');
    document.getElementById('resume-status').innerText = 'Resume saved to storage.';
  } catch (e) {
    console.error(e);
    document.getElementById('resume-status').innerText = 'Error saving resume.';
  }
});

async function extractJobTextFromActiveTab() {
  const tabs = await B.tabs.query({ active: true, currentWindow: true });
  const active = tabs[0];
  if (!active) throw new Error('No active tab');
  const results = await B.tabs.executeScript(active.id, {
    code: `
      (function() {
        const el = document.getElementById('jobDescriptionText')
          || document.querySelector('[id^="jobDescriptionText"]')
          || document.querySelector('.jobsearch-JobComponent-description')
          || document.querySelector('.jobsearch-jobDescriptionText');
        return el ? el.innerText : null;
      })();
    `
  });
  return results && results[0] ? results[0] : null;
}

async function callAnalyzeAPI(resumeText, jobText) {
  const resp = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume: resumeText, job: jobText })
  });
  if (!resp.ok) throw new Error('API returned ' + resp.status);
  return resp.json();
}

document.getElementById('extract-button').addEventListener('click', async () => {
  document.getElementById('result').innerText = 'Working...';
  try {
    const jobText = await extractJobTextFromActiveTab();
    if (!jobText) {
      document.getElementById('result').innerText = 'Job description not found on page.';
      return;
    }
    const resumeText = await getStoredResume();
    if (!resumeText) {
      document.getElementById('result').innerText = 'resume.txt not found in extension storage. Upload a resume first.';
      return;
    }
    const data = await callAnalyzeAPI(resumeText, jobText);

	  const formattedResponse = `
<strong>Score:</strong> ${(data.score * 100).toFixed(2)}%
<strong>Strengths:</strong><ul>
${data.strengths
.map(strength => strength.trim())
.filter(strength => strength) // Remove empty entries
.map(strength => {
const parts = strength.split('|');
const description = parts[0]?.trim() || '';
const similarity = parts[parts.length - 1]?.trim().replace('similarity: ', '') || '';
return `<li>${description}</li>`;
}).join('')}</ul><strong>Weaknesses:</strong><ul>
${data.weaknesses
.map(weakness => weakness.trim())
.filter(weakness => weakness) // Remove empty entries
.map(weakness => `<li>${weakness}</li>`).join('')}</ul>
`;

    document.getElementById('result').innerHTML = formattedResponse;
  } catch (err) {
    console.error(err);
    document.getElementById('result').innerText = String(err);
  }
});

document.getElementById('analyze-manual-button').addEventListener('click', async () => {
  const jobText = document.getElementById('manual-job-text').value.trim();
  if (!jobText) {
    document.getElementById('result').innerText = 'Please paste job text or use Extract.';
    return;
  }
  try {
    const resumeText = await getStoredResume();
    if (!resumeText) {
      document.getElementById('result').innerText = 'resume.txt not found in extension storage. Upload a resume first.';
      return;
    }
    document.getElementById('result').innerText = 'Analyzing...';

    const data = await callAnalyzeAPI(resumeText, jobText);

	  const formattedResponse = `
<strong>Score:</strong> ${(data.score * 100).toFixed(2)}%
<strong>Strengths:</strong><ul>
${data.strengths
.map(strength => strength.trim())
.filter(strength => strength) // Remove empty entries
.map(strength => {
const parts = strength.split('|');
const description = parts[0]?.trim() || '';
const similarity = parts[parts.length - 1]?.trim().replace('similarity: ', '') || '';
return `<li>${description}</li>`;
}).join('')}</ul><strong>Weaknesses:</strong><ul>
${data.weaknesses
.map(weakness => weakness.trim())
.filter(weakness => weakness) // Remove empty entries
.map(weakness => `<li>${weakness}</li>`).join('')}</ul>
`;
    document.getElementById('result').innerHTML = formattedResponse;
  } catch (err) {
    console.error(err);
    document.getElementById('result').innerText = 'An error occurred while analyzing the data.';
  }
});




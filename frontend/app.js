/**
 * ResolveOps — Technical RAG Frontend Client
 * Completely decoupled from backend: communicates purely via HTTP REST API.
 */

// ─────────────────────────────────────────────────────────────────────────────
// State & Configuration
// ─────────────────────────────────────────────────────────────────────────────

const defaultApiBase = (typeof window !== "undefined" && window.location && window.location.origin && window.location.origin !== "null")
  ? window.location.origin
  : "http://localhost:8000";

const STATE = {
  apiBase: localStorage.getItem("resolveops_api_url") || defaultApiBase,
  candidatePool: parseInt(localStorage.getItem("resolveops_candidate_pool") || "10", 10),
  topK: parseInt(localStorage.getItem("resolveops_top_k") || "3", 10),
  activeCitations: [],
  isQuerying: false,
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ─────────────────────────────────────────────────────────────────────────────

const DOM = {
  statusPill: document.getElementById("backend-status"),
  chatHistory: document.getElementById("chat-history"),
  chatForm: document.getElementById("chat-form"),
  queryInput: document.getElementById("query-input"),
  sendBtn: document.getElementById("send-btn"),
  presetChips: document.querySelectorAll(".preset-chip"),
  citationsContainer: document.getElementById("citations-container"),
  citationCountBadge: document.getElementById("citation-count-badge"),
  metricRetrieval: document.getElementById("metric-retrieval"),
  metricRerank: document.getElementById("metric-rerank"),
  metricLLM: document.getElementById("metric-llm"),
  metricTotal: document.getElementById("metric-total"),
  configBtn: document.getElementById("config-btn"),
  settingsModal: document.getElementById("settings-modal"),
  closeModalBtn: document.getElementById("close-modal-btn"),
  saveSettingsBtn: document.getElementById("save-settings-btn"),
  apiUrlInput: document.getElementById("api-url-input"),
  candidatePoolSlider: document.getElementById("candidate-pool-slider"),
  candidatePoolVal: document.getElementById("candidate-pool-val"),
  topKSlider: document.getElementById("top-k-slider"),
  topKVal: document.getElementById("top-k-val"),
  scrollBottomBtn: document.getElementById("scroll-bottom-btn"),
};

// ─────────────────────────────────────────────────────────────────────────────
// Health Probe
// ─────────────────────────────────────────────────────────────────────────────

async function checkBackendHealth() {
  try {
    const res = await fetch(`${STATE.apiBase}/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    DOM.statusPill.className = "status-pill status-online";
    DOM.statusPill.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-label">Online (${data.indexed_chunks} Chunks Indexed)</span>
    `;
  } catch (err) {
    DOM.statusPill.className = "status-pill status-offline";
    DOM.statusPill.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-label">Offline (${STATE.apiBase})</span>
    `;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format Text with Interactive Citations
// ─────────────────────────────────────────────────────────────────────────────

function formatAnswerWithCitations(rawText) {
  let text = rawText
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks ```lang ... ```
  text = text.replace(/```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (m, code) => {
    return `§§CODEBLOCK§§<pre><code>${code.trim()}</code></pre>§§ENDCODEBLOCK§§`;
  });

  // Inline code `...`
  text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Bold **...**
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Citations [1], [2] -> clickable interactive chips
  text = text.replace(/\[(\d+)\]/g, (m, num) => {
    return `<span class="citation-link" data-cite-index="${num}">[${num}]</span>`;
  });

  // Line-by-line processing for lists, paragraphs, and code blocks
  const lines = text.split("\n");
  let html = "";
  let inUl = false;
  let inOl = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (inOl) { html += "</ol>"; inOl = false; }
      continue;
    }

    if (line.includes("§§CODEBLOCK§§")) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (inOl) { html += "</ol>"; inOl = false; }
      html += line.replace("§§CODEBLOCK§§", "").replace("§§ENDCODEBLOCK§§", "");
      continue;
    }

    // Bullet list: -, *, •
    if (/^[-*•]\s+/.test(line)) {
      if (inOl) { html += "</ol>"; inOl = false; }
      if (!inUl) { html += "<ul>"; inUl = true; }
      html += `<li>${line.replace(/^[-*•]\s+/, "")}</li>`;
      continue;
    }

    // Numbered list: 1., 2.
    if (/^\d+\.\s+/.test(line)) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (!inOl) { html += "<ol>"; inOl = true; }
      html += `<li>${line.replace(/^\d+\.\s+/, "")}</li>`;
      continue;
    }

    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }

    html += `<p>${line}</p>`;
  }

  if (inUl) html += "</ul>";
  if (inOl) html += "</ol>";

  return html;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render Citations in Audit Aside Panel
// ─────────────────────────────────────────────────────────────────────────────

function renderCitations(citations) {
  STATE.activeCitations = citations || [];
  DOM.citationCountBadge.textContent = `${STATE.activeCitations.length} Sources`;

  if (!STATE.activeCitations.length) {
    DOM.citationsContainer.innerHTML = `
      <div class="empty-citations-placeholder">
        <div class="empty-icon">📑</div>
        <h4>No Active Citations</h4>
        <p>No citation references were generated for this response.</p>
      </div>
    `;
    return;
  }

  DOM.citationsContainer.innerHTML = "";

  STATE.activeCitations.forEach(cit => {
    const card = document.createElement("div");
    card.className = "citation-card";
    card.id = `citation-card-${cit.index}`;

    const isPdf = cit.source.endsWith(".pdf");
    const pagesHtml = cit.page_numbers && cit.page_numbers.length
      ? `<span class="citation-pages">Page ${escapeHtml(cit.page_numbers.join(", "))}</span>`
      : "";

    const headingHtml = cit.heading_context && cit.heading_context.length
      ? `<div class="citation-breadcrumb">${escapeHtml(cit.heading_context.join(" ▸ "))}</div>`
      : "";

    card.innerHTML = `
      <div class="citation-card-top">
        <div class="citation-source">
          <span>${isPdf ? "📄" : "📝"}</span>
          <span>${escapeHtml(cit.source)}</span>
        </div>
        <div style="display: flex; gap: 6px; align-items: center;">
          ${pagesHtml}
          <span class="citation-badge">[${cit.index}]</span>
        </div>
      </div>
      ${headingHtml}
      <div class="citation-snippet">${escapeHtml(cit.snippet)}</div>
    `;

    DOM.citationsContainer.appendChild(card);
  });
}

function attachCitationListeners(container = document) {
  container.querySelectorAll(".citation-link").forEach(link => {
    const index = link.dataset.citeIndex;
    const parentMsg = link.closest(".assistant-message");

    const ensureCitationsActive = () => {
      if (parentMsg && parentMsg._citations && parentMsg._citations !== STATE.activeCitations) {
        renderCitations(parentMsg._citations);
        if (parentMsg._timings) {
          updateTelemetry(parentMsg._timings);
        }
      }
    };

    link.addEventListener("mouseenter", () => {
      ensureCitationsActive();
      const targetCard = document.getElementById(`citation-card-${index}`);
      if (targetCard) {
        targetCard.classList.add("highlighted");
        targetCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });

    link.addEventListener("mouseleave", () => {
      const targetCard = document.getElementById(`citation-card-${index}`);
      if (targetCard) targetCard.classList.remove("highlighted");
    });

    link.addEventListener("click", () => {
      ensureCitationsActive();
      const targetCard = document.getElementById(`citation-card-${index}`);
      if (targetCard) {
        targetCard.classList.add("highlighted");
        targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
        setTimeout(() => targetCard.classList.remove("highlighted"), 2000);
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Update Telemetry Metrics
// ─────────────────────────────────────────────────────────────────────────────

function updateTelemetry(timings, modelName) {
  if (!timings) return;
  DOM.metricRetrieval.textContent = `${timings.retrieval_ms.toFixed(1)} ms`;
  DOM.metricRerank.textContent = `${timings.rerank_ms.toFixed(1)} ms`;
  DOM.metricLLM.textContent = `${timings.generation_ms.toFixed(1)} ms`;
  DOM.metricTotal.textContent = `${timings.total_ms.toFixed(1)} ms`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Query Execution
// ─────────────────────────────────────────────────────────────────────────────

async function submitQuery(queryText) {
  const clean = queryText.trim();
  if (!clean || STATE.isQuerying) return;

  STATE.isQuerying = true;
  DOM.sendBtn.disabled = true;
  DOM.queryInput.value = "";
  DOM.queryInput.style.height = "auto";

  // 1. Add User Message to History
  appendUserMessage(clean);

  // 2. Add Loading Indicator
  const loadingElem = appendLoadingMessage();

  try {
    const t0 = performance.now();
    const response = await fetch(`${STATE.apiBase}/api/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: clean,
        candidate_pool: STATE.candidatePool,
        top_k: STATE.topK,
        temperature: 0.0,
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    loadingElem.remove();

    // 3. Render Assistant Response
    appendAssistantMessage(data);

    // 4. Render Citations & Telemetry
    renderCitations(data.citations);
    updateTelemetry(data.timings, data.model_name);

  } catch (err) {
    loadingElem.remove();
    appendErrorMessage(err.message);
  } finally {
    STATE.isQuerying = false;
    DOM.sendBtn.disabled = false;
    DOM.queryInput.focus();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat DOM Helpers
// ─────────────────────────────────────────────────────────────────────────────

function scrollChatToBottom(smooth = true) {
  const doScroll = () => {
    if (DOM.chatHistory) {
      if (smooth) {
        DOM.chatHistory.scrollTo({
          top: DOM.chatHistory.scrollHeight + 5000,
          behavior: "smooth",
        });
      } else {
        DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight + 5000;
      }
    }
  };

  doScroll();
  requestAnimationFrame(doScroll);
  setTimeout(doScroll, 80);
  setTimeout(doScroll, 250);
}

function appendUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "chat-message user-message";
  msg.innerHTML = `<div class="message-body">${escapeHtml(text)}</div>`;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(false);
}

function appendAssistantMessage(data) {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message";
  msg._citations = data.citations || [];
  msg._timings = data.timings;

  const formattedContent = formatAnswerWithCitations(data.answer);

  let validationWarningHtml = "";
  if (data.hallucinated_markers && data.hallucinated_markers.length) {
    validationWarningHtml = `<span class="meta-pill" style="color: var(--accent-rose); border: 1px solid var(--accent-rose);">⚠️ Invalid Citations: [${escapeHtml(data.hallucinated_markers.join(", "))}]</span>`;
  } else if (data.citation_validation_failed && (!data.citations || !data.citations.length)) {
    validationWarningHtml = `<span class="meta-pill" style="color: var(--accent-amber);">⚠️ No Sources Cited</span>`;
  }

  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">🤖</span>
      <span class="agent-name">${escapeHtml(data.model_name || "ResolveOps")}</span>
      <span class="message-time">${data.timings.total_ms.toFixed(0)}ms</span>
    </div>
    <div class="message-body">${formattedContent}</div>
    <div class="message-meta-bar">
      <span class="meta-pill">Candidates: ${data.candidate_count}</span>
      <span class="meta-pill">Reranked: ${data.reranked_count}</span>
      <span class="meta-pill">Citations: ${data.citations ? data.citations.length : 0}</span>
      ${validationWarningHtml}
    </div>
  `;

  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
  attachCitationListeners(msg);
}

function appendLoadingMessage() {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message loading-state";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">⚡</span>
      <span class="agent-name">ResolveOps Pipeline</span>
    </div>
    <div class="message-body">
      <p style="color: var(--text-muted); font-style: italic;">
        Stage 1: Hybrid Search ➔ Stage 2: Nemotron 8B Rerank ➔ Stage 3: DeepSeek V3...
      </p>
    </div>
  `;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
  return msg;
}

function appendErrorMessage(errText) {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">⚠️</span>
      <span class="agent-name" style="color: var(--accent-rose)">Pipeline Error</span>
    </div>
    <div class="message-body" style="border-color: var(--accent-rose);">
      <p>Could not retrieve an answer: <strong>${escapeHtml(errText)}</strong></p>
      <p style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
        Make sure the backend server is running on <code>${STATE.apiBase}</code>:
        <br><code>.venv\\Scripts\\uvicorn resolveops.api.main:app --port 8000</code>
      </p>
    </div>
  `;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────────────────────────────────────

DOM.chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  submitQuery(DOM.queryInput.value);
});

// Auto-expand textarea on typing & Enter to send (Shift+Enter for newline)
DOM.queryInput.addEventListener("input", () => {
  DOM.queryInput.style.height = "auto";
  DOM.queryInput.style.height = `${Math.min(DOM.queryInput.scrollHeight, 120)}px`;
});

DOM.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    DOM.chatForm.requestSubmit();
  }
});

// Floating Scroll-to-Bottom Button Handler
if (DOM.chatHistory && DOM.scrollBottomBtn) {
  DOM.chatHistory.addEventListener("scroll", () => {
    const distanceFromBottom = DOM.chatHistory.scrollHeight - DOM.chatHistory.scrollTop - DOM.chatHistory.clientHeight;
    if (distanceFromBottom > 180) {
      DOM.scrollBottomBtn.classList.remove("hidden");
    } else {
      DOM.scrollBottomBtn.classList.add("hidden");
    }
  });

  DOM.scrollBottomBtn.addEventListener("click", () => {
    scrollChatToBottom(true);
    DOM.scrollBottomBtn.classList.add("hidden");
  });
}

// Preset Query Chips
DOM.presetChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const q = chip.dataset.query;
    if (q) submitQuery(q);
  });
});

// Settings Modal
DOM.configBtn.addEventListener("click", () => {
  DOM.apiUrlInput.value = STATE.apiBase;
  DOM.candidatePoolSlider.value = STATE.candidatePool;
  DOM.candidatePoolVal.textContent = STATE.candidatePool;
  DOM.topKSlider.value = STATE.topK;
  DOM.topKVal.textContent = STATE.topK;
  DOM.settingsModal.classList.remove("hidden");
});

DOM.closeModalBtn.addEventListener("click", () => {
  DOM.settingsModal.classList.add("hidden");
});

DOM.candidatePoolSlider.addEventListener("input", (e) => {
  DOM.candidatePoolVal.textContent = e.target.value;
});

DOM.topKSlider.addEventListener("input", (e) => {
  DOM.topKVal.textContent = e.target.value;
});

DOM.saveSettingsBtn.addEventListener("click", () => {
  STATE.apiBase = DOM.apiUrlInput.value.trim() || "http://localhost:8000";
  STATE.candidatePool = parseInt(DOM.candidatePoolSlider.value, 10);
  STATE.topK = parseInt(DOM.topKSlider.value, 10);

  localStorage.setItem("resolveops_api_url", STATE.apiBase);
  localStorage.setItem("resolveops_candidate_pool", STATE.candidatePool.toString());
  localStorage.setItem("resolveops_top_k", STATE.topK.toString());

  DOM.settingsModal.classList.add("hidden");
  checkBackendHealth();
});

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

checkBackendHealth();
setInterval(checkBackendHealth, 8000);
DOM.queryInput.focus();


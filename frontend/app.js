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
  reviewVoiceTranscript: localStorage.getItem("resolveops_voice_review") === "true",
  activeCitations: [],
  isQuerying: false,
  isRecording: false,
  mediaRecorder: null,
  mediaStream: null,
  audioChunks: [],
  recordStartTime: 0,
  recordTimerId: null,
  activeAudio: null,
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
  voiceBtn: document.getElementById("voice-btn"),
  voiceStatusBar: document.getElementById("voice-status-bar"),
  voiceStatusText: document.getElementById("voice-status-text"),
  stopSpeakingBtn: document.getElementById("stop-speaking-btn"),
  presetChips: document.querySelectorAll(".preset-chip"),
  citationsContainer: document.getElementById("citations-container"),
  citationCountBadge: document.getElementById("citation-count-badge"),
  faithfulnessBadge: document.getElementById("faithfulness-badge"),
  metricSttRow: document.getElementById("metric-stt-row"),
  metricStt: document.getElementById("metric-stt"),
  metricRetrieval: document.getElementById("metric-retrieval"),
  metricRerank: document.getElementById("metric-rerank"),
  metricLLM: document.getElementById("metric-llm"),
  metricAudit: document.getElementById("metric-audit"),
  metricTtsRow: document.getElementById("metric-tts-row"),
  metricTts: document.getElementById("metric-tts"),
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
  voiceReviewToggle: document.getElementById("voice-review-toggle"),
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
        renderGroundingBadge(parentMsg._faithfulnessScore, parentMsg._claimAudit);
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

function renderGroundingBadge(faithfulnessScore, claimAudit) {
  if (!DOM.faithfulnessBadge) return;
  if (faithfulnessScore === null || faithfulnessScore === undefined) {
    DOM.faithfulnessBadge.classList.add("hidden");
    return;
  }
  DOM.faithfulnessBadge.classList.remove("hidden");
  const pct = Math.round(faithfulnessScore * 100);
  if (claimAudit && claimAudit.unsupported_claims > 0) {
    DOM.faithfulnessBadge.className = "badge faithfulness-badge warning";
    DOM.faithfulnessBadge.textContent = `⚠️ ${claimAudit.unsupported_claims} Unsupported (${pct}%)`;
    if (claimAudit.unsupported_details && claimAudit.unsupported_details.length) {
      DOM.faithfulnessBadge.title = claimAudit.unsupported_details.map(u => `Claim: "${u.sentence}"\nReason: ${u.explanation}`).join("\n\n");
    }
  } else {
    DOM.faithfulnessBadge.className = "badge faithfulness-badge";
    DOM.faithfulnessBadge.textContent = `✅ ${pct}% Grounded`;
    DOM.faithfulnessBadge.title = "All claims factually verified against source documentation.";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Update Telemetry Metrics
// ─────────────────────────────────────────────────────────────────────────────

function updateTelemetry(timings, modelName) {
  if (!timings) return;

  const isVoice = timings.stt_ms !== undefined && timings.stt_ms !== null;
  if (DOM.metricSttRow && DOM.metricStt) {
    if (isVoice) {
      DOM.metricSttRow.classList.remove("hidden");
      DOM.metricStt.textContent = `${timings.stt_ms.toFixed(1)} ms`;
    } else {
      DOM.metricSttRow.classList.add("hidden");
    }
  }

  DOM.metricRetrieval.textContent = `${timings.retrieval_ms.toFixed(1)} ms`;
  DOM.metricRerank.textContent = `${timings.rerank_ms.toFixed(1)} ms`;
  DOM.metricLLM.textContent = `${timings.generation_ms.toFixed(1)} ms`;
  if (DOM.metricAudit && timings.claim_audit_ms !== undefined) {
    DOM.metricAudit.textContent = `${timings.claim_audit_ms.toFixed(1)} ms`;
  }

  if (DOM.metricTtsRow && DOM.metricTts) {
    if (isVoice) {
      DOM.metricTtsRow.classList.remove("hidden");
      DOM.metricTts.textContent = `${(timings.tts_ms || 0).toFixed(1)} ms`;
    } else {
      DOM.metricTtsRow.classList.add("hidden");
    }
  }

  DOM.metricTotal.textContent = `${timings.total_ms.toFixed(1)} ms`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Voice Agent: Recording, Barge-In Interruption, and Natural Audio Playback
// ─────────────────────────────────────────────────────────────────────────────

function stopActiveSpeech() {
  if (STATE.activeAudio) {
    STATE.activeAudio.pause();
    STATE.activeAudio.currentTime = 0;
    STATE.activeAudio = null;
  }
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if (DOM.voiceBtn) DOM.voiceBtn.classList.remove("speaking");
  if (!STATE.isRecording && DOM.voiceStatusBar) {
    DOM.voiceStatusBar.classList.add("hidden");
    DOM.voiceStatusBar.classList.remove("speaking");
  }
  if (DOM.stopSpeakingBtn) DOM.stopSpeakingBtn.classList.add("hidden");
}

function playNaturalAudio(audioElem, fallbackText) {
  stopActiveSpeech();

  if (audioElem) {
    STATE.activeAudio = audioElem;
    if (DOM.voiceBtn) DOM.voiceBtn.classList.add("speaking");
    if (DOM.voiceStatusBar) {
      DOM.voiceStatusBar.classList.remove("hidden", "recording");
      DOM.voiceStatusBar.classList.add("speaking");
    }
    if (DOM.voiceStatusText) DOM.voiceStatusText.textContent = "Speaking answer... (click Mic or Stop to interrupt)";
    if (DOM.stopSpeakingBtn) {
      DOM.stopSpeakingBtn.textContent = "⏹ Stop Speaking";
      DOM.stopSpeakingBtn.classList.remove("hidden");
    }

    const onEnd = () => {
      if (STATE.activeAudio === audioElem) {
        stopActiveSpeech();
      }
    };
    audioElem.addEventListener("ended", onEnd, { once: true });
    audioElem.addEventListener("pause", onEnd, { once: true });
    audioElem.play().catch(() => {
      // Browser autoplay policy blocked playback; user can still press Play in the inline audio bar
      onEnd();
    });
    return;
  }

  // Graceful fallback to Browser Web Speech if backend TTS degraded to text-only
  if (fallbackText && typeof window !== "undefined" && "speechSynthesis" in window) {
    const utter = new SpeechSynthesisUtterance(fallbackText);
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => /Natural|Neural|Google US English|Aria|Ava/i.test(v.name)) || voices.find(v => v.lang.startsWith("en"));
    if (preferred) utter.voice = preferred;
    utter.rate = 1.0;

    if (DOM.voiceBtn) DOM.voiceBtn.classList.add("speaking");
    if (DOM.voiceStatusBar) {
      DOM.voiceStatusBar.classList.remove("hidden", "recording");
      DOM.voiceStatusBar.classList.add("speaking");
    }
    if (DOM.voiceStatusText) DOM.voiceStatusText.textContent = "Speaking (Browser Voice Fallback)...";
    if (DOM.stopSpeakingBtn) {
      DOM.stopSpeakingBtn.textContent = "⏹ Stop Speaking";
      DOM.stopSpeakingBtn.classList.remove("hidden");
    }

    utter.onend = () => stopActiveSpeech();
    utter.onerror = () => stopActiveSpeech();
    window.speechSynthesis.speak(utter);
  }
}

function pickSupportedAudioMime() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  if (typeof MediaRecorder === "undefined") return "";
  return candidates.find(m => MediaRecorder.isTypeSupported(m)) || "";
}

async function toggleVoiceRecording() {
  if (STATE.isQuerying) return;

  // If currently recording -> stop and submit
  if (STATE.isRecording && STATE.mediaRecorder) {
    STATE.isRecording = false;
    clearInterval(STATE.recordTimerId);
    STATE.mediaRecorder.stop();
    if (DOM.voiceBtn) {
      DOM.voiceBtn.classList.remove("recording");
      DOM.voiceBtn.innerHTML = `<span class="mic-icon">🎙️</span>`;
    }
    if (DOM.voiceStatusBar) {
      DOM.voiceStatusBar.classList.add("hidden");
      DOM.voiceStatusBar.classList.remove("recording");
    }
    return;
  }

  // Instant Barge-In: stop any ongoing agent playback before listening
  stopActiveSpeech();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    appendErrorMessage("Your browser does not support microphone capture (MediaRecorder API).");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    STATE.mediaStream = stream;
    STATE.audioChunks = [];

    const mimeType = pickSupportedAudioMime();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    STATE.mediaRecorder = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        STATE.audioChunks.push(event.data);
      }
    };

    recorder.onstop = () => {
      // Release hardware mic immediately
      if (STATE.mediaStream) {
        STATE.mediaStream.getTracks().forEach(t => t.stop());
        STATE.mediaStream = null;
      }
      const elapsedMs = Date.now() - STATE.recordStartTime;
      const blob = new Blob(STATE.audioChunks, { type: mimeType || "audio/webm" });
      STATE.audioChunks = [];
      if (elapsedMs < 450 || blob.size < 1000) {
        appendNoticeMessage("Recording was too short to capture speech. Please hold and speak for at least 1 second, then click Stop.");
        return;
      }
      submitVoiceAudio(blob);
    };

    recorder.start(250);
    STATE.isRecording = true;
    STATE.recordStartTime = Date.now();

    if (DOM.voiceBtn) {
      DOM.voiceBtn.classList.add("recording");
      DOM.voiceBtn.innerHTML = `<span class="mic-icon">⏹</span>`;
    }
    if (DOM.voiceStatusBar) {
      DOM.voiceStatusBar.classList.remove("hidden", "speaking");
      DOM.voiceStatusBar.classList.add("recording");
    }
    if (DOM.stopSpeakingBtn) {
      DOM.stopSpeakingBtn.textContent = "⏹ Stop & Send";
      DOM.stopSpeakingBtn.classList.remove("hidden");
    }

    const updateTimer = () => {
      const secs = Math.floor((Date.now() - STATE.recordStartTime) / 1000);
      const mins = Math.floor(secs / 60);
      const rem = String(secs % 60).padStart(2, "0");
      if (DOM.voiceStatusText) {
        DOM.voiceStatusText.textContent = `Listening... ${mins}:${rem} (click ⏹ to send)`;
      }
    };
    updateTimer();
    STATE.recordTimerId = setInterval(updateTimer, 500);

  } catch (err) {
    appendNoticeMessage("Microphone access was blocked or unavailable. Please allow microphone permission in your browser settings and try again.");
  }
}

function startVoiceProcessingIndicator() {
  const steps = [
    { label: "🎙️ Transcribing speech", status: "Transcribing your voice..." },
    { label: "🔍 Searching docs", status: "Searching & reranking AWS documentation..." },
    { label: "✅ Grounding answer", status: "Generating & verifying cited claims..." },
    { label: "🔊 Synthesizing voice", status: "Generating natural spoken response..." },
  ];

  // Show live glowing processing bar right above the chat box
  if (DOM.voiceBtn) {
    DOM.voiceBtn.innerHTML = `<span class="voice-spinner"></span>`;
  }
  if (DOM.voiceStatusBar) {
    DOM.voiceStatusBar.classList.remove("hidden", "recording", "speaking");
    DOM.voiceStatusBar.classList.add("processing");
  }
  if (DOM.stopSpeakingBtn) DOM.stopSpeakingBtn.classList.add("hidden");

  // Append rich animated pipeline progress card in chat history
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message loading-state";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">🎙️</span>
      <span class="agent-name">ResolveOps Voice Agent</span>
      <span class="message-time processing-elapsed">0.0s</span>
    </div>
    <div class="message-body">
      <div class="processing-card">
        <div class="processing-header-row">
          <span class="voice-spinner"></span>
          <span class="processing-headline">${steps[0].status}</span>
        </div>
        <div class="processing-bar-track"><div class="processing-bar-fill"></div></div>
        <div class="processing-steps">
          ${steps.map((s, idx) => `<span class="processing-step ${idx === 0 ? "active" : ""}" data-step="${idx}">${s.label}</span>`).join("")}
        </div>
      </div>
    </div>
  `;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);

  const startedAt = performance.now();
  const elapsedEl = msg.querySelector(".processing-elapsed");
  const headlineEl = msg.querySelector(".processing-headline");
  const stepEls = msg.querySelectorAll(".processing-step");

  const updateProgress = () => {
    const sec = (performance.now() - startedAt) / 1000;
    const stepIdx = sec < 1.2 ? 0 : (sec < 2.6 ? 1 : (sec < 5.0 ? 2 : 3));

    if (elapsedEl) elapsedEl.textContent = `${sec.toFixed(1)}s`;
    if (headlineEl) headlineEl.textContent = steps[stepIdx].status;
    if (DOM.voiceStatusText) {
      DOM.voiceStatusText.textContent = `${steps[stepIdx].status} (${sec.toFixed(1)}s)`;
    }

    stepEls.forEach((el, idx) => {
      el.classList.toggle("done", idx < stepIdx);
      el.classList.toggle("active", idx === stepIdx);
    });
  };

  updateProgress();
  const intervalId = setInterval(updateProgress, 200);

  return () => {
    clearInterval(intervalId);
    msg.remove();
    if (DOM.voiceBtn) {
      DOM.voiceBtn.innerHTML = `<span class="mic-icon">🎙️</span>`;
    }
    if (DOM.voiceStatusBar) {
      DOM.voiceStatusBar.classList.remove("processing");
      if (!STATE.isRecording && !STATE.activeAudio) {
        DOM.voiceStatusBar.classList.add("hidden");
      }
    }
  };
}

async function submitVoiceAudio(audioBlob) {
  if (STATE.isQuerying) return;

  STATE.isQuerying = true;
  DOM.sendBtn.disabled = true;
  if (DOM.voiceBtn) DOM.voiceBtn.disabled = true;

  const stopProcessingUI = startVoiceProcessingIndicator();

  try {
    const ext = (audioBlob.type || "").includes("mp4") ? "mp4" : ((audioBlob.type || "").includes("ogg") ? "ogg" : "webm");

    // Optional Review Mode: transcribe into input box so the user can verify/fix technical terms before querying
    if (STATE.reviewVoiceTranscript) {
      const transcribeUrl = `${STATE.apiBase}/api/v1/voice/transcribe`;
      const response = await fetch(transcribeUrl, {
        method: "POST",
        headers: {
          "Content-Type": audioBlob.type || "audio/webm",
          "X-Audio-Filename": `recording.${ext}`,
        },
        body: audioBlob,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        if (response.status === 422) {
          stopProcessingUI();
          appendNoticeMessage(errData.detail || "No speech detected in your recording. Please speak clearly and try again.");
          return;
        }
        throw new Error(errData.detail || "Something went wrong while transcribing your voice.");
      }

      const sttData = await response.json();
      stopProcessingUI();
      DOM.queryInput.value = sttData.text || "";
      DOM.queryInput.dispatchEvent(new Event("input"));
      DOM.queryInput.focus();
      DOM.queryInput.setSelectionRange(DOM.queryInput.value.length, DOM.queryInput.value.length);
      return;
    }

    const url = `${STATE.apiBase}/api/v1/voice/query?candidate_pool=${STATE.candidatePool}&top_k=${STATE.topK}&temperature=0.0`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": audioBlob.type || "audio/webm",
        "X-Audio-Filename": `recording.${ext}`,
      },
      body: audioBlob,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      if (response.status === 422) {
        stopProcessingUI();
        appendNoticeMessage(errData.detail || "No speech detected in your recording. Please speak clearly and try again.");
        return;
      }
      throw new Error(errData.detail || "Something went wrong while processing your voice request.");
    }

    const data = await response.json();
    stopProcessingUI();

    // 1. Render transcribed user voice question
    appendUserMessage(data.transcript, true);

    // 2. Render verified assistant answer + inline audio player
    const audioElem = appendAssistantMessage(data);

    // 3. Render Citations, Grounding Badge & Voice Telemetry Waterfall
    renderCitations(data.citations);
    renderGroundingBadge(data.faithfulness_score, data.claim_audit);
    updateTelemetry(data.timings, data.model_name);

    // 4. Auto-play natural voice response (with graceful browser fallback)
    playNaturalAudio(audioElem, data.audio_url ? null : data.spoken_text);

  } catch (err) {
    stopProcessingUI();
    appendErrorMessage(err.message);
  } finally {
    STATE.isQuerying = false;
    DOM.sendBtn.disabled = false;
    if (DOM.voiceBtn) DOM.voiceBtn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Query Execution
// ─────────────────────────────────────────────────────────────────────────────

async function submitQuery(queryText) {
  const clean = queryText.trim();
  if (!clean || STATE.isQuerying) return;

  stopActiveSpeech();
  STATE.isQuerying = true;
  DOM.sendBtn.disabled = true;
  if (DOM.voiceBtn) DOM.voiceBtn.disabled = true;
  DOM.queryInput.value = "";
  DOM.queryInput.style.height = "auto";

  // 1. Add User Message to History
  appendUserMessage(clean, false);

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
    renderGroundingBadge(data.faithfulness_score, data.claim_audit);
    updateTelemetry(data.timings, data.model_name);

  } catch (err) {
    loadingElem.remove();
    appendErrorMessage(err.message);
  } finally {
    STATE.isQuerying = false;
    DOM.sendBtn.disabled = false;
    if (DOM.voiceBtn) DOM.voiceBtn.disabled = false;
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

function appendUserMessage(text, isVoice = false) {
  const msg = document.createElement("div");
  msg.className = "chat-message user-message";
  const prefix = isVoice ? `<span class="voice-badge" style="margin-right: 6px;">🎙️ Voice</span>` : "";
  const editBtnHtml = isVoice
    ? `<button type="button" class="edit-transcript-btn" title="Edit transcript in input box and re-ask">✏️ Edit</button>`
    : "";
  msg.innerHTML = `<div class="message-body">${prefix}${escapeHtml(text)}${editBtnHtml}</div>`;
  if (isVoice) {
    const editBtn = msg.querySelector(".edit-transcript-btn");
    if (editBtn) {
      editBtn.addEventListener("click", () => {
        DOM.queryInput.value = text;
        DOM.queryInput.dispatchEvent(new Event("input"));
        DOM.queryInput.focus();
        DOM.queryInput.setSelectionRange(DOM.queryInput.value.length, DOM.queryInput.value.length);
      });
    }
  }
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(false);
}

function appendAssistantMessage(data) {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message";
  msg._citations = data.citations || [];
  msg._timings = data.timings;
  msg._faithfulnessScore = data.faithfulness_score;
  msg._claimAudit = data.claim_audit;

  const formattedContent = formatAnswerWithCitations(data.answer);

  let validationWarningHtml = "";
  if (data.hallucinated_markers && data.hallucinated_markers.length) {
    validationWarningHtml = `<span class="meta-pill" style="color: var(--accent-rose); border: 1px solid var(--accent-rose);">⚠️ Invalid Citations: [${escapeHtml(data.hallucinated_markers.join(", "))}]</span>`;
  } else if (data.citation_validation_failed && (!data.citations || !data.citations.length)) {
    validationWarningHtml = `<span class="meta-pill" style="color: var(--accent-amber);">⚠️ No Sources Cited</span>`;
  }

  let faithfulnessPill = "";
  if (data.faithfulness_score !== null && data.faithfulness_score !== undefined) {
    const pct = Math.round(data.faithfulness_score * 100);
    const isWarn = data.claim_audit && data.claim_audit.unsupported_claims > 0;
    const pillColor = isWarn ? "#f59e0b" : "#10b981";
    faithfulnessPill = `<span class="meta-pill" style="color: ${pillColor}; border: 1px solid ${pillColor};">Grounded: ${pct}%</span>`;
  }

  let audioBarHtml = "";
  if (data.audio_url) {
    const fullAudioUrl = `${STATE.apiBase}${data.audio_url}`;
    const durLabel = data.audio_duration_seconds ? `${data.audio_duration_seconds.toFixed(1)}s` : "";
    audioBarHtml = `
      <div class="voice-audio-bar">
        <span class="voice-badge">🔊 ${escapeHtml(data.tts_model || "Voice")} ${durLabel}</span>
        <audio controls preload="auto" src="${escapeHtml(fullAudioUrl)}"></audio>
      </div>
    `;
  } else if (data.tts_error) {
    audioBarHtml = `
      <div class="voice-audio-bar" style="border-color: rgba(245, 158, 11, 0.35);">
        <span class="voice-badge" style="color: var(--accent-amber);">🔊 Browser Voice Fallback</span>
        <span style="font-size: 11px; color: var(--text-muted);">Backend TTS unavailable (${escapeHtml(data.tts_error)})</span>
      </div>
    `;
  }

  const retrievalPills = (data.candidate_count !== undefined && data.reranked_count !== undefined)
    ? `<span class="meta-pill">Candidates: ${data.candidate_count}</span><span class="meta-pill">Reranked: ${data.reranked_count}</span>`
    : `<span class="meta-pill">STT: ${escapeHtml(data.stt_model || "whisper")}</span>`;

  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">${data.audio_url || data.stt_model ? "🎙️" : "🤖"}</span>
      <span class="agent-name">${escapeHtml(data.model_name || "ResolveOps")}</span>
      <span class="message-time">${data.timings.total_ms.toFixed(0)}ms</span>
    </div>
    <div class="message-body">${formattedContent}${audioBarHtml}</div>
    <div class="message-meta-bar">
      ${retrievalPills}
      <span class="meta-pill">Citations: ${data.citations ? data.citations.length : 0}</span>
      ${faithfulnessPill}
      ${validationWarningHtml}
    </div>
  `;

  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
  attachCitationListeners(msg);
  return msg.querySelector("audio");
}

function appendLoadingMessage(statusText = "Searching docs ➔ Reranking ➔ Generating grounded answer...") {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message loading-state";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">⚡</span>
      <span class="agent-name">ResolveOps Pipeline</span>
    </div>
    <div class="message-body">
      <div class="processing-card">
        <div class="processing-header-row">
          <span class="voice-spinner"></span>
          <span>${escapeHtml(statusText)}</span>
        </div>
        <div class="processing-bar-track"><div class="processing-bar-fill"></div></div>
      </div>
    </div>
  `;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
  return msg;
}

function appendNoticeMessage(noticeText) {
  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">🎙️</span>
      <span class="agent-name" style="color: var(--accent-amber)">Voice Notice</span>
    </div>
    <div class="message-body" style="border-color: rgba(245, 158, 11, 0.45);">
      <p>${escapeHtml(noticeText)}</p>
    </div>
  `;
  DOM.chatHistory.appendChild(msg);
  scrollChatToBottom(true);
}

function appendErrorMessage(errText) {
  const isNetworkErr = /failed to fetch|networkerror|load failed/i.test(errText || "");
  const friendlyText = isNetworkErr
    ? "Unable to reach the assistant right now. Please try again in a moment."
    : (errText || "Something went wrong while generating your answer. Please try again.");

  const msg = document.createElement("div");
  msg.className = "chat-message assistant-message";
  msg.innerHTML = `
    <div class="message-header">
      <span class="agent-avatar">⚠️</span>
      <span class="agent-name" style="color: var(--accent-rose)">Unable to Respond</span>
    </div>
    <div class="message-body" style="border-color: rgba(244, 63, 94, 0.45);">
      <p>${escapeHtml(friendlyText)}</p>
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

if (DOM.voiceBtn) {
  DOM.voiceBtn.addEventListener("click", () => {
    toggleVoiceRecording();
  });
}

if (DOM.stopSpeakingBtn) {
  DOM.stopSpeakingBtn.addEventListener("click", () => {
    if (STATE.isRecording) {
      toggleVoiceRecording();
    } else {
      stopActiveSpeech();
    }
  });
}

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
  if (DOM.voiceReviewToggle) {
    DOM.voiceReviewToggle.checked = STATE.reviewVoiceTranscript;
  }
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
  if (DOM.voiceReviewToggle) {
    STATE.reviewVoiceTranscript = DOM.voiceReviewToggle.checked;
  }

  localStorage.setItem("resolveops_api_url", STATE.apiBase);
  localStorage.setItem("resolveops_candidate_pool", STATE.candidatePool.toString());
  localStorage.setItem("resolveops_top_k", STATE.topK.toString());
  localStorage.setItem("resolveops_voice_review", STATE.reviewVoiceTranscript ? "true" : "false");

  DOM.settingsModal.classList.add("hidden");
  checkBackendHealth();
});

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

checkBackendHealth();
setInterval(checkBackendHealth, 8000);
DOM.queryInput.focus();


const state = {
  conversation: [],
  loading: false,
  toastTimer: null,
};

const elements = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#questionInput"),
  role: document.querySelector("#roleSelect"),
  department: document.querySelector("#departmentSelect"),
  asOfDate: document.querySelector("#asOfDate"),
  send: document.querySelector("#sendButton"),
  welcome: document.querySelector("#welcomeState"),
  messages: document.querySelector("#messageList"),
  scroll: document.querySelector("#conversationScroll"),
  count: document.querySelector("#characterCount"),
  strategy: document.querySelector("#strategyMetric"),
  cache: document.querySelector("#cacheMetric"),
  corrections: document.querySelector("#correctionMetric"),
  latency: document.querySelector("#latencyMetric"),
  badge: document.querySelector("#verificationBadge"),
  citationCount: document.querySelector("#citationCount"),
  citationList: document.querySelector("#citationList"),
  citationEmpty: document.querySelector("#citationEmpty"),
  sourceCount: document.querySelector("#sourceCount"),
  sourceList: document.querySelector("#sourceList"),
  sourceEmpty: document.querySelector("#sourceEmpty"),
  verifierReason: document.querySelector("#verifierReason"),
  serviceStatus: document.querySelector("#serviceStatus"),
  sideStatusDot: document.querySelector("#sideStatusDot"),
  sideStatusText: document.querySelector("#sideStatusText"),
  sidebar: document.querySelector("#sidebar"),
  menuButton: document.querySelector("#menuButton"),
  newSession: document.querySelector("#newSessionButton"),
  toast: document.querySelector("#toast"),
};

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswer(value = "") {
  const safe = escapeHtml(value).trim();
  if (!safe) return "No answer was returned.";
  return safe
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${paragraph.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function showToast(message, type = "default") {
  clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast visible ${type === "error" ? "error" : ""}`;
  state.toastTimer = setTimeout(() => {
    elements.toast.className = "toast";
  }, 4200);
}

function setLoading(loading) {
  state.loading = loading;
  elements.send.disabled = loading;
  elements.input.disabled = loading;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.scroll.scrollTo({ top: elements.scroll.scrollHeight, behavior: "smooth" });
  });
}

function createUserMessage(question) {
  const node = document.createElement("article");
  node.className = "message user";
  node.innerHTML = `
    <div class="message-body">${formatAnswer(question)}</div>
    <div class="avatar" aria-hidden="true">YOU</div>
  `;
  return node;
}

function createLoadingMessage() {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.id = "loadingMessage";
  node.innerHTML = `
    <div class="avatar" aria-hidden="true">TS</div>
    <div class="message-body">
      <div class="message-label">TakaSecure intelligence</div>
      <div class="loading-card">
        <div class="loading-dots"><span></span><span></span><span></span></div>
        <span>Retrieving authorized policy evidence and verifying the response…</span>
      </div>
    </div>
  `;
  return node;
}

function createAssistantMessage(response) {
  const citations = Array.isArray(response.citations) ? response.citations : [];
  const passed = Boolean(response.verification?.passed);
  const tool = response.requires_tool && response.tool_name
    ? `<span class="tool-pill">Approved tool: ${escapeHtml(response.tool_name)}</span>`
    : "";
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar" aria-hidden="true">TS</div>
    <div class="message-body">
      <div class="message-label">TakaSecure intelligence</div>
      <div class="answer-card ${passed ? "" : "unverified"}">
        ${formatAnswer(response.answer)}
        ${tool}
        ${citations.length ? `<div class="inline-citations">${citations.map((item) => `<span class="citation-pill">${escapeHtml(item)}</span>`).join("")}</div>` : ""}
      </div>
    </div>
  `;
  return node;
}

function showConversation() {
  elements.welcome.hidden = true;
  elements.messages.classList.add("visible");
}

function renderDiagnostics(response, elapsedMs) {
  const citations = Array.isArray(response.citations) ? response.citations : [];
  const sources = Array.isArray(response.sources) ? response.sources : [];
  const passed = Boolean(response.verification?.passed);

  elements.strategy.textContent = response.retrieval_strategy === "multi_query" ? "Multi-query" : "Direct";
  const cacheLabels = {
    hit: "Hit",
    miss: "Miss",
    disabled: "Disabled",
    error: "Bypassed",
    bypass: "Bypassed",
  };
  elements.cache.textContent = cacheLabels[response.cache_status]
    || (response.cache_hit ? "Hit" : "Miss");
  elements.corrections.textContent = String(response.correction_attempts ?? 0);
  elements.latency.textContent = elapsedMs < 1000 ? `${elapsedMs} ms` : `${(elapsedMs / 1000).toFixed(1)} s`;

  elements.badge.textContent = response.access_denied
    ? "Access denied"
    : (passed ? "Verified" : "Review required");
  elements.badge.className = `verified-badge ${response.access_denied ? "denied" : (passed ? "passed" : "failed")}`;
  const verifierReason = response.verification?.reasoning || "No verifier explanation was returned.";
  elements.verifierReason.textContent = verifierReason.length > 900
    ? `${verifierReason.slice(0, 897)}…`
    : verifierReason;

  elements.citationCount.textContent = String(citations.length);
  elements.citationEmpty.hidden = citations.length > 0;
  elements.citationList.innerHTML = citations
    .map((citation) => `<span class="citation-pill">${escapeHtml(citation)}</span>`)
    .join("");

  elements.sourceCount.textContent = String(sources.length);
  elements.sourceEmpty.hidden = sources.length > 0;
  elements.sourceList.innerHTML = "";
  sources.forEach((source, index) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const sourceName = source.source ? String(source.source).split(/[\\/]/).pop() : "Policy corpus";
    const page = source.page ?? "—";
    card.innerHTML = `
      <button class="source-button" type="button" aria-expanded="false">
        <strong>${escapeHtml(sourceName || `Evidence ${index + 1}`)}</strong>
        <span>Page ${escapeHtml(page)}</span>
      </button>
      <p class="source-content">${escapeHtml(source.preview || "No preview available.")}</p>
    `;
    const button = card.querySelector("button");
    button.addEventListener("click", () => {
      const open = card.classList.toggle("open");
      button.setAttribute("aria-expanded", String(open));
    });
    elements.sourceList.appendChild(card);
  });
}

function resetDiagnostics() {
  elements.strategy.textContent = "—";
  elements.cache.textContent = "—";
  elements.corrections.textContent = "—";
  elements.latency.textContent = "—";
  elements.badge.textContent = "Awaiting query";
  elements.badge.className = "verified-badge neutral";
  elements.citationCount.textContent = "0";
  elements.citationList.innerHTML = "";
  elements.citationEmpty.hidden = false;
  elements.sourceCount.textContent = "0";
  elements.sourceList.innerHTML = "";
  elements.sourceEmpty.hidden = false;
  elements.verifierReason.textContent = "The grounded-answer verifier has not run yet.";
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
  elements.count.textContent = `${elements.input.value.length} / 4000`;
}

async function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || state.loading) return;
  if (!elements.role.value) {
    showToast("Select an authorized role before asking a policy question.", "error");
    elements.role.focus();
    return;
  }

  showConversation();
  elements.messages.appendChild(createUserMessage(trimmed));
  const loadingNode = createLoadingMessage();
  elements.messages.appendChild(loadingNode);
  scrollToBottom();
  setLoading(true);

  const payload = {
    question: trimmed,
    conversation_context: state.conversation.slice(-10),
    user_role: elements.role.value,
    department: elements.department.value || null,
    response_format: "text",
    as_of_date: elements.asOfDate.value || null,
  };

  const start = performance.now();
  try {
    const result = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!result.ok) {
      let detail = `Request failed with status ${result.status}`;
      try {
        const errorBody = await result.json();
        detail = errorBody.detail || detail;
      } catch (_) {
        // The fallback status message is sufficient.
      }
      throw new Error(detail);
    }

    const response = await result.json();
    const elapsed = Math.round(performance.now() - start);
    loadingNode.replaceWith(createAssistantMessage(response));
    renderDiagnostics(response, elapsed);
    state.conversation.push(`User: ${trimmed}`, `Assistant: ${response.answer}`);

    if (response.escalation_required) {
      showToast("The response requires human escalation.");
    }
  } catch (error) {
    loadingNode.remove();
    const failure = {
      answer: "The policy service could not complete this request. Confirm that FastAPI and the vLLM endpoint are running, then try again.",
      citations: [],
      verification: { passed: false },
    };
    elements.messages.appendChild(createAssistantMessage(failure));
    showToast(error.message || "Unable to reach the policy service.", "error");
  } finally {
    setLoading(false);
    elements.input.value = "";
    resizeInput();
    elements.input.focus();
    scrollToBottom();
  }
}

async function checkHealth() {
  const statusDot = elements.serviceStatus.querySelector(".status-pulse");
  const statusText = elements.serviceStatus.querySelector("span:last-child");
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const response = await fetch("/health", { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error("Service unavailable");
    statusDot.className = "status-pulse online";
    statusText.textContent = "Service online";
    elements.sideStatusDot.className = "status-pulse online";
    elements.sideStatusText.textContent = "Service online";
  } catch (_) {
    statusDot.className = "status-pulse offline";
    statusText.textContent = "Service offline";
    elements.sideStatusDot.className = "status-pulse offline";
    elements.sideStatusText.textContent = "Service offline";
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(elements.input.value);
});

elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.role.value = button.dataset.role || "employee";
    elements.department.value = button.dataset.department || "";
    submitQuestion(button.dataset.question || "");
  });
});

elements.newSession.addEventListener("click", () => {
  state.conversation = [];
  elements.messages.innerHTML = "";
  elements.messages.classList.remove("visible");
  elements.welcome.hidden = false;
  resetDiagnostics();
  showToast("New policy session started.");
});

elements.menuButton.addEventListener("click", () => {
  const open = elements.sidebar.classList.toggle("open");
  elements.menuButton.setAttribute("aria-expanded", String(open));
});

document.addEventListener("click", (event) => {
  if (window.innerWidth > 960) return;
  if (!elements.sidebar.contains(event.target) && !elements.menuButton.contains(event.target)) {
    elements.sidebar.classList.remove("open");
    elements.menuButton.setAttribute("aria-expanded", "false");
  }
});

resizeInput();
resetDiagnostics();
checkHealth();

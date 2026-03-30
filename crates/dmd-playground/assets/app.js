const MONACO_VERSION = "0.52.2";
const MONACO_BASE_URL = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min`;
const MONACO_VS_URL = `${MONACO_BASE_URL}/vs`;
const LANGUAGE_ID = "dmdDsl";
const THEME_ID = "dmdWorkbench";
const MONACO_TIMEOUT_MS = 8000;

const defaultSource = `params M, N, K;
array A[M, K];
array B[K, N];
array C[M, N];

for i in 0 .. M {
  for j in 0 .. N {
    for k in 0 .. K {
      read C[i, j];
      read A[i, k];
      read B[k, j];
      write C[i, j];
    }
  }
}`;

const form = document.getElementById("job-form");
const runButton = document.getElementById("run-button");
const taskList = document.getElementById("task-list");
const resultPanel = document.getElementById("result-panel");
const statusChip = document.getElementById("status-chip");
const editorHost = document.getElementById("source-editor");
const fallbackSource = document.getElementById("source-fallback");
const editorStatus = document.getElementById("editor-status");
const jobs = new Map();

let editor = null;
let monacoPromise = null;

fallbackSource.value = defaultSource;
renderTaskList();
void initializeEditor();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Submitting", "running");
  runButton.disabled = true;
  try {
    const response = await fetch("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: getSourceValue(),
        block_size: Number(document.getElementById("block-size").value),
        num_sets: Number(document.getElementById("num-sets").value),
        max_operations: Number(document.getElementById("max-operations").value),
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const task = await response.json();
    jobs.set(task.task_id, { status: task.status });
    renderTaskList();
    void pollTask(task.task_id).catch((error) => {
      renderFailure(error.message);
      setStatus("Failed", "failed");
    });
  } catch (error) {
    setStatus("Failed", "failed");
    renderFailure(error.message);
  } finally {
    runButton.disabled = false;
  }
});

taskList.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }

  const row = target.closest("[data-task-id]");
  if (!(row instanceof HTMLElement)) {
    return;
  }

  const snapshot = jobs.get(row.dataset.taskId);
  if (!snapshot) {
    return;
  }

  if (snapshot.status === "completed" && snapshot.result) {
    renderReport(snapshot.result);
    setStatus("Completed", "completed");
    return;
  }

  if (snapshot.status === "failed") {
    renderFailure(snapshot.error || "analysis failed");
    setStatus("Failed", "failed");
  }
});

async function initializeEditor() {
  setEditorStatus("Loading Monaco", "loading");

  try {
    const monaco = await loadMonaco();
    registerDslLanguage(monaco);
    defineTheme(monaco);

    editor = monaco.editor.create(editorHost, {
      value: fallbackSource.value,
      language: LANGUAGE_ID,
      theme: THEME_ID,
      automaticLayout: true,
      minimap: { enabled: false },
      smoothScrolling: true,
      scrollBeyondLastLine: false,
      lineNumbersMinChars: 3,
      lineHeight: 24,
      fontSize: 15,
      fontFamily: "IBM Plex Mono, SFMono-Regular, Consolas, monospace",
      padding: { top: 18, bottom: 18 },
      tabSize: 2,
      insertSpaces: true,
      bracketPairColorization: { enabled: true },
      guides: {
        indentation: true,
        bracketPairs: true,
      },
      renderWhitespace: "selection",
      wordWrap: "on",
      cursorBlinking: "smooth",
      contextmenu: false,
      quickSuggestions: {
        other: true,
        comments: false,
        strings: false,
      },
      suggest: {
        showWords: false,
        snippetsPreventQuickSuggestions: false,
      },
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => form.requestSubmit());
    editor.onDidChangeModelContent(() => {
      fallbackSource.value = editor.getValue();
    });

    fallbackSource.classList.add("is-hidden");
    editorHost.classList.add("is-ready");
    editor.focus();
    setEditorStatus("Monaco Ready", "ready");
  } catch (error) {
    console.error("Failed to initialize Monaco", error);
    editorHost.classList.remove("is-ready");
    fallbackSource.classList.remove("is-hidden");
    setEditorStatus("Fallback Editor", "fallback");
  }
}

async function loadMonaco() {
  if (window.monaco?.editor) {
    return window.monaco;
  }

  if (monacoPromise) {
    return monacoPromise;
  }

  if (typeof window.require !== "function") {
    throw new Error("Monaco loader is unavailable");
  }

  window.MonacoEnvironment = {
    getWorkerUrl() {
      const workerSource = `
self.MonacoEnvironment = { baseUrl: "${MONACO_BASE_URL}/" };
importScripts("${MONACO_VS_URL}/base/worker/workerMain.js");
`;
      return `data:text/javascript;charset=utf-8,${encodeURIComponent(workerSource)}`;
    },
  };

  monacoPromise = new Promise((resolve, reject) => {
    window.require.config({
      paths: { vs: MONACO_VS_URL },
    });

    const timeoutId = window.setTimeout(() => {
      reject(new Error("Timed out while loading Monaco"));
    }, MONACO_TIMEOUT_MS);

    window.require(
      ["vs/editor/editor.main"],
      () => {
        window.clearTimeout(timeoutId);
        resolve(window.monaco);
      },
      (error) => {
        window.clearTimeout(timeoutId);
        reject(error);
      },
    );
  }).catch((error) => {
    monacoPromise = null;
    throw error;
  });

  return monacoPromise;
}

function registerDslLanguage(monaco) {
  if (monaco.languages.getLanguages().some((language) => language.id === LANGUAGE_ID)) {
    return;
  }

  monaco.languages.register({
    id: LANGUAGE_ID,
    aliases: ["DMD DSL", "Loop-Tree DSL"],
  });

  monaco.languages.setLanguageConfiguration(LANGUAGE_ID, {
    comments: {
      lineComment: "//",
    },
    brackets: [
      ["{", "}"],
      ["[", "]"],
      ["(", ")"],
    ],
    autoClosingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
    ],
    surroundingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
    ],
    indentationRules: {
      increaseIndentPattern: /^.*\{\s*$/,
      decreaseIndentPattern: /^\s*\}/,
    },
  });

  monaco.languages.setMonarchTokensProvider(LANGUAGE_ID, {
    defaultToken: "",
    tokenPostfix: ".dmd",
    tokenizer: {
      root: [
        [/\bparams\b/, { token: "keyword", next: "@paramsList" }],
        [/\barray\b/, { token: "keyword", next: "@arrayName" }],
        [/\bfor\b/, { token: "keyword", next: "@loopVar" }],
        [/\b(read|write|update)\b/, { token: "keyword", next: "@accessName" }],
        [/\b(if|else|in|step)\b/, "keyword"],
        [/[A-Za-z_][A-Za-z0-9_]*/, "identifier"],
        [/[0-9]+/, "number"],
        { include: "@whitespace" },
        [/[{}()[\]]/, "@brackets"],
        [/[;,]/, "delimiter"],
        [/\.\./, "operator.range"],
        [/&&|<=|>=|==|<|>|\+|-|\*|\//, "operator"],
      ],
      whitespace: [
        [/[ \t\r\n]+/, "white"],
        [/\/\/.*$/, "comment"],
      ],
      paramsList: [
        { include: "@whitespace" },
        [/[A-Za-z_][A-Za-z0-9_]*/, "parameter"],
        [/,/, "delimiter"],
        [/;/, { token: "delimiter", next: "@pop" }],
      ],
      arrayName: [
        { include: "@whitespace" },
        [/[A-Za-z_][A-Za-z0-9_]*/, { token: "type.identifier", next: "@pop" }],
      ],
      loopVar: [
        { include: "@whitespace" },
        [/[A-Za-z_][A-Za-z0-9_]*/, { token: "variable.parameter", next: "@pop" }],
      ],
      accessName: [
        { include: "@whitespace" },
        [/[A-Za-z_][A-Za-z0-9_]*/, { token: "type.identifier", next: "@pop" }],
      ],
    },
  });

  monaco.languages.registerCompletionItemProvider(LANGUAGE_ID, {
    provideCompletionItems(model, position) {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };

      return {
        suggestions: [
          keywordSuggestion(monaco, "params", "Declare symbolic parameters.", range),
          keywordSuggestion(monaco, "array", "Declare an array shape.", range),
          keywordSuggestion(monaco, "for", "Start an affine loop.", range),
          keywordSuggestion(monaco, "if", "Guard a subtree with affine comparisons.", range),
          keywordSuggestion(monaco, "read", "Read from an array element.", range),
          keywordSuggestion(monaco, "write", "Write an array element.", range),
          keywordSuggestion(monaco, "update", "Read-modify-write an array element.", range),
          snippetSuggestion(
            monaco,
            "for-loop",
            ["for ${1:i} in ${2:0} .. ${3:N} {", "  $0", "}"].join("\n"),
            "Affine loop skeleton.",
            range,
          ),
          snippetSuggestion(
            monaco,
            "if-guard",
            ["if ${1:i} < ${2:N} {", "  $0", "} else {", "  ", "}"].join("\n"),
            "Conditional loop-tree branch.",
            range,
          ),
          snippetSuggestion(
            monaco,
            "array-decl",
            "array ${1:A}[${2:N}, ${3:M}];",
            "Array declaration.",
            range,
          ),
          snippetSuggestion(
            monaco,
            "access",
            "${1|read,write,update|} ${2:A}[${3:i}, ${4:j}];",
            "Array access statement.",
            range,
          ),
        ],
      };
    },
  });
}

function keywordSuggestion(monaco, label, detail, range) {
  return {
    label,
    kind: monaco.languages.CompletionItemKind.Keyword,
    insertText: label,
    detail,
    range,
  };
}

function snippetSuggestion(monaco, label, insertText, detail, range) {
  return {
    label,
    kind: monaco.languages.CompletionItemKind.Snippet,
    insertText,
    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
    detail,
    range,
  };
}

function defineTheme(monaco) {
  monaco.editor.defineTheme(THEME_ID, {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "F7B267", fontStyle: "bold" },
      { token: "parameter", foreground: "F6D8AE" },
      { token: "type.identifier", foreground: "8BD3C7" },
      { token: "variable.parameter", foreground: "B4E1A5" },
      { token: "number", foreground: "F08A5D" },
      { token: "comment", foreground: "7D8A84", fontStyle: "italic" },
      { token: "operator", foreground: "F3B673" },
      { token: "operator.range", foreground: "FFD29D" },
      { token: "delimiter", foreground: "CDBBA6" },
    ],
    colors: {
      "editor.background": "#121614",
      "editor.foreground": "#F5EBDD",
      "editorLineNumber.foreground": "#7D8A84",
      "editorLineNumber.activeForeground": "#F7B267",
      "editorCursor.foreground": "#F7B267",
      "editor.selectionBackground": "#9A341244",
      "editor.inactiveSelectionBackground": "#9A341222",
      "editorIndentGuide.background1": "#2B332F",
      "editorIndentGuide.activeBackground1": "#4E5B55",
      "editorLineHighlightBackground": "#FFFFFF08",
      "editorBracketMatch.background": "#FFFFFF0A",
      "editorBracketMatch.border": "#F7B26766",
      "editorSuggestWidget.background": "#1A201D",
      "editorSuggestWidget.foreground": "#F5EBDD",
      "editorSuggestWidget.selectedBackground": "#9A341233",
      "editorWidget.background": "#171C19",
      "scrollbarSlider.background": "#F7B26733",
      "scrollbarSlider.hoverBackground": "#F7B26755",
      "scrollbarSlider.activeBackground": "#F7B26777",
    },
  });
}

function getSourceValue() {
  return editor ? editor.getValue() : fallbackSource.value;
}

async function pollTask(taskId) {
  setStatus("Running", "running");
  while (true) {
    const response = await fetch(`/api/tasks/${taskId}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }
    const snapshot = await response.json();
    jobs.set(taskId, snapshot);
    renderTaskList();
    if (snapshot.status === "completed") {
      renderReport(snapshot.result);
      setStatus("Completed", "completed");
      return;
    }
    if (snapshot.status === "failed") {
      renderFailure(snapshot.error || "analysis failed");
      setStatus("Failed", "failed");
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 400));
  }
}

function renderTaskList() {
  const rows = [...jobs.entries()].reverse().map(([taskId, snapshot]) => {
    const taskState = snapshot.status || "queued";
    return `<button type="button" class="task-row" data-state="${taskState}" data-task-id="${taskId}">
      <span class="task-meta">
        <span class="task-id">${taskId.slice(0, 8)}</span>
        <span class="task-caption">${taskCaption(taskState)}</span>
      </span>
      <span class="task-state">${taskState}</span>
    </button>`;
  });
  taskList.innerHTML = rows.join("") || `<p class="muted">No tasks yet. The next submission will appear here.</p>`;
}

function taskCaption(status) {
  switch (status) {
    case "completed":
      return "Result ready";
    case "failed":
      return "Solver reported an error";
    case "running":
      return "Analysis in progress";
    default:
      return "Queued for execution";
  }
}

function renderFailure(message) {
  resultPanel.className = "result-panel";
  resultPanel.innerHTML = `<div class="callout error">
    <p class="label">Analysis Error</p>
    <p>${escapeHtml(message)}</p>
  </div>`;
}

function renderReport(report) {
  const riRows = renderDistribution(report.ri_distribution);
  const rdRows = renderDistribution(report.rd_distribution);
  const terms = report.dmd_terms
    .map((term) => `<tr>
      <td>${escapeHtml(term.domain_plain)}</td>
      <td>${renderMath(term.multiplicity_latex, term.multiplicity_plain, { className: "math-cell" })}</td>
      <td>${renderMath(term.reuse_distance_latex, term.reuse_distance_plain, { className: "math-cell" })}</td>
      <td>${renderMath(term.term_latex, term.term_plain, { className: "math-cell" })}</td>
    </tr>`)
    .join("");
  const notes = report.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");

  resultPanel.className = "result-panel";
  resultPanel.innerHTML = `
    <section class="formula-band">
      <div class="formula-card">
        <p class="label">DMD Formula</p>
        ${renderMath(report.dmd_formula_latex, report.dmd_formula_plain, { displayMode: true, className: "math-block" })}
        <details class="formula-plain">
          <summary>Plain text</summary>
          <pre>${escapeHtml(report.dmd_formula_plain)}</pre>
        </details>
      </div>
      <div class="formula-card">
        <p class="label">Counts</p>
        <dl>
          <div><dt>Total</dt><dd>${renderMath(report.total_accesses_latex, report.total_accesses_plain, { className: "math-inline" })}</dd></div>
          <div><dt>Warm</dt><dd>${renderMath(report.warm_accesses_latex, report.warm_accesses_plain, { className: "math-inline" })}</dd></div>
          <div><dt>Compulsory</dt><dd>${renderMath(report.compulsory_accesses_latex, report.compulsory_accesses_plain, { className: "math-inline" })}</dd></div>
        </dl>
      </div>
    </section>
    <section class="detail-grid">
      <div class="detail-card">
        <h3>RI Distribution</h3>
        ${riRows}
      </div>
      <div class="detail-card">
        <h3>RD Distribution</h3>
        ${rdRows}
      </div>
    </section>
    <section class="detail-card">
      <h3>DMD Terms</h3>
      <table>
        <thead><tr><th>Domain</th><th>Multiplicity</th><th>RD</th><th>Term</th></tr></thead>
        <tbody>${terms || `<tr><td colspan="4">No warm terms</td></tr>`}</tbody>
      </table>
    </section>
    <section class="detail-card">
      <h3>Notes</h3>
      <ul>${notes}</ul>
    </section>
  `;
  hydrateMath(resultPanel);
  scheduleMathHydration(resultPanel);
}

function renderDistribution(entries) {
  if (!entries.length) {
    return `<p class="muted">No entries.</p>`;
  }
  return entries
    .map((entry) => {
      const rows = entry.regions
        .map((region) => `<tr><td>${escapeHtml(region.domain_plain)}</td><td>${renderMath(region.count_latex, region.count_plain, { className: "math-cell" })}</td></tr>`)
        .join("");
      return `
        <div class="distribution-entry">
          <p class="label">Value</p>
          ${renderMath(entry.value_latex, entry.value_plain, { className: "math-inline math-inline-strong" })}
          <table>
            <thead><tr><th>Domain</th><th>Count</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    })
    .join("");
}

function setStatus(text, tone = "idle") {
  statusChip.textContent = text;
  statusChip.dataset.tone = tone;
}

function setEditorStatus(text, tone) {
  editorStatus.textContent = text;
  editorStatus.dataset.tone = tone;
}

function renderMath(latex, plain, options = {}) {
  const {
    className = "math-inline",
    displayMode = false,
  } = options;

  return `<div class="${className}" data-katex="${escapeAttribute(latex)}" data-plain="${escapeAttribute(plain)}" data-display-mode="${displayMode ? "true" : "false"}">${escapeHtml(plain)}</div>`;
}

function hydrateMath(root) {
  const nodes = root.querySelectorAll("[data-katex]");
  for (const node of nodes) {
    const latex = node.dataset.katex || "";
    const displayMode = node.dataset.displayMode === "true";
    const plain = node.dataset.plain || node.textContent || "";
    const fallback = fallbackMathText(latex, plain);

    if (typeof window.katex?.render !== "function") {
      node.classList.add("math-fallback");
      node.textContent = fallback;
      continue;
    }

    try {
      window.katex.render(latex, node, {
        displayMode,
        strict: "ignore",
        throwOnError: false,
      });
      node.classList.add("math-ready");
    } catch (error) {
      console.error("Failed to render KaTeX formula", error);
      node.classList.add("math-fallback");
      node.textContent = fallback;
    }
  }
}

function scheduleMathHydration(root, attempt = 0) {
  if (typeof window.katex?.render === "function") {
    hydrateMath(root);
    return;
  }

  if (attempt >= 10) {
    return;
  }

  window.setTimeout(() => {
    scheduleMathHydration(root, attempt + 1);
  }, 300);
}

function fallbackMathText(latex, plain) {
  if (!latex) {
    return plain;
  }

  return latex
    .replaceAll("\\left", "")
    .replaceAll("\\right", "")
    .replaceAll("\\cdot", "·")
    .replace(/\\sqrt\{([^{}]+)\}/g, "√($1)")
    .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1)/($2)");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttribute(value) {
  return escapeHtml(value)
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

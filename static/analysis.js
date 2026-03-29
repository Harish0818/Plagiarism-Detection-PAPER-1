document.addEventListener("DOMContentLoaded", () => {
    const loader = document.getElementById("pageLoader");
    const loaderText = document.getElementById("loaderText");
    const revealNodes = document.querySelectorAll(".reveal");

    const loadMessages = [
        "Initializing integrity engines...",
        "Aligning semantic manifolds...",
        "Preparing secure workspace..."
    ];

    let msgIndex = 0;
    const loadTimer = setInterval(() => {
        msgIndex = (msgIndex + 1) % loadMessages.length;
        loaderText.textContent = loadMessages[msgIndex];
    }, 600);

    setTimeout(() => {
        clearInterval(loadTimer);
        loader.classList.remove("active");
        revealNodes.forEach((node, i) => {
            setTimeout(() => node.classList.add("visible"), i * 100);
        });
    }, 1500);

    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const fileNameDisplay = document.getElementById("fileNameDisplay");
    const loadingHud = document.getElementById("loadingHud");
    const statusText = document.getElementById("statusText");
    const results = document.getElementById("results");

    const integrityScore = document.getElementById("integrityScore");
    const aiScore = document.getElementById("aiScore");
    const plagScore = document.getElementById("plagScore");
    const citeScore = document.getElementById("citeScore");
    const downloadMenuBtn = document.getElementById("downloadMenuBtn");
    const downloadMenu = document.getElementById("downloadMenu");
    const downloadItems = document.querySelectorAll(".dropdown-item");

    const tabButtons = document.querySelectorAll(".tab-btn");
    const workspaceTabs = document.querySelectorAll(".workspace-tab");
    const workspacePanels = document.querySelectorAll("[data-workspace-panel]");
    const workspaceTriggers = document.querySelectorAll("[data-workspace-target]");
    const panels = {
        summary: document.getElementById("tab-summary"),
        ai: document.getElementById("tab-ai"),
        plagiarism: document.getElementById("tab-plagiarism"),
        citations: document.getElementById("tab-citations"),
        system: document.getElementById("tab-system")
    };
    let selectedFile = null;
    let latestAnalysisData = null;

    const setWorkspace = (name) => {
        workspaceTabs.forEach((tab) => {
            const isActive = tab.dataset.workspaceTab === name;
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });
        workspacePanels.forEach((panel) => {
            panel.classList.toggle("active", panel.dataset.workspacePanel === name);
        });
    };

    const openWorkspaceFromTrigger = (trigger) => {
        const workspaceName = trigger.dataset.workspaceTarget || "home";
        const scrollId = trigger.dataset.scrollTo || "";
        setWorkspace(workspaceName);
        if (scrollId) {
            const section = document.getElementById(scrollId);
            if (section) {
                section.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }
    };

    workspaceTabs.forEach((tab) => {
        tab.addEventListener("click", () => setWorkspace(tab.dataset.workspaceTab || "home"));
    });

    workspaceTriggers.forEach((trigger) => {
        trigger.addEventListener("click", (event) => {
            event.preventDefault();
            openWorkspaceFromTrigger(trigger);
        });
    });

    if (window.location.hash === "#analyzer") {
        setWorkspace("analyzer");
        const analyzerSection = document.getElementById("analyzer");
        if (analyzerSection) {
            setTimeout(() => analyzerSection.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
        }
    } else {
        setWorkspace("home");
    }

    const setTab = (name) => {
        tabButtons.forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.tab === name);
        });
        Object.entries(panels).forEach(([key, panel]) => {
            panel.classList.toggle("active", key === name);
        });
    };

    tabButtons.forEach((btn) => {
        btn.addEventListener("click", () => setTab(btn.dataset.tab));
    });

    uploadBtn.addEventListener("click", () => fileInput.click());
    downloadMenuBtn.addEventListener("click", (event) => {
        if (downloadMenuBtn.disabled) return;
        event.stopPropagation();
        downloadMenu.classList.toggle("open");
    });
    downloadItems.forEach((item) => {
        item.addEventListener("click", () => {
            const reportType = item.dataset.reportType || "full";
            downloadMenu.classList.remove("open");
            downloadReport(reportType);
        });
    });
    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!downloadMenu.contains(target) && target !== downloadMenuBtn) {
            downloadMenu.classList.remove("open");
        }
        const isPanelButton = target instanceof Element && target.classList.contains("panel-menu-btn");
        document.querySelectorAll(".panel-dropdown-menu.open").forEach((menu) => {
            if (!menu.contains(target) && !isPanelButton) {
                menu.classList.remove("open");
            }
        });
    });

    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) return;
        const btn = event.target.closest(".panel-menu-btn");
        if (btn) {
            const menu = btn.parentElement.querySelector(".panel-dropdown-menu");
            if (menu) {
                document.querySelectorAll(".panel-dropdown-menu.open").forEach((m) => {
                    if (m !== menu) m.classList.remove("open");
                });
                menu.classList.toggle("open");
            }
            return;
        }
        const actionBtn = event.target.closest(".panel-dropdown-item");
        if (actionBtn) {
            const action = actionBtn.dataset.action;
            const reportType = actionBtn.dataset.reportType || "full";
            const panelId = actionBtn.dataset.panelId || "";
            if (action === "download") {
                downloadReport(reportType);
            } else if (action === "copy") {
                copyPanelText(panelId);
            }
            const parentMenu = actionBtn.closest(".panel-dropdown-menu");
            if (parentMenu) parentMenu.classList.remove("open");
        }
    });

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) {
            return;
        }
        selectedFile = file;
        const isPdf = file.name.toLowerCase().endsWith(".pdf");
        downloadMenuBtn.disabled = !isPdf;
        latestAnalysisData = null;

        fileNameDisplay.textContent = `Selected: ${file.name}`;
        loadingHud.classList.remove("hidden");
        results.classList.add("hidden");
        statusText.textContent = "Extracting text and validating document...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            statusText.textContent = "Running AI, plagiarism, and citation analysis...";
            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Analysis failed");
            }

            latestAnalysisData = data;
            renderResults(data);
            results.classList.remove("hidden");
            setTab("summary");
        } catch (err) {
            statusText.textContent = `Error: ${err.message}`;
        } finally {
            loadingHud.classList.add("hidden");
        }
    });

    function renderResults(data) {
        const analyses = data.analyses || {};
        const ai = analyses.ai_detection || data.ai || {};
        const plag = analyses.plagiarism || {};
        const cite = analyses.citations || {};
        const evalData = data.evaluation || {};
        const overview = data.overview || {};

        const aiPct = Number(ai.confidence ?? ai.ai_percentage ?? 0);
        const plagiarismResults = Array.isArray(plag.results)
            ? plag.results
            : (Array.isArray(data.plagiarism) ? data.plagiarism : []);
        const citeResults = cite.results || data.citations || {};
        const fraudulent = Array.isArray(citeResults.fraudulent) ? citeResults.fraudulent : [];
        const valid = Array.isArray(citeResults.valid) ? citeResults.valid : [];
        const edges = Array.isArray(data.citation_edges)
            ? data.citation_edges
            : (Array.isArray(cite.edges) ? cite.edges : []);

        integrityScore.textContent = `${(Number(evalData.overall_accuracy || 0) * 100).toFixed(1)}%`;
        aiScore.textContent = `${(aiPct * 100).toFixed(1)}%`;
        plagScore.textContent = String(plag.match_count ?? plagiarismResults.length);
        citeScore.textContent = String(cite.fraud_count ?? fraudulent.length);

        panels.summary.innerHTML = panelShell("Summary", `
            <div class="list reveal-list">
                <div class="item"><strong>File:</strong> ${escapeHtml(data.filename || "Uploaded document")}</div>
                <div class="item"><strong>Docs Processed:</strong> ${escapeHtml(String(overview.docs_processed ?? 1))}</div>
                <div class="item"><strong>Mean AI Content:</strong> ${(Number(overview.mean_ai_content || aiPct) * 100).toFixed(1)}%</div>
                <div class="item"><strong>Semantic Matches:</strong> ${escapeHtml(String(overview.semantic_matches ?? plagiarismResults.length))}</div>
                <div class="item"><strong>Citation Contradictions:</strong> ${escapeHtml(String(overview.citation_contradictions ?? fraudulent.length))}</div>
            </div>
        `, "summary", "full");

        const aiSegments = Array.isArray(ai.segments) ? ai.segments : (Array.isArray(ai.ai_segments) ? ai.ai_segments : []);
        const aiStyl = ai.stylometrics || {};
        panels.ai.innerHTML = panelShell("AI Analysis", `
            <div class="analysis-stat-grid">
                <div class="analysis-stat-card tone-ai">
                    <span>AI Content</span>
                    <strong>${(aiPct * 100).toFixed(1)}%</strong>
                </div>
                <div class="analysis-stat-card tone-mid">
                    <span>Mean Risk</span>
                    <strong>${(Number(ai.mean_risk || data.ai?.mean_risk || 0) * 100).toFixed(1)}%</strong>
                </div>
                <div class="analysis-stat-card tone-soft">
                    <span>Flagged Segments</span>
                    <strong>${aiSegments.length}</strong>
                </div>
            </div>
            <div class="analysis-grid">
                <div class="analysis-section-card">
                    <h4>Stylometric Signals</h4>
                    <div class="kv-grid">
                        <div class="kv-row"><span>Avg Sentence Length</span><strong>${formatNum(aiStyl.avg_sentence_length)}</strong></div>
                        <div class="kv-row"><span>Burstiness</span><strong>${formatNum(aiStyl.burstiness)}</strong></div>
                        <div class="kv-row"><span>Lexical Diversity</span><strong>${formatNum(aiStyl.lexical_diversity)}</strong></div>
                    </div>
                </div>
                <div class="analysis-section-card">
                    <h4>Stylometric Graph</h4>
                    <div id="stylometricsGraph" class="graph-box"></div>
                </div>
            </div>
            <div class="analysis-section-card">
                <h4>Flagged Segment Previews</h4>
                ${aiSegments.length ? aiSegments.slice(0, 6).map((seg, idx) => `
                    <details class="dropdown-detail">
                        <summary>Segment ${idx + 1}</summary>
                        <div class="preview-box">${escapeHtml(seg.slice(0, 1200))}</div>
                    </details>
                `).join("") : "<div class='empty-state'>No AI segments highlighted.</div>"}
            </div>
        `, "ai", "ai");
        renderStylometricsGraph(aiStyl);

        panels.plagiarism.innerHTML = plagiarismResults.length
            ? panelShell("Plagiarism", `
                <div class="analysis-stat-grid">
                    <div class="analysis-stat-card tone-warn">
                        <span>Total Matches</span>
                        <strong>${plagiarismResults.length}</strong>
                    </div>
                    <div class="analysis-stat-card tone-mid">
                        <span>Average Similarity</span>
                        <strong>${formatNum(average(plagiarismResults.map((x) => x.score)))}</strong>
                    </div>
                </div>
                <div class="list reveal-list">${plagiarismResults.slice(0, 20).map((m) => `
                <div class="item plagiarism-item">
                    <div class="match-head">
                        <strong>${escapeHtml(m.source || "Unknown Source")}</strong>
                        <span class="match-badge">Similarity ${formatNum(m.score)}</span>
                    </div>
                    <div class="match-meta">Type: ${escapeHtml(m.match_type || "N/A")}</div>
                    <div class="match-snippet">${escapeHtml((m.text_chunk || "").slice(0, 260))}</div>
                    <details class="dropdown-detail plagiarism-item-detail">
                        <summary>Show details</summary>
                        <div class="preview-box">
                            <strong>Source:</strong> ${escapeHtml(m.source || "Unknown Source")}<br>
                            <strong>Similarity:</strong> ${formatNum(m.score)}<br>
                            <strong>Match Type:</strong> ${escapeHtml(m.match_type || "N/A")}<br><br>
                            ${escapeHtml(m.text_chunk || "No snippet available.")}
                        </div>
                    </details>
                </div>
            `).join("")}</div>`, "plagiarism", "plagiarism")
            : panelShell("Plagiarism", "<div class='item'>No significant semantic matches found.</div>", "plagiarism", "plagiarism");

        panels.citations.innerHTML = panelShell("Citations", `
            <div class="analysis-stat-grid">
                <div class="analysis-stat-card tone-ok">
                    <span>Valid Citations</span>
                    <strong>${valid.length}</strong>
                </div>
                <div class="analysis-stat-card tone-danger">
                    <span>Contradictions</span>
                    <strong>${fraudulent.length}</strong>
                </div>
                <div class="analysis-stat-card tone-soft">
                    <span>Invalid / Irrelevant</span>
                    <strong>${(Array.isArray(citeResults.invalid) ? citeResults.invalid.length : 0) + (Array.isArray(citeResults.irrelevant) ? citeResults.irrelevant.length : 0)}</strong>
                </div>
            </div>
            <div class="analysis-grid">
                <div class="analysis-section-card">
                    <h4>Citation Network Graph</h4>
                    <div id="citationGraph" class="graph-box"></div>
                </div>
                <div class="analysis-section-card">
                    <h4>Fraud / Contradiction Alerts</h4>
                ${fraudulent.slice(0, 8).map((f) => `
                    <div class="item compact-item"><strong>${escapeHtml(f.citation || "Unknown citation")}</strong><br>
                    Status: ${escapeHtml(f.status || "Contradiction")} | Confidence: ${formatNum(f.confidence)}</div>
                `).join("")}
                ${fraudulent.length ? "" : "<div class='empty-state'>No citation contradictions detected.</div>"}
                </div>
            </div>
        `, "citations", "citations");
        renderCitationGraph(edges);

        const evidenceRows = Array.isArray(data.evidence_table) ? data.evidence_table : [];
        const evidenceHtml = evidenceRows.length
            ? evidenceRows.map((r) => `
                <div class="item">
                    <strong>${escapeHtml(r.Metric || "Metric")}</strong><br>
                    Value: ${escapeHtml(r.Value || "-")}<br>
                    Confidence: ${escapeHtml(r.Confidence || "-")}<br>
                    Interpretation: ${escapeHtml(r.Interpretation || "-")}
                </div>
            `).join("")
            : "<div class='item'>No evidence table generated.</div>";

        panels.system.innerHTML = panelShell("System Accuracy", `
            <div class="list reveal-list">
                <div class="item"><strong>Integrity Score:</strong> ${(Number(evalData.overall_accuracy || 0) * 100).toFixed(1)}%</div>
                <div class="item"><strong>F1-Score:</strong> ${formatNum(evalData.f1_score)}</div>
                <div class="item"><strong>Precision:</strong> ${formatNum(evalData.precision)}</div>
                <div class="item"><strong>Recall:</strong> ${formatNum(evalData.recall)}</div>
                <div class="item"><strong>AI Confidence:</strong> ${(Number(evalData.ai_confidence || 0) * 100).toFixed(1)}%</div>
                <div class="item"><strong>Plagiarism Confidence:</strong> ${(Number(evalData.plagiarism_confidence || 0) * 100).toFixed(1)}%</div>
                <div class="item"><strong>Citation Confidence:</strong> ${(Number(evalData.citation_confidence || 0) * 100).toFixed(1)}%</div>
                ${evidenceHtml}
            </div>
        `, "system", "full");
    }

    async function downloadReport(reportType) {
        if (!selectedFile) {
            statusText.textContent = "Select and analyze a file first.";
            return;
        }
        if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
            statusText.textContent = "Report download works for PDF input only.";
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("report_type", reportType);
        if (latestAnalysisData) {
            formData.append("analysis_payload", JSON.stringify({
                raw_text: latestAnalysisData.raw_text || "",
                analyses: latestAnalysisData.analyses || {},
            }));
        }

        try {
            statusText.textContent = `Generating ${reportType.toUpperCase()} report...`;
            loadingHud.classList.remove("hidden");
            const response = await fetch("/generate-report", { method: "POST", body: formData });
            if (!response.ok) {
                let err = "Report generation failed";
                try {
                    const data = await response.json();
                    err = data.error || err;
                } catch (_) {
                    // ignore parse errors
                }
                throw new Error(err);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            const baseName = selectedFile.name.replace(/\.pdf$/i, "");
            a.href = url;
            a.download = `${reportType}_report_${baseName}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            statusText.textContent = `${reportType.toUpperCase()} report downloaded.`;
        } catch (err) {
            statusText.textContent = `Error: ${err.message}`;
        } finally {
            loadingHud.classList.add("hidden");
        }
    }

    function renderCitationGraph(edges) {
        const graphNode = document.getElementById("citationGraph");
        if (!graphNode) {
            return;
        }
        if (!window.Plotly) {
            graphNode.innerHTML = "<p>Graph library unavailable.</p>";
            return;
        }
        if (!edges || !edges.length) {
            graphNode.innerHTML = "<p>No citation links available for network graph.</p>";
            return;
        }

        const nodeMap = new Map();
        edges.forEach((edge) => {
            const source = String(edge[0] || "Document");
            const target = String(edge[1] || "Citation");
            if (!nodeMap.has(source)) nodeMap.set(source, { label: source, type: "doc" });
            if (!nodeMap.has(target)) nodeMap.set(target, { label: target, type: "cite" });
        });

        const nodes = [...nodeMap.values()];
        const count = nodes.length;
        const angleStep = (Math.PI * 2) / Math.max(count, 1);
        const positions = nodes.map((_, i) => ({
            x: Math.cos(i * angleStep),
            y: Math.sin(i * angleStep),
        }));

        const indexByLabel = new Map(nodes.map((n, i) => [n.label, i]));
        const edgeX = [];
        const edgeY = [];
        edges.forEach((edge) => {
            const si = indexByLabel.get(String(edge[0] || "Document"));
            const ti = indexByLabel.get(String(edge[1] || "Citation"));
            if (si === undefined || ti === undefined) return;
            edgeX.push(positions[si].x, positions[ti].x, null);
            edgeY.push(positions[si].y, positions[ti].y, null);
        });

        const edgeTrace = {
            x: edgeX,
            y: edgeY,
            mode: "lines",
            line: { width: 1.2, color: "#9fc6d3" },
            hoverinfo: "skip",
            type: "scatter",
        };
        const nodeTrace = {
            x: positions.map((p) => p.x),
            y: positions.map((p) => p.y),
            mode: "markers+text",
            text: nodes.map((n) => n.label.length > 34 ? `${n.label.slice(0, 34)}...` : n.label),
            textposition: "top center",
            hovertext: nodes.map((n) => n.label),
            hoverinfo: "text",
            marker: {
                size: nodes.map((n) => n.type === "doc" ? 22 : 14),
                color: nodes.map((n) => n.type === "doc" ? "#0e4a5a" : "#14b8a6"),
                line: { color: "#ffffff", width: 1.5 },
            },
            type: "scatter",
        };

        const layout = {
            margin: { l: 20, r: 20, t: 20, b: 20 },
            xaxis: { visible: false },
            yaxis: { visible: false },
            paper_bgcolor: "#f7fbfd",
            plot_bgcolor: "#f7fbfd",
            showlegend: false,
        };

        Plotly.newPlot(graphNode, [edgeTrace, nodeTrace], layout, { responsive: true, displayModeBar: false });
    }

    function panelShell(title, contentHtml, panelId, reportType) {
        return `
            <div class="panel-head">
                <h3>${escapeHtml(title)}</h3>
                <div class="panel-dropdown">
                    <button type="button" class="panel-menu-btn">Options</button>
                    <div class="panel-dropdown-menu">
                        <button type="button" class="panel-dropdown-item" data-action="copy" data-panel-id="${escapeHtml(panelId)}">Copy section</button>
                        <button type="button" class="panel-dropdown-item" data-action="download" data-report-type="${escapeHtml(reportType)}">Download report</button>
                    </div>
                </div>
            </div>
            ${contentHtml}
        `;
    }

    async function copyPanelText(panelId) {
        const panel = document.getElementById(`tab-${panelId}`);
        if (!panel) return;
        const text = panel.innerText || "";
        if (!text.trim()) return;
        try {
            await navigator.clipboard.writeText(text);
            statusText.textContent = "Section content copied.";
        } catch (_) {
            statusText.textContent = "Clipboard access denied.";
        }
    }

    function renderStylometricsGraph(styl) {
        const graphNode = document.getElementById("stylometricsGraph");
        if (!graphNode) return;
        if (!window.Plotly) {
            graphNode.innerHTML = "<p>Graph library unavailable.</p>";
            return;
        }

        const avgSentenceLength = Number(styl.avg_sentence_length || 0);
        const burstiness = Number(styl.burstiness || 0);
        const lexicalDiversity = Number(styl.lexical_diversity || 0);
        const labels = ["Avg Sentence Length", "Burstiness", "Lexical Diversity"];
        const values = [avgSentenceLength, burstiness, lexicalDiversity];

        const trace = {
            x: labels,
            y: values,
            type: "bar",
            marker: {
                color: ["#0ea5e9", "#14b8a6", "#f59e0b"],
                line: { color: "#ffffff", width: 1.2 },
            },
            text: values.map((v) => Number(v).toFixed(3)),
            textposition: "outside",
            hovertemplate: "%{x}: %{y:.3f}<extra></extra>",
        };
        const layout = {
            margin: { l: 40, r: 20, t: 14, b: 52 },
            paper_bgcolor: "#f7fbfd",
            plot_bgcolor: "#f7fbfd",
            yaxis: { gridcolor: "#dce8ee", zerolinecolor: "#dce8ee", rangemode: "tozero" },
            xaxis: { tickfont: { size: 11 } },
            showlegend: false,
        };
        Plotly.newPlot(graphNode, [trace], layout, { responsive: true, displayModeBar: false });
    }

    function formatNum(value) {
        const n = Number(value || 0);
        return Number.isFinite(n) ? n.toFixed(3) : "0.000";
    }

    function average(values) {
        const nums = values.map((v) => Number(v)).filter((v) => Number.isFinite(v));
        if (!nums.length) return 0;
        return nums.reduce((acc, v) => acc + v, 0) / nums.length;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }
});

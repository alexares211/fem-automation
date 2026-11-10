/* Fixed-size orientation workflow. The imported boundary conditions remain in the solver deck. */
(() => {
  "use strict";
  const el = id => document.getElementById(id);
  const run = el("orientationRun"), stop = el("orientationStop"), progress = el("orientationProgress");
  const method = el("orientationMethod");
  let jobId = null, polling = false, timer = null, running = false;
  let disabledBeforeRun = new Map();
  const number = value => Number(value).toLocaleString(undefined, {maximumSignificantDigits: 6});
  const sequence = values => values.map(number).join(" / ");
  const table = (headers, rows) => `<table class="resultsTable"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  function describeMethod() {
    const equal = method.value === "equal_stress";
    el("equalStressSettings").hidden = !equal;
    el("methodExplanation").textContent = equal
      ? "Maximize stiffness while keeping every ply within its stress limit and balancing the peak normalized stress utilization between plies."
      : "Maximize stiffness while keeping every ply within its maximum-stress limit. This orientation-only adaptation cannot force all plies to full utilization when material sizing is fixed.";
  }
  function fixedSizing() {
    const tokens = el("layupAngles").value.split(/[\/,]/).map(s => s.trim()).filter(Boolean);
    const count = tokens.length * (el("layupSym").checked ? 2 : 1);
    const thickness = Number(el("layupThk").value);
    el("fixedPlyCount").value = count;
    el("fixedPlyThickness").value = thickness > 0 ? number(thickness) : "Enter thickness above";
    el("fixedTotalThickness").value = thickness > 0 && count ? number(count*thickness) : "—";
  }
  function setRunning(value) {
    running = value;
    if (value) {
      disabledBeforeRun = new Map();
      document.querySelectorAll("input, select, button").forEach(control => {
        disabledBeforeRun.set(control, control.disabled);
        control.disabled = true;
      });
      stop.disabled = false;
    } else {
      disabledBeforeRun.forEach((disabled, control) => { control.disabled = disabled; });
      disabledBeforeRun.clear();
      run.disabled = false;
      stop.disabled = true;
    }
  }
  function drawChart(history) {
    const canvas = el("orientationChart");
    const feasible = history.filter(row => row.constraints_met);
    if (!feasible.length) { canvas.style.display = "none"; return; }
    canvas.style.display = "block";
    const context = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
    const baseline = history[0].objective_value;
    let best = Infinity;
    const values = feasible.map(row => { best = Math.min(best, row.objective_value); return [row.evaluation, baseline/best]; });
    const low = Math.min(1, ...values.map(v => v[1])), high = Math.max(1.01, ...values.map(v => v[1]));
    const x = n => 65+(w-90)*(n-1)/Math.max(1, history.length-1);
    const y = v => h-40-(h-80)*(v-low)/Math.max(0.01, high-low);
    context.clearRect(0, 0, w, h);
    context.font = "12px Segoe UI"; context.fillStyle = "#a9b9cb"; context.strokeStyle = "#2c3d50";
    for (let i=0; i<=4; i++) {
      const v = low+(high-low)*i/4;
      context.beginPath(); context.moveTo(60, y(v)); context.lineTo(w-20, y(v)); context.stroke();
      context.fillText(`${v.toFixed(2)}×`, 8, y(v)+4);
    }
    context.fillText("Best feasible stiffness / baseline stiffness", 65, 18);
    context.fillText("FEM evaluation", w/2-30, h-8);
    context.fillText("1", 60, h-22); context.fillText(String(history.length), w-35, h-22);
    context.strokeStyle = "#12c2e9"; context.lineWidth = 2; context.beginPath();
    values.forEach(([n,v],i) => { if (i === 0) context.moveTo(x(n),y(v)); else context.lineTo(x(n),y(v)); });
    context.stroke();
  }
  function renderHistory(history) {
    el("orientationHistory").innerHTML = table(["Solve", "Source", "Ply angles (°)", "2U", "Peak utilization", "Ply CV", "Feasible"], history.map(row => [
      row.evaluation, escapeHtml(row.source), escapeHtml(sequence(row.angles)), number(row.compliance),
      number(row.max_utilization), number(row.utilization_cv), row.constraints_met ? "Yes" : "No"
    ]));
    drawChart(history);
  }
  function renderResult(result) {
    if (!result || !result.best) { el("orientationSummary").textContent = "Stopped before a completed FEM evaluation."; return; }
    const best = result.best, baseline = result.baseline;
    const force = best.mode === "force";
    const label = force ? "Compliance (N·mm), lower is stiffer" : "2 × strain energy (N·mm), higher is stiffer";
    el("orientationSummary").innerHTML = `<p style="color:var(--${result.constraints_met ? "ok" : "warn"});"><strong>${result.constraints_met ? "A feasible design was found." : "No feasible design was found within this search."}</strong> ${escapeHtml(result.termination)}.</p>` +
      table(["Quantity", "Initial layup", "Best design"], [
        ["Ply angles (°)", baseline ? escapeHtml(sequence(baseline.angles)) : "—", escapeHtml(sequence(best.angles))],
        [label, baseline ? number(baseline.compliance) : "—", number(best.compliance)],
        ["Maximum nodal displacement (mm)", baseline ? number(baseline.max_displacement) : "—", number(best.max_displacement)],
        ["In-plane maximum-stress utilization", baseline ? number(baseline.max_utilization) : "—", number(best.max_utilization)],
        ["Variation between plies (CV)", baseline ? number(baseline.utilization_cv) : "—", number(best.utilization_cv)],
        ["Stiffness relative to initial design", "1.000×", result.stiffness_ratio === null ? "—" : `${number(result.stiffness_ratio)}× (${number(result.stiffness_gain_percent)}%)`],
        ["Fixed sizing", `${result.ply_count} plies`, `${result.ply_count} plies; ${number(result.total_thickness)} mm total`]
      ]) + `<p class="hint">${escapeHtml(result.output_folder)}. A local search result, with ${result.evaluations} real FEM evaluations; no global optimum is claimed.${result.method === "fully_stressed" && !result.fully_stressed_achieved ? " Full utilization was not reached with fixed material sizing." : ""}</p>`;
    el("orientationPlyResults").innerHTML = table(["Ply", "Angle (°)", "Peak |σ11|", "Peak |σ22|", "Peak |τ12|", "Utilization"], best.plies.map(p => [
      p.ply, number(best.angles[p.ply-1]), number(p.max_abs_S11), number(p.max_abs_S22), number(p.max_abs_S12), number(p.utilization)
    ]));
    const files = el("orientationFiles"); files.replaceChildren();
    const open = document.createElement("button"); open.textContent = "Open results in PrePoMax";
    open.addEventListener("click", async () => {
      open.disabled = true; progress.textContent = "Opening and verifying the optimized result in PrePoMax…";
      try { const r = await fetch(`/orientation/open/${jobId}`, {method:"POST"}); const data = await r.json(); if (!data.ok) throw new Error(data.error); progress.textContent = "Optimized results sent to PrePoMax."; }
      catch (error) { progress.textContent = error.message; }
      finally { open.disabled = false; }
    }); files.appendChild(open);
    const apply = document.createElement("button"); apply.className = "ghost"; apply.textContent = "Use optimized layup";
    apply.addEventListener("click", () => {
      el("layupAngles").value = best.free_angles.join("/");
      el("layupSym").checked = result.symmetry_enforced;
      el("layupThk").value = result.ply_thickness;
      fixedSizing();
    }); files.appendChild(apply);
    ["optimized.inp", "optimized.frd", "report.json", "history.csv"].forEach(name => {
      const link = document.createElement("a"); link.href = `/orientation/artifact/${jobId}/${name}`;
      link.textContent = name; link.style.marginTop = "18px"; files.appendChild(link);
    });
  }
  async function poll() {
    if (polling || !jobId) return;
    polling = true;
    try {
      const response = await fetch(`/orientation/status/${jobId}`);
      const data = await response.json();
      if (response.status === 404) {
        clearInterval(timer); timer = null; setRunning(false);
        progress.textContent = "This job is no longer active in the server. Completed solver files remain in the selected output folder.";
        return;
      }
      if (!data.ok) throw new Error(data.error);
      renderHistory(data.history);
      progress.textContent = `${data.history.length} FEM evaluations completed (budget ${data.settings.max_evaluations}). ${data.done ? "Run finished." : "Solving…"}`;
      if (data.phase === "opening_results") {
        stop.disabled = true;
        progress.textContent = `${data.history.length} FEM evaluations completed. Opening and verifying results in PrePoMax…`;
      }
      if (data.done) {
        clearInterval(timer); timer = null; setRunning(false);
        if (data.error) { progress.textContent = `Optimization failed: ${data.error}. Completed evaluation files: ${data.output_folder}`; }
        else { renderResult(data.result); progress.textContent += ` ${data.viewer || data.viewer_error || ""}`; }
      }
    } catch (error) {
      // A transient network failure must not lose the job or enable a duplicate start.
      progress.textContent = `Connection interrupted: ${error.message}. Retrying…`;
    } finally { polling = false; }
  }
  run.addEventListener("click", async () => {
    if (running) return;
    const model = window.femCollectModel();
    const strengths = Object.fromEntries(["S11t", "S11c", "S22t", "S22c", "S21s"].map(key => [key, el(`puck_${key}`).value]));
    const payload = {...model, strengths, layup: el("layupAngles").value,
      output_folder: el("run_output_folder").value, ccx_exe: el("run_ccx_exe").value,
      method: method.value, max_evaluations: Number(el("optBudget").value), angle_step: Number(el("optAngleStep").value),
      stress_limit: Number(el("optStressLimit").value), equal_tolerance: Number(el("optEqualTolerance").value), auto_open: el("optAutoOpen").checked};
    ["orientationSummary", "orientationPlyResults", "orientationFiles", "orientationHistory"].forEach(id => el(id).replaceChildren());
    el("orientationChart").style.display = "none";
    setRunning(true); progress.textContent = "Validating the imported model and starting the baseline solve…";
    try {
      const response = await fetch("/orientation/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await response.json(); if (!data.ok) throw new Error(data.error);
      jobId = data.job_id; await poll(); if (running) timer = setInterval(poll, 1200);
    } catch (error) { setRunning(false); progress.textContent = error.message; }
  });
  stop.addEventListener("click", async () => {
    if (!jobId) return;
    stop.disabled = true;
    try { const r = await fetch(`/orientation/stop/${jobId}`, {method:"POST"}); const data = await r.json(); if (!data.ok) throw new Error(data.error); progress.textContent = "Stopping the solver and keeping the best completed design…"; }
    catch (error) { stop.disabled = false; progress.textContent = error.message; }
  });
  el("loadBeamBtn").addEventListener("click", async () => {
    const button = el("loadBeamBtn"); button.disabled = true;
    try {
      const response = await fetch("/examples/cantilever"); const data = await response.json(); if (!data.ok) throw new Error(data.error);
      const part = data.suggested_elset || data.elsets[0];
      const support = data.nsets.find(name => /FIXED|CLAMP/i.test(name)) || null;
      window.femImportState = {path:data.import_inp_path, elset:part, rfNset:support};
      const picker = el("inpElsetPick"), rfPicker = el("inpRfNsetPick");
      picker.replaceChildren(...data.elsets.map(name => new Option(name, name, name === part, name === part)));
      rfPicker.replaceChildren(new Option("(skip)", "(skip)"), ...data.nsets.map(name => new Option(name, name, name === support, name === support)));
      const update = () => { window.femImportState = {path:data.import_inp_path, elset:picker.value, rfNset:rfPicker.value === "(skip)" ? null : rfPicker.value}; };
      picker.onchange = update; rfPicker.onchange = update;
      el("inpPickers").style.display = "flex";
      el("inpMsg").textContent = `Loaded PrePoMax export: ${data.import_inp_path}`;
      el("layupAngles").value = "45/-45/90/0"; el("layupThk").value = "0.25"; el("layupSym").checked = true;
      fixedSizing();
    } catch (error) { el("inpMsg").textContent = error.message; }
    finally { button.disabled = false; }
  });
  method.addEventListener("change", describeMethod);
  ["layupAngles", "layupThk", "layupSym"].forEach(id => el(id).addEventListener("input", fixedSizing));
  fixedSizing(); describeMethod();
})();

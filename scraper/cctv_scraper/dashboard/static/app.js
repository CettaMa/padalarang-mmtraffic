(function () {
  "use strict";

  var REFRESH_MS = 20000;
  var COVERAGE_DAYS = 7;

  var HEALTH = {
    online: { label: "Online", tone: "ok" },
    stale: { label: "Segmen basi", tone: "warn" },
    restarting: { label: "Restart", tone: "warn" },
    offline: { label: "Offline", tone: "bad" },
    expired_url: { label: "URL kedaluwarsa", tone: "crit" },
    unknown: { label: "Belum ada data", tone: "idle" }
  };

  var $ = function (id) { return document.getElementById(id); };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function pct(value, digits) {
    if (value === null || value === undefined) return "—";
    return (value * 100).toFixed(digits === undefined ? 1 : digits);
  }

  function bytes(value) {
    if (!value) return "0 B";
    var units = ["B", "KB", "MB", "GB", "TB"];
    var index = 0;
    var size = value;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return size.toFixed(size >= 100 || index === 0 ? 0 : 1) + " " + units[index];
  }

  function age(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    if (seconds < 60) return seconds + " dtk";
    if (seconds < 3600) return Math.floor(seconds / 60) + " mnt";
    if (seconds < 86400) return Math.floor(seconds / 3600) + " jam";
    return Math.floor(seconds / 86400) + " hari";
  }

  function duration(seconds) {
    if (!seconds) return "0j";
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.round((seconds % 3600) / 60);
    return hours + "j " + minutes + "m";
  }

  function clock(iso) {
    if (!iso) return "—";
    return String(iso).replace("T", " ").slice(0, 19);
  }

  /** Ramp from the neutral surface to the mist accent, matching the CSS legend. */
  function heatColor(ratio) {
    var alpha = 0.06 + Math.min(1, Math.max(0, ratio)) * 0.88;
    return "rgba(183, 206, 206, " + alpha.toFixed(3) + ")";
  }

  // ------------------------------------------------------------------ fetch

  function api(path) {
    return fetch(path, { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) throw new Error(path + " -> HTTP " + response.status);
        return response.json();
      });
  }

  function setLive(ok, text) {
    var live = $("live");
    live.classList.toggle("is-error", !ok);
    $("live-label").textContent = text;
  }

  // ------------------------------------------------------------------ hero

  function renderHero(data) {
    var summary = data.summary;
    $("hero-uptime").textContent = pct(summary.uptime_24h);
    $("hero-total").textContent = summary.total;
    $("hero-clock").textContent = clock(data.generated_at);

    var chips = $("hero-chips");
    clear(chips);
    ["online", "stale", "restarting", "offline", "expired_url", "unknown"].forEach(function (key) {
      var count = summary[key] || 0;
      if (!count) return;
      var chip = el("span", "chip");
      chip.setAttribute("data-health", key);
      chip.appendChild(el("i", "dot"));
      chip.appendChild(el("span", null, count + " " + HEALTH[key].label.toLowerCase()));
      chips.appendChild(chip);
    });
    if (!chips.childNodes.length) {
      chips.appendChild(el("span", "chip chip-quiet", "belum ada kamera terdeteksi"));
    }
  }

  function kpi(label, value, unit, sub, tone) {
    var card = el("div", "kpi");
    if (tone) card.setAttribute("data-tone", tone);
    card.appendChild(el("div", "kpi-label", label));
    var valueNode = el("div", "kpi-value", value);
    if (unit) valueNode.appendChild(el("small", null, unit));
    card.appendChild(valueNode);
    card.appendChild(el("div", "kpi-sub", sub));
    return card;
  }

  function renderKpis(data) {
    var summary = data.summary;
    var disk = data.disk;
    var target = $("kpis");
    clear(target);

    var down = (summary.offline || 0) + (summary.expired_url || 0);

    target.appendChild(kpi(
      "Kamera aktif",
      summary.online + "/" + summary.total,
      null,
      down ? down + " butuh tindakan" : "semua titik merekam",
      down ? "bad" : null
    ));
    target.appendChild(kpi(
      "Coverage hari ini",
      pct(summary.coverage_today_ratio),
      "%",
      "porsi waktu berjalan yang terekam",
      summary.coverage_today_ratio < 0.9 ? "warn" : null
    ));
    target.appendChild(kpi(
      "Backlog mentah",
      summary.raw_backlog,
      ".ts",
      "segmen menunggu paket 7z harian",
      summary.raw_backlog > 2000 ? "warn" : null
    ));
    target.appendChild(kpi(
      "Arsip harian gagal",
      summary.archive_failures,
      null,
      "hari yang gagal dipaketkan ke 7z",
      summary.archive_failures > 0 ? "crit" : null
    ));
    target.appendChild(kpi(
      "Disk bebas",
      disk.free_gb,
      "GB",
      "ambang " + disk.min_free_gb + " GB",
      disk.low_space ? "crit" : null
    ));
  }

  // ---------------------------------------------------------------- cameras

  function cameraCard(camera) {
    var meta = HEALTH[camera.health] || HEALTH.unknown;
    var card = el("button", "cam");
    card.type = "button";
    card.setAttribute("data-health", camera.health);
    card.setAttribute("aria-label", "Detail " + camera.name);

    var top = el("div", "cam-top");
    var left = el("div");
    left.appendChild(el("div", "cam-name", camera.name));
    if (camera.reason) left.appendChild(el("div", "cam-reason", camera.reason));
    top.appendChild(left);

    var chip = el("span", "chip");
    chip.setAttribute("data-health", camera.health);
    chip.appendChild(el("i", "dot"));
    chip.appendChild(el("span", null, meta.label));
    top.appendChild(chip);
    card.appendChild(top);

    var metrics = el("div", "cam-metrics");
    metrics.appendChild(metricBlock("Uptime 24j", pct(camera.uptime_24h) + "%", false,
      "Porsi 24 jam terakhir dengan segmen benar-benar masuk"));
    metrics.appendChild(metricBlock("Segmen", age(camera.seconds_since_segment),
      camera.seconds_since_segment !== null && camera.seconds_since_segment > 600,
      "Usia segmen terakhir: " + (camera.last_segment_at || "belum ada")));
    metrics.appendChild(metricBlock("Backlog", camera.raw_backlog + " ts", false,
      "Segmen mentah menunggu paket 7z harian"));
    card.appendChild(metrics);

    var bar = el("div", "bar");
    var fill = el("div", "bar-fill");
    fill.style.width = Math.round((camera.uptime_24h || 0) * 100) + "%";
    bar.appendChild(fill);
    card.appendChild(bar);

    card.addEventListener("click", function () { openDrawer(camera.name); });
    return card;
  }

  function metricBlock(label, value, flagged, hint) {
    var wrap = el("div");
    if (hint) wrap.title = hint;
    wrap.appendChild(el("div", "metric-label", label));
    wrap.appendChild(el("div", "metric-value" + (flagged ? " is-flag" : ""), value));
    return wrap;
  }

  function renderCameras(data) {
    var grid = $("camera-grid");
    clear(grid);
    if (!data.cameras.length) {
      grid.appendChild(el("p", "empty", "Belum ada kamera yang menulis data ke dataset."));
      return;
    }
    data.cameras.forEach(function (camera) { grid.appendChild(cameraCard(camera)); });
  }

  // ------------------------------------------------------------ disk & rest

  function renderDisk(disk, diskSeries) {
    var target = $("disk");
    clear(target);

    var meter = el("div", "meter");
    var fill = el("div", "meter-fill" + (disk.low_space ? " is-low" : ""));
    fill.style.width = Math.min(100, disk.percent_used) + "%";
    meter.appendChild(fill);
    target.appendChild(meter);

    var list = el("dl");
    [
      ["Terpakai", bytes(disk.used_bytes) + " (" + disk.percent_used + "%)"],
      ["Bebas", bytes(disk.free_bytes)],
      ["Total", bytes(disk.total_bytes)],
      ["Ambang minimum", disk.min_free_gb + " GB"]
    ].forEach(function (pair) {
      var row = el("div", "stat-row");
      row.appendChild(el("dt", null, pair[0]));
      row.appendChild(el("dd", null, pair[1]));
      list.appendChild(row);
    });
    target.appendChild(list);

    if (disk.low_space) {
      target.appendChild(el("p", "cam-reason",
        "Di bawah ambang: DiskMonitor akan memangkas tanggal terlama."));
    }

    var series = diskSeries || [];
    target.appendChild(el("p", "section-note", "Tren penyimpanan · 72 jam terakhir."));
    if (!series.length) {
      target.appendChild(el("p", "empty", "Belum ada riwayat disk untuk dihitung."));
      return;
    }
    var strip = el("div", "strip strip--lg");
    series.forEach(function (entry) {
      var total = entry.free_gb + entry.used_gb;
      var usedPct = total > 0 ? (entry.used_gb / total) * 100 : 0;
      var bar = el("div", "strip-bar");
      bar.style.height = Math.max(6, Math.round(usedPct)) + "%";
      if (entry.free_gb < disk.min_free_gb) bar.setAttribute("data-low", "1");
      bar.title = clock(entry.at) + " · " + entry.used_gb.toFixed(1) + " GB terpakai / " +
        entry.free_gb.toFixed(1) + " GB bebas";
      strip.appendChild(bar);
    });
    target.appendChild(strip);
  }

  function renderArchives(archives) {
    var target = $("archives");
    clear(target);
    if (!archives.length) {
      target.appendChild(el("p", "empty", "Belum ada arsip harian .7z."));
      return;
    }
    archives.forEach(function (entry) {
      var item = el("div", "list-item");
      var head = el("div", "list-head");
      head.appendChild(el("span", "list-title", entry.day));
      head.appendChild(el("span", "list-time", bytes(entry.size_bytes)));
      item.appendChild(head);
      item.appendChild(el("div", "list-body", "dikemas " + clock(entry.modified_at)));
      target.appendChild(item);
    });
  }

  function incidentTone(status) {
    if (status === "expired_url_escalated") return "crit";
    if (status === "offline") return "bad";
    return "warn";
  }

  function renderIncidents(incidents) {
    var target = $("incident-list");
    clear(target);
    if (!incidents.length) {
      target.appendChild(el("p", "empty", "Tidak ada insiden pada jendela status terbaru."));
      return;
    }
    incidents.forEach(function (row) {
      var item = el("div", "list-item");
      item.setAttribute("data-tone", incidentTone(row.status));
      var head = el("div", "list-head");
      head.appendChild(el("span", "list-title", row.camera + " · " + row.status));
      head.appendChild(el("span", "list-time", row.timestamp));
      item.appendChild(head);
      item.appendChild(el("div", "list-body",
        (row.reason || "tanpa alasan") + (row.http_status ? " · HTTP " + row.http_status : "")));
      target.appendChild(item);
    });
  }

  // ---------------------------------------------------------------- heatmap

  function renderHeatmap(coverage) {
    var target = $("heatmap");
    clear(target);
    var days = coverage.days;
    var cameras = coverage.cameras;

    if (!days.length || !cameras.length) {
      target.appendChild(el("p", "empty", "Belum ada segmen mentah untuk dihitung."));
      return;
    }

    var header = el("div", "heat-row");
    header.style.setProperty("--cols", days.length);
    header.appendChild(el("div", "heat-name", ""));
    days.forEach(function (day) {
      var head = el("div", "heat-head", day.day.slice(5) + (day.archive_failed ? " !" : ""));
      if (day.archive_failed) head.title = "Paket 7z harian gagal untuk " + day.day;
      header.appendChild(head);
    });
    target.appendChild(header);

    cameras.forEach(function (camera) {
      var row = el("div", "heat-row");
      row.style.setProperty("--cols", days.length);
      var nameCell = el("div", "heat-name", camera);
      nameCell.title = camera;
      row.appendChild(nameCell);
      days.forEach(function (day) {
        var entry = day.per_camera[camera] || { ratio: 0, covered_seconds: 0 };
        var cell = el("div", "heat-cell", entry.ratio ? Math.round(entry.ratio * 100) : "");
        cell.style.background = heatColor(entry.ratio);
        cell.setAttribute("data-empty", entry.ratio ? "0" : "1");
        cell.title = camera + " · " + day.day + " · " + duration(entry.covered_seconds) +
          " terekam (" + pct(entry.ratio) + "%)";
        if (entry.ratio < 0.5) cell.style.color = "rgba(226, 226, 226, 0.62)";
        row.appendChild(cell);
      });
      target.appendChild(row);
    });
  }

  // ------------------------------------------------------------- trend charts

  function renderUptimeTrend(series) {
    var target = $("uptime-trend-body");
    clear(target);
    var points = (series && series.points) || [];
    if (!points.length) {
      target.appendChild(el("p", "empty", "Belum ada riwayat uptime untuk dihitung."));
      return;
    }
    var strip = el("div", "strip strip--lg");
    points.forEach(function (point) {
      var bar = el("div", "strip-bar");
      bar.style.height = Math.max(6, Math.round(point.ratio * 100)) + "%";
      if (point.ratio < 0.9) bar.setAttribute("data-low", "1");
      bar.title = clock(point.at) + " · " + pct(point.ratio) + "%";
      strip.appendChild(bar);
    });
    target.appendChild(strip);
  }

  function renderCoverageTrend(coverage) {
    var target = $("coverage-trend");
    clear(target);
    var days = coverage.days || [];
    if (!days.length) {
      target.appendChild(el("p", "empty", "Belum ada data coverage untuk dihitung."));
      return;
    }
    target.appendChild(el("p", "section-note", "Rasio coverage harian · 7 hari terakhir."));
    var strip = el("div", "strip strip--lg");
    days.forEach(function (day) {
      var bar = el("div", "strip-bar");
      bar.style.height = Math.max(6, Math.round(day.ratio * 100)) + "%";
      if (day.ratio < 0.9) bar.setAttribute("data-low", "1");
      if (day.archive_failed) bar.setAttribute("data-archive-failed", "1");
      bar.title = day.day + " · " + pct(day.ratio) + "%" +
        (day.archive_failed ? " · paket 7z gagal" : "");
      strip.appendChild(bar);
    });
    target.appendChild(strip);
  }

  // ----------------------------------------------------------------- drawer

  var drawer = $("drawer");
  var scrim = $("scrim");

  function closeDrawer() {
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    scrim.hidden = true;
  }

  function openDrawer(name) {
    drawer.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    scrim.hidden = false;
    $("drawer-title").textContent = name;
    $("drawer-eyebrow").textContent = "Kamera";
    var body = $("drawer-body");
    clear(body);
    body.appendChild(el("p", "empty", "Memuat detail…"));

    api("/api/cameras/" + encodeURIComponent(name) + "?days=" + COVERAGE_DAYS)
      .then(function (detail) { renderDrawer(detail); })
      .catch(function (error) {
        clear(body);
        body.appendChild(el("p", "empty", "Gagal memuat detail: " + error.message));
      });
  }

  function drawerSection(title, content) {
    var section = el("section");
    section.appendChild(el("h3", "sub-title", title));
    section.appendChild(content);
    return section;
  }

  function renderDrawer(detail) {
    var body = $("drawer-body");
    var camera = detail.camera;
    clear(body);

    $("drawer-eyebrow").textContent = (HEALTH[camera.health] || HEALTH.unknown).label;

    var grid = el("div", "drawer-grid");
    [
      ["Uptime 24 jam", pct(camera.uptime_24h) + "%"],
      ["Segmen terakhir", age(camera.seconds_since_segment)],
      ["Coverage hari ini", pct(camera.coverage_today_ratio) + "%"],
      ["Backlog .ts", String(camera.raw_backlog)]
    ].forEach(function (pair) {
      var block = el("div");
      block.appendChild(el("div", "metric-label", pair[0]));
      block.appendChild(el("div", "metric-value", pair[1]));
      grid.appendChild(block);
    });
    body.appendChild(drawerSection("Ringkasan", grid));

    if (camera.url) {
      var source = el("div");
      source.appendChild(el("div", "list-body", camera.url));
      source.appendChild(el("div", "list-body",
        "Status terakhir: " + (camera.status || "—") + " @ " + (camera.last_status_at || "—")));
      body.appendChild(drawerSection("Sumber stream (dimask)", source));
    }

    var points = (detail.uptime_series && detail.uptime_series.points) || [];
    if (points.length) {
      var strip = el("div", "strip");
      points.forEach(function (point) {
        var bar = el("div", "strip-bar");
        bar.style.height = Math.max(6, Math.round(point.ratio * 100)) + "%";
        if (point.ratio < 0.9) bar.setAttribute("data-low", "1");
        bar.title = clock(point.at) + " · " + pct(point.ratio) + "%";
        strip.appendChild(bar);
      });
      body.appendChild(drawerSection("Uptime 24 jam (dari riwayat SQLite)", strip));
    }

    var hours = el("div");
    detail.coverage.forEach(function (day) {
      var row = el("div", "hour-row");
      row.appendChild(el("div", "hour-label", day.day.slice(5) + " · " + pct(day.ratio, 0) + "%"));
      for (var hour = 0; hour < 24; hour += 1) {
        var seconds = (day.hourly && day.hourly[hour]) || 0;
        var cell = el("div", "hour-cell");
        cell.style.background = heatColor(Math.min(1, seconds / 3600));
        cell.title = day.day + " jam " + hour + ":00 · " + Math.round(seconds / 60) + " menit";
        row.appendChild(cell);
      }
      hours.appendChild(row);
    });
    body.appendChild(drawerSection("Coverage per jam (" + COVERAGE_DAYS + " hari)", hours));

    var table = el("table", "rows");
    var thead = el("thead");
    var headRow = el("tr");
    ["Waktu", "Status", "Alasan", "HTTP"].forEach(function (label) {
      headRow.appendChild(el("th", null, label));
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = el("tbody");
    detail.status_rows.slice(0, 80).forEach(function (row) {
      var tr = el("tr");
      tr.appendChild(el("td", "mono", row.timestamp));
      tr.appendChild(el("td", "mono", row.status));
      tr.appendChild(el("td", null, row.reason || "—"));
      tr.appendChild(el("td", "mono", row.http_status || "—"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    var tableWrap = el("div", "scroll-y");
    tableWrap.appendChild(table);
    body.appendChild(drawerSection("Riwayat status", tableWrap));

    var logs = el("pre", "logs", detail.log_tail.join("\n") || "Log kosong.");
    body.appendChild(drawerSection("Log kamera", logs));
  }

  scrim.addEventListener("click", closeDrawer);
  $("drawer-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeDrawer();
  });

  // ------------------------------------------------------------------- load

  function refresh() {
    return Promise.all([
      api("/api/overview"),
      api("/api/coverage?days=" + COVERAGE_DAYS),
      api("/api/history/uptime?hours=24&buckets=96"),
      api("/api/history/disk?hours=72")
    ])
      .then(function (results) {
        var overview = results[0];
        var coverage = results[1];
        var uptimeSeries = results[2];
        var diskSeries = results[3];
        renderHero(overview);
        renderKpis(overview);
        renderUptimeTrend(uptimeSeries);
        renderCameras(overview);
        renderDisk(overview.disk, diskSeries);
        renderArchives(overview.archives);
        renderIncidents(overview.incidents);
        renderCoverageTrend(coverage);
        renderHeatmap(coverage);
        $("foot-updated").textContent = "diperbarui " + clock(overview.generated_at);
        setLive(true, "live · " + REFRESH_MS / 1000 + "s");
      })
      .catch(function (error) {
        setLive(false, "gagal memuat");
        $("foot-updated").textContent = error.message;
      });
  }

  $("refresh").addEventListener("click", refresh);
  refresh();
  setInterval(function () {
    if (document.visibilityState === "visible") refresh();
  }, REFRESH_MS);
})();

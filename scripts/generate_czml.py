"""
generate_czml.py
----------------
Converts SpaceTech episode CSV data to CesiumJS CZML format.
Run: python scripts/generate_czml.py
Output: outputs/constellation.czml  +  outputs/cesium_viewer.html
"""

import json
import math
import os
import random
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
EP_CSV = "outputs/episode_data.csv"
GS_CSV = "outputs/gs_positions.csv"
OUT_CZML = "outputs/constellation.czml"
OUT_HTML = "cesium_viewer.html"

EPOCH = "2025-01-01T00:00:00Z"   # simulation start time
STEP_SECONDS = 10                 # real seconds per simulation step

# Health state colours (RGBA 0-255)
COLOR_NOMINAL  = {"rgba": [30,  240,  90, 255]}
COLOR_DEGRADED = {"rgba": [255, 210,  0,  255]}
COLOR_CRITICAL = {"rgba": [255,  50,  40, 255]}
COLOR_SAFE     = {"rgba": [230,  25, 230, 255]}
COLOR_ECLIPSE  = {"rgba": [ 80, 180, 255, 255]}
COLOR_GS       = {"rgba": [255, 215,   0, 255]}
COLOR_DEBRIS   = {"rgba": [255,  80,  30, 180]}


def health_color(row):
    if row.in_safe_mode:
        return COLOR_SAFE
    if row.fault_wheel or row.fault_thruster or row.fault_sensor:
        return COLOR_CRITICAL
    if row.battery_pct < 30 or row.fuel_pct < 15:
        return COLOR_DEGRADED
    if row.eclipse:
        return COLOR_ECLIPSE
    return COLOR_NOMINAL


def km_to_m(v):
    return v * 1000.0


def build_satellite_packet(sat_id, sat_df):
    """Build a CZML packet for one satellite with time-varying position + color."""
    n_steps = len(sat_df)

    # Time-stamped Cartesian positions (t, x, y, z) in seconds + metres
    cartesian = []
    for _, row in sat_df.iterrows():
        t = (row.step - 1) * STEP_SECONDS
        cartesian += [t,
                      km_to_m(row.ecef_x_km),
                      km_to_m(row.ecef_y_km),
                      km_to_m(row.ecef_z_km)]

    # Time-varying color intervals
    color_intervals = []
    prev_clr = None
    interval_start = 0
    for _, row in sat_df.iterrows():
        t = (row.step - 1) * STEP_SECONDS
        clr = health_color(row)
        if clr != prev_clr:
            if prev_clr is not None:
                color_intervals[-1]["interval"] = (
                    color_intervals[-1]["interval"].split("/")[0]
                    + f"/{EPOCH.split('T')[0]}T{format_t(t)}")
            color_intervals.append({
                "interval": f"{EPOCH.split('T')[0]}T{format_t(t)}/9999-12-31T24:00:00Z",
                "rgba": clr["rgba"]
            })
            prev_clr = clr

    # Fault label
    label_text = []
    for _, row in sat_df.iterrows():
        t = (row.step - 1) * STEP_SECONDS
        parts = [f"S{sat_id}"]
        if row.in_safe_mode:   parts.append("⚠ SAFE")
        if row.fault_wheel:    parts.append("🔧 WHEEL")
        if row.fault_thruster: parts.append("🔧 THRUSTER")
        if row.eclipse:        parts.append("🌑 ECLIPSE")
        label_text.append({"interval": one_step_interval(t), "string": " | ".join(parts)})

    packet = {
        "id": f"satellite_{sat_id}",
        "name": f"Satellite {sat_id}",
        "availability": f"{EPOCH}/{n_steps * STEP_SECONDS}",
        "position": {
            "epoch": EPOCH,
            "referenceFrame": "FIXED",
            "cartesian": cartesian
        },
        "point": {
            "pixelSize": {
                "epoch": EPOCH,
                "number": build_size_timeline(sat_df)
            },
            "color": {"epoch": EPOCH, "rgba": build_color_timeline(sat_df)},
            "outlineColor": {"rgba": [255, 255, 255, 180]},
            "outlineWidth": 1.5
        },
        "path": {
            "show": True,
            "width": 1.8,
            "material": {
                "polylineGlow": {
                    "color": {"epoch": EPOCH, "rgba": build_color_timeline(sat_df)},
                    "glowPower": 0.15
                }
            },
            "trailTime": STEP_SECONDS * 25,   # 25-step trail
            "leadTime": 0
        },
        "label": {
            "show": True,
            "text": f"S{sat_id}",
            "font": "bold 11px sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 220]},
            "outlineColor": {"rgba": [0, 0, 0, 200]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "pixelOffset": {"cartesian2": [12, -12]}
        },
        "description": f"<b>Satellite {sat_id}</b><br>Plane {int(sat_df.iloc[0].plane_id)}"
    }
    return packet


def build_color_timeline(sat_df):
    """Build flat [t, r, g, b, a, ...] array for CZML time-varying color."""
    result = []
    for _, row in sat_df.iterrows():
        t = (row.step - 1) * STEP_SECONDS
        c = health_color(row)["rgba"]
        result += [t] + c
    return result


def build_size_timeline(sat_df):
    """Build flat [t, size, ...] array — larger when fault active."""
    result = []
    for _, row in sat_df.iterrows():
        t = (row.step - 1) * STEP_SECONDS
        if row.fault_wheel or row.fault_thruster or row.fault_sensor:
            sz = 18
        elif row.in_safe_mode:
            sz = 14
        else:
            sz = 9
        result += [t, sz]
    return result


def one_step_interval(t):
    return f"{EPOCH.split('T')[0]}T{format_t(t)}/9999-12-31T24:00:00Z"


def format_t(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}Z"


def build_ground_station_packet(row):
    return {
        "id": f"gs_{row['name']}",
        "name": row['name'],
        "position": {
            "cartesian": [
                km_to_m(row.ecef_x_km),
                km_to_m(row.ecef_y_km),
                km_to_m(row.ecef_z_km)
            ]
        },
        "billboard": {
            "image": "data:image/svg+xml;base64," + dish_svg_b64(),
            "width": 28, "height": 28,
            "verticalOrigin": "BOTTOM"
        },
        "label": {
            "show": True,
            "text": row['name'],
            "font": "bold 12px sans-serif",
            "fillColor": {"rgba": [255, 220, 80, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 200]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "pixelOffset": {"cartesian2": [0, -32]}
        }
    }


def dish_svg_b64():
    import base64
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><circle cx="12" cy="20" r="3" fill="#FFD700"/><path d="M12 17 Q6 10 4 4 Q10 6 12 17 Z" fill="#FFD700" opacity="0.8"/><path d="M12 17 Q18 10 20 4 Q14 6 12 17 Z" fill="#FFD700" opacity="0.6"/><line x1="12" y1="17" x2="12" y2="4" stroke="#FFD700" stroke-width="1.5"/></svg>'
    return base64.b64encode(svg.encode()).decode()


def build_debris_packets(n=80):
    """Simulate a debris field at LEO altitude ~550 km."""
    random.seed(42)
    packets = []
    EARTH_R_M = 6371000.0
    for i in range(n):
        alt = EARTH_R_M + 550000 + random.gauss(0, 40000)
        az  = random.uniform(0, 2 * math.pi)
        el  = random.gauss(0, math.radians(18))
        x = alt * math.cos(el) * math.cos(az)
        y = alt * math.cos(el) * math.sin(az)
        z = alt * math.sin(el)
        packets.append({
            "id": f"debris_{i}",
            "name": f"Debris {i}",
            "position": {"cartesian": [x, y, z]},
            "point": {
                "pixelSize": 3,
                "color": {"rgba": [255, 70, 20, 160]},
                "disableDepthTestDistance": 1e9
            }
        })
    return packets


OUT_SANDCASTLE = "outputs/sandcastle.js"
REWARD_CSV = "outputs/reward_history_new.csv"


def main():
    print("Loading CSV data...")
    ep = pd.read_csv(EP_CSV)
    gs = pd.read_csv(GS_CSV)
    
    # Load reward history
    if os.path.exists(REWARD_CSV):
        rh = pd.read_csv(REWARD_CSV)
    else:
        rh = pd.read_csv("outputs/reward_history.csv")

    n_sats  = ep.sat_id.max() + 1
    n_steps = ep.step.max()
    print(f"  Satellites: {n_sats}  |  Steps: {n_steps}")

    czml = [
        {
            "id": "document",
            "name": "SpaceTech AI Constellation",
            "version": "1.0",
            "clock": {
                "interval": f"{EPOCH}/{n_steps * STEP_SECONDS}",
                "currentTime": EPOCH,
                "multiplier": 30,
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER"
            }
        }
    ]

    # Satellites
    print("  Building satellite packets...")
    for sid in range(n_sats):
        sat_df = ep[ep.sat_id == sid].sort_values("step").reset_index(drop=True)
        czml.append(build_satellite_packet(sid, sat_df))

    # Ground stations
    print("  Building ground station packets...")
    for _, row in gs.iterrows():
        czml.append(build_ground_station_packet(row))

    # Debris field
    print("  Building debris field...")
    czml.extend(build_debris_packets(80))

    # Save CZML
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_CZML, "w") as f:
        json.dump(czml, f, indent=2)
    size_kb = os.path.getsize(OUT_CZML) / 1024
    print(f"\n✓ CZML saved: {OUT_CZML}  ({size_kb:.1f} KB)")


    # Build HTML viewer
    build_html(n_steps)
    print(f"✓ Viewer saved: {OUT_HTML}")

    # Build Sandcastle JS (paste directly into sandcastle.cesium.com)
    build_sandcastle(czml, rh)
    print(f"✓ Sandcastle JS saved: {OUT_SANDCASTLE}")
    print(f"\n  → Paste contents of {OUT_SANDCASTLE} into sandcastle.cesium.com")
    print(f"\nOpen {OUT_HTML} in Chrome/Firefox to view the animation!")


def build_html(n_steps):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpaceTech AI — Satellite Constellation</title>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#080810; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  #cesiumContainer {{ width:100vw; height:100vh; }}

  #hud {{
    position:fixed; top:16px; left:16px; z-index:100;
    background:rgba(8,8,20,0.85);
    border:1px solid rgba(80,120,255,0.4);
    border-radius:12px; padding:14px 18px;
    color:#e0e8ff; min-width:260px;
    backdrop-filter:blur(8px);
  }}
  #hud h2 {{ font-size:14px; color:#7ab4ff; margin-bottom:10px; letter-spacing:1px; }}
  .hud-row {{ display:flex; justify-content:space-between; margin:4px 0; font-size:12px; }}
  .hud-val {{ font-weight:bold; color:#fff; }}
  .hud-val.warn {{ color:#ff6060; }}
  .hud-val.ok   {{ color:#40ff80; }}

  #legend {{
    position:fixed; bottom:60px; left:16px; z-index:100;
    background:rgba(8,8,20,0.85);
    border:1px solid rgba(80,120,255,0.4);
    border-radius:10px; padding:12px 16px;
    color:#e0e8ff;
    backdrop-filter:blur(8px);
  }}
  #legend h3 {{ font-size:12px; color:#7ab4ff; margin-bottom:8px; }}
  .leg-item {{ display:flex; align-items:center; gap:8px; font-size:11px; margin:3px 0; }}
  .leg-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}

  #title-bar {{
    position:fixed; top:16px; left:50%; transform:translateX(-50%);
    z-index:100; text-align:center;
    color:white; font-size:16px; font-weight:bold;
    text-shadow:0 0 20px rgba(100,150,255,0.8);
    pointer-events:none;
  }}
  #step-info {{
    position:fixed; top:50px; left:50%; transform:translateX(-50%);
    z-index:100; font-size:12px; color:#aabbff;
    pointer-events:none;
  }}
</style>
</head>
<body>
<div id="cesiumContainer"></div>

<div id="title-bar">🛰 SpaceTech AI — Walker Delta Constellation (10 Satellites)</div>
<div id="step-info">AI-controlled LEO orbital swarm | Phase-adaptive PPO Brain | 200 Training Cycles</div>

<div id="hud">
  <h2>📡 FLEET STATUS</h2>
  <div class="hud-row"><span>Simulation Time</span><span class="hud-val" id="hud-time">—</span></div>
  <div class="hud-row"><span>Active Satellites</span><span class="hud-val ok" id="hud-sats">10 / 10</span></div>
  <div class="hud-row"><span>Ground Contacts</span><span class="hud-val" id="hud-los">—</span></div>
  <div class="hud-row"><span>Debris Field</span><span class="hud-val warn">80 objects at 550 km</span></div>
  <div class="hud-row"><span>Orbit Altitude</span><span class="hud-val">~550 km LEO</span></div>
  <div class="hud-row"><span>Training Cycles</span><span class="hud-val ok">200 ✓</span></div>
</div>

<div id="legend">
  <h3>HEALTH STATUS</h3>
  <div class="leg-item"><div class="leg-dot" style="background:#1ef05a"></div> Nominal</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ffd200"></div> Degraded (low battery/fuel)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ff3228"></div> Critical fault active</div>
  <div class="leg-item"><div class="leg-dot" style="background:#e619e6"></div> Safe mode</div>
  <div class="leg-item"><div class="leg-dot" style="background:#50b4ff"></div> Eclipse</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ff4614"></div> Debris object</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ffd700; border-radius:0"></div> Ground station</div>
</div>

<script>
// ── Free CesiumJS token (get yours at cesium.com/ion — it's free) ──
Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYTU2YjlhOC0xMmNkLTQ1ZjAtOGU5OC0wNjBhYTVjZDI5OTAiLCJpZCI6MjU5LCJpYXQiOjE1MTgxNzI0NTN9.u4W4e2sfp6QSN32aTDGfAuiD6VPUj0LJlJBXNkNQ7c4';

const viewer = new Cesium.Viewer('cesiumContainer', {{
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    animation: true,
    timeline: true,
    fullscreenButton: true,
    imageryProvider: new Cesium.TileMapServiceImageryProvider({{
        url: Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
    }})
}});

// Dark space background
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#040408');
viewer.scene.skyBox.show = true;
viewer.scene.sun.show = true;
viewer.scene.moon.show = false;
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0d1a3a');

// Load CZML
const czmlDataSource = new Cesium.CzmlDataSource();
czmlDataSource.load('outputs/constellation.czml').then(function(ds) {{
    viewer.dataSources.add(ds);
    viewer.zoomTo(ds);
}}).otherwise(function(err) {{
    console.error('CZML load error:', err);
    alert('Could not load constellation.czml\\nMake sure it is in the outputs/ folder next to this HTML file.');
}});

// Update HUD every second
setInterval(function() {{
    const t = viewer.clock.currentTime;
    if (t) {{
        const iso = Cesium.JulianDate.toIso8601(t, 0);
        document.getElementById('hud-time').textContent = iso.split('T')[1].split('Z')[0];
    }}
}}, 1000);

// Camera: start at nice angle
viewer.camera.setView({{
    destination: Cesium.Cartesian3.fromDegrees(0, 20, 18000000),
    orientation: {{ heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 }}
}});
</script>
</body>
</html>
"""
    with open(OUT_HTML, "w") as f:
        f.write(html)




def build_sandcastle(czml, rh_df):
    """
    Generate a self-contained JavaScript file ready to paste into
    sandcastle.cesium.com — no server or HTML needed.
    """
    czml_json = json.dumps(czml, separators=(',', ':'))   # compact, no spaces
    
    # Extract reward data
    cycles = [int(x) for x in rh_df['step'].values]
    # Calculate mean reward across all sats for each step
    reward_cols = [c for c in rh_df.columns if '_reward' in c]
    if reward_cols:
        rewards = [float(x) for x in rh_df[reward_cols].mean(axis=1).values]
    else:
        rewards = [0.0] * len(cycles)

    js = f"""// ================================================================
//  SpaceTech AI — Satellite Constellation
//  Paste this entire file into sandcastle.cesium.com and click Run
// ================================================================

const czmlData = {czml_json};
const rewardCycles = {cycles};
const rewardValues = {rewards};

// ── Create Viewer ──────────────────────────────────────────────
const viewer = new Cesium.Viewer("cesiumContainer", {{
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  animation: true,
  timeline: true,
  fullscreenButton: true,
  imageryProvider: new Cesium.TileMapServiceImageryProvider({{
    url: Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"),
  }}),
}});

// ── Appearance ─────────────────────────────────────────────────
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#040408");
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#0d1a3a");
viewer.scene.skyAtmosphere.show = true;

// ── Load inline CZML ───────────────────────────────────────────
const dataSource = new Cesium.CzmlDataSource();
dataSource.load(czmlData).then(function (ds) {{
  viewer.dataSources.add(ds);

  // Zoom to fit all satellites on first load
  viewer.zoomTo(ds);

  // Show entity info on click
  viewer.selectedEntityChanged.addEventListener(function (entity) {{
    if (!entity) return;
    const pos = entity.position
      ? entity.position.getValue(viewer.clock.currentTime)
      : null;
    if (pos) {{
      const carto = Cesium.Ellipsoid.WGS84.cartesianToCartographic(pos);
      const alt = (carto.height / 1000).toFixed(1);
      console.log(`${{entity.name}} | Altitude: ${{alt}} km`);
    }}
  }});
}}).catch(function (err) {{
  console.error("CZML load error:", err);
}});

// ── Camera: isometric angle over Earth ────────────────────────
viewer.camera.setView({{
  destination: Cesium.Cartesian3.fromDegrees(0, 20, 18000000),
  orientation: {{
    heading: 0,
    pitch: Cesium.Math.toRadians(-45),
    roll: 0,
  }},
}});

// ── Legend overlay (injected into the page) ───────────────────
const legend = document.createElement("div");
legend.style.cssText = `
  position:absolute; bottom:60px; left:12px; z-index:999;
  background:rgba(8,8,24,0.88); color:#e0e8ff;
  border:1px solid rgba(80,120,255,0.45); border-radius:10px;
  padding:12px 16px; font-family:sans-serif; font-size:12px;
  backdrop-filter:blur(6px);
`;
legend.innerHTML = `
  <b style="color:#7ab4ff;font-size:13px;">🛰 SpaceTech AI Fleet</b><br><br>
  <span style="color:#1ef05a">●</span> Nominal &nbsp;
  <span style="color:#ffd200">●</span> Degraded<br>
  <span style="color:#ff3228">●</span> Fault Active &nbsp;
  <span style="color:#e619e6">●</span> Safe Mode<br>
  <span style="color:#50b4ff">●</span> Eclipse &nbsp;
  <span style="color:#ff4614">●</span> Debris (80 obj)<br>
  <span style="color:#ffd700">▼</span> Ground Station<br><br>
  <span style="color:#aabbff">10 satellites | 360 steps | 200 cycles</span>
`;
document.body.appendChild(legend);

// ── Reward Graph overlay (injected into the page) ─────────────
const chartContainer = document.createElement("div");
chartContainer.style.cssText = `
  position:absolute; bottom:60px; right:12px; z-index:999;
  background:rgba(8,8,24,0.88); border:1px solid rgba(80,120,255,0.45);
  border-radius:10px; padding:10px; width:400px; height:250px;
  backdrop-filter:blur(6px);
`;
chartContainer.innerHTML = '<canvas id="rewardChart"></canvas>';
document.body.appendChild(chartContainer);

// Dynamically load Chart.js and draw graph
const script = document.createElement('script');
script.src = "https://cdn.jsdelivr.net/npm/chart.js";
script.onload = () => {{
  const ctx = document.getElementById('rewardChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: rewardCycles,
      datasets: [{{
        label: 'Mean Reward',
        data: rewardValues,
        borderColor: '#40ff80',
        backgroundColor: 'rgba(64, 255, 128, 0.1)',
        borderWidth: 2, fill: true, pointRadius: 0
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'Training Reward History', color: '#7ab4ff', font: {{ size: 14 }} }},
        legend: {{ display: false }}
      }},
      scales: {{
        x: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#ccc' }}, title: {{ display: true, text: 'Cycle', color: '#ccc' }} }},
        y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#ccc' }}, title: {{ display: true, text: 'Reward', color: '#ccc' }} }}
      }}
    }}
  }});
}};
document.head.appendChild(script);
"""
    with open(OUT_SANDCASTLE, "w") as f:
        f.write(js)
    size_kb = os.path.getsize(OUT_SANDCASTLE) / 1024
    print(f"  File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()

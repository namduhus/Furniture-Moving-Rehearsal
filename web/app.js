// Furniture-moving rehearsal viewer: replays a recorded Isaac Sim rehearsal (per-decision link and item poses) in the
// room built from the Meta scene spec, with the real G1 meshes placed at the simulated link poses. Simulation frame:
// right-handed, z up, metres, quaternions [x, y, z, w]; the room is centred on the origin.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const DEMOS = [
  { id: "pedestal_carry", label: "Light pedestal · carry" },
  { id: "bookcase_push", label: "Heavy bookcase · try to lift, then push" },
];
const PHASES = {
  start: "Start", approach: "Walk to the item", reach: "Reach", grasp: "Grasp", lift: "Lift", hold: "Hold",
  carry: "Carry", align: "Align", lower: "Set down", release: "Release", withdraw: "Withdraw arms",
  step_back: "Step back", reposition: "Walk to the push side", push: "Push on sliders", back_away: "Step away",
};

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = (id) => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef0f2);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3ad, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2, -3, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 0.5, far: 20 });
scene.add(sun);

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function originMatrix(element) {
  // URDF <origin xyz rpy>: rotation Rz(yaw) Ry(pitch) Rx(roll), i.e. three.js Euler order ZYX.
  const xyz = (element?.getAttribute("xyz") ?? "0 0 0").trim().split(/\s+/).map(Number);
  const rpy = (element?.getAttribute("rpy") ?? "0 0 0").trim().split(/\s+/).map(Number);
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rpy[0], rpy[1], rpy[2], "ZYX"));
  return new THREE.Matrix4().compose(new THREE.Vector3(...xyz), q, new THREE.Vector3(1, 1, 1));
}

async function loadRobot() {
  const text = await (await fetch("../assets/references/g1_wbc.urdf")).text();
  const xml = new DOMParser().parseFromString(text, "application/xml");
  const loader = new STLLoader();
  const cache = new Map();
  const links = new Map();
  const jobs = [];
  for (const link of xml.querySelectorAll("robot > link")) {
    const visuals = [...link.querySelectorAll(":scope > visual")];
    if (!visuals.length) continue;
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    for (const visual of visuals) {
      const file = visual.querySelector("geometry > mesh")?.getAttribute("filename");
      if (!file) continue;
      const rgba = (visual.querySelector("material > color")?.getAttribute("rgba") ?? "0.7 0.7 0.7 1").split(/\s+/).map(Number);
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), metalness: 0.2, roughness: 0.55 });
      const name = file.split("/").pop();
      if (!cache.has(name)) cache.set(name, loader.loadAsync(`./vendor/g1/meshes/${name}`));
      const offset = originMatrix(visual.querySelector(":scope > origin"));
      jobs.push(cache.get(name).then((geometry) => {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.applyMatrix4(offset);
        mesh.castShadow = true;
        group.add(mesh);
      }));
    }
    links.set(link.getAttribute("name"), group);
  }
  // Fixed joints place links that the simulation merged into their parent body (head, logo, pelvis contour).
  const fixed = new Map();
  for (const joint of xml.querySelectorAll("robot > joint[type='fixed']")) {
    fixed.set(joint.querySelector("child").getAttribute("link"),
              { parent: joint.querySelector("parent").getAttribute("link"), origin: originMatrix(joint.querySelector("origin")) });
  }
  await Promise.all(jobs);
  const root = new THREE.Group();
  for (const group of links.values()) root.add(group);
  return { root, links, fixed };
}

function box(size, color, opacity = 1) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size),
                              new THREE.MeshStandardMaterial({ color, roughness: 0.8, transparent: opacity < 1, opacity }));
  mesh.castShadow = mesh.receiveShadow = opacity === 1;
  return mesh;
}

function buildRoom(rehearsal, size) {
  const group = new THREE.Group();
  const { width_m: w, depth_m: d, boxes } = rehearsal.layout.room;
  const { thickness_m: t, height_m: h } = rehearsal.walls;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(w + 2 * t, d + 2 * t), new THREE.MeshStandardMaterial({ color: 0xd8c3a5, roughness: 0.9 }));
  floor.receiveShadow = true;
  group.add(floor);
  for (const [cx, cy, sx, sy] of [[0, -d / 2 - t / 2, w + 2 * t, t], [0, d / 2 + t / 2, w + 2 * t, t], [-w / 2 - t / 2, 0, t, d], [w / 2 + t / 2, 0, t, d]]) {
    const wall = box([sx, sy, h], 0xf4f4f2, 0.55);
    wall.position.set(cx, cy, h / 2);
    group.add(wall);
  }
  for (const [cx, cy, sx, sy, sz] of boxes) {
    const item = box([sx, sy, sz], 0xe6e2da);
    item.position.set(cx, cy, sz / 2);
    group.add(item);
  }
  // Requested final pose of the item, as an outline.
  const [px, py, yaw] = rehearsal.layout.placed;
  const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), new THREE.LineBasicMaterial({ color: 0x1f8a4c }));
  ghost.position.set(px, py, size[2] / 2);
  ghost.rotation.z = yaw;
  group.add(ghost);
  return group;
}

const state = { demo: null, t: 0, playing: true, speed: 1, lang: "en" };
let robot;
let world = new THREE.Group();
scene.add(world);

function segments(frames, dt) {
  const out = [];
  frames.forEach((frame, i) => {
    if (!out.length || out[out.length - 1].phase !== frame.phase) out.push({ phase: frame.phase, start: i * dt, end: i * dt });
    out[out.length - 1].end = i * dt;
  });
  return out;
}

function outcomeNote(rehearsal, phase) {
  const r = rehearsal.phases?.[phase];
  if (!r || !r.outcome) return "";
  return `${r.outcome === "success" ? "✓" : "✗ " + r.outcome} ${r.seconds?.toFixed(1) ?? ""} s`;
}

function describe(rehearsal, scene) {
  const req = rehearsal.request;
  const item = scene.scene.furniture.find((f) => f.id === req.target);
  const [w, d, h] = req.corrections.size_m;
  $("request").innerHTML = [
    ["Item", `${item?.category ?? req.target}`],
    ["Meta estimate", item ? `${item.size_m.join(" × ")} m, ${item.estimated_mass_kg} kg` : "–"],
    ["User correction", `${w} × ${d} × ${h} m, ${req.corrections.mass_kg} kg`],
    ["Destination", `(${req.destination.center_m.join(", ")}) m, ${req.destination.yaw_deg}°`],
    ["Strategy", req.strategy === "auto" ? "try to lift, else decide" : req.strategy],
  ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
  const p = rehearsal.placement;
  const rows = [["Status", rehearsal.status.replaceAll("_", " ")]];
  if (rehearsal.lift && !rehearsal.lift.lifted) rows.push(["Lift attempt", "stayed on the floor"]);
  if (p) rows.push(["Placement error", `<span class="result ${p.within_tolerance ? "ok" : "bad"}">${(p.position_m * 100).toFixed(1)} cm, ${p.yaw_deg.toFixed(1)}°</span>`]);
  $("result").innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
}

function showMeta(rehearsal, visible) {
  const advice = rehearsal.advice;
  $("meta-box").hidden = !advice;
  if (!advice) return;
  $("meta").hidden = !visible;
  $("meta-pending").hidden = visible;
  const decision = advice.decision;
  $("meta-who").textContent = advice.mode === "rules_only"
    ? `Capability rules (Meta unavailable) → ${decision.strategy}`
    : `Muse Spark → ${decision.strategy}${decision.use_sliders ? " on sliders" : ""}${decision.model_choice_accepted ? " · accepted by the capability rules" : " · overridden by the rules"}`;
  $("meta-text").textContent = state.lang === "ko" ? advice.advice?.explanation_ko ?? "" : advice.advice?.explanation ?? "";
  $("lang").textContent = state.lang === "ko" ? "English" : "한국어";
}

async function load(id) {
  $("status").hidden = false;
  $("status").textContent = "Loading…";
  const get = async (name) => (await fetch(`./demos/${id}/${name}`)).json();
  const [replay, rehearsal, sceneSpec] = await Promise.all([get("replay.json"), get("rehearsal.json"), get("scene.json")]);
  robot ??= await loadRobot();
  scene.remove(world);
  world = new THREE.Group();
  world.add(buildRoom(rehearsal, replay.cabinet_size_m));
  const item = box(replay.cabinet_size_m, 0xb8864b);
  world.add(item, robot.root);
  scene.add(world);
  const index = new Map(replay.bodies.map((name, i) => [name, i]));
  const segs = segments(replay.frames, replay.decision_dt);
  $("phases").innerHTML = segs.map((s, i) => `<li data-i="${i}"><span>${PHASES[s.phase] ?? s.phase}</span><small>${outcomeNote(rehearsal, s.phase)}</small></li>`).join("");
  describe(rehearsal, sceneSpec);
  $("photo").src = `./demos/${id}/photo.png`;
  const start = replay.frames[0].cabinet, placed = rehearsal.layout.placed, d = rehearsal.layout.room.depth_m;
  const focus = new THREE.Vector3((start[0] + placed[0]) / 2, (start[1] + placed[1]) / 2, 0.3);
  camera.position.set(focus.x, -d / 2 - 2.5, 4.5);
  controls.target.copy(focus);
  state.demo = { id, replay, rehearsal, item, index, segs, duration: (replay.frames.length - 1) * replay.decision_dt,
                 diagnosis: segs.find((s) => s.phase === "reposition")?.start ?? Infinity };
  state.t = 0;
  state.playing = true;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  $("status").hidden = true;
}

const a = new THREE.Vector3(), b = new THREE.Vector3(), qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
const unit = new THREE.Vector3(1, 1, 1), worldMatrix = new Map();

function blend(target, p0, p1, f) {
  // Pose between two recorded decisions: linear position, spherical rotation.
  a.set(p0[0], p0[1], p0[2]).lerp(b.set(p1[0], p1[1], p1[2]), f);
  qa.set(p0[3], p0[4], p0[5], p0[6]).slerp(qb.set(p1[3], p1[4], p1[5], p1[6]), f);
  return target.compose(a, qa, unit);
}

function update() {
  const demo = state.demo;
  if (!demo) return;
  const frames = demo.replay.frames;
  const x = Math.min(state.t / demo.replay.decision_dt, frames.length - 1);
  const i = Math.floor(x), f = x - i, now = frames[i], next = frames[Math.min(i + 1, frames.length - 1)];
  worldMatrix.clear();
  for (const [name, group] of robot.links) {
    if (demo.index.has(name)) {
      const k = demo.index.get(name);
      worldMatrix.set(name, blend(group.matrix, now.bodies[k], next.bodies[k], f));
    }
  }
  for (const [name, group] of robot.links) {
    const joint = robot.fixed.get(name);
    if (!demo.index.has(name) && joint && worldMatrix.has(joint.parent)) group.matrix.multiplyMatrices(worldMatrix.get(joint.parent), joint.origin);
  }
  demo.item.matrixAutoUpdate = false;
  blend(demo.item.matrix, now.cabinet, next.cabinet, f);
  const seg = demo.segs.findIndex((s) => state.t >= s.start && state.t <= s.end + demo.replay.decision_dt);
  document.querySelectorAll("#phases li").forEach((li, j) => { li.classList.toggle("now", j === seg); li.classList.toggle("done", j < seg); });
  $("phase").textContent = PHASES[demo.segs[seg]?.phase] ?? "";
  $("time").textContent = `${state.t.toFixed(1)} / ${demo.duration.toFixed(1)} s`;
  $("scrub").value = String(Math.round((state.t / demo.duration) * 1000));
  showMeta(demo.rehearsal, state.t >= demo.diagnosis);
}

let last = performance.now();
renderer.setAnimationLoop((now) => {
  const dt = (now - last) / 1000;
  last = now;
  if (state.demo && state.playing) {
    state.t = Math.min(state.t + dt * state.speed, state.demo.duration);
    if (state.t >= state.demo.duration) state.playing = false;
  }
  $("play").textContent = state.playing ? "Pause" : "Play";
  update();
  controls.update();
  renderer.render(scene, camera);
});

$("tabs").innerHTML = DEMOS.map((d) => `<button type="button" data-id="${d.id}">${d.label}</button>`).join("");
$("tabs").addEventListener("click", (e) => { if (e.target.dataset.id) load(e.target.dataset.id).catch(fail); });
$("play").addEventListener("click", () => {
  if (!state.demo) return;
  if (!state.playing && state.t >= state.demo.duration) state.t = 0;
  state.playing = !state.playing;
});
$("speed").addEventListener("click", () => { state.speed = state.speed === 4 ? 1 : state.speed * 2; $("speed").textContent = `${state.speed}×`; });
$("scrub").addEventListener("input", (e) => { if (state.demo) { state.t = (Number(e.target.value) / 1000) * state.demo.duration; state.playing = false; } });
$("lang").addEventListener("click", () => { state.lang = state.lang === "en" ? "ko" : "en"; });

function fail(error) {
  $("status").hidden = false;
  $("status").textContent = `Could not load the rehearsal: ${error.message}. Serve the repository root (python3 -m http.server) and run scripts/fetch_g1_meshes.py first.`;
}

load(new URLSearchParams(location.search).get("demo") ?? DEMOS[0].id).catch(fail);

// Furniture-moving rehearsal viewer: replays a recorded Isaac Sim rehearsal (per-decision link and item poses) in the
// room built from the Meta scene spec, with the real G1 meshes placed at the simulated link poses. Furniture and floor
// take their colours from the room photo (inside each item's SAM 3.1 mask). Simulation frame: right-handed, z up,
// metres, quaternions [x, y, z, w]; the room is centred on the origin.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const DEMOS = [
  { id: "pedestal_carry", label: "Light pedestal · carry" },
  { id: "bookcase_push", label: "Heavy bookcase · try to lift, then push" },
  { id: "pedestal_three", label: "Pedestal room · two moves in a row" },
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
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9ebee);
// Image-based light from a neutral studio room for soft reflections, plus a key light for shadows.
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.7;
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xfff6ea, 0x8a8f96, 0.5));
const sun = new THREE.DirectionalLight(0xfff3e0, 2.2);
sun.position.set(2.5, -3, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 0.5, far: 20 });
scene.add(sun);
// Ambient occlusion darkens contacts and corners (feet on the floor, furniture against walls).
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const ao = new GTAOPass(scene, camera, window.innerWidth, window.innerHeight);
composer.addPass(ao);
composer.addPass(new OutputPass());

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
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
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), metalness: 0.35, roughness: 0.4 });
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

function canvasTexture(width, height, draw, repeat = [1, 1]) {
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  draw(canvas.getContext("2d"), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(...repeat);
  texture.anisotropy = 8;
  return texture;
}

function shade(color, factor) {
  return `rgb(${color.map((c) => Math.max(0, Math.min(255, Math.round(c * factor)))).join(",")})`;
}

function woodTexture(rgb) {
  // Fine vertical grain over the photo colour.
  return canvasTexture(256, 256, (g, w, h) => {
    g.fillStyle = shade(rgb, 1);
    g.fillRect(0, 0, w, h);
    for (let x = 0; x < w; x += 2) {
      const k = 0.9 + 0.12 * Math.sin(x * 0.21) * Math.sin(x * 0.047 + 1.3) + 0.05 * Math.sin(x * 1.7);
      g.fillStyle = shade(rgb, k);
      g.fillRect(x, 0, 2, h);
    }
  });
}

function floorTexture(rgb, w, d) {
  // Planks 0.2 m wide and 1.2 m long on a 1.6 m tile, each plank's tone varied around the photo's floor colour.
  const tile = 1.6;
  return canvasTexture(512, 512, (g, size) => {
    const plank = size / 8, length = size * 0.75;
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let row = 0; row < 8; row += 1) {
      let x = -random() * length;
      while (x < size) {
        g.fillStyle = shade(rgb, 0.9 + 0.2 * random());
        g.fillRect(x, row * plank, length, plank);
        g.fillStyle = shade(rgb, 0.7);
        g.fillRect(x, row * plank, 2, plank);
        for (let k = 0; k < 6; k += 1) {
          g.fillStyle = shade(rgb, 0.93 + 0.1 * random());
          g.fillRect(x, row * plank + random() * plank, length, 1);
        }
        x += length;
      }
      g.fillStyle = shade(rgb, 0.68);
      g.fillRect(0, row * plank, size, 2);
    }
  }, [w / tile, d / tile]);
}

const FABRIC = /sofa|armchair|chair|bed|ottoman/i, SCREEN = /^\s*(tv|television|monitor|screen)\s*$/i;     // not "tv stand"

function richer(rgb, saturation, lightness) {
  // A photo colour seen under the room's own bright light, made deeper so it survives the renderer's lighting.
  const color = new THREE.Color().setRGB(...rgb.map((c) => c / 255), THREE.SRGBColorSpace);
  const hsl = color.getHSL({}, THREE.SRGBColorSpace);
  color.setHSL(hsl.h, Math.min(1, hsl.s * saturation), hsl.l * lightness, THREE.SRGBColorSpace);
  const out = color.getRGB({}, THREE.SRGBColorSpace);
  return [out.r, out.g, out.b].map((c) => Math.round(255 * c));
}

function furnitureMaterial(category, rgb) {
  if (SCREEN.test(category ?? "")) return new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.25, metalness: 0.2 });
  rgb ??= FABRIC.test(category ?? "") ? [205, 200, 192] : [168, 124, 82];
  if (FABRIC.test(category ?? "")) {
    return new THREE.MeshPhysicalMaterial({ color: new THREE.Color().setRGB(...rgb.map((c) => c / 255), THREE.SRGBColorSpace),
                                            roughness: 0.95, sheen: 0.6, sheenRoughness: 0.8, sheenColor: 0xffffff });
  }
  return new THREE.MeshStandardMaterial({ map: woodTexture(rgb), roughness: 0.55 });
}

function roundedBox(size, material) {
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(size[0], size[1], size[2], 3, Math.min(0.025, ...size.map((v) => v / 4))), material);
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

async function photoColors(url, appearance) {
  // Mean photo colour inside each item's SAM mask (grid cells over its pixel box) and over a floor patch near the
  // bottom of the photo.
  const image = new Image();
  image.src = url;
  await image.decode();
  const canvas = Object.assign(document.createElement("canvas"), { width: 480, height: Math.round(480 * image.height / image.width) });
  const g = canvas.getContext("2d", { willReadFrequently: true });
  g.drawImage(image, 0, 0, canvas.width, canvas.height);
  const pixels = g.getImageData(0, 0, canvas.width, canvas.height).data;
  const sx = canvas.width / image.width, sy = canvas.height / image.height;
  const mean = (points) => {
    const sum = [0, 0, 0];
    for (const [x, y] of points) {
      const k = 4 * (Math.min(canvas.height - 1, Math.round(y * sy)) * canvas.width + Math.min(canvas.width - 1, Math.round(x * sx)));
      sum[0] += pixels[k]; sum[1] += pixels[k + 1]; sum[2] += pixels[k + 2];
    }
    return sum.map((v) => v / Math.max(1, points.length));
  };
  const colors = {};
  for (const [itemId, look] of Object.entries(appearance.items ?? {})) {
    const [x1, y1, x2, y2] = look.box_px, n = look.grid.length, points = [];
    look.grid.forEach((row, i) => [...row].forEach((bit, j) => {
      if (bit === "1") points.push([x1 + (j + 0.5) * (x2 - x1) / n, y1 + (i + 0.5) * (y2 - y1) / n]);
    }));
    if (points.length) colors[itemId] = mean(points);
  }
  const floor = [];
  for (let i = 0; i < 12; i += 1) for (let j = 0; j < 24; j += 1) floor.push([image.width * (0.35 + 0.3 * j / 23), image.height * (0.8 + 0.12 * i / 11)]);
  return { items: colors, floor: mean(floor) };
}

function buildShell(w, d, walls, floorRgb) {
  // Floor and walls; the wall nearest the camera is translucent.
  const group = new THREE.Group();
  const { thickness_m: t, height_m: h } = walls;
  const colors = { floor: floorRgb };
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(w + 2 * t, d + 2 * t),
                               new THREE.MeshStandardMaterial({ map: floorTexture(richer(colors.floor, 1.35, 0.78), w + 2 * t, d + 2 * t),
                                                                roughness: 0.6 }));
  floor.receiveShadow = true;
  group.add(floor);
  const plaster = new THREE.MeshStandardMaterial({ color: 0xf3f1ec, roughness: 0.95 });
  const front = new THREE.MeshStandardMaterial({ color: 0xf3f1ec, roughness: 0.95, transparent: true, opacity: 0.18, depthWrite: false });
  for (const [cx, cy, sx, sy, material] of [[0, -d / 2 - t / 2, w + 2 * t, t, front], [0, d / 2 + t / 2, w + 2 * t, t, plaster],
                                            [-w / 2 - t / 2, 0, t, d, plaster], [w / 2 + t / 2, 0, t, d, plaster]]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, h), material);
    wall.position.set(cx, cy, h / 2);
    wall.receiveShadow = material === plaster;
    group.add(wall);
  }
  group.userData.floor = floor;
  return group;
}

function buildRoom(rehearsal, sceneSpec, colors, ghosts) {
  const { width_m: w, depth_m: d, boxes } = rehearsal.layout.room;
  const group = buildShell(w, d, rehearsal.walls, colors.floor);
  // Each static box is the spec item (other than the moved ones) nearest in centre and height, in the simulation frame.
  const { x: [sxSign, xOffset], y: [sySign, yOffset] } = rehearsal.layout.spec_transform;
  const moved = new Set((rehearsal.moves ?? [rehearsal]).map((m) => m.request.target));
  const specItems = sceneSpec.scene.furniture.filter((f) => !moved.has(f.id))
    .map((f) => ({ ...f, sim: [(f.center_m[0] - xOffset) / sxSign, (f.center_m[1] - yOffset) / sySign] }));
  const score = (f, [cx, cy, , , sz]) => Math.hypot(f.sim[0] - cx, f.sim[1] - cy) + Math.abs(f.size_m[2] - sz);
  for (const b of boxes) {
    const [cx, cy, sx, sy, sz] = b;
    const spec = specItems.reduce((best, f) => (!best || score(f, b) < score(best, b) ? f : best), null);
    // Display only: a screen standing inside a lower box's footprint is drawn on top of it, as in the photo (the
    // simulation keeps every box on the floor as an obstacle).
    const under = SCREEN.test(spec?.category ?? "") ? boxes.find(([ux, uy, usx, usy, usz]) => usz < sz &&
      Math.abs(ux - cx) <= (usx - sx) / 2 + 0.05 && Math.abs(uy - cy) <= (usy - sy) / 2 + 0.05) : null;
    const item = roundedBox([sx, sy, sz], furnitureMaterial(spec?.category, colors.items[spec?.id]));
    item.position.set(cx, cy, sz / 2 + (under ? under[4] : 0));
    group.add(item);
  }
  // Requested final pose of each moved item, as an outline.
  for (const { placed: [px, py, yaw], size } of ghosts) {
    const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), new THREE.LineBasicMaterial({ color: 0x1f8a4c }));
    ghost.position.set(px, py, size[2] / 2);
    ghost.rotation.z = yaw;
    group.add(ghost);
  }
  return group;
}

const state = { demo: null, t: 0, playing: true, speed: 1, lang: "en" };
let robot;
let world = new THREE.Group();
scene.add(world);

function segments(frames, dt, label = (frame) => frame.phase) {
  const out = [];
  frames.forEach((frame, i) => {
    const phase = label(frame);
    if (!out.length || out[out.length - 1].phase !== phase) out.push({ phase, start: i * dt, end: i * dt });
    out[out.length - 1].end = i * dt;
  });
  return out;
}

function placementText(p) {
  return p ? `<span class="result ${p.within_tolerance ? "ok" : "bad"}">${(p.position_m * 100).toFixed(1)} cm, ${p.yaw_deg.toFixed(1)}°</span>` : "–";
}

function outcomeNote(rehearsal, phase) {
  const r = rehearsal.phases?.[phase];
  if (!r || !r.outcome) return "";
  return `${r.outcome === "success" ? "✓" : "✗ " + r.outcome} ${r.seconds?.toFixed(1) ?? ""} s`;
}

function describe(rehearsal, scene) {
  const req = rehearsal.request, moves = rehearsal.moves;
  const item = scene.scene.furniture.find((f) => f.id === req.target);
  const [w, d, h] = req.corrections.size_m;
  if (moves) {
    $("request").innerHTML = [
      ["Items", `${moves.length} × ${item?.category ?? "item"}, one after another`],
      ["Meta estimate", item ? `${item.size_m.join(" × ")} m, ${item.estimated_mass_kg} kg` : "–"],
      ["User correction", `${w} × ${d} × ${h} m, ${req.corrections.mass_kg} kg each`],
      ["Strategy", req.strategy],
    ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
    // Moves the rehearsal could not finish are listed with how they ended.
    const leftOut = (rehearsal.left_out ?? []).map((m) => {
      const kind = scene.scene.furniture.find((f) => f.id === m.target)?.category ?? m.target;
      return [`${kind}`, `<span class="result bad">not moved: ${(m.carry ?? m.status).replaceAll("_", " ")} (${m.attempts} tries)</span>`];
    });
    $("result").innerHTML = [["Status", rehearsal.status.replaceAll("_", " ")], ...moves.map((m, k) => [`Box ${k + 1} placement`, placementText(m.placement)]),
                             ...leftOut].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
    return;
  }
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
  if (p) rows.push(["Placement error", placementText(p)]);
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
  const [replay, rehearsal, sceneSpec, appearance] = await Promise.all([get("replay.json"), get("rehearsal.json"), get("scene.json"),
                                                                       get("appearance.json")]);
  robot ??= await loadRobot();
  const colors = await photoColors(`./demos/${id}/photo.png`, appearance);
  setMode("replay");
  scene.remove(world);
  world = new THREE.Group();
  // A sequence demo moves several items one after another (frame.item); a single rehearsal moves one.
  const sequence = Boolean(rehearsal.moves);
  const moves = rehearsal.moves ?? [{ request: rehearsal.request, layout: rehearsal.layout, size_m: replay.cabinet_size_m,
                                      frames: [0, replay.frames.length - 1] }];
  world.add(buildRoom(rehearsal, sceneSpec, colors, moves.map((m) => ({ placed: m.layout.placed, size: m.size_m }))));
  const items = moves.map((m) => {
    const mesh = roundedBox(m.size_m, new THREE.MeshStandardMaterial({ map: woodTexture(colors.items[m.request.target] ?? [184, 134, 75]),
                                                                       roughness: 0.5 }));
    mesh.matrixAutoUpdate = false;
    world.add(mesh);
    return { mesh, start: replay.frames[m.frames[0]].cabinet, end: replay.frames[m.frames[1]].cabinet };
  });
  world.add(robot.root);
  scene.add(world);
  const index = new Map(replay.bodies.map((name, i) => [name, i]));
  const segs = segments(replay.frames, replay.decision_dt, sequence ? (frame) => `Box ${frame.item + 1}` : undefined);
  $("phases").innerHTML = segs.map((s, i) => `<li data-i="${i}"><span>${PHASES[s.phase] ?? s.phase}</span><small>${
    sequence ? placementText(moves[i]?.placement) : outcomeNote(rehearsal, s.phase)}</small></li>`).join("");
  describe(rehearsal, sceneSpec);
  $("photo").src = `./demos/${id}/photo.png`;
  const d = rehearsal.layout.room.depth_m, ends = items.flatMap((it, k) => [it.start, moves[k].layout.placed]);
  const focus = new THREE.Vector3(ends.reduce((sum, p) => sum + p[0], 0) / ends.length, ends.reduce((sum, p) => sum + p[1], 0) / ends.length, 0.3);
  camera.position.set(focus.x, -d / 2 - 2.5, 4.5);
  controls.target.copy(focus);
  state.demo = { id, replay, rehearsal, items, sequence, index, segs, duration: (replay.frames.length - 1) * replay.decision_dt,
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
  // The item being moved follows its recorded pose; the others wait at their start or stay where they were set down.
  const k = now.item ?? 0;
  demo.items.forEach((it, j) => {
    if (j === k) blend(it.mesh.matrix, now.cabinet, (next.item ?? 0) === k ? next.cabinet : now.cabinet, f);
    else blend(it.mesh.matrix, j < k ? it.end : it.start, j < k ? it.end : it.start, 0);
  });
  const seg = demo.segs.findIndex((s) => state.t >= s.start && state.t <= s.end + demo.replay.decision_dt);
  document.querySelectorAll("#phases li").forEach((li, j) => { li.classList.toggle("now", j === seg); li.classList.toggle("done", j < seg); });
  $("phase").textContent = demo.sequence ? `Box ${k + 1} · ${PHASES[now.phase] ?? now.phase}` : PHASES[demo.segs[seg]?.phase] ?? "";
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
  composer.render();
});

// ---- Try it: pick items and destinations one after another (each checked with the rehearsal rules and Muse Spark,
// given the moves queued before it), then rehearse them all on Isaac ----
const plan = { view: null, items: new Map(), floor: null, selected: null, ghost: null, pose: null, checked: null, queue: [] };
const raycaster = new THREE.Raycaster();
let standing = null;

function setMode(mode) {
  $("plan-panel").hidden = mode !== "plan";
  $("replay-panel").hidden = mode !== "replay";
  document.querySelector("footer").hidden = mode !== "replay";
}

const simSize = ([w, d, h]) => [d, w, h];
const specYaw = (yaw) => THREE.MathUtils.radToDeg(Math.atan2(-Math.cos(yaw), Math.sin(yaw)));    // inverse of facing_yaw

async function placeStanding(start) {
  // The G1's first recorded frame, moved rigidly from that rehearsal's start to `start`.
  standing ??= (await (await fetch("./demos/pedestal_carry/replay.json")).json());
  const index = new Map(standing.bodies.map((name, i) => [name, i]));
  const frame = standing.frames[0].bodies, pelvis = frame[index.get("pelvis")];
  const yaw0 = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(...pelvis.slice(3)), "ZYX").z;
  const delta = new THREE.Matrix4().makeTranslation(start[0], start[1], 0)
    .multiply(new THREE.Matrix4().makeRotationZ(start[2] - yaw0)).multiply(new THREE.Matrix4().makeTranslation(-pelvis[0], -pelvis[1], 0));
  const matrices = new Map();
  for (const [name, group] of robot.links) {
    if (index.has(name)) matrices.set(name, group.matrix.multiplyMatrices(delta, blend(new THREE.Matrix4(), frame[index.get(name)], frame[index.get(name)], 0)));
  }
  for (const [name, group] of robot.links) {
    const joint = robot.fixed.get(name);
    if (!index.has(name) && joint && matrices.has(joint.parent)) group.matrix.multiplyMatrices(matrices.get(joint.parent), joint.origin);
  }
}

async function loadPlan(sceneId) {
  $("status").hidden = false;
  $("status").textContent = "Loading…";
  const view = await (await fetch(`/api/scene?id=${encodeURIComponent(sceneId)}`)).json();
  if (view.error) throw new Error(view.error);
  robot ??= await loadRobot();
  const colors = view.photo ? await photoColors(view.photo, view.appearance) : { items: {}, floor: [196, 160, 120] };
  state.demo = null;
  scene.remove(world);
  world = new THREE.Group();
  const shell = buildShell(view.room.width_m, view.room.depth_m, view.walls, colors.floor);
  world.add(shell);
  plan.items.clear();
  const boxes = view.furniture.map((f) => ({ f, size: simSize(f.size_m) }));
  for (const { f, size } of boxes) {
    const mesh = roundedBox(size, furnitureMaterial(f.category, colors.items[f.id]));
    // Display only: a screen inside a lower item's footprint stands on it, as in the photo.
    const under = SCREEN.test(f.category) ? boxes.find((o) => o.f !== f && o.f.size_m[2] < f.size_m[2] &&
      Math.hypot(o.f.sim[0] - f.sim[0], o.f.sim[1] - f.sim[1]) < 0.3) : null;
    mesh.position.set(f.sim[0], f.sim[1], size[2] / 2 + (under ? under.f.size_m[2] : 0));
    mesh.rotation.z = f.sim[2];
    mesh.userData.item = f;
    plan.items.set(f.id, mesh);
    world.add(mesh);
  }
  await placeStanding(view.start);
  world.add(robot.root);
  scene.add(world);
  Object.assign(plan, { view, floor: shell.userData.floor, selected: null, ghost: null, pose: null, checked: null, queue: [] });
  renderQueue();
  $("photo").src = view.photo ?? "";
  camera.position.set(0, -view.room.depth_m / 2 - 2.5, 4.5);
  controls.target.set(0, 0, 0.3);
  setMode("plan");
  $("plan-item").hidden = true;
  $("plan-hint").textContent = "Click the furniture you want to move.";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.scene === sceneId));
  $("status").hidden = true;
}

function corrections() {
  return { size_m: ["fix-w", "fix-d", "fix-h"].map((id) => Number($(id).value)), mass_kg: Number($("fix-m").value) };
}

function inCarryTier() {
  const tier = plan.view.carry_tier, c = corrections();
  return c.size_m.every((v, i) => Math.abs(v - tier.size_m[i]) <= tier.tolerance_m) && c.mass_kg >= tier.mass_kg[0] && c.mass_kg <= tier.mass_kg[1];
}

function highlight(f) {
  for (const mesh of plan.items.values()) {
    mesh.material.emissive?.setHex(mesh.userData.item === f ? 0x2f6fde : 0x000000);
    if (mesh.material.emissive) mesh.material.emissiveIntensity = 0.35;
  }
}

function resizeItem(id, sizeM) {
  // An item drawn at a (corrected) spec size, standing on the floor at its spec pose.
  const mesh = plan.items.get(id), size = simSize(sizeM);
  if (size.some((v) => !(v > 0))) return;
  mesh.geometry.dispose();
  mesh.geometry = new RoundedBoxGeometry(size[0], size[1], size[2], 3, Math.min(0.025, ...size.map((v) => v / 4)));
  mesh.position.z = size[2] / 2;
}

const queued = () => new Set(plan.queue.map((q) => q.request.target));
// Other items of the selected one's kind, not yet queued: they can take the same corrected size.
const alikeItems = () => plan.view.furniture.filter((o) => o.id !== plan.selected.id && o.category === plan.selected.category && !queued().has(o.id));
const alikeIds = () => ($("fix-alike").checked && !$("alike-row").hidden ? alikeItems().map((o) => o.id) : []);

function resizeSelected() {
  resizeItem(plan.selected.id, corrections().size_m);
  for (const o of alikeItems()) resizeItem(o.id, alikeIds().includes(o.id) ? corrections().size_m : o.size_m);
}

function selectItem(f) {
  if (queued().has(f.id)) { $("plan-hint").textContent = `${f.category} is already in the moves.`; return; }
  highlight(f);
  plan.selected = f;
  $("plan-item").hidden = false;
  $("plan-hint").textContent = `Selected: ${f.category}.`;
  $("plan-meta").innerHTML = `<span>Meta estimate</span><span>${f.size_m.join(" × ")} m, ${f.estimated_mass_kg} kg</span>`;
  [["fix-w", f.size_m[0]], ["fix-d", f.size_m[1]], ["fix-h", f.size_m[2]], ["fix-m", f.estimated_mass_kg]].forEach(([id, v]) => { $(id).value = v; });
  const others = alikeItems();
  $("alike-row").hidden = !others.length;
  $("alike-text").textContent = `Same size for the other ${others.length} ${f.category}${others.length > 1 ? "s" : ""}`;
  clearGhost();
  showFeasible().catch((error) => { $("dest-hint").textContent = error.message; });
}

function renderQueue() {
  const v = plan.view;
  $("queue-box").hidden = !plan.queue.length;
  $("queue").innerHTML = plan.queue.map((q, k) => `<li><span>${k + 1}. ${q.item.category} → (${q.request.destination.center_m.map((c) => c.toFixed(2)).join(", ")})</span><small>${q.request.strategy}</small></li>`).join("");
  // Muse Spark's explanation of the latest move stays visible after the item panel closes.
  const advice = plan.queue[plan.queue.length - 1]?.advice;
  $("queue-meta").hidden = !advice;
  $("queue-meta").textContent = advice ? `Muse Spark: ${state.lang === "ko" ? advice.explanation_ko : advice.explanation}` : "";
  $("rehearse").textContent = `Rehearse all (${plan.queue.length})`;
  $("rehearse").disabled = !plan.queue.length;
}

function queueMove(result) {
  // The checked move joins the queue: its outline stays, the item fades where it stands until the rehearsal.
  plan.queue.push({ request: result.request, ghost: plan.ghost, item: plan.selected, advice: result.advice?.advice });
  const mesh = plan.items.get(plan.selected.id);
  Object.assign(mesh.material, { transparent: true, opacity: 0.45, needsUpdate: true });
  highlight(null);
  if (plan.dots) world.remove(plan.dots);
  Object.assign(plan, { ghost: null, pose: null, checked: null, selected: null, dots: null });
  $("plan-item").hidden = true;
  $("plan-hint").textContent = `Added as move ${plan.queue.length}. Click another item, or rehearse all.`;
  renderQueue();
}

function undoMove() {
  const last = plan.queue.pop();
  if (!last) return;
  world.remove(last.ghost);
  Object.assign(plan.items.get(last.item.id).material, { transparent: false, opacity: 1, needsUpdate: true });
  $("plan-hint").textContent = "Removed the last move. Click the furniture you want to move.";
  renderQueue();
}

function clearGhost() {
  if (plan.ghost) world.remove(plan.ghost);
  Object.assign(plan, { ghost: null, pose: null, checked: null, turned: false });
  $("plan-result").innerHTML = "";
  $("check").disabled = true;
  refreshHint();
}

function refreshHint() {
  if (!plan.selected) return;
  $("rot-l").disabled = $("rot-r").disabled = !inCarryTier() || !plan.pose;
}

const facingYaw = (deg) => Math.atan2(Math.cos(THREE.MathUtils.degToRad(deg)), -Math.sin(THREE.MathUtils.degToRad(deg)));

async function showFeasible() {
  // Destinations that pass the rehearsal layout checks for the selected item and correction, as dots on the floor.
  if (plan.dots) world.remove(plan.dots);
  plan.feasible = [];
  const v = plan.view;
  const body = { scene: v.id, target: plan.selected.id, corrections: corrections(), alike: alikeIds(),
                 previous: plan.queue.map((q) => q.request) };
  const result = await (await fetch("/api/feasible", { method: "POST", body: JSON.stringify(body) })).json();
  if (result.error) throw new Error(result.error);
  plan.feasible = result.points.map((p) => ({ ...p, sim: [p.center_m[0] - v.spec_transform.x[1], v.spec_transform.y[1] - p.center_m[1], facingYaw(p.yaw_deg)] }));
  const dots = new THREE.InstancedMesh(new THREE.CircleGeometry(0.035, 12), new THREE.MeshBasicMaterial({ color: 0x22c55e, toneMapped: false }),
                                       Math.max(1, plan.feasible.length));
  dots.count = plan.feasible.length;
  plan.feasible.forEach((p, i) => dots.setMatrixAt(i, new THREE.Matrix4().makeTranslation(p.sim[0], p.sim[1], 0.004)));
  plan.dots = dots;
  world.add(dots);
  // With no destination, say why (the rehearsal rules' reasons for the carry, lift-then-push and push strategies).
  const why = Object.entries(result.reasons ?? {}).map(([strategy, reason]) => `${strategy}: ${reason}`).join("; ");
  $("dest-hint").textContent = plan.feasible.length
    ? `Green dots: where it can go (${inCarryTier() ? "carried" : "lifted if it can be, else pushed on sliders"}). Click near one.`
    : `No reachable destination with this correction (${why || "no candidate"}).${inCarryTier() ? "" : " Try “Use carry size”."}`;
}

function placeGhost(point) {
  // Snap to the nearest feasible destination; a carried item can then be turned before the check.
  const near = plan.feasible.reduce((best, p) => (!best || Math.hypot(p.sim[0] - point.x, p.sim[1] - point.y) <
                                                  Math.hypot(best.sim[0] - point.x, best.sim[1] - point.y) ? p : best), null);
  if (!near) return;
  plan.turned = false;
  drawGhost(near.sim);
}

function drawGhost(pose) {
  if (plan.ghost) world.remove(plan.ghost);
  const size = simSize(corrections().size_m);
  const geometry = new THREE.BoxGeometry(...size);
  const ghost = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x1f8a4c, transparent: true, opacity: 0.18, depthWrite: false }));
  ghost.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: 0x1f8a4c })));
  ghost.position.set(pose[0], pose[1], size[2] / 2);
  ghost.rotation.z = pose[2];
  world.add(ghost);
  Object.assign(plan, { ghost, pose, checked: null });
  $("check").disabled = false;
  $("plan-result").innerHTML = "";
  refreshHint();
}

async function checkMove() {
  const v = plan.view, [x, y, yaw] = plan.pose;
  $("plan-result").innerHTML = `<p class="hint">Checking the move and asking Muse Spark…</p>`;
  const body = { scene: v.id, target: plan.selected.id, corrections: corrections(), ask_meta: true, alike: alikeIds(),
                 previous: plan.queue.map((q) => q.request),
                 destination: { center_m: [x * v.spec_transform.x[0] + v.spec_transform.x[1], y * v.spec_transform.y[0] + v.spec_transform.y[1]],
                                yaw_deg: specYaw(yaw) } };
  const result = await (await fetch("/api/check", { method: "POST", body: JSON.stringify(body) })).json();
  if (result.error) throw new Error(result.error);
  plan.checked = result;
  const label = { carry: "carry it", auto: "try to lift it, else decide", push: "push it on sliders" }[result.strategy];
  const advice = result.advice?.advice, decision = result.advice?.decision;
  $("plan-result").innerHTML = result.ok
    ? `<p class="ok-text">Possible: ${label}.</p>` + (advice ? `<div class="meta"><div class="who"><span>Muse Spark → ${decision.strategy}${decision.model_choice_accepted ? " · accepted by the rules" : " · overridden by the rules"}</span></div><p>${state.lang === "ko" ? advice.explanation_ko : advice.explanation}</p></div>` : "")
    : `<p class="bad-text">Not possible here: ${Object.values(result.reasons)[0]}</p>`;
  if (result.ok) queueMove(result);
}

async function rehearseMove() {
  const started = performance.now();
  $("rehearse").disabled = $("undo").disabled = $("check").disabled = true;
  const moves = plan.queue.length;
  const response = await (await fetch("/api/rehearse", { method: "POST", body: JSON.stringify({ moves: plan.queue.map((q) => q.request) }) })).json();
  if (response.error) throw new Error(response.error);
  const show = (text) => { $("progress").hidden = false; $("progress").textContent = text; };
  show("Rehearsing on Isaac Sim…");
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 4000));
    const job = await (await fetch(`/api/job?id=${response.job}`)).json();
    const elapsed = Math.round((performance.now() - started) / 1000);
    if (job.error) { show(`Rehearsal error: ${job.error}`); return; }
    if (job.status === "running") {
      show(`Rehearsing on Isaac Sim… ${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}` +
           (moves > 1 ? ` · move ${job.move ?? 1} of ${moves}` : "") +
           (job.phases?.length ? ` · ${PHASES[job.phases[job.phases.length - 1]] ?? job.phases[job.phases.length - 1]}` : ""));
      continue;
    }
    $("progress").hidden = true;
    if (!job.demo) { show(`Rehearsal finished without a replay (${job.summary?.status ?? "unknown"}).`); return; }
    DEMOS.push({ id: job.demo, label: `Your ${moves > 1 ? `${moves} moves` : "move"} (${job.summary.status.replaceAll("_", " ")})` });
    renderTabs();
    await load(job.demo);
    return;
  }
}

renderer.domElement.addEventListener("pointerdown", (e) => { plan.down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!plan.view || state.demo || !plan.down || Math.hypot(e.clientX - plan.down[0], e.clientY - plan.down[1]) > 5) return;
  const rect = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
  const hit = raycaster.intersectObjects([...plan.items.values(), plan.floor], false)[0];
  if (!hit) return;
  if (hit.object.userData.item) selectItem(hit.object.userData.item);
  else if (plan.selected) placeGhost(hit.point);
});
$("rot-l").addEventListener("click", () => { if (plan.pose) { plan.turned = true; drawGhost([plan.pose[0], plan.pose[1], plan.pose[2] + Math.PI / 12]); } });
$("rot-r").addEventListener("click", () => { if (plan.pose) { plan.turned = true; drawGhost([plan.pose[0], plan.pose[1], plan.pose[2] - Math.PI / 12]); } });
["fix-w", "fix-d", "fix-h", "fix-m", "fix-alike"].forEach((id) => $(id).addEventListener("change", () => {
  resizeSelected();
  clearGhost();
  showFeasible().catch((error) => { $("dest-hint").textContent = error.message; });
}));
$("check").addEventListener("click", () => checkMove().catch((error) => { $("plan-result").innerHTML = `<p class="bad-text">${error.message}</p>`; }));
$("rehearse").addEventListener("click", () => rehearseMove().catch((error) => { $("progress").hidden = false; $("progress").textContent = error.message; }));
$("undo").addEventListener("click", undoMove);
$("fix-carry").addEventListener("click", () => {
  // The trained carry tier: the cabinet size (spec order) and a mass inside its range.
  const tier = plan.view.carry_tier;
  ["fix-w", "fix-d", "fix-h"].forEach((id, i) => { $(id).value = tier.size_m[i]; });
  $("fix-m").value = 3;
  $("fix-m").dispatchEvent(new Event("change"));
});

let scenes = [];
// Scenes kept out of the Try it tabs (their demos and requests still use them).
const HIDDEN_TRY = new Set(["living_room_bookcase", "living_room_cabinet"]);
function renderTabs() {
  $("tabs").innerHTML = DEMOS.map((d) => `<button type="button" data-id="${d.id}">${d.label}</button>`).join("") +
    scenes.map((id) => `<button type="button" class="try" data-scene="${id}">Try it · ${id.replace("living_room_", "")} room</button>`).join("");
}
renderTabs();
// The Try it tabs need the local rehearsal server (scripts/rehearsal_server.py); a plain static server has no /api.
fetch("/api/scenes").then((r) => (r.ok ? r.json() : [])).then((list) => {
  scenes = Array.isArray(list) ? list.filter((id) => !HIDDEN_TRY.has(id)) : [];
  renderTabs();
}).catch(() => {});
$("tabs").addEventListener("click", (e) => {
  if (e.target.dataset.id) load(e.target.dataset.id).catch(fail);
  if (e.target.dataset.scene) loadPlan(e.target.dataset.scene).catch(fail);
});
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

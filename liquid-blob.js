import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

/* ------------------------------------------------------------------
   Tweak these to play with size / shape / motion / look.
   ------------------------------------------------------------------ */
const CONFIG = {
  // shape
  scale: 0.6,             // overall size
  amplitude: 0.42,        // how far lobes bulge out
  frequency: 0.7,         // lower = fewer, bigger lobes
  warp: 0.75,             // domain warp: higher = more swirling, folded, liquid shapes
  detail: 0.06,           // small secondary ripples on top of the big lobes (0 = none)

  // motion
  flowSpeed: 0.3,         // how fast the surface flows/morphs
  rotationSpeed: 0.12,    // tumble speed (radians/sec)
  stretchAmount: 0.14,    // slow squash/stretch along drifting axes (0 = off)
  stretchSpeed: 0.3,

  // look
  color: 0x020202,
  roughness: 0.48,        // lower = sharper reflections
  clearcoat: 0.2,         // glossy lacquer layer
  clearcoatRoughness: 0.4,
  envIntensity: 0.14,     // studio reflection strength
  keyLight: 0.6,          // top-right highlight strength

  cameraDistance: 3.4,
  fov: 35,
  segments: 220,          // mesh resolution (higher = smoother, more GPU)
  pixelRatioCap: 1.75,
};

const DISPLACE_GLSL = /* glsl */ `
  uniform float uTime;
  uniform float uAmplitude;
  uniform float uFrequency;
  uniform float uWarp;
  uniform float uDetail;

  vec3 mod289(vec3 x){return x - floor(x * (1.0/289.0)) * 289.0;}
  vec4 mod289(vec4 x){return x - floor(x * (1.0/289.0)) * 289.0;}
  vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

  float snoise(vec3 v){
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  // Domain-warped noise: the sample point is pushed around by another noise
  // field that itself drifts over time, which gives folding, water-like flow.
  float blobField(vec3 dir){
    float t = uTime;
    vec3 q = dir * uFrequency;
    vec3 w = vec3(
      snoise(q + vec3(0.0, t * 0.7, 0.0)),
      snoise(q + vec3(5.2, 1.3, t * 0.6)),
      snoise(q + vec3(t * 0.5, 9.1, 2.7))
    );
    q += w * uWarp;
    float n = snoise(q + vec3(t * 0.35));
    n += snoise(q * 2.2 - vec3(t * 0.25)) * uDetail;
    return n;
  }

  vec3 blobDisplace(vec3 dir){
    return dir * (1.0 + blobField(dir) * uAmplitude);
  }

  // Branchless orthonormal basis (Duff et al. 2017), stable at the poles.
  void blobBasis(vec3 n, out vec3 b1, out vec3 b2){
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    b2 = vec3(b, s + n.y * n.y * a, -n.y);
  }
`;

const NORMAL_GLSL = /* glsl */ `
  vec3 blobDir = normalize(position);
  vec3 blobT, blobB;
  blobBasis(blobDir, blobT, blobB);
  const float BLOB_EPS = 0.004;
  vec3 blobP  = blobDisplace(blobDir);
  vec3 blobPT = blobDisplace(normalize(blobDir + blobT * BLOB_EPS));
  vec3 blobPB = blobDisplace(normalize(blobDir + blobB * BLOB_EPS));
  vec3 objectNormal = normalize(cross(blobPT - blobP, blobPB - blobP));
  if (dot(objectNormal, blobDir) < 0.0) objectNormal = -objectNormal;
`;

class LiquidBlob {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.config = { ...CONFIG, ...opts };
    this.clock = new THREE.Clock(false);
    this.running = false;
    this.elapsed = 0;

    this._initScene();
    this._bindEvents();
    this.resize();
    this.start();
  }

  _initScene() {
    const c = this.config;

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, c.pixelRatioCap));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = this.envTexture;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(c.fov, 1, 0.1, 100);
    this.camera.position.set(0, 0, c.cameraDistance);

    const key = new THREE.DirectionalLight(0xffffff, c.keyLight);
    key.position.set(2.5, 3, 2);
    this.scene.add(key);

    this.uniforms = {
      uTime: { value: 0 },
      uAmplitude: { value: c.amplitude },
      uFrequency: { value: c.frequency },
      uWarp: { value: c.warp },
      uDetail: { value: c.detail },
    };

    const material = new THREE.MeshPhysicalMaterial({
      color: c.color,
      metalness: 0.0,
      roughness: c.roughness,
      clearcoat: c.clearcoat,
      clearcoatRoughness: c.clearcoatRoughness,
      envMapIntensity: c.envIntensity,
    });

    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${DISPLACE_GLSL}`)
        .replace('#include <beginnormal_vertex>', NORMAL_GLSL)
        .replace('#include <begin_vertex>', 'vec3 transformed = blobP;');
    };

    const geometry = new THREE.SphereGeometry(1, c.segments, Math.round(c.segments * 0.75));
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.mesh.scale.setScalar(c.scale);
    this.scene.add(this.mesh);

    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  _bindEvents() {
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);

    this._onVisibility = () => (document.hidden ? this.stop() : this.start());
    document.addEventListener('visibilitychange', this._onVisibility);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.clock.start();
    this._tick();
  }

  stop() {
    this.running = false;
    this.clock.stop();
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  _tick = () => {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this._tick);

    const c = this.config;
    // clamp so a dropped frame never makes the shape jump
    const dt = Math.min(this.clock.getDelta(), 1 / 30) * (this.reducedMotion ? 0.15 : 1);
    this.elapsed += dt;

    this.uniforms.uTime.value += dt * c.flowSpeed;
    this.mesh.rotation.y += c.rotationSpeed * dt;
    this.mesh.rotation.x = Math.sin(this.elapsed * c.rotationSpeed * 0.6) * 0.35;

    const st = this.elapsed * c.stretchSpeed;
    this.mesh.scale.set(
      c.scale * (1 + Math.sin(st) * c.stretchAmount),
      c.scale * (1 + Math.sin(st * 1.3 + 2.1) * c.stretchAmount),
      c.scale * (1 + Math.sin(st * 0.7 + 4.2) * c.stretchAmount)
    );

    this.renderer.render(this.scene, this.camera);
  };

  destroy() {
    this.stop();
    window.removeEventListener('resize', this._onResize);
    document.removeEventListener('visibilitychange', this._onVisibility);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.envTexture.dispose();
    this.renderer.dispose();
  }
}

function mountAll() {
  document.querySelectorAll('canvas.silk-gif').forEach((canvas) => new LiquidBlob(canvas));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mountAll);
} else {
  mountAll();
}

export { LiquidBlob, CONFIG };

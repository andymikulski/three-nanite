import {
  Color,
  DirectionalLight,
  Fog,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  Vector3,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const SKY = 0x9ec3e6;
const SUN_DIRECTION = new Vector3(0.45, 1, 0.3).normalize();

/** Scene, camera, lights and ground. `fit` resizes the shadow camera, the ground and the view to the content. */
export class Stage {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(45, 1, 0.05, 1000);
  readonly sun = new DirectionalLight(0xfff1de, 3);
  readonly ground: Mesh;
  controls: OrbitControls | null = null;
  private readonly fog = new Fog(SKY, 10, 100);

  constructor() {
    this.scene.background = new Color(SKY);
    this.scene.fog = this.fog;
    this.scene.add(new HemisphereLight(0xcfe4ff, 0x6b5f52, 1.1));

    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.01;
    this.scene.add(this.sun, this.sun.target);

    this.ground = new Mesh(new PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new MeshStandardMaterial({ color: 0xb9b3a6, roughness: 1 }));
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
  }

  attach(domElement: HTMLElement): void {
    const controls = new OrbitControls(this.camera, domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.495;
    this.controls = controls;
  }

  /** @param half Half-size of the square footprint of the content (it stands on y = 0, centred on the origin). */
  fit(half: number, height: number): void {
    const radius = half * Math.SQRT2 + height;

    const shadow = this.sun.shadow.camera;
    shadow.left = shadow.bottom = -radius;
    shadow.right = shadow.top = radius;
    shadow.near = 0.1;
    shadow.far = radius * 4;
    shadow.updateProjectionMatrix();
    this.sun.position.copy(SUN_DIRECTION).multiplyScalar(radius * 2);
    this.sun.target.position.set(0, 0, 0);

        const distance = (radius * 0.8) / Math.tan((this.camera.fov * Math.PI) / 360);
    this.ground.scale.setScalar(radius * 50);
    this.fog.near = distance * 2.5;
    this.fog.far = distance * 8;
    this.camera.near = Math.max(0.02, distance / 400);
    this.camera.far = distance * 12;
    this.camera.position.set(0.55, 0.5, 1).normalize().multiplyScalar(distance);
    this.camera.updateProjectionMatrix();
    const controls = this.controls;
    if (controls) {
      controls.target.set(0, height * 0.4, 0);
      controls.maxDistance = distance * 3;
      controls.update();
    }
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
}

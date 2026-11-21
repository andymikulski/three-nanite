import { CatmullRomCurve3, Vector3 } from "three";
import type { PerspectiveCamera } from "three";
import { heightAt, TRAIL } from "./terrain.js";

export type CameraMode = "tour" | "fly" | "walk";

/**
 * Drag to look, W A S D to move, Q / E down and up, Shift faster, wheel to move along the view.
 * "walk" keeps the eye 1.7 m above the ground; "tour" glides along the trail by itself until
 * something is pressed.
 */
export class Controls {
  mode: CameraMode = "tour";
  speed = 20;
  /** Called when input takes the camera off the tour. */
  onLeaveTour: (() => void) | null = null;

  private yaw = 0;
  private pitch = 0;
  private readonly keys = new Set<string>();
  private dragging = false;
  private tourT = 0;
  private readonly tour: CatmullRomCurve3;
  private readonly look: CatmullRomCurve3;

  constructor(
    private readonly camera: PerspectiveCamera,
    element: HTMLElement,
  ) {
    // the tour: along the trail, a few metres up, looking ahead and a little down
    const points = TRAIL.map(([x, z]) => new Vector3(x, 0, z));
    const path = new CatmullRomCurve3(points, false, "centripetal");
    const eye: Vector3[] = [];
    const ahead: Vector3[] = [];
    const n = 160;
    for (let i = 0; i <= n; i++) {
      const p = path.getPointAt(i / n);
      const sway = Math.sin(i / 9) * 14;
      const t = path.getTangentAt(i / n);
      const x = p.x - t.z * sway;
      const z = p.z + t.x * sway;
      eye.push(new Vector3(x, heightAt(x, z) + 6 + 10 * (0.5 + 0.5 * Math.sin(i / 23)), z));
      const q = path.getPointAt(Math.min(1, i / n + 0.04));
      ahead.push(new Vector3(q.x + t.x * 40, heightAt(q.x, q.z) + 2, q.z + t.z * 40));
    }
    // there and back again
    this.tour = new CatmullRomCurve3([...eye, ...eye.slice(1, -1).reverse()], true);
    const back = ahead.map((a, i) => {
      const e = eye[i];
      return new Vector3(2 * e.x - a.x, a.y, 2 * e.z - a.z);
    });
    this.look = new CatmullRomCurve3([...ahead, ...back.slice(1, -1).reverse()], true);

    element.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      element.setPointerCapture(e.pointerId);
      this.takeOver();
    });
    element.addEventListener("pointerup", () => (this.dragging = false));
    element.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      this.yaw -= e.movementX * 0.003;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch - e.movementY * 0.003));
    });
    element.addEventListener(
      "wheel",
      (e) => {
        this.takeOver();
        this.camera.position.addScaledVector(this.forward(new Vector3()), -e.deltaY * 0.05);
      },
      { passive: true },
    );
    window.addEventListener("keydown", (e) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      this.keys.add(e.code);
      if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(e.code)) this.takeOver();
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
  }

  /** Starts the tour where the camera is closest to it. */
  startTour(): void {
    this.mode = "tour";
    let best = Infinity;
    for (let i = 0; i < 200; i++) {
      const d = this.tour.getPointAt(i / 200).distanceTo(this.camera.position);
      if (d < best) {
        best = d;
        this.tourT = i / 200;
      }
    }
  }

  private takeOver(): void {
    if (this.mode !== "tour") return;
    this.mode = "fly";
    this.syncAngles();
    this.onLeaveTour?.();
  }

  private syncAngles(): void {
    const f = this.camera.getWorldDirection(new Vector3());
    this.yaw = Math.atan2(-f.x, -f.z);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, f.y)));
  }

  setView(position: Vector3, target: Vector3): void {
    this.camera.position.copy(position);
    this.camera.lookAt(target);
    this.syncAngles();
  }

  private forward(v: Vector3): Vector3 {
    return v.set(-Math.sin(this.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(this.yaw) * Math.cos(this.pitch));
  }

  update(dt: number): void {
    const camera = this.camera;
    if (this.mode === "tour") {
      // about 9 m/s along the way
      this.tourT = (this.tourT + (dt * 9) / this.tour.getLength()) % 1;
      camera.position.copy(this.tour.getPointAt(this.tourT));
      const ground = heightAt(camera.position.x, camera.position.z) + 2;
      if (camera.position.y < ground) camera.position.y = ground;
      camera.lookAt(this.look.getPointAt(this.tourT));
      return;
    }
    const f = this.forward(new Vector3());
    const flat = new Vector3(f.x, 0, f.z).normalize();
    const right = new Vector3(-flat.z, 0, flat.x);
    const move = new Vector3();
    const k = this.keys;
    const along = this.mode === "walk" ? flat : f;
    if (k.has("KeyW")) move.add(along);
    if (k.has("KeyS")) move.sub(along);
    if (k.has("KeyD")) move.add(right);
    if (k.has("KeyA")) move.sub(right);
    if (k.has("KeyE")) move.y += 1;
    if (k.has("KeyQ")) move.y -= 1;
    const speed = (this.mode === "walk" ? 4 : this.speed) * (k.has("ShiftLeft") || k.has("ShiftRight") ? 4 : 1);
    if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), speed * dt);
    const ground = heightAt(camera.position.x, camera.position.z);
    if (this.mode === "walk") camera.position.y = ground + 1.7;
    else if (camera.position.y < ground + 0.5) camera.position.y = ground + 0.5;
    camera.lookAt(camera.position.clone().add(f));
  }
}

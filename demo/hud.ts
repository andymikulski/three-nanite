/** Plain-DOM heads-up display (top-left) and build/status overlay (bottom-left). */
const hud = document.getElementById("hud") as HTMLElement;
const overlay = document.getElementById("overlay") as HTMLElement;

export interface HudInfo {
  renderer: string;
  fps: number;
  instances: number;
  clusters: number;
  /** Triangles drawn by the LOD selection (or by the reference mesh in comparison mode). */
  triangles: number;
  fullTriangles: number;
  slots: number;
  drawCapacity: number;
  updateMs: number;
  drawCalls: number;
  comparing: boolean;
}

const fmt = (n: number): string => Math.round(n).toLocaleString("en-US");

export function renderHud(i: HudInfo): void {
  const saved = i.fullTriangles > 0 ? (1 - i.triangles / i.fullTriangles) * 100 : 0;
  hud.textContent = [
    i.renderer,
    `fps          ${i.fps.toFixed(0)}`,
    `instances    ${fmt(i.instances)}`,
    `clusters     ${fmt(i.clusters)}`,
    `triangles    ${fmt(i.triangles)}`,
    `full detail  ${fmt(i.fullTriangles)}`,
    `saved        ${i.comparing ? "(full detail shown)" : `${saved.toFixed(1)} %`}`,
    `slots        ${fmt(i.slots)} / ${fmt(i.drawCapacity)}`,
    `LOD update   ${i.updateMs.toFixed(2)} ms`,
    `draw calls   ${fmt(i.drawCalls)}`,
  ].join("\n");
}

export function setStatus(text: string, isError = false): void {
  overlay.textContent = text;
  overlay.classList.toggle("error", isError);
}

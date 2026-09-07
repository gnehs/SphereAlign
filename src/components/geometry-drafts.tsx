import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { Box, LoaderCircle, Square } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { localiseUserMessage, type Task } from "@/lib/pipeline";

interface Settings { modelPath: string; frameLimit: number; validityThreshold: number; keepIntermediates: boolean }
interface Frame {
  id: number; name: string; cameraId: number; width: number; height: number;
  validPixels: number; totalPixels: number; rejectionCounts: number[];
  intermediateState?: "retained" | "cleanupPending" | "removed"; removedIntermediateBytes?: number;
  faces: Array<{ face: number; shift: number | null; scale: number; error: string | null }>;
}
interface Run {
  activeForTraining?: boolean;
  id: string; status: string; jobId: string | null; message: string;
  settings: Settings; total: number; completed: number; frames: Frame[];
  datasetSha256: string; outputPath: string; updatedMs: number;
}
interface Status { runs: Run[]; busy: boolean; job: { jobId: string; running: boolean; error: string | null } | null }
interface Preflight {
  registered: number; selected: number; estimatedOutputBytes: number; estimatedPeakBytes: number; modelDownloadBytes: number;
  modelIdentity: string; cameras: Array<{ id: number; model: string; width: number; height: number }>;
}
const selectClass = "h-9 min-w-0 rounded-md border bg-background px-2 text-sm";
const size = (n: number) => `${(n / 1024 ** 3).toFixed(2)} GiB`;

export function GeometryDrafts({ task }: { task: Task }) {
  const [enabled, setEnabled] = useState(false);
  const [settings, setSettings] = useState<Settings>({ modelPath: "", frameLimit: 8, validityThreshold: 0.5, keepIntermediates: false });
  const [status, setStatus] = useState<Status>({ runs: [], busy: false, job: null });
  const [preflight, setPreflight] = useState<Preflight | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [runId, setRunId] = useState("");
  const [frameId, setFrameId] = useState<number>();
  const [kind, setKind] = useState("normal");
  const [image, setImage] = useState("");
  const [loadingImage, setLoadingImage] = useState(false);
  const [stopping, setStopping] = useState(false);
  const initialized = useRef(false);
  const root = task.rootPath;
  const available = !task.previewOnly && task.stages.align.status === "completed";
  const run = status.runs.find((r) => r.id === runId) ?? status.runs[0];
  const frame = run?.frames.find((f) => f.id === frameId) ?? run?.frames[0];
  const running = status.job?.running || status.runs.some((r) => r.status === "running");
  const jobId = status.job?.running ? status.job.jobId : status.runs.find((r) => r.status === "running")?.jobId;

  useEffect(() => {
    if (task.previewOnly) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await invoke<Status>("geometry_status", { projectPath: root });
        if (!active) return;
        setStatus(result);
        if (!result.job?.running) setStopping(false);
        if (!initialized.current && result.runs[0]) {
          initialized.current = true;
          setSettings({ ...result.runs[0].settings, keepIntermediates: result.runs[0].settings.keepIntermediates ?? false });
        }
      } catch (e) { if (active) setError(String(e)); }
      if (active) timer = setTimeout(() => void poll(), 2000);
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [root, task.previewOnly]);

  useEffect(() => {
    setImage("");
    if (!run || !frame) return;
    let active = true;
    let url = "";
    setLoadingImage(true);
    void invoke<ArrayBuffer>("geometry_preview", { projectPath: root, runId: run.id, frameId: frame.id, kind })
      .then((bytes) => {
        if (!active) return;
        url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
        setImage(url);
      }).catch((e) => { if (active) setError(String(e)); })
      .finally(() => { if (active) setLoadingImage(false); });
    return () => { active = false; if (url) URL.revokeObjectURL(url); };
  }, [root, run?.id, frame?.id, kind]);

  if (task.previewOnly) return null;
  const change = (next: Settings) => { setSettings(next); setPreflight(null); };
  const act = async (action: () => Promise<void>) => {
    setWorking(true); setError("");
    try { await action(); } catch (e) { setError(String(e)); } finally { setWorking(false); }
  };
  const start = (config: Settings) => act(async () => {
    const id = await invoke<string>("start_geometry", { projectPath: root, settings: config });
    setStatus((s) => ({ ...s, busy: true, job: { jobId: id, running: true, error: null } }));
  });
  const disabled = working || status.busy || Boolean(running);
  const views = [
    ["rgb", t`Original RGB`], ["normal", t`Camera-space normals`], ["range", t`Relative range · false colour`],
    ["validity", t`Model validity`], ["reasons", t`Rejection reasons`],
    ...Array.from({ length: 6 }, (_, f) => [`face${f}`, t`Perspective ${f}`]),
  ];
  return <section className="border-b py-5" aria-labelledby="geometry-title">
    <div className="mb-3 flex items-center justify-between gap-3">
      <h3 id="geometry-title" className="flex items-center gap-2 font-semibold"><Box className="size-4" /><Trans>Normal maps</Trans></h3>
      <Badge variant="secondary"><Trans>Pipeline stage 4</Trans></Badge>
    </div>
    <p className="text-sm leading-relaxed text-muted-foreground"><Trans>The Normals stage exports all registered images to the training normals folder. Inspect the maps here; relative range remains a preview.</Trans></p>
    {task.stages.normals.status === "completed" && <Button className="mt-3" variant="outline" onClick={() => void act(async () => {
      await openPath(`${task.outputPath}/geometry/spirula-normals.json`);
    })}><Trans>Open Spirula training configuration</Trans></Button>}
    <label className="mt-4 flex items-center gap-2 text-sm">
      <input type="checkbox" checked={enabled} disabled={disabled} onChange={(e) => setEnabled(e.target.checked)} />
      <Trans>Advanced: generate a separate preview sample</Trans>
    </label>
    {!available && <p className="mt-2 text-xs text-muted-foreground"><Trans>Complete alignment to use the final registered cameras and original images.</Trans></p>}
    {enabled && <div className="mt-4 space-y-3">
      <div className="rounded-md border bg-muted/20 p-3 text-xs leading-relaxed">
        <strong>MoGe-2 ViT-B · DirectML</strong>
        <p className="mt-1"><Trans>Windows GPU · 768 × 768 per perspective · 1,800 tokens. The model runs locally. An empty model path downloads the pinned 420 MB model on first generation.</Trans></p>
      </div>
      <label className="block text-xs"><Trans>Model file (optional)</Trans>
        <div className="mt-1 flex gap-2"><Input aria-label={t`Model file (optional)`} value={settings.modelPath} disabled={disabled} onChange={(e) => change({ ...settings, modelPath: e.target.value })} placeholder={t`Download pinned model automatically`} />
          <Button variant="outline" disabled={disabled} onClick={() => void act(async () => {
            const path = await open({ multiple: false, filters: [{ name: "ONNX", extensions: ["onnx"] }] });
            if (typeof path === "string") change({ ...settings, modelPath: path });
          })}><Trans>Browse</Trans></Button></div>
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="text-xs"><Trans>Image limit (0 = all)</Trans><Input className="mt-1" type="number" min={0} max={1000000} disabled={disabled} value={settings.frameLimit} onChange={(e) => change({ ...settings, frameLimit: Math.max(0, Math.trunc(Number(e.target.value))) })} /></label>
        <label className="text-xs"><Trans>Model validity threshold</Trans><Input className="mt-1" type="number" min={0.1} max={0.95} step={0.05} disabled={disabled} value={settings.validityThreshold} onChange={(e) => change({ ...settings, validityThreshold: Number(e.target.value) })} /></label>
      </div>
      <p className="text-xs text-muted-foreground"><Trans>Images are selected in name order. Validity measures model support, not accuracy. Range has not been aligned to the reconstruction scale.</Trans></p>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={settings.keepIntermediates} disabled={disabled} onChange={(e) => change({ ...settings, keepIntermediates: e.target.checked })} />
        <Trans>Keep intermediate files for debugging</Trans>
      </label>
      <p className="text-xs text-muted-foreground"><Trans>After a complete, verified run, large intermediate files are removed automatically. PNG maps, previews and run history are kept. Regenerate with this option enabled if you later need the original floating-point data.</Trans></p>
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={disabled || !available} onClick={() => void act(async () => setPreflight(await invoke<Preflight>("geometry_preflight", { projectPath: root, settings })))}><Trans>Check inputs</Trans></Button>
        <Button disabled={disabled || !available || !preflight} onClick={() => void start(settings)}><Trans>Generate / resume draft</Trans></Button>
      </div>
      {preflight && <div className="rounded-md border p-3 text-xs leading-relaxed">
        <p><Trans>{preflight.selected} of {preflight.registered} registered images</Trans> · <Trans>Estimated output</Trans>: {size(preflight.estimatedOutputBytes)}</p>
        <p><Trans>Estimated peak disk space during generation</Trans>: {size(preflight.estimatedPeakBytes)}</p>
        {preflight.cameras.map((c) => <p key={c.id}>Camera {c.id} · {c.model} · {c.width} × {c.height}</p>)}
        {preflight.modelDownloadBytes > 0 && <p><Trans>First-run model storage</Trans>: {size(preflight.modelDownloadBytes * 2)}</p>}
      </div>}
    </div>}
    {running && <div className="mt-4 rounded-md border p-3 text-sm" role="status">
      <p className="flex items-center gap-2"><LoaderCircle className="size-4 animate-spin" />{status.runs.find((r) => r.status === "running")?.message || t`Preparing Geometry`}</p>
      <Button className="mt-2" variant="outline" disabled={stopping || !jobId} onClick={() => void act(async () => {
        const cancelled = await invoke<boolean>("cancel_job", { jobId });
        if (!cancelled) throw new Error(t`This job belongs to another process. Stop it in that process.`);
        setStopping(true);
      })}><Square className="size-3" />{stopping ? t`Stopping after current GPU call…` : t`Cancel`}</Button>
    </div>}
    {run && <div className="mt-4 space-y-3">
      <label className="flex flex-col gap-1 text-xs"><Trans>Generation history</Trans>
        <select className={selectClass} value={run.id} onChange={(e) => { setRunId(e.target.value); setFrameId(undefined); }}>
          {status.runs.map((r) => <option value={r.id} key={r.id}>{new Date(r.updatedMs).toLocaleString()} · {r.completed}/{r.total} · {r.status}</option>)}
        </select>
      </label>
      <p className="break-words text-xs text-muted-foreground">{localiseUserMessage(run.message)}</p>
      {run.frames.some((f) => f.intermediateState === "removed") && <p className="text-xs text-muted-foreground"><Trans>Intermediate files cleaned</Trans> · {size(run.frames.reduce((total, f) => total + (f.removedIntermediateBytes ?? 0), 0))}</p>}
      {preflight && preflight.modelIdentity !== run.datasetSha256 && <p className="text-xs text-warning"><Trans>The final reconstruction has changed. Generate a new draft before reviewing this dataset.</Trans></p>}
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={disabled} onClick={() => void act(async () => { await openPath(run.outputPath); })}><Trans>Open maps folder</Trans></Button>
        {run.status !== "running" && run.status !== "completed" && <Button variant="outline" disabled={disabled || !available || !enabled} onClick={() => { change(run.settings); void start(run.settings); }}><Trans>Resume this configuration</Trans></Button>}
      </div>
      {frame && <>
        <label className="flex flex-col gap-1 text-xs"><Trans>Registered image</Trans><select className={selectClass} value={frame.id} onChange={(e) => setFrameId(Number(e.target.value))}>{run.frames.map((f) => <option key={f.id} value={f.id}>{f.name} · camera {f.cameraId}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-xs"><Trans>Preview</Trans><select className={selectClass} value={kind} onChange={(e) => setKind(e.target.value)}>{views.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
        <div className="flex min-h-40 items-center justify-center overflow-hidden rounded-md border bg-black">
          {loadingImage ? <LoaderCircle className="size-5 animate-spin text-white" /> : image && <img src={image} alt={`${frame.name} — ${views.find(([id]) => id === kind)?.[1]}`} className="max-h-[30rem] w-full object-contain" />}
        </div>
        <p className="text-xs text-muted-foreground">{frame.width} × {frame.height} · <Trans>Valid support</Trans>: {(100 * frame.validPixels / frame.totalPixels).toFixed(1)}% · <Trans>Review scene quality after training.</Trans></p>
        {kind === "range" && <p className="text-xs text-muted-foreground"><Trans>Blue → green → red: near → far, with a separate display scale for each image. Black pixels are invalid. Colours are not comparable between images.</Trans></p>}
        {kind === "reasons" && <p className="text-xs text-muted-foreground"><Trans>Green: valid · black: outside lens · amber: source mask · grey: model/recovery invalid · magenta: perspective disagreement</Trans></p>}
        <details className="rounded-md border p-2 text-xs"><summary className="cursor-pointer"><Trans>Inference details</Trans></summary>
          <p className="mt-2"><Trans>Rejection counts (lens / mask / model / overlap)</Trans>: {frame.rejectionCounts.slice(1).join(" / ")}</p>
          {frame.faces.map((f) => <p className="mt-1" key={f.face}>Perspective {f.face}: {f.error || `Z shift ${f.shift?.toFixed(4)} · model scale ${f.scale.toFixed(4)}`}</p>)}
        </details>
      </>}
    </div>}
    {(error || status.job?.error) && <p className="mt-3 break-words text-sm text-destructive" role="alert">{error || status.job?.error}</p>}
  </section>;
}

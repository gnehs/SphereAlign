import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { openPath } from "@tauri-apps/plugin-opener";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { AlertTriangle, FileText, LoaderCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Task } from "@/lib/pipeline";
import { useAppStore } from "@/stores/app-store";

interface QualityReport {
  status: "issues_found" | "needs_visual_review" | "stale";
  expectedFrames: number;
  completeFrames: number;
  sources: Array<{
    source: string; expectedFrames: number; completeFrames: number;
    coverageRatio: number; longestMissingRun: number;
    weakFrames: string[]; pathJumpFrames: string[];
  }>;
  observations: {
    invalidFeatureCoordinates: number; invalidTrackReferences: number;
    invalidDepthOrProjection: number; maskedObservations: number; reprojectionOutliers: number;
    medianReprojectionErrorPx: number | null; p95ReprojectionErrorPx: number | null;
  };
  issues: string[];
  examples: string[];
}

export function ReconstructionQuality({ task }: { task: Task }) {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);
  const colmapPath = useAppStore((state) => state.colmapPath);
  const status = task.stages.align.status;
  const available = status === "completed" || status === "failed";
  useEffect(() => {
    let active = true;
    setReport(null);
    setError("");
    if (!available || task.previewOnly) return;
    void invoke<QualityReport | null>("read_reconstruction_quality", { projectPath: task.rootPath })
      .then((value) => { if (active) setReport(value); })
      .catch((reason) => { if (active) setError(String(reason)); });
    return () => { active = false; };
  }, [available, task.previewOnly, task.rootPath, task.stages.align.finishedAtMs, task.stages.align.updatedAtMs]);

  if (!available || task.previewOnly) return null;
  const checkQuality = async () => {
    setChecking(true);
    setError("");
    try { setReport(await invoke<QualityReport>("audit_existing_alignment", { projectPath: task.rootPath, colmapPath: colmapPath || null })); }
    catch (reason) { setError(String(reason)); }
    finally { setChecking(false); }
  };
  const openReport = async () => {
    try { await openPath(`${task.outputPath}/metadata/reconstruction_quality.json`); }
    catch (reason) { setError(String(reason)); }
  };
  const pixels = (value: number | null) => value === null ? "—" : `${value.toFixed(2)} px`;
  const obs = report?.observations;
  return <section className="border-b py-5" aria-labelledby="reconstruction-quality-title">
    <div className="mb-3 flex items-center justify-between gap-2">
      <h3 id="reconstruction-quality-title" className="font-semibold"><Trans>Reconstruction quality</Trans></h3>
      <Badge variant={report?.status === "issues_found" ? "destructive" : "secondary"}>
        {report?.status === "issues_found" ? t`Issues found` : report?.status === "stale" ? t`Report out of date` : t`Visual review required`}
      </Badge>
    </div>
    {!report && <p className="text-sm text-muted-foreground"><Trans>Check the existing model to generate a quality report without rerunning alignment.</Trans></p>}
    {report?.status === "stale" ? <p className="text-sm text-warning"><Trans>The model or feature database has changed. Check quality again before using these results.</Trans></p> : report && obs && <>
      <p className="mb-3 text-sm text-muted-foreground"><Trans>Camera registration is complete for {report.completeFrames} of {report.expectedFrames} frame groups. Check each capture separately.</Trans></p>
      <table className="w-full text-left text-xs tabular-nums">
        <thead><tr className="border-b text-muted-foreground">
          <th className="py-2 font-medium"><Trans>Capture</Trans></th>
          <th className="py-2 text-right font-medium"><Trans>Coverage</Trans></th>
          <th className="py-2 text-right font-medium"><Trans>Longest gap</Trans></th>
        </tr></thead>
        <tbody>{report.sources.map((source) => <tr key={source.source} className="border-b">
          <td className="py-2 font-mono">{source.source}</td>
          <td className="py-2 text-right">{source.completeFrames}/{source.expectedFrames} · {(source.coverageRatio * 100).toFixed(1)}%</td>
          <td className="py-2 text-right">{source.longestMissingRun}</td>
        </tr>)}</tbody>
      </table>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-xs">
        <div><dt className="text-muted-foreground"><Trans>Native error · median / P95</Trans></dt><dd className="mt-1 font-mono">{pixels(obs.medianReprojectionErrorPx)} / {pixels(obs.p95ReprojectionErrorPx)}</dd></div>
        <div><dt className="text-muted-foreground"><Trans>Observations inside masks</Trans></dt><dd className="mt-1 font-mono">{obs.maskedObservations.toLocaleString()}</dd></div>
        <div><dt className="text-muted-foreground"><Trans>Feature / track inconsistencies</Trans></dt><dd className="mt-1 font-mono">{(obs.invalidFeatureCoordinates + obs.invalidTrackReferences).toLocaleString()}</dd></div>
        <div><dt className="text-muted-foreground"><Trans>Weak frames / possible jumps</Trans></dt><dd className="mt-1 font-mono">{report.sources.reduce((n, source) => n + source.weakFrames.length, 0)} / {report.sources.reduce((n, source) => n + source.pathJumpFrames.length, 0)}</dd></div>
      </dl>
      {report.issues.length > 0 && <details className="mt-3 rounded-md border p-2 text-xs">
        <summary className="cursor-pointer font-medium"><Trans>Inspect quality findings</Trans></summary>
        <ul className="mt-2 list-disc space-y-1 break-words pl-4">{report.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>
      </details>}
    </>}
    <div className="mt-4 flex gap-2 text-sm text-muted-foreground">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <p><Trans>Low pixel error and connected matches do not prove room topology or scale. Compare doors and walls with the original captures before training.</Trans></p>
    </div>
    <details className="mt-3 text-sm">
      <summary className="cursor-pointer font-medium"><Trans>Before tile training</Trans></summary>
      <ul className="mt-2 list-disc space-y-2 pl-4 text-muted-foreground">
        <li><Trans>Keep a single global alignment and paired lens frames. Tiles share its camera poses and calibration.</Trans></li>
        <li><Trans>Train with overlapping halos, then crop to non-overlapping cores. Check both sides of doorways and tile boundaries.</Trans></li>
        <li><Trans>Use the points training frame and save evaluation views. Check ghosting, people and floaters from original camera positions.</Trans></li>
        <li><Trans>Keep display scaling separate from physical calibration. A completed training run still needs visual review.</Trans></li>
      </ul>
    </details>
    <div className="mt-3 flex flex-wrap gap-2">
      <Button variant="outline" size="sm" disabled={checking} onClick={() => void checkQuality()}>{checking && <LoaderCircle className="animate-spin" />}{checking ? t`Checking quality…` : t`Check existing model`}</Button>
      {report && <Button variant="outline" size="sm" onClick={() => void openReport()}><FileText /><Trans>Open quality report</Trans></Button>}
    </div>
    {error && <p className="mt-2 break-words text-xs text-destructive" role="alert">{error}</p>}
  </section>;
}

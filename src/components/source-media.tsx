import { invoke } from "@tauri-apps/api/core";
import { CircleAlert, CircleDashed, Video, X, type LucideIcon } from "lucide-react";
import { t } from "@lingui/core/macro";
import { useEffect, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { localiseUserMessage, type SourceIssue, type SourceMedia } from "@/lib/pipeline";
import { cn } from "@/lib/utils";

const IS_TAURI_RUNTIME = typeof window !== "undefined" && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
const MAX_SOURCE_PREVIEW_CACHE_ENTRIES = 64;
const sourcePreviewRequests = new Map<string, Promise<ArrayBuffer>>();

function loadSourcePreview(path: string) {
  const cached = sourcePreviewRequests.get(path);
  if (cached) return cached;
  const request = invoke<ArrayBuffer>("source_preview", { path });
  sourcePreviewRequests.set(path, request);
  void request.then(() => {
    while (sourcePreviewRequests.size > MAX_SOURCE_PREVIEW_CACHE_ENTRIES) {
      const oldestPath = sourcePreviewRequests.keys().next().value;
      if (oldestPath === undefined) break;
      sourcePreviewRequests.delete(oldestPath);
    }
  }, () => {
    if (sourcePreviewRequests.get(path) === request) sourcePreviewRequests.delete(path);
  });
  return request;
}

function sourceIssueSeverityLabel(severity: SourceIssue["severity"]) {
  return severity === "error" ? t`Error` : t`Warning`;
}

function sourceIssueImpactLabel(impact: string) {
  const normalised = impact.trim().toLowerCase().replace(/[\s_-]+/g, "");
  if (["sourcedecode", "decoding", "decodeunavailable"].includes(normalised)) {
    return t`Source decoding may be unavailable`;
  }
  if (["dualfisheyeextraction", "fisheyeextraction"].includes(normalised)) {
    return t`Dual-fisheye extraction may be unavailable`;
  }
  if (["groundalignment", "groundcorrection", "groundcorrectionunavailable", "groundalignmentunavailable"].includes(normalised)) {
    return t`Ground correction may be unavailable`;
  }
  if (["reconstructionacceleration", "reconstructionaccelerationunavailable", "imukeyframe", "imukeyframeselection"].includes(normalised)) {
    return t`Reconstruction acceleration may be unavailable`;
  }
  if (["imufusion", "fusedattitude", "fusedattitudeunavailable"].includes(normalised)) {
    return t`IMU-assisted orientation may be unavailable`;
  }
  return localiseUserMessage(impact);
}

function sourceIssueCopy(issue: SourceIssue) {
  switch (issue.code) {
    case "unsupported-format":
      return { title: t`Unsupported source format`, detail: t`This file type is not supported by the current source adapters.` };
    case "file-not-found":
      return { title: t`Source file not found`, detail: t`The file may have been moved, renamed, or deleted.` };
    case "file-unreadable":
      return { title: t`Source file cannot be read`, detail: t`Check the file permissions and confirm that it is a regular file.` };
    case "ffprobe-unavailable":
      return { title: t`Source decoding was not verified`, detail: t`FFprobe is unavailable, so decoding cannot be checked before processing.` };
    case "decode-failed":
      return { title: t`Source cannot be decoded`, detail: t`FFprobe could not read this file as a supported video source.` };
    case "no-video-stream":
      return { title: t`No video stream found`, detail: t`The container does not include a video stream that can be processed.` };
    case "incomplete-dual-fisheye":
      return { title: t`Matching lens file is missing`, detail: t`Select the matching _00_ or _10_ INSV file to complete the dual-fisheye source.` };
    case "metadata-unavailable":
      return { title: t`Source metadata is unavailable`, detail: t`Camera metadata could not be decoded from this file.` };
    case "imu-unavailable":
      return { title: t`IMU metadata is unavailable`, detail: t`No usable IMU or fused-attitude samples were found.` };
    case "fused-attitude-unavailable":
      return { title: t`Orientation metadata is unavailable`, detail: t`IMU samples were found, but fused-attitude samples were not available.` };
    default:
      return {
        title: localiseUserMessage(issue.message),
        detail: localiseUserMessage(issue.detail),
      };
  }
}

function SourceIssueIndicator({ source }: { source: SourceMedia }) {
  const issues = source.issues ?? [];
  if (!issues.length) return null;
  const hasError = issues.some((issue) => issue.severity === "error");
  const issueLabel = issues.length === 1 ? t`Show source issue` : t`Show ${issues.length} source issues`;
  return (
    <Popover>
      <PopoverTrigger
        render={(
          <button
            type="button"
            className={cn(
              "inline-grid size-5 shrink-0 place-items-center rounded-full outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
              hasError
                ? "text-destructive hover:bg-destructive/10"
                : "text-amber-600 hover:bg-amber-500/10 dark:text-amber-400",
            )}
            aria-label={issueLabel}
          />
        )}
      >
        <CircleAlert className="size-4" strokeWidth={2} aria-hidden="true" />
      </PopoverTrigger>
      <PopoverContent className="w-[min(380px,calc(100vw-32px))] max-h-[min(520px,calc(100vh-32px))] overflow-y-auto p-3" side="right" align="start" sideOffset={8}>
        <PopoverTitle className="mb-2 flex items-center gap-2">
          <CircleAlert className={cn("size-4", hasError ? "text-destructive" : "text-amber-600 dark:text-amber-400")} aria-hidden="true" />
          {t`Source inspection issues`}
        </PopoverTitle>
        <div className="flex flex-col gap-2">
          {issues.map((issue, index) => {
            const impacts = issue.impacts.map(sourceIssueImpactLabel).filter(Boolean);
            const copy = sourceIssueCopy(issue);
            return (
              <Alert key={`${issue.code}-${index}`} variant={issue.severity === "error" ? "destructive" : "default"} className={issue.severity === "warning" ? "border-amber-500/30 bg-amber-500/5 text-amber-950 dark:text-amber-100" : undefined}>
                <CircleAlert aria-hidden="true" />
                <AlertTitle className="flex min-w-0 flex-wrap items-center gap-1.5">
                  <span className="min-w-0 break-words">{copy.title}</span>
                  <Badge variant={issue.severity === "error" ? "destructive" : "outline"} className={issue.severity === "warning" ? "border-amber-500/40 text-amber-700 dark:text-amber-300" : undefined}>{sourceIssueSeverityLabel(issue.severity)}</Badge>
                </AlertTitle>
                <AlertDescription>
                  <p className="break-words">{copy.detail}</p>
                  {impacts.length > 0 && (
                    <div className="mt-2 border-t border-current/15 pt-2">
                      <p className="mb-1 text-xs font-medium text-current">{t`Possible impact`}</p>
                      <ul className="list-disc space-y-0.5 pl-4 text-xs text-current/80">
                        {impacts.map((impact, impactIndex) => <li key={`${impact}-${impactIndex}`}>{impact}</li>)}
                      </ul>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            );
          })}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function SourceThumbnail({
  source,
  previewSide = "right",
  size = "default",
}: {
  source: SourceMedia;
  previewSide?: "left" | "right";
  size?: "default" | "compact";
}) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!IS_TAURI_RUNTIME) {
      setFailed(true);
      return;
    }
    let active = true;
    let objectUrl: string | null = null;
    setPreviewUrl(null);
    setFailed(false);
    void loadSourcePreview(source.path)
      .then((bytes) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
        setPreviewUrl(objectUrl);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [source.path]);

  const alt = t`${source.detail}: first-frame preview from the first video stream`;

  if (!previewUrl) {
    return (
      <div
        className={cn(
          "grid aspect-square shrink-0 place-items-center overflow-hidden rounded-full border bg-muted text-muted-foreground",
          size === "compact" ? "size-10.5" : "size-12",
        )}
        title={failed ? t`Unable to generate a first-frame preview` : undefined}
      >
        {failed ? <Video className="size-4.5" aria-hidden="true" /> : <CircleDashed className="size-4.5 animate-spin [animation-duration:900ms]" aria-hidden="true" />}
      </div>
    );
  }

  return (
    <Popover>
      <PopoverTrigger
        openOnHover
        delay={120}
        closeDelay={120}
        render={(
          <button
            type="button"
            className={cn(
              "aspect-square shrink-0 cursor-default overflow-hidden rounded-full border bg-muted p-0 text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              size === "compact" ? "size-10.5" : "size-12",
            )}
            aria-label={t`Preview ${alt}`}
          />
        )}
      >
        <img className="size-full object-cover" src={previewUrl} alt={alt} />
      </PopoverTrigger>
      <PopoverContent className="w-[min(272px,calc(100vw-32px))] overflow-hidden rounded-full p-1.5 shadow-[0_18px_48px_rgb(0_0_0/18%),0_2px_8px_rgb(0_0_0/10%)]" side={previewSide} sideOffset={12}>
        <PopoverTitle className="sr-only">{t`${source.detail} source snapshot`}</PopoverTitle>
        <img className="block aspect-square w-full rounded-full object-cover" src={previewUrl} alt="" />
      </PopoverContent>
    </Popover>
  );
}

export function SupportedFormatCard({ icon: Icon, title, detail }: { icon: LucideIcon; title: ReactNode; detail: ReactNode }) {
  return (
    <article className="flex min-w-0 items-center gap-3 rounded-xl border bg-card px-3.5 py-3">
      <Icon className="size-5 shrink-0 text-muted-foreground" strokeWidth={1.6} aria-hidden="true" />
      <span className="flex min-w-0 flex-col gap-0.5">
        <strong className="truncate text-sm font-semibold text-foreground">{title}</strong>
        <small className="truncate text-xs text-muted-foreground">{detail}</small>
      </span>
    </article>
  );
}

export function SourceListItem({
  source,
  title,
  detail,
  previewSide = "right",
  onRemove,
  removeLabel,
}: {
  source: SourceMedia;
  title: ReactNode;
  detail: ReactNode;
  previewSide?: "left" | "right";
  onRemove?: () => void;
  removeLabel?: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2.5 border-t px-2 py-2 first:border-t-0">
      <SourceThumbnail source={source} previewSide={previewSide} />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex min-w-0 items-center gap-1.5">
          <strong className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">{title}</strong>
          <SourceIssueIndicator source={source} />
        </span>
        <small className="truncate font-mono text-xs text-muted-foreground" title={typeof detail === "string" ? detail : undefined}>{detail}</small>
      </span>
      {onRemove && (
        <Button type="button" variant="ghost" size="icon-xs" aria-label={removeLabel} onClick={onRemove}>
          <X />
        </Button>
      )}
    </div>
  );
}

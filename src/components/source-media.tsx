import { invoke } from "@tauri-apps/api/core";
import { CircleDashed, Video, X, type LucideIcon } from "lucide-react";
import { t } from "@lingui/core/macro";
import { useEffect, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import type { OsvSource } from "@/lib/pipeline";
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

export function SourceThumbnail({
  source,
  previewSide = "right",
  size = "default",
}: {
  source: OsvSource;
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

  const alt = t`${source.detail}: first-frame preview from the first lens`;

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
        <PopoverTitle className="sr-only">{t`${source.detail} dual-fisheye snapshot`}</PopoverTitle>
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
  source: OsvSource;
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
        <strong className="truncate text-sm font-medium text-foreground">{title}</strong>
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

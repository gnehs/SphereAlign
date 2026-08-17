import {
  AlertTriangle,
  CircleDashed,
  CircleHelp,
  Film,
  FileVideoCamera,
  FileStack,
  Folder,
  FolderOpen,
  Gpu,
  Info,
  LoaderCircle,
  MonitorCog,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  ScanLine,
  ScanSearch,
  Settings2,
  Square,
  Gauge,
  Minus,
  CheckCircle2,
  Copy,
  Trash2,
  Upload,
  Video,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, useReducedMotion } from "motion/react";
import * as m from "motion/react-m";
import { i18n, type I18n, type MessageDescriptor } from "@lingui/core";
import { msg, plural, t } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { Plural, Trans } from "@lingui/react/macro";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow, ProgressBarStatus } from "@tauri-apps/api/window";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { writeText as writeClipboardText } from "@tauri-apps/plugin-clipboard-manager";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress, ProgressValue } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { TASK_DETAIL_DRAWER_TRANSITION, TaskDetailPanel } from "@/components/task-detail-panel";
import { useTheme, type Theme } from "@/components/theme-provider";
import { getEnglishI18n, getLocale, localeLabels, setLocale, supportedLocales } from "@/i18n";
import { cn } from "@/lib/utils";
import "./App.css";

type StageKey = "extract" | "mask" | "align";
type StageStatus = "pending" | "running" | "completed" | "cancelled" | "failed";
type DiagnosticStatus = "ready" | "warning" | "unknown";
type ExtractColorMode = "auto" | "dlogMRec709" | "native";

function translate(descriptor: MessageDescriptor) {
  return i18n._(descriptor);
}

const LANGUAGE_OPTIONS = supportedLocales.map((value) => ({ value, label: localeLabels[value] }));
const APP_NOTICE_EASE = [0.22, 1, 0.36, 1] as const;

interface ColorInspection {
  shouldApply?: boolean;
}

interface ColorInspectionSummary {
  files: ColorInspection[];
  shouldApply?: boolean;
}

interface StageState {
  status: StageStatus;
  progress: number;
  message: string;
  jobId?: string;
  phase?: string;
  completed?: number;
  total?: number;
  currentItem?: string;
  timestampMs?: number;
  updatedAtMs?: number;
  startedAtMs?: number;
  finishedAtMs?: number;
  durationMs?: number;
}

interface PipelineSettings {
  extract: {
    baseFps: number;
    denseFps: number;
    skipBlurry: boolean;
    colorMode: ExtractColorMode;
    lutPath?: string;
  };
  mask: { yoloEnabled: boolean; classes: string[]; maskSky: boolean; modelDir: string };
  align: {
    useGpu: boolean;
    gpuIndex: string;
  };
}

interface OsvSource {
  id: string;
  path: string;
  label: string;
  detail: string;
  status?: "ready" | "warning" | "unknown";
}

interface ProjectManifest {
  projectId: string;
  name: string;
  rootPath: string;
  inputPaths: string[];
  outputPath: string;
  settings: PipelineSettings;
  stages: Record<StageKey, StageState>;
  logs: TaskLog[];
  warnings: string[];
  createdAt?: string;
  updatedAt?: string;
}

interface Task extends ProjectManifest {
  previewOnly?: boolean;
}

interface DiagnosticItem {
  label: string;
  value: string;
  detail: string;
  details?: string[];
  status: DiagnosticStatus;
}

interface SystemInfo {
  osName: string;
  osVersion: string;
  architecture: string;
  processors: string[];
  graphicsAdapters: string[];
}

type TaskLogKind = "progress" | "message" | "summary";
type TaskLogLevel = "info" | "warning" | "error";

interface TaskLog {
  id: string;
  kind: TaskLogKind;
  stage?: StageKey;
  phase?: string;
  jobId?: string;
  level: TaskLogLevel;
  message: string;
  timestampMs: number;
  startedAtMs?: number;
  finishedAtMs?: number;
  durationMs?: number;
  completed?: number;
  total?: number;
  currentItem?: string;
}

interface DoctorReport {
  platform: string;
  systemInfo: SystemInfo;
  summary: string;
  checkedAt: string;
  items: DiagnosticItem[];
  warnings: string[];
  colmapCapabilities?: Record<string, unknown>;
  gpuAvailable?: boolean;
  gpuDevices: Array<{ index: number; name: string }>;
}

interface ProgressEventPayload {
  stage?: StageKey;
  progress?: number;
  status?: StageStatus;
  message?: string;
  jobId?: string;
  phase?: string;
  completed?: number;
  total?: number;
  currentItem?: string;
  timestampMs?: number;
  elapsedMs?: number;
  done?: boolean;
}

interface LogEventPayload {
  jobId?: string;
  level: TaskLogLevel;
  message: string;
  stage?: StageKey;
  phase?: string;
  timestampMs: number;
  completed?: number;
  total?: number;
  currentItem?: string;
}

interface AutoPipelineRun {
  task: Pick<Task, "rootPath" | "outputPath" | "settings">;
  colmapPath: string;
  nextStage: StageKey;
  paused?: boolean;
  stage?: StageKey;
  jobId?: string;
}

type StageDefinition = { key: StageKey; label: MessageDescriptor; description: MessageDescriptor; icon: LucideIcon };

const STAGES: StageDefinition[] = [
  {
    key: "extract",
    label: msg({ message: "Frame extraction", context: "pipeline stage label", comment: "The stage that extracts selected frames from source media." }),
    description: msg({ message: "Dual-fisheye frames, intrinsics, and IMU", context: "pipeline stage description", comment: "Technical description of the frame extraction stage; keep IMU as the acronym." }),
    icon: ScanLine,
  },
  {
    key: "mask",
    label: msg({ message: "Masking", context: "pipeline stage label", comment: "The stage that creates masks for dynamic objects and sky." }),
    description: msg({ message: "Dynamic-object and sky masks", context: "pipeline stage description", comment: "Technical description of the masking stage." }),
    icon: CircleDashed,
  },
  {
    key: "align",
    label: msg({ message: "Alignment", context: "pipeline stage label", comment: "The stage that aligns multiple OSV sources and camera rigs." }),
    description: msg({ message: "Multi-source OSV and camera-rig alignment", context: "pipeline stage description", comment: "Technical description of the alignment stage; keep OSV as the product format acronym." }),
    icon: Workflow,
  },
];

function stageLabel(stage?: StageDefinition) {
  return stage ? translate(stage.label) : translate(msg({ message: "Stage", context: "pipeline stage fallback", comment: "Generic pipeline stage label." }));
}

function stageDescription(stage: StageDefinition) {
  return translate(stage.description);
}

// Aggregate durations from three completed runs. Keeping the raw
// observations makes the overall progress weighting auditable and easy to tune.
const STAGE_OBSERVED_DURATION_MS: Record<StageKey, number> = {
  extract: 1_498_380,
  mask: 287_773,
  align: 4_941_579,
};
const TOTAL_OBSERVED_DURATION_MS = Object.values(STAGE_OBSERVED_DURATION_MS)
  .reduce((total, duration) => total + duration, 0);

const MASK_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];
const MASK_CLASS_LABELS: Record<string, MessageDescriptor> = {
  person: msg({ message: "Person", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
  bicycle: msg({ message: "Bicycle", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
  car: msg({ message: "Car", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
  motorcycle: msg({ message: "Motorcycle", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
  bus: msg({ message: "Bus", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
  truck: msg({ message: "Truck", context: "mask class", comment: "Object class that can be excluded from reconstruction masks." }),
};
const MIN_CANDIDATE_MULTIPLIER = 2;
const MAX_CANDIDATE_MULTIPLIER = 10;
const DEFAULT_CANDIDATE_MULTIPLIER = 4;
const DEFAULT_SETTINGS: PipelineSettings = {
  extract: {
    baseFps: 3,
    denseFps: 12,
    skipBlurry: true,
    colorMode: "auto",
  },
  mask: { yoloEnabled: false, classes: [], maskSky: false, modelDir: "" },
  align: {
    useGpu: true,
    gpuIndex: "-1",
  },
};
const COLMAP_PATH_STORAGE_KEY = "gs360studio.colmapPath";

function normaliseExtractColorMode(value: unknown): ExtractColorMode {
  const raw = String(value ?? "").trim().toLowerCase().replace(/[\s_-]+/g, "");
  if (["dlogmrec709", "dlogm709", "dlogm709lut", "dlogmtorec709"].includes(raw)) return "dlogMRec709";
  if (["native", "original", "none", "off"].includes(raw)) return "native";
  return "auto";
}

function customLutPathIsInvalid(path?: string) {
  const trimmed = path?.trim() ?? "";
  return Boolean(trimmed) && !trimmed.toLowerCase().endsWith(".cube");
}

function candidateMultiplierFor(extract: PipelineSettings["extract"]): number {
  if (!Number.isFinite(extract.baseFps) || extract.baseFps <= 0 || !Number.isFinite(extract.denseFps)) {
    return DEFAULT_CANDIDATE_MULTIPLIER;
  }
  return Math.min(
    MAX_CANDIDATE_MULTIPLIER,
    Math.max(MIN_CANDIDATE_MULTIPLIER, Math.round(extract.denseFps / extract.baseFps)),
  );
}

function normalisePipelineSettings(value: unknown): PipelineSettings {
  const source = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const extract = source.extract && typeof source.extract === "object" ? source.extract as Record<string, unknown> : {};
  const finiteNumber = (candidate: unknown, fallback: number, min: number, max: number) => {
    const parsed = typeof candidate === "number" ? candidate : Number.NaN;
    return Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : fallback;
  };
  const mask = source.mask && typeof source.mask === "object" ? source.mask as Record<string, unknown> : {};
  const classes = Array.isArray(mask.classes)
    ? mask.classes.filter((item): item is string => typeof item === "string" && MASK_CLASSES.includes(item))
    : DEFAULT_SETTINGS.mask.classes;
  const yoloEnabled = (typeof mask.yoloEnabled === "boolean" ? mask.yoloEnabled : DEFAULT_SETTINGS.mask.yoloEnabled) && classes.length > 0;
  const align = source.align && typeof source.align === "object" ? source.align as Record<string, unknown> : {};
  const rawGpuIndex = align.gpuIndex;
  const gpuIndex = typeof rawGpuIndex === "string"
    ? rawGpuIndex
    : typeof rawGpuIndex === "number" && Number.isFinite(rawGpuIndex)
      ? String(rawGpuIndex)
      : DEFAULT_SETTINGS.align.gpuIndex;
  const baseFps = finiteNumber(extract.baseFps, DEFAULT_SETTINGS.extract.baseFps, 1, 30);
  const lutPath = typeof extract.lutPath === "string" && extract.lutPath.trim()
    ? extract.lutPath.trim()
    : undefined;
  return {
    extract: {
      baseFps,
      denseFps: finiteNumber(extract.denseFps, baseFps * DEFAULT_CANDIDATE_MULTIPLIER, baseFps * MIN_CANDIDATE_MULTIPLIER, baseFps * MAX_CANDIDATE_MULTIPLIER),
      skipBlurry: typeof extract.skipBlurry === "boolean" ? extract.skipBlurry : DEFAULT_SETTINGS.extract.skipBlurry,
      colorMode: normaliseExtractColorMode(extract.colorMode),
      ...(lutPath ? { lutPath } : {}),
    },
    mask: {
      yoloEnabled,
      classes,
      maskSky: typeof mask.maskSky === "boolean" ? mask.maskSky : DEFAULT_SETTINGS.mask.maskSky,
      modelDir: typeof mask.modelDir === "string" ? mask.modelDir : DEFAULT_SETTINGS.mask.modelDir,
    },
    align: {
      useGpu: typeof align.useGpu === "boolean" ? align.useGpu : DEFAULT_SETTINGS.align.useGpu,
      gpuIndex,
    },
  };
}

function selectAvailableGpu(settings: PipelineSettings, devices: DoctorReport["gpuDevices"]): PipelineSettings {
  if (devices.length === 0) return settings;
  const requestedIndex = settings.align.gpuIndex.split(",")[0]?.trim();
  const gpuIndex = devices.some((device) => String(device.index) === requestedIndex)
    ? requestedIndex
    : String(devices[0].index);
  if (gpuIndex === settings.align.gpuIndex) return settings;
  return { ...settings, align: { ...settings.align, gpuIndex } };
}

function gpuDeviceLabel(device: DoctorReport["gpuDevices"][number], devices: DoctorReport["gpuDevices"]): string {
  const matchingDevices = devices.filter((candidate) => candidate.name === device.name);
  if (matchingDevices.length === 1) return device.name;
  const ordinal = matchingDevices.findIndex((candidate) => candidate.index === device.index) + 1;
  return t`${device.name} (GPU ${ordinal})`;
}

const COLMAP_CUDA_DIAGNOSTIC_LABEL = "CUDA acceleration";
const HARDWARE_ACCELERATION_LABEL = "Hardware acceleration";

function emptyDoctor(): DoctorReport {
  return {
    platform: "Platform not checked yet",
    systemInfo: {
      osName: "Not checked yet",
      osVersion: "Not checked yet",
      architecture: "Not checked yet",
      processors: [],
      graphicsAdapters: [],
    },
    summary: "Run an environment check to confirm available capabilities",
    checkedAt: "Not checked yet",
    items: [
      { label: "COLMAP", value: "Not checked yet", detail: "Check the COLMAP executable and version", status: "unknown" },
      { label: COLMAP_CUDA_DIAGNOSTIC_LABEL, value: "Not checked yet", detail: "Check the COLMAP CUDA build and whether an NVIDIA GPU is available", status: "unknown" },
      { label: "FFmpeg", value: "Not checked yet", detail: "Confirm FFmpeg is available on the system PATH", status: "unknown" },
      { label: HARDWARE_ACCELERATION_LABEL, value: "Not checked yet", detail: "Confirm FFmpeg hardware decoding capabilities", status: "unknown" },
    ],
    warnings: [],
    gpuDevices: [],
  };
}

function diagnosticItemLabel(label: string) {
  if (label === COLMAP_CUDA_DIAGNOSTIC_LABEL) return t`CUDA acceleration`;
  if (label === HARDWARE_ACCELERATION_LABEL) return t`Hardware acceleration`;
  return label;
}

const IS_TAURI_RUNTIME = typeof window !== "undefined" && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
const IS_WINDOWS_RUNTIME = IS_TAURI_RUNTIME && typeof navigator !== "undefined" && /Windows/i.test(navigator.userAgent);
const IS_MACOS_RUNTIME = IS_TAURI_RUNTIME && typeof navigator !== "undefined" && /Macintosh|Mac OS X/i.test(navigator.userAgent);

const PREPARING_WORK_SOURCE = "Preparing work";
const BROWSER_PREVIEW_TASK_SOURCE = "Browser preview task";
const BROWSER_PREVIEW_NOT_CONNECTED_SOURCE = "Browser preview: not connected to the local runtime";

const UNKNOWN_BACKEND_ERROR = msg({
  message: "The local runtime returned an unexpected error. See diagnostics for details.",
  context: "backend error fallback",
  comment: "Shown when a backend or Tauri error has no localized mapping. Do not include the raw error text here.",
});

const HIDDEN_TECHNICAL_ERROR_DETAILS = msg({
  message: "Technical details are available in diagnostics.",
  context: "backend error details",
  comment: "Shown when technical details contain an untranslated message in a non-Traditional-Chinese locale.",
});

const HIDDEN_BACKEND_STATUS = msg({
  message: "A backend status message is unavailable in this language.",
  context: "backend status fallback",
  comment: "Shown when an untranslated backend progress or status message contains CJK and has no localized mapping; hide the original to avoid misleading users.",
});

function containsCjk(value: string): boolean {
  return /[\u3400-\u9fff]/.test(value);
}

function localiseTechnicalErrorDetail(value: string): string {
  const detail = value.trim();
  if (!detail) return "";
  // Traditional Chinese is the backend's source locale. English technical
  // output remains useful in every locale, but untranslated CJK diagnostics
  // must not leak into Simplified Chinese or Japanese UI.
  if (getLocale() === "zh-TW" || !containsCjk(detail)) return detail;
  return translate(HIDDEN_TECHNICAL_ERROR_DETAILS);
}

function isLikelyBackendError(value: string): boolean {
  return /(error|failed|failure|unable|cannot|could not|not found|unsupported|missing|unavailable|錯誤|失敗|找不到|無法|不能|不支援|缺少|未能|不符|未提供|不成立|不提供)/i.test(value);
}

const USER_MESSAGE_TRANSLATIONS: Record<string, MessageDescriptor> = {
  "scanning paired fisheye candidates": msg({ message: "Scanning dual-fisheye frame candidates", comment: "Backend progress message: scanning candidate frame pairs." }),
  "cancelled before interval": msg({ message: "Cancelled before processing the next interval", comment: "Backend progress message; interval is a processing window, not a time duration." }),
  "cancelled while scoring candidates": msg({ message: "Cancelled while scoring frame candidates", comment: "Backend progress message: scoring candidate frames for selection." }),
  "selected pair already exists; skipped": msg({ message: "Selected frame pair already exists; skipped", comment: "Backend message: an output pair was already present, so no duplicate was written." }),
  "cancelled before output commit": msg({ message: "Cancelled before committing output", comment: "Backend progress message: output files were not committed." }),
  "copying selected pair": msg({ message: "Copying selected frame pair", comment: "Backend progress message: copying selected source frames." }),
  "selected pair committed": msg({ message: "Selected frame pair committed", comment: "Backend progress message: selected source frames were written to output." }),
  "extraction cancelled": msg({ message: "Frame extraction cancelled", context: "backend stage status", comment: "The frame-extraction stage was cancelled." }),
  "scanning native fisheye images": msg({ message: "Scanning native dual-fisheye images", comment: "Backend progress message: native fisheye images are being inspected." }),
  "loading YOLO11/skyseg models": msg({ message: "Loading YOLO11/SkySeg models", comment: "Keep YOLO11 and SkySeg as model names." }),
  "masking cancelled": msg({ message: "Masking cancelled", context: "backend stage status", comment: "The masking stage was cancelled." }),
  "verified mask exists; skipped": msg({ message: "Verified that the mask exists; skipped", comment: "Backend message: an existing mask file was reused." }),
  "running YOLO11/skyseg": msg({ message: "Running YOLO11/SkySeg inference", comment: "Keep YOLO11 and SkySeg as model names." }),
  "masking cancelled before output commit": msg({ message: "Masking cancelled before committing output", comment: "Backend progress message: mask output files were not committed." }),
  "committing mask files": msg({ message: "Committing mask files", comment: "Backend progress message: mask files are being written." }),
  "mask completed": msg({ message: "Masking completed", context: "backend stage status", comment: "The masking stage completed successfully." }),
  "Stage started": msg({ message: "Stage started", context: "backend stage status", comment: "Generic backend stage status message." }),
  "Stage cancelled; committed artifacts are resumable": msg({ message: "Stage cancelled; committed artifacts can be resumed", comment: "Committed artifacts remain valid and can be reused by a later run." }),
  "Stage completed": msg({ message: "Stage completed", context: "backend stage status", comment: "Generic backend stage status message." }),
  "Existing dual-fisheye frames were discovered": msg({ message: "Existing dual-fisheye frames were discovered", comment: "A resumable project already contains extracted dual-fisheye frames." }),
  "Existing masks were discovered": msg({ message: "Existing masks were discovered", comment: "A resumable project already contains mask files." }),
  "Existing COLMAP reconstruction was discovered": msg({ message: "Existing COLMAP reconstruction was discovered", comment: "A resumable project already contains COLMAP reconstruction output." }),
  "Previous run was interrupted; this stage can be resumed": msg({ message: "The previous run was interrupted; this stage can be resumed", comment: "A resumable project was recovered after an interrupted run." }),
  "This project manifest was recovered from existing artifacts": msg({ message: "This project manifest was recovered from existing artifacts", comment: "Project metadata was reconstructed from files already on disk." }),
  "Extract requires both system ffmpeg and ffprobe": msg({ message: "Frame extraction requires both system FFmpeg and ffprobe", comment: "Keep FFmpeg and ffprobe as command-line tool names." }),
  "COLMAP is unavailable; alignment will remain in a resumable pending state": msg({ message: "COLMAP is unavailable; alignment will remain pending and resumable", comment: "The alignment stage cannot start without COLMAP, but the project remains resumable." }),
  "FFmpeg was found without VideoToolbox support; extraction will use the CPU decoder": msg({ message: "FFmpeg was found without VideoToolbox support; extraction will use the CPU decoder", comment: "Keep FFmpeg, VideoToolbox, and CPU as technical names." }),

  // Rust emits these messages in Traditional Chinese. Keep the source text as
  // the lookup key so persisted manifests and event payloads remain locale
  // independent; the descriptor is resolved only when the UI renders it.
  "正在掃描雙魚眼配對候選影格": msg({ message: "Scanning dual-fisheye frame candidates", comment: "Backend progress message: scanning candidate frame pairs." }),
  "已在評分候選影格時取消": msg({ message: "Cancelled while scoring frame candidates", comment: "Backend progress message: scoring candidate frames for selection." }),
  "已在選定結果前取消": msg({ message: "Cancelled before committing the selected result", comment: "Backend progress message: selection was cancelled before the selected result was committed." }),
  "已選定配對影格（未複製輸出）": msg({ message: "Selected frame pair (output copy disabled)", comment: "Backend progress message: a frame pair was selected without copying output files." }),
  "選定的配對影格已存在，已略過": msg({ message: "Selected frame pair already exists; skipped", comment: "Backend message: an output pair was already present, so no duplicate was written." }),
  "已在寫入輸出前取消": msg({ message: "Cancelled before committing output", comment: "Backend progress message: output files were not committed." }),
  "正在複製選定的配對影格": msg({ message: "Copying selected frame pair", comment: "Backend progress message: copying selected source frames." }),
  "已寫入選定的配對影格": msg({ message: "Selected frame pair committed", comment: "Backend progress message: selected source frames were written to output." }),
  "影格擷取已取消": msg({ message: "Frame extraction cancelled", context: "backend stage status", comment: "The frame-extraction stage was cancelled." }),
  "正在掃描原生雙魚眼影像": msg({ message: "Scanning native dual-fisheye images", comment: "Backend progress message: native fisheye images are being inspected." }),
  "正在載入 YOLO11／SkySeg 模型": msg({ message: "Loading YOLO11/SkySeg models", comment: "Keep YOLO11 and SkySeg as model names." }),
  "已確認遮罩存在，已略過": msg({ message: "Verified that the mask exists; skipped", comment: "Backend message: an existing mask file was reused." }),
  "正在執行 YOLO11／SkySeg 推論": msg({ message: "Running YOLO11/SkySeg inference", comment: "Keep YOLO11 and SkySeg as model names." }),
  "遮罩處理已取消": msg({ message: "Mask operation cancelled", context: "backend stage status", comment: "The masking operation was cancelled." }),
  "已在寫入遮罩前取消": msg({ message: "Masking cancelled before committing output", comment: "Backend progress message: mask output files were not committed." }),
  "正在寫入遮罩檔案": msg({ message: "Writing mask files", comment: "Backend progress message: mask files are being written." }),
  "未啟用遮罩，正在略過": msg({ message: "Masking is disabled; skipping", comment: "Backend stage status when no object or sky mask is enabled." }),
  "處理階段已開始": msg({ message: "Stage started", context: "backend stage status", comment: "Generic backend stage status message." }),
  "處理階段已取消，已寫入的結果可繼續使用": msg({ message: "Stage cancelled; committed artifacts can be resumed", comment: "Committed artifacts remain valid and can be reused by a later run." }),
  "工作已取消，可稍後續作": msg({ message: "Cancelled; you can resume later", comment: "A cancelled stage can be resumed later." }),
  "未啟用 YOLO 或天空過濾，已略過遮罩階段": msg({ message: "YOLO and sky filtering are disabled; masking was skipped", comment: "Backend stage summary when no mask filters are enabled." }),
  "處理階段已完成": msg({ message: "Stage completed", context: "backend stage status", comment: "Generic backend stage status message." }),
  "遮罩處理完成": msg({ message: "Masking completed", context: "backend stage status", comment: "Backend completion message from the masking stage; this is not a request to start masking." }),
  "找不到來源檔案": msg({ message: "Source file was not found", comment: "Media preview error when the requested source file does not exist." }),
  "不支援此來源格式": msg({ message: "This source format is not supported", comment: "Media preview error for an unsupported source format." }),
  "無法讀取影片第一幀": msg({ message: "Unable to read the first video frame", comment: "Media preview error when the first frame cannot be decoded." }),
  "尚未安裝 FFmpeg": msg({ message: "FFmpeg is not installed", comment: "Media preview error when FFmpeg is unavailable." }),
  "無法啟動 FFmpeg": msg({ message: "Unable to start FFmpeg", comment: "Media preview error when FFmpeg cannot be launched." }),
  "已找到現有遮罩": msg({ message: "Existing masks were discovered", comment: "A resumable project already contains mask files." }),
  "已找到現有的 COLMAP 重建結果": msg({ message: "Existing COLMAP reconstruction was discovered", comment: "A resumable project already contains COLMAP reconstruction output." }),
  "上次處理中斷，此階段可繼續執行": msg({ message: "The previous run was interrupted; this stage can be resumed", comment: "A resumable project was recovered after an interrupted run." }),
  "已依現有處理結果復原專案資訊": msg({ message: "This project manifest was recovered from existing artifacts", comment: "Project metadata was reconstructed from files already on disk." }),
  "來源不能位於輸出資料夾內，請先將來源移到其他位置": msg({ message: "Sources cannot be inside the output folder; move them elsewhere first", comment: "Project validation error: source media cannot be nested under reconstruction output." }),
  "專案操作鎖暫時無法使用": msg({ message: "The project operation lock is temporarily unavailable", comment: "Project operation error while another mutation is in progress." }),
  "找不到可修改的專案資訊": msg({ message: "No editable project information was found", context: "project edit error", comment: "Shown when the selected folder does not contain a project manifest that can be edited." }),
  "無法確認專案根目錄": msg({ message: "Unable to determine the project root", context: "project edit error", comment: "Project root means the folder containing the loaded project manifest." }),
  "任務名稱不能是空白": msg({ message: "Task name cannot be blank", context: "project edit validation", comment: "Validation error shown when a queued task name contains only whitespace." }),
  "專案資訊中的根目錄與實際位置不一致，已取消重新命名": msg({ message: "The project root in the manifest does not match its actual location; rename cancelled", context: "project edit error", comment: "The rename is cancelled to avoid changing a project whose manifest path is inconsistent." }),
  "輸出資料夾沒有可用的父目錄": msg({ message: "The output folder has no usable parent directory", context: "project edit error", comment: "Shown when the output folder cannot be renamed because its parent directory is unavailable." }),
  "復原專案資訊": msg({ message: "Recovering project information", comment: "Project recovery status shown while reconstructing a manifest from existing artifacts." }),
  "可繼續執行": msg({ message: "Can be resumed", comment: "Short project recovery status indicating that processing can continue." }),
  "預覽服務暫時無法使用": msg({ message: "The preview service is temporarily unavailable", comment: "Media preview error when the preview service cannot be used." }),
  "在系統 PATH 中找不到 FFmpeg": msg({ message: "FFmpeg was not found on the system PATH", comment: "Backend tool error: FFmpeg is missing from PATH." }),
  "在系統 PATH 中找不到 ffprobe": msg({ message: "ffprobe was not found on the system PATH", comment: "Backend tool error: ffprobe is missing from PATH." }),
  "影像配對完成": msg({ message: "Image matching completed", comment: "Backend alignment progress message after image matching." }),
  "正在匯入受限影像配對": msg({ message: "Importing constrained image matches", comment: "Backend alignment progress message while importing constrained matches." }),
  "正在擷取影像特徵": msg({ message: "Extracting image features", comment: "Backend alignment progress message while extracting image features." }),
  "影像特徵擷取完成": msg({ message: "Image feature extraction completed", comment: "Backend alignment progress message after feature extraction." }),
  "相機組已提供完整外參": msg({ message: "The camera rig already provides complete extrinsics", comment: "Backend alignment message when a rig configuration supplies all extrinsics." }),
  "至少需要兩組同名的 lens0/lens1 影格才能對齊": msg({ message: "At least two matching lens0/lens1 frame pairs are required for alignment", comment: "Alignment validation error for dual-fisheye frame pairs." }),
  "COLMAP 初始建模未產生任何 sparse 子模型": msg({ message: "COLMAP bootstrap produced no sparse submodel", comment: "Alignment error when COLMAP bootstrap produces no sparse model." }),
  "此資料夾沒有原始影片；可直接執行現有影格適用的遮罩或對齊階段": msg({ message: "This folder has no source video; masking or alignment can run directly on existing frames", comment: "Backend project message when an existing frame dataset has no source video." }),
  "影格擷取需要系統已安裝 FFmpeg 與 ffprobe": msg({ message: "Frame extraction requires both system FFmpeg and ffprobe", comment: "Backend diagnostic warning: both command-line tools are required." }),
  "指定的 COLMAP 支援 CUDA，但未確認 Ceres GPU 求解器；Bundle Adjustment 會使用 CPU": msg({ message: "The selected COLMAP supports CUDA, but the Ceres GPU solver was not confirmed; Bundle Adjustment will use the CPU", comment: "Diagnostic warning: COLMAP CUDA support does not prove Ceres GPU availability." }),
  "指定的 COLMAP 未確認 CUDA 建置；特徵擷取與配對會使用 CPU，且無法啟用 Ceres GPU 求解器": msg({ message: "The selected COLMAP CUDA build was not confirmed; feature extraction and matching will use the CPU, and the Ceres GPU solver cannot be enabled", comment: "Diagnostic warning for a COLMAP build without confirmed CUDA support." }),
  "尚未選取可執行的 COLMAP；無法判定 COLMAP CUDA 建置能力": msg({ message: "No usable COLMAP executable is selected; COLMAP CUDA build capability cannot be determined", comment: "Diagnostic note when no COLMAP executable is available." }),
  "無法讀取指定 COLMAP 的 version banner；能力判定採保守的未知/不可用": msg({ message: "Unable to read the selected COLMAP version banner; capabilities are conservatively treated as unknown or unavailable", comment: "Diagnostic warning when COLMAP capability probing cannot read the version banner." }),
  "目前流程需要 COLMAP 4.1.1 或更新版本；較舊版本不支援完整的 rig 與全域對齊流程": msg({ message: "This workflow requires COLMAP 4.1.1 or newer; older versions do not support the complete rig and global alignment workflow", comment: "Diagnostic warning for an unsupported COLMAP version." }),
  "此 FFmpeg build 未回報硬體加速元件；影格擷取將使用 CPU 解碼": msg({ message: "This FFmpeg build reported no hardware acceleration components; frame extraction will use CPU decoding", comment: "Diagnostic note when FFmpeg hardware decoding is unavailable." }),
  "指定的 COLMAP 未同時提供 FeatureExtraction/FeatureMatching GPU 選項；特徵階段會使用 CPU": msg({ message: "The selected COLMAP does not provide both FeatureExtraction and FeatureMatching GPU options; feature stages will use the CPU", context: "backend diagnostic warning", comment: "FeatureExtraction and FeatureMatching are COLMAP command-line options; keep these names unchanged." }),
  "指定的 COLMAP 不提供 global_mapper；只能使用增量對齊": msg({ message: "The selected COLMAP does not provide global_mapper; only incremental alignment can be used", context: "backend diagnostic warning", comment: "global_mapper is the COLMAP command name; incremental alignment is the fallback workflow." }),
  "指定的 global_mapper 未提供 gravity rotation averaging 選項；global gravity 模式不可用": msg({ message: "The selected global_mapper does not provide the gravity rotation averaging option; global-gravity mode is unavailable", context: "backend diagnostic warning", comment: "Keep global_mapper and gravity rotation averaging as COLMAP technical terms." }),
  "指定的 global_mapper 未提供 GPU global positioning 選項；global positioning 會使用 CPU": msg({ message: "The selected global_mapper does not provide GPU global positioning; global positioning will use the CPU", context: "backend diagnostic warning", comment: "Keep global_mapper and global positioning as COLMAP technical terms." }),
  "指定的 global_mapper 未提供 GPU Ceres BA 選項；global Bundle Adjustment 會使用 CPU": msg({ message: "The selected global_mapper does not provide GPU Ceres BA; global Bundle Adjustment will use the CPU", context: "backend diagnostic warning", comment: "BA means Bundle Adjustment; keep global_mapper, Ceres, and Bundle Adjustment as technical terms." }),
  "指定的 global_mapper 未提供 fixed-rotation/joint BA stage 選項；無法使用固定旋轉實驗模式": msg({ message: "The selected global_mapper does not provide the fixed-rotation/joint BA stage option; fixed-rotation experimental mode is unavailable", context: "backend diagnostic warning", comment: "Keep global_mapper and fixed-rotation/joint BA stage as COLMAP technical terms." }),
  "指定的 COLMAP 不提供 view_graph_calibrator；global mapper 的 focal prior 必須由外部校正提供": msg({ message: "The selected COLMAP does not provide view_graph_calibrator; the global mapper's focal prior must be supplied by external calibration", context: "backend diagnostic warning", comment: "Keep view_graph_calibrator, global mapper, and focal prior as COLMAP technical terms." }),
  "指定的 COLMAP 缺少 feature_extractor、matches_importer、mapper、model_converter 或 rig_configurator；雙鏡頭對齊流程無法完成": msg({ message: "The selected COLMAP is missing feature_extractor, matches_importer, mapper, model_converter, or rig_configurator; the dual-camera alignment workflow cannot complete", context: "backend diagnostic warning", comment: "Keep the five COLMAP command names unchanged; dual-camera alignment refers to the dual-fisheye rig workflow." }),
  "FFmpeg 未回報硬體加速元件；影格擷取將使用 CPU 解碼": msg({ message: "FFmpeg reported no hardware acceleration components; frame extraction will use CPU decoding", context: "backend diagnostic warning", comment: "Diagnostic note when FFmpeg reports no hardware acceleration components." }),
};

const STAGE_SUMMARY_TRANSLATIONS: Record<string, MessageDescriptor> = {
  "Frame extraction: pending": msg({ message: "Frame extraction: Pending", comment: "Synthesized pipeline log summary for the frame-extraction stage." }),
  "Frame extraction: running": msg({ message: "Frame extraction: Running", comment: "Synthesized pipeline log summary for the frame-extraction stage." }),
  "Frame extraction: completed": msg({ message: "Frame extraction: Completed", comment: "Synthesized pipeline log summary for the frame-extraction stage." }),
  "Frame extraction: cancelled": msg({ message: "Frame extraction: Cancelled", comment: "Synthesized pipeline log summary for the frame-extraction stage." }),
  "Frame extraction: failed": msg({ message: "Frame extraction: Failed", comment: "Synthesized pipeline log summary for the frame-extraction stage." }),
  "Masking: pending": msg({ message: "Masking: Pending", comment: "Synthesized pipeline log summary for the masking stage." }),
  "Masking: running": msg({ message: "Masking: Running", comment: "Synthesized pipeline log summary for the masking stage." }),
  "Masking: completed": msg({ message: "Masking: Completed", comment: "Synthesized pipeline log summary for the masking stage." }),
  "Masking: cancelled": msg({ message: "Masking: Cancelled", comment: "Synthesized pipeline log summary for the masking stage." }),
  "Masking: failed": msg({ message: "Masking: Failed", comment: "Synthesized pipeline log summary for the masking stage." }),
  "Alignment: pending": msg({ message: "Alignment: Pending", comment: "Synthesized pipeline log summary for the alignment stage." }),
  "Alignment: running": msg({ message: "Alignment: Running", comment: "Synthesized pipeline log summary for the alignment stage." }),
  "Alignment: completed": msg({ message: "Alignment: Completed", comment: "Synthesized pipeline log summary for the alignment stage." }),
  "Alignment: cancelled": msg({ message: "Alignment: Cancelled", comment: "Synthesized pipeline log summary for the alignment stage." }),
  "Alignment: failed": msg({ message: "Alignment: Failed", comment: "Synthesized pipeline log summary for the alignment stage." }),
};

// Frontend-generated dynamic values use the same source strings as the `t`
// calls in the view. Keeping descriptors here lets state store those source
// strings without freezing the active locale into a task, toast, or doctor
// report.
const APP_MESSAGE_TRANSLATIONS: Record<string, MessageDescriptor> = {
  "Platform not checked yet": msg({ message: "Platform not checked yet" }),
  "Not checked yet": msg({ message: "Not checked yet" }),
  "Run an environment check to confirm available capabilities": msg({ message: "Run an environment check to confirm available capabilities" }),
  "Check the COLMAP executable and version": msg({ message: "Check the COLMAP executable and version" }),
  "Check the COLMAP CUDA build and whether an NVIDIA GPU is available": msg({ message: "Check the COLMAP CUDA build and whether an NVIDIA GPU is available" }),
  "Confirm FFmpeg is available on the system PATH": msg({ message: "Confirm FFmpeg is available on the system PATH" }),
  "Confirm FFmpeg hardware decoding capabilities": msg({ message: "Confirm FFmpeg hardware decoding capabilities" }),
  "Detected": msg({ message: "Detected" }),
  "Supported": msg({ message: "Supported" }),
  "Unsupported": msg({ message: "Unsupported" }),
  "Not reported": msg({ message: "Not reported" }),
  "Processing": msg({ message: "Processing" }),
  "Pipeline log": msg({ message: "Pipeline log" }),
  "Not run yet": msg({ message: "Not run yet" }),
  "Unnamed reconstruction": msg({ message: "Unnamed reconstruction" }),
  "Frame extraction": msg({ message: "Frame extraction" }),
  "Masking": msg({ message: "Masking" }),
  "Alignment": msg({ message: "Alignment" }),
  "Not detected": msg({ message: "Not detected" }),
  "CUDA build: Supported": msg({ message: "CUDA build: Supported" }),
  "CUDA build: Unsupported": msg({ message: "CUDA build: Unsupported" }),
  "SIFT extraction: Supported": msg({ message: "SIFT extraction: Supported" }),
  "SIFT extraction: Unsupported": msg({ message: "SIFT extraction: Unsupported" }),
  "SIFT matching: Supported": msg({ message: "SIFT matching: Supported" }),
  "SIFT matching: Unsupported": msg({ message: "SIFT matching: Unsupported" }),
  "Ceres BA: May be available (runtime CUDA/cuDSS check required)": msg({ message: "Ceres BA: May be available (runtime CUDA/cuDSS check required)" }),
  "Ceres BA: CPU only": msg({ message: "Ceres BA: CPU only" }),
  "Ceres BA: Not reported": msg({ message: "Ceres BA: Not reported" }),
  "The legacy diagnostic did not report the COLMAP build; FFmpeg CUDA/VideoToolbox does not imply COLMAP CUDA": msg({ message: "The legacy diagnostic did not report the COLMAP build; FFmpeg CUDA/VideoToolbox does not imply COLMAP CUDA" }),
  "CUDA acceleration available": msg({ message: "CUDA acceleration available" }),
  "CUDA acceleration partially available": msg({ message: "CUDA acceleration partially available" }),
  "No usable CUDA GPU detected": msg({ message: "No usable CUDA GPU detected" }),
  "CUDA status not confirmed": msg({ message: "CUDA status not confirmed" }),
  "Native dual-fisheye camera-rig alignment is available": msg({ message: "Native dual-fisheye camera-rig alignment is available" }),
  "The executable was found, but the complete alignment workflow was not confirmed": msg({ message: "The executable was found, but the complete alignment workflow was not confirmed" }),
  "The alignment stage will remain pending": msg({ message: "The alignment stage will remain pending" }),
  "COLMAP alignment capability not confirmed": msg({ message: "COLMAP alignment capability not confirmed" }),
  "COLMAP not detected": msg({ message: "COLMAP not detected" }),
  "Executable: system PATH": msg({ message: "Executable: system PATH" }),
  "COLMAP CUDA capabilities were checked": msg({ message: "COLMAP CUDA capabilities were checked" }),
  "COLMAP CUDA capability check result": msg({ message: "COLMAP CUDA capability check result" }),
  "FFmpeg tools incomplete": msg({ message: "FFmpeg tools incomplete" }),
  "FFmpeg and ffprobe are both available": msg({ message: "FFmpeg and ffprobe are both available" }),
  "Frame extraction requires FFmpeg and ffprobe": msg({ message: "Frame extraction requires FFmpeg and ffprobe" }),
  "FFmpeg hardware acceleration supported": msg({ message: "FFmpeg hardware acceleration supported" }),
  "FFmpeg hardware acceleration not enabled": msg({ message: "FFmpeg hardware acceleration not enabled" }),
  "Hardware decoding status not reported": msg({ message: "Hardware decoding status not reported" }),
  "Available": msg({ message: "Available" }),
  "FFmpeg: Available": msg({ message: "FFmpeg: Available" }),
  "FFmpeg: Not detected": msg({ message: "FFmpeg: Not detected" }),
  "ffprobe: Available": msg({ message: "ffprobe: Available" }),
  "ffprobe: Not detected": msg({ message: "ffprobe: Not detected" }),
  "Unable to generate a first-frame preview": msg({ message: "Unable to generate a first-frame preview" }),
  "Executable detected (full path hidden)": msg({ message: "Executable detected (full path hidden)" }),
  "Some CUDA or hardware acceleration capabilities are unavailable; affected stages will use the CPU.": msg({ message: "Some CUDA or hardware acceleration capabilities are unavailable; affected stages will use the CPU." }),
  [PREPARING_WORK_SOURCE]: msg({ message: "Preparing work", context: "pipeline stage status", comment: "Initial status stored while a backend stage is being prepared; keep this source value locale-independent in task state." }),
  [BROWSER_PREVIEW_TASK_SOURCE]: msg({ message: "Browser preview task", context: "browser preview task", comment: "Fallback task name used when the app runs without the local Tauri runtime." }),
  [BROWSER_PREVIEW_NOT_CONNECTED_SOURCE]: msg({ message: "Browser preview: not connected to the local runtime", context: "browser preview task warning", comment: "Warning stored on a browser-only preview task because it is not connected to the local Tauri runtime." }),
  "Sources found; you can create a new reconstruction task": msg({ message: "Sources found; you can create a new reconstruction task" }),
  "Source inspection results are not available yet": msg({ message: "Source inspection results are not available yet" }),
  "New reconstruction task": msg({ message: "New reconstruction task" }),
  "This task has started and can no longer be edited": msg({ message: "This task has started and can no longer be edited" }),
  "Browser preview keeps the custom output path": msg({ message: "Browser preview keeps the custom output path" }),
  "Browser preview does not read local project information": msg({ message: "Browser preview does not read local project information" }),
  "No loadable project information was found": msg({ message: "No loadable project information was found" }),
  "Failed to open project": msg({ message: "Failed to open project" }),
  "Browser preview is not connected to the local runtime": msg({ message: "Browser preview is not connected to the local runtime" }),
  "Diagnostics copied; you can paste them into a bug report": msg({ message: "Diagnostics copied; you can paste them into a bug report" }),
  "Unable to copy diagnostics; check clipboard permissions": msg({ message: "Unable to copy diagnostics; check clipboard permissions" }),
  "The COLMAP path is used by the local Windows runtime": msg({ message: "The COLMAP path is used by the local Windows runtime" }),
  "Browser preview does not read local LUT files; paste a .cube path directly": msg({ message: "Browser preview does not read local LUT files; paste a .cube path directly" }),
  "Unable to start the stage automatically; check the runtime message": msg({ message: "Unable to start the stage automatically; check the runtime message" }),
  "Select at least one OSV or dual-fisheye source first": msg({ message: "Select at least one OSV or dual-fisheye source first" }),
  "The custom LUT must be a .cube file": msg({ message: "The custom LUT must be a .cube file" }),
  "Failed to create the task; check the runtime message": msg({ message: "Failed to create the task; check the runtime message" }),
  "Keep at least one source": msg({ message: "Keep at least one source" }),
  "Preview task updated": msg({ message: "Preview task updated" }),
  "Failed to save task changes; check the runtime message": msg({ message: "Failed to save task changes; check the runtime message" }),
  "Queued task updated": msg({ message: "Queued task updated" }),
  "This task has started and cannot be removed": msg({ message: "This task has started and cannot be removed" }),
  "Task removed from the queue; the output folder is kept": msg({ message: "Task removed from the queue; the output folder is kept" }),
  "This task has already started": msg({ message: "This task has already started" }),
  "Task added to the run queue": msg({ message: "Task added to the run queue" }),
  "Browser preview does not run backend work": msg({ message: "Browser preview does not run backend work" }),
  "A processing stage is already running for this task; please wait": msg({ message: "A processing stage is already running for this task; please wait" }),
  "Unable to start the stage; check the runtime message": msg({ message: "Unable to start the stage; check the runtime message" }),
  "Browser preview does not cancel backend work": msg({ message: "Browser preview does not cancel backend work" }),
  "Cancelled; you can resume later": msg({ message: "Cancelled; you can resume later" }),
  "Automatic pipeline completed frame extraction, masking, and alignment": msg({ message: "Automatic pipeline completed frame extraction, masking, and alignment" }),
  "A processing stage is already running; please wait": msg({ message: "A processing stage is already running; please wait" }),
};

function localiseUserMessage(value: string): string {
  const exact = USER_MESSAGE_TRANSLATIONS[value];
  if (exact) return translate(exact);
  const stageSummary = STAGE_SUMMARY_TRANSLATIONS[value];
  if (stageSummary) return translate(stageSummary);
  const appMessage = APP_MESSAGE_TRANSLATIONS[value];
  if (appMessage) return translate(appMessage);
  const translated = value
    .replace(/cancelled before interval (\d+)/g, (_match, interval) => t`Cancelled before interval ${interval}`)
    .replace(/scoring (\d+) paired candidates/g, (_match, count) => t`Scoring ${count} paired candidates`)
    .replace(/FFmpeg 有 (\d+) 張候選影格未回報 PTS，已使用 candidate FPS 時間估算/g, (_match, count) => t({
      message: `FFmpeg had ${count} candidate frames without PTS; estimated timing using candidate FPS`,
      context: "backend progress message",
      comment: "PTS is the presentation timestamp; candidate FPS is the frame-rate estimate used when FFmpeg omits PTS.",
    }))
    .replace(/FFmpeg 記憶體候選硬體與軟體解碼皆失敗：硬體錯誤：(.+?)；軟體錯誤：(.+)$/g, (_match, hardwareError, softwareError) => t({
      message: `Both hardware and software decoding failed for FFmpeg in-memory candidates: hardware error: ${localiseTechnicalErrorDetail(hardwareError)}; software error: ${localiseTechnicalErrorDetail(softwareError)}`,
      context: "backend error",
      comment: "The in-memory candidate decoder tried hardware and software paths; technical error details may be shown only when safe for the active locale.",
    }))
    .replace(/FFmpeg 硬體解碼失敗後，無法準備軟體解碼回退：(.+)$/g, (_match, error) => t({
      message: `After FFmpeg hardware decoding failed, unable to prepare the software-decoding fallback: ${localiseTechnicalErrorDetail(error)}`,
      context: "backend error",
      comment: "Fallback means retrying the same operation with software decoding after hardware decoding fails.",
    }))
    .replace(/FFmpeg 候選影格硬體與軟體解碼皆失敗：硬體錯誤：(.+?)；軟體錯誤：(.+)$/g, (_match, hardwareError, softwareError) => {
      const cleanup = /^(.*)；且無法清理不完整輸出：(.+)$/.exec(softwareError);
      if (cleanup) {
        return t({
          message: `Both hardware and software decoding failed for FFmpeg candidate frames: hardware error: ${localiseTechnicalErrorDetail(hardwareError)}; software error: ${localiseTechnicalErrorDetail(cleanup[1])}; cleanup also failed for incomplete output: ${localiseTechnicalErrorDetail(cleanup[2])}`,
          context: "backend error",
          comment: "The candidate-frame decoder failed on both paths and could not clean up its incomplete output.",
        });
      }
      return t({
        message: `Both hardware and software decoding failed for FFmpeg candidate frames: hardware error: ${localiseTechnicalErrorDetail(hardwareError)}; software error: ${localiseTechnicalErrorDetail(softwareError)}`,
        context: "backend error",
        comment: "The candidate-frame decoder tried hardware and software paths; technical error details may be shown only when safe for the active locale.",
      });
    })
    .replace(/^(.+) 未包含兩路可辨識的雙魚眼 video stream$/g, (_match, source) => t({
      message: `${source} does not contain two recognizable dual-fisheye video streams`,
      context: "source validation error",
      comment: "A source must expose two recognizable video streams for the dual-fisheye workflow.",
    }))
    .replace(/^無法讀回來源 (\d+) 的標準化 telemetry，IMU keyframe term 已停用：(.+)$/g, (_match, source, error) => t({
      message: `Unable to read normalized telemetry for source ${source}; the IMU keyframe term was disabled: ${localiseTechnicalErrorDetail(error)}`,
      context: "backend diagnostic warning",
      comment: "Telemetry is normalized camera metadata; disabling the IMU keyframe term leaves visual keyframe selection available.",
    }))
    .replace(/^無法解析來源 (\d+) 的標準化 telemetry；改用 visual novelty＋max gap：(.+)$/g, (_match, source, error) => t({
      message: `Unable to parse normalized telemetry for source ${source}; using visual novelty and the maximum-gap fallback: ${localiseTechnicalErrorDetail(error)}`,
      context: "backend diagnostic warning",
      comment: "Visual novelty and maximum-gap fallback are keyframe-selection strategies used when telemetry parsing fails.",
    }))
    .replace(/已在第\s*(\d+)\s*個區間前取消/g, (_match, interval) => t`Cancelled before processing interval ${interval}`)
    .replace(/正在評分\s*(\d+)\s*組配對候選影格/g, (_match, count) => t`Scoring ${count} paired candidates`)
    .replace(/首次使用，正在下載\s*(.+?)\s*模型（(\d+)%）/g, (_match, model, percent) => t`First use: downloading the ${model} model (${percent}%)`)
    .replace(/模型已載入\s*(.+?)，CPU 推論回退已停用/g, (_match, provider) => t`Model loaded with ${provider}; CPU inference fallback is disabled`)
    .replace(/正在同步解碼並評分來源\s*(\d+)（已處理\s*(\d+)\s*組候選影格）/g, (_match, source, count) => t`Decoding and scoring source ${source} (${count} candidate pairs processed)`)
    .replace(/已完成配對區塊\s*(\d+)\s*\/\s*(\d+)/g, (_match, current, total) => t`Pairing block ${current} / ${total} completed`)
    .replace(/正在處理配對區塊\s*(\d+)\s*\/\s*(\d+)/g, (_match, current, total) => t`Processing pairing block ${current} / ${total}`)
    .replace(/對齊處理完成：已註冊\s*(\d+)\s*\/\s*(\d+)\s*組相機組影格（([\d.]+)%）/g, (_match, registered, total, percentage) => t`Alignment completed: ${registered} / ${total} camera-rig frames registered (${percentage}%)`)
    .replace(/無法讀取來源影格\s*(.+?)：(.+)$/g, (_match, frame, error) => t`Unable to read source frame ${frame}: ${error}`)
    .replace(/無法建立暫存影格\s*(.+?)：(.+)$/g, (_match, frame, error) => t`Unable to create temporary frame ${frame}: ${error}`)
    .replace(/無法複製影格\s*(.+?) 至 (.+?)：(.+)$/g, (_match, source, destination, error) => t`Unable to copy frame ${source} to ${destination}: ${error}`)
    .replace(/無法同步暫存影格\s*(.+?)：(.+)$/g, (_match, frame, error) => t`Unable to sync temporary frame ${frame}: ${error}`)
    .replace(/無法建立暫存 metadata\s*(.+?)：(.+)$/g, (_match, path, error) => t`Unable to create temporary metadata ${path}: ${error}`)
    .replace(/無法寫入暫存 metadata\s*(.+?)：(.+)$/g, (_match, path, error) => t`Unable to write temporary metadata ${path}: ${error}`)
    .replace(/無法同步暫存 metadata\s*(.+?)：(.+)$/g, (_match, path, error) => t`Unable to sync temporary metadata ${path}: ${error}`)
    .replace(/COLMAP (.+?) GPU 執行失敗，(?:移除不完整輸出並)?改用 CPU 重試：(.+)$/g, (_match, component, error) => t`COLMAP ${component} GPU execution failed; retrying with CPU: ${error}`)
    .replace(/COLMAP (.+?) 的 Ceres GPU 不可用，已由 Ceres 改用 CPU：(.+)$/g, (_match, component, error) => t`COLMAP ${component} Ceres GPU is unavailable; Ceres switched to CPU: ${error}`)
    .replace(/FFmpeg 未產生任何記憶體候選影格/g, () => t`FFmpeg did not produce any in-memory candidate frames`)
    .replace(/FFmpeg rawvideo 在 frame 中途結束：收到 (\d+)\/(\d+) bytes/g, (_match, received, expected) => t`FFmpeg rawvideo ended mid-frame: received ${received}/${expected} bytes`)
    .replace(/讀取 FFmpeg rawvideo 失敗：(.+)$/g, (_match, error) => t`Failed to read FFmpeg rawvideo: ${error}`)
    .replace(/FFmpeg 硬體解碼(?:記憶體候選|候選影格)失敗，(?:將)?改用 CPU(?: 軟體解碼)?重試：(.+)$/g, (_match, error) => t`FFmpeg hardware decoding failed; retrying with CPU software decoding: ${error}`)
    .replace(/FFmpeg 候選影格已安全回退至 CPU 軟體解碼/g, () => t`FFmpeg candidate frames fell back safely to CPU software decoding`)
    .replace(/FFmpeg 產生 (\d+) 張同步影格，但預期選定 (\d+) 張/g, (_match, actual, expected) => t`FFmpeg produced ${actual} synchronized frames, but ${expected} were expected`)
    .replace(/來源 (\d+) 色彩處理：(.+)$/g, (_match, source, warning) => t`Source ${source} color processing: ${warning}`)
    .replace(/來源 (\d+) 已沿用記憶體評分 checkpoint/g, (_match, source) => t`Source ${source} reused the in-memory scoring checkpoint`)
    .replace(/正在記憶體中同步解碼並評分來源 (\d+) 的雙魚眼候選影格/g, (_match, source) => t`Decoding and scoring source ${source} dual-fisheye candidates in memory`)
    .replace(/來源 (\d+) 已完成 (\d+) 組候選影格評分/g, (_match, source, count) => t`Source ${source} completed scoring ${count} candidate pairs`)
    .replace(/來源 (\d+) 的動態 keyframe 剪枝保留 (\d+) \/ (\d+) 組 base-FPS 候選（移除 (\d+) 組）/g, (_match, source, kept, total, removed) => t`Source ${source} dynamic keyframe pruning kept ${kept} / ${total} base-FPS candidates (${removed} removed)`)
    .replace(/已完成\s*(\d+)\s*個來源/g, (_match, count) => t`${count} sources completed`)
    .replace(/已處理\s*(\d+)\s*組候選影格/g, (_match, count) => t`${count} candidate pairs processed`)
    .replace(/已寫入\s*(\d+)\s*個遮罩/g, (_match, count) => t`${count} masks written`)
    .replace(/(\d+)\s*個遮罩處理失敗，請查看處理紀錄/g, (_match, count) => t`${count} masks failed; see the pipeline log`)
    .replace(/(\d+) masks failed; see pipeline log/g, (_match, count) => t`${count} masks failed; see the pipeline log`)
    .replace(/^(\d+) sources · (\d+) passed inspection$/g, (_match, total, valid) => t`${total} sources · ${valid} passed inspection`)
    .replace(/^(\d+) sources · browser preview$/g, (_match, count) => t`${count} sources · browser preview`)
    .replace(/^(\d+) sources$/g, (_match, count) => t`${count} sources`)
    .replace(/^Loaded resumable project (.+)$/g, (_match, name) => t`Loaded resumable project ${name}`)
    .replace(/^Loaded unfinished project with (\d+) warnings$/g, (_match, count) => t`Loaded unfinished project with ${count} warnings`)
    .replace(/^Loaded unfinished project: (.+)$/g, (_match, name) => t`Loaded unfinished project: ${name}`)
    .replace(/^Opened (.+)$/g, (_match, name) => t`Opened ${name}`)
    .replace(/^Created (.+)$/g, (_match, name) => t`Created ${name}`)
    .replace(/^Preview task added: (.+)$/g, (_match, name) => t`Preview task added: ${name}`)
    .replace(/^((?:CUDA build|SIFT extraction|SIFT matching|Alignment workflow)): (Supported|Unsupported|Not reported)$/g, (_match, label, status) => t`${label}: ${localiseUserMessage(status)}`)
    .replace(/^Ceres BA: (May be available \(runtime CUDA\/cuDSS check required\)|CPU only|Not reported)$/g, (_match, status) => t`Ceres BA: ${localiseUserMessage(status)}`)
    .replace(/^(FFmpeg|ffprobe): (Available|Not detected)$/g, (_match, tool, status) => t`${tool}: ${localiseUserMessage(status)}`)
    .replace(/^(.+) · FFmpeg hardware decoding capability not reported$/g, (_match, platform) => t`${platform} · FFmpeg hardware decoding capability not reported`)
    .replace(/^(.+): (enabled in build|Unsupported)$/g, (_match, label, status) => t`${label}: ${localiseUserMessage(status)}`)
    .replace(/^Complete prerequisite: (extract|mask|align)$/g, (_match, stageKey: StageKey) => t({
      message: `Complete ${stageLabel(STAGES.find((stage) => stage.key === stageKey))} first`,
      comment: "Shown when a pipeline stage is blocked by an incomplete prerequisite stage.",
    }))
    .replace(/影格擷取需要系統已安裝 FFmpeg 與 ffprobe/g, () => t`Frame extraction requires both system FFmpeg and ffprobe`)
    .replace(/找不到 COLMAP；對齊階段會維持可繼續的待執行狀態/g, () => t`COLMAP was not found; alignment will remain pending and resumable`)
    .replace(/FFmpeg 候選影格已安全回退至 CPU 軟體解碼/g, () => t`FFmpeg candidate frames fell back safely to CPU software decoding`)
    .replace(/指定的 COLMAP 路徑不存在或不是檔案/g, () => t`The selected COLMAP path does not exist or is not a file`)
    .replace(/指定的 COLMAP 未在 version\/help banner 標示 CUDA；將使用 CPU 特徵擷取與配對/g, () => t`The selected COLMAP did not report CUDA in its version/help banner; CPU feature extraction and matching will be used`)
    .replace(/未偵測到可供 COLMAP 使用的 NVIDIA GPU；將使用 CPU/g, () => t`No NVIDIA GPU usable by COLMAP was detected; the CPU will be used`)
    .replace(/影格擷取將使用 CPU 解碼/g, () => t`Frame extraction will use CPU decoding`)
    .replace(/System ffmpeg was not found on PATH/g, () => t`System FFmpeg was not found on PATH`)
    .replace(/System ffprobe was not found on PATH/g, () => t`System ffprobe was not found on PATH`)
    .replace(/COLMAP was not found on PATH/g, () => t`COLMAP was not found on PATH`)
    .replace(/COLMAP bootstrap did not produce sparse\/0/g, () => t`COLMAP bootstrap did not produce sparse/0`)
    .replace(/^invalid extraction input: /, () => t`Invalid extraction input: `)
    .replace(/^extraction image error: /, () => t`Extraction image error: `)
    .replace(/^extraction I\/O error: /, () => t`Extraction I/O error: `)
    .replace(/^invalid mask input: /, () => t`Invalid mask input: `)
    .replace(/^mask model error: /, () => t`Mask model error: `)
    .replace(/^mask inference error: /, () => t`Mask inference error: `)
    .replace(/^mask image error: /, () => t`Mask image error: `)
    .replace(/^mask I\/O error: /, () => t`Mask I/O error: `)
    .replace(/^mask operation cancelled$/, () => t`Mask operation cancelled`)
    .replace(/^無效的擷取輸入：\s*/g, () => t`Invalid extraction input: `)
    .replace(/^擷取影像錯誤：\s*/g, () => t`Extraction image error: `)
    .replace(/^擷取 I\/O 錯誤：\s*/g, () => t`Extraction I/O error: `)
    .replace(/^無效的遮罩輸入：\s*/g, () => t`Invalid mask input: `)
    .replace(/^遮罩模型錯誤：\s*/g, () => t`Mask model error: `)
    .replace(/^遮罩推論錯誤：\s*/g, () => t`Mask inference error: `)
    .replace(/^遮罩影像錯誤：\s*/g, () => t`Mask image error: `)
    .replace(/^遮罩 I\/O 錯誤：\s*/g, () => t`Mask I/O error: `)
    .replace(/^遮罩操作已取消$/g, () => t`Mask operation cancelled`)
    .replace(/^無法檢查專案根目錄：(.+)$/g, (_match, error) => t({
      message: `Unable to inspect the project root: ${localiseTechnicalErrorDetail(error)}`,
      context: "project edit error",
      comment: "Project root means the folder containing the loaded project manifest; technical details may be shown only when safe for the active locale.",
    }))
    .replace(/^儲存任務修改失敗，輸出資料夾已復原：(.+)$/g, (_match, error) => t({
      message: `Saving task changes failed; the output folder was restored: ${localiseTechnicalErrorDetail(error)}`,
      context: "project edit error",
      comment: "The rename/save operation failed, but the output folder was rolled back successfully.",
    }))
    .replace(/^儲存任務修改失敗，且輸出資料夾無法復原：(.+)；(.+)$/g, (_match, error, rollbackError) => t({
      message: `Saving task changes failed, and the output folder could not be restored: ${localiseTechnicalErrorDetail(error)}; rollback error: ${localiseTechnicalErrorDetail(rollbackError)}`,
      context: "project edit error",
      comment: "Both saving the task and rolling the output folder back failed; technical details may be shown only when safe for the active locale.",
    }));
  if (translated !== value) {
    // Exact and parameterized mappings may legitimately translate to CJK in
    // Simplified Chinese or Japanese. Keep every successful translation.
    return translated;
  }
  // Only an unchanged CJK source is an unmatched backend message. This keeps
  // legitimate CJK translations from being mistaken for untranslated raw
  // backend output.
  if (getLocale() !== "zh-TW" && containsCjk(value)) {
    return translate(isLikelyBackendError(value) ? UNKNOWN_BACKEND_ERROR : HIDDEN_BACKEND_STATUS);
  }
  if (getLocale() !== "en" && getLocale() !== "zh-TW" && isLikelyBackendError(value)) {
    return translate(UNKNOWN_BACKEND_ERROR);
  }
  return translated;
}

function backendErrorMessage(error: unknown): string {
  const raw = typeof error === "string"
    ? error.trim()
    : error instanceof Error
      ? error.message.trim()
      : "";
  if (!raw) return translate(UNKNOWN_BACKEND_ERROR);
  const translated = localiseUserMessage(raw);
  if (translated !== raw) return translated;
  // English may retain raw technical diagnostics, while Traditional Chinese
  // is the backend source locale. Other locales use the generic message so a
  // Tauri/Rust error cannot leak untranslated Chinese into a toast.
  if (getLocale() === "en" || getLocale() === "zh-TW") return raw;
  return translate(UNKNOWN_BACKEND_ERROR);
}

function platformLabel(value: string) {
  const labels: Record<string, string> = { macos: "macOS", windows: "Windows", linux: "Linux" };
  return labels[value.toLowerCase()] ?? value;
}

function timestampMs(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 0 && value < 100_000_000_000 ? Math.round(value * 1000) : Math.round(value);
  }
  if (typeof value !== "string" || !value.trim()) return undefined;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return numeric > 0 && numeric < 100_000_000_000 ? Math.round(numeric * 1000) : Math.round(numeric);
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function nonNegativeInteger(value: unknown): number | undefined {
  if ((typeof value !== "number" && typeof value !== "string") || (typeof value === "string" && !value.trim())) return undefined;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.round(number) : undefined;
}

function basename(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  const clean = value.trim().replace(/[\\/]+$/, "");
  return clean.split(/[\\/]/).filter(Boolean).pop() || clean;
}

function normaliseLogLevel(value: unknown): TaskLogLevel {
  const raw = String(value ?? "info").toLowerCase();
  if (raw.includes("error") || raw.includes("fail")) return "error";
  if (raw.includes("warn")) return "warning";
  return "info";
}

const PHASE_LABELS: Record<string, MessageDescriptor> = {
  starting: msg({ message: "Preparing", context: "pipeline phase", comment: "Short label for the preparation phase." }),
  scanning: msg({ message: "Scanning", context: "pipeline phase", comment: "Short label for a scan phase." }),
  scoring: msg({ message: "Scoring candidates", context: "pipeline phase", comment: "Short label for scoring candidate frames." }),
  "selecting-in-memory": msg({ message: "Scoring in-memory candidates", context: "pipeline phase", comment: "Short label for candidate scoring in memory." }),
  "decoding-full-resolution": msg({ message: "Decoding full resolution", context: "pipeline phase", comment: "Short label for full-resolution decoding." }),
  committing: msg({ message: "Committing output", context: "pipeline phase", comment: "Short label for writing output files." }),
  masking: msg({ message: "Mask inference", context: "pipeline phase", comment: "Short label for running mask inference." }),
  matching: msg({ message: "Image matching", context: "pipeline phase", comment: "Short label for matching images." }),
  "feature-extraction": msg({ message: "Feature extraction", context: "pipeline phase", comment: "Short label for extracting image features." }),
  bootstrap: msg({ message: "Bootstrap reconstruction", context: "pipeline phase", comment: "Short label for the initial reconstruction bootstrap." }),
  "final-mapping": msg({ message: "Final reconstruction", context: "pipeline phase", comment: "Short label for the final mapping/reconstruction pass." }),
  rig: msg({ message: "Camera-rig estimation", context: "pipeline phase", comment: "Short label for estimating the camera rig." }),
  completed: msg({ message: "Completed", context: "pipeline phase status", comment: "Short completed status label." }),
  cancelled: msg({ message: "Cancelled", context: "pipeline phase status", comment: "Short cancelled status label." }),
  failed: msg({ message: "Failed", context: "pipeline phase status", comment: "Short failed status label." }),
  summary: msg({ message: "Stage summary", context: "pipeline phase", comment: "Short label for a stage summary." }),
};

function phaseLabel(value?: string) {
  if (!value) return t`Processing`;
  const label = PHASE_LABELS[value];
  return label ? translate(label) : value.replace(/[-_]+/g, " ");
}

function formatDuration(value?: number) {
  if (!Number.isFinite(value) || value === undefined || value < 0) return t`Not timed yet`;
  const totalSeconds = Math.max(0, Math.floor(value / 1000));
  if (totalSeconds < 1) return t`Less than 1 second`;
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60) % 60;
  const hours = Math.floor(totalSeconds / 3600);
  if (hours > 0) return t`${hours} hr ${String(minutes).padStart(2, "0")} min`;
  if (minutes > 0) return t`${minutes} min ${String(seconds).padStart(2, "0")} sec`;
  return t`${seconds} sec`;
}

function formatTimestamp(value?: number, includeDate = false) {
  if (!value) return t`Not recorded yet`;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return t`Not recorded yet`;
  return date.toLocaleString(getLocale(), includeDate
    ? { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }
    : { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function timestampDateTime(value?: number) {
  if (!value) return undefined;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString();
}

function formatDoctorCheckedAt(value: string) {
  if (value === "Not checked yet") return localiseUserMessage(value);
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Date(parsed).toLocaleTimeString(getLocale(), { hour: "2-digit", minute: "2-digit" });
}

function stageSummaryLogs(stages: Record<StageKey, StageState>): TaskLog[] {
  return STAGES.flatMap(({ key, label }) => {
    const stage = stages[key];
    const updated = stage.updatedAtMs;
    const timestamp = stage.finishedAtMs ?? updated ?? stage.startedAtMs;
    if (stage.status === "pending" || (!timestamp && stage.durationMs === undefined)) return [];
    return [{
      id: `summary:${key}:${timestamp ?? 0}`,
      kind: "summary" as const,
      stage: key,
      phase: "summary",
      level: stage.status === "failed" ? "error" as const : stage.status === "cancelled" ? "warning" as const : "info" as const,
      // Keep this synthesized log locale independent too. The stage/status
      // pair is translated by localiseUserMessage when the row is rendered.
      message: `${label.message ?? key}: ${stage.status}`,
      timestampMs: timestamp ?? Date.now(),
      startedAtMs: stage.startedAtMs,
      finishedAtMs: stage.finishedAtMs,
      durationMs: stage.durationMs,
    }];
  });
}

function parseTaskLog(value: unknown, index: number): TaskLog | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const time = timestampMs(body.timestampMs ?? body.timestamp ?? body.updatedAt) ?? Date.now();
  // Persist the backend's raw message. Translating while parsing would make
  // old logs stick to whichever locale was active when the project loaded.
  const message = typeof body.message === "string" && body.message ? body.message : "Pipeline log";
  const stage = normaliseStage(body.stage);
  const kind: TaskLogKind = body.kind === "summary" || body.kind === "message" ? body.kind : "progress";
  const completed = nonNegativeInteger(body.completed);
  const total = nonNegativeInteger(body.total);
  return {
    id: typeof body.id === "string" && body.id ? body.id : `log:${time}:${index}`,
    kind,
    stage,
    phase: typeof body.phase === "string" && body.phase ? body.phase : undefined,
    jobId: typeof body.jobId === "string" ? body.jobId : undefined,
    level: normaliseLogLevel(body.level),
    message,
    timestampMs: time,
    startedAtMs: timestampMs(body.startedAtMs ?? body.startedAt),
    finishedAtMs: timestampMs(body.finishedAtMs ?? body.finishedAt),
    durationMs: nonNegativeInteger(body.durationMs ?? body.elapsedMs),
    completed,
    total,
    currentItem: basename(body.currentItem),
  };
}

function parseTaskLogs(value: unknown, stages: Record<StageKey, StageState>): TaskLog[] {
  const parsed = Array.isArray(value)
    ? value.map((entry, index) => parseTaskLog(entry, index)).filter((entry): entry is TaskLog => Boolean(entry))
    : [];
  return parsed.length ? parsed.slice(-100) : stageSummaryLogs(stages).slice(-100);
}

function taskStageLabel(stage?: StageKey) {
  return stageLabel(STAGES.find((item) => item.key === stage));
}

function logCountLabel(completed?: number, total?: number) {
  const hasCompleted = completed !== undefined && completed > 0;
  const hasTotal = total !== undefined && total > 0;
  if (completed !== undefined && hasTotal) return t`${completed.toLocaleString(getLocale())} / ${total.toLocaleString(getLocale())}`;
  if (hasTotal) return t({
    message: plural(total, { one: "Total #", other: "Total #" }),
    comment: "Count of all items reported by a pipeline stage.",
  });
  if (hasCompleted) return t({
    message: plural(completed, { one: "Processed #", other: "Processed #" }),
    comment: "Count of items already processed by a pipeline stage.",
  });
  return undefined;
}

function taskProgress(task: Task) {
  const completedDuration = STAGES.reduce((total, { key }) => {
    const stage = task.stages[key];
    const completion = stage.status === "completed"
      ? 1
      : Math.max(0, Math.min(100, stage.progress)) / 100;
    return total + STAGE_OBSERVED_DURATION_MS[key] * completion;
  }, 0);
  return Math.round(completedDuration / TOTAL_OBSERVED_DURATION_MS * 100);
}

function taskHasNotStarted(task: Task) {
  return STAGES.every(({ key }) => {
    const stage = task.stages[key];
    return stage.status === "pending"
      && stage.progress === 0
      && stage.startedAtMs === undefined
      && stage.finishedAtMs === undefined
      && stage.durationMs === undefined
      && !stage.jobId;
  });
}

function taskIsCompleted(task: Task) {
  return STAGES.every(({ key }) => task.stages[key].status === "completed");
}

function taskProgressSummary(task: Task) {
  const runningIndex = STAGES.findIndex(({ key }) => task.stages[key].status === "running");
  if (runningIndex >= 0) {
    const label = stageLabel(STAGES[runningIndex]);
    return t`Stage ${runningIndex + 1} of ${STAGES.length} · ${label}`;
  }
  const interruptedIndex = STAGES.findIndex(({ key }) => ["failed", "cancelled"].includes(task.stages[key].status));
  if (interruptedIndex >= 0) {
    const label = stageLabel(STAGES[interruptedIndex]);
    return t`Stopped at stage ${interruptedIndex + 1} of ${STAGES.length} · ${label}`;
  }
  const nextIndex = STAGES.findIndex(({ key }) => task.stages[key].status !== "completed");
  if (nextIndex >= 0) {
    const label = stageLabel(STAGES[nextIndex]);
    return t`Waiting for stage ${nextIndex + 1} of ${STAGES.length} · ${label}`;
  }
  return t`${STAGES.length} of ${STAGES.length} stages complete`;
}

function taskCurrentStage(task: Task) {
  const running = STAGES.find(({ key }) => task.stages[key].status === "running");
  if (running) return running;
  const interrupted = STAGES.find(({ key }) => ["failed", "cancelled"].includes(task.stages[key].status));
  if (interrupted) return interrupted;
  const next = STAGES.find(({ key }) => task.stages[key].status !== "completed");
  return next ?? STAGES[STAGES.length - 1];
}

function stagePrerequisiteKey(task: Task, stageKey: StageKey): StageKey | undefined {
  const stageIndex = STAGES.findIndex(({ key }) => key === stageKey);
  if (stageIndex <= 0) return undefined;
  return STAGES.slice(0, stageIndex).find(({ key }) => task.stages[key].status !== "completed")?.key;
}

function stagePrerequisiteLabel(task: Task, stageKey: StageKey) {
  const prerequisiteKey = stagePrerequisiteKey(task, stageKey);
  return prerequisiteKey ? stageLabel(STAGES.find(({ key }) => key === prerequisiteKey)) : undefined;
}

function taskHasRunningStage(task: Task, except?: StageKey) {
  return STAGES.some(({ key }) => key !== except && task.stages[key].status === "running");
}

function stageActionState(task: Task, stageKey: StageKey, globallyRunning: boolean) {
  const current = task.stages[stageKey];
  const prerequisite = stagePrerequisiteLabel(task, stageKey);
  const blocked = Boolean(prerequisite) || taskHasRunningStage(task, stageKey) || (globallyRunning && current.status !== "running");
  const label = current.status === "running"
    ? t`Stop ${stageLabel(STAGES.find((stage) => stage.key === stageKey))}`
    : prerequisite
      ? t`Wait for ${prerequisite} to finish`
      : blocked
        ? t`Wait for the current stage to finish`
        : stageAction(current.status);
  return { blocked, label, prerequisite };
}

function normaliseStageStatus(value: unknown): StageStatus {
  const raw = String(value ?? "pending").toLowerCase();
  if (raw.includes("run")) return "running";
  if (raw.includes("complete") || raw.includes("done")) return "completed";
  if (raw.includes("cancel") || raw.includes("pause")) return "cancelled";
  if (raw.includes("fail") || raw.includes("error")) return "failed";
  return "pending";
}

function normaliseStage(value: unknown): StageKey | undefined {
  const raw = String(value ?? "").toLowerCase();
  if (raw.includes("extract") || raw.includes("feature")) return "extract";
  if (raw.includes("mask") || raw.includes("segment")) return "mask";
  if (raw.includes("align") || raw.includes("mapper") || raw.includes("register")) return "align";
  return undefined;
}

function toProgress(value: unknown) {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return 0;
  return Math.round(Math.max(0, Math.min(1, number <= 1 ? number : number / 100)) * 100);
}

function cloneStages(raw: unknown): Record<StageKey, StageState> {
  const source = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  return STAGES.reduce((result, stage) => {
    const item = source[stage.key] && typeof source[stage.key] === "object" ? (source[stage.key] as Record<string, unknown>) : {};
    const updatedAtMs = timestampMs(item.updatedAtMs ?? item.updatedAt);
    result[stage.key] = {
      status: normaliseStageStatus(item.status),
      progress: toProgress(item.progress),
      message: typeof item.message === "string" && item.message ? item.message : "Not run yet",
      jobId: typeof item.jobId === "string" ? item.jobId : undefined,
      phase: typeof item.phase === "string" && item.phase ? item.phase : undefined,
      completed: nonNegativeInteger(item.completed),
      total: nonNegativeInteger(item.total),
      currentItem: basename(item.currentItem),
      timestampMs: timestampMs(item.timestampMs ?? item.updatedAtMs ?? item.updatedAt),
      updatedAtMs,
      startedAtMs: timestampMs(item.startedAtMs ?? item.startedAt),
      finishedAtMs: timestampMs(item.finishedAtMs ?? item.finishedAt),
      durationMs: nonNegativeInteger(item.durationMs ?? item.elapsedMs),
    };
    return result;
  }, {} as Record<StageKey, StageState>);
}

function manifestFromUnknown(value: unknown): ProjectManifest | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const inputPaths = Array.isArray(body.inputPaths) ? body.inputPaths.filter((path): path is string => typeof path === "string") : [];
  const outputPath = typeof body.outputPath === "string" ? body.outputPath : typeof body.rootPath === "string" ? body.rootPath : "";
  if (!outputPath && !inputPaths.length) return null;
  const stages = cloneStages(body.stages);
  return {
    projectId: typeof body.projectId === "string" ? body.projectId : `project-${Date.now()}`,
    name: typeof body.name === "string" && body.name ? body.name : outputPath.split(/[\\/]/).filter(Boolean).pop() || "Unnamed reconstruction",
    rootPath: typeof body.rootPath === "string" ? body.rootPath : outputPath,
    inputPaths,
    outputPath,
    settings: normalisePipelineSettings(body.settings),
    stages,
    logs: parseTaskLogs(body.logs ?? body.pipelineLogs, stages),
    warnings: Array.isArray(body.warnings) ? body.warnings.map((warning) => String(warning)) : [],
    createdAt: typeof body.createdAt === "string" ? body.createdAt : undefined,
    updatedAt: typeof body.updatedAt === "string" ? body.updatedAt : undefined,
  };
}

function taskCreatedAtMs(task: Task) {
  return timestampMs(task.createdAt)
    ?? task.logs.reduce<number | undefined>((oldest, log) => oldest === undefined ? log.timestampMs : Math.min(oldest, log.timestampMs), undefined)
    ?? timestampMs(task.updatedAt)
    ?? Number.MAX_SAFE_INTEGER;
}

function readProgress(payload: unknown): ProgressEventPayload {
  const body = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  const rawDone = body.done;
  const rawCompleted = body.completed;
  const done = typeof rawDone === "boolean"
    ? rawDone
    : typeof rawCompleted === "boolean"
      ? rawCompleted
      : false;
  return {
    stage: normaliseStage(body.stage ?? body.name),
    progress: body.progress !== undefined || body.percent !== undefined ? toProgress(body.progress ?? body.percent) : undefined,
    status: body.status !== undefined || body.state !== undefined ? normaliseStageStatus(body.status ?? body.state) : undefined,
    message: typeof body.message === "string" ? body.message : undefined,
    jobId: typeof body.jobId === "string" ? body.jobId : undefined,
    phase: typeof body.phase === "string" && body.phase ? body.phase : undefined,
    completed: nonNegativeInteger(typeof rawCompleted === "boolean" ? undefined : rawCompleted),
    total: nonNegativeInteger(body.total),
    currentItem: basename(body.currentItem),
    timestampMs: timestampMs(body.timestampMs ?? body.timestamp),
    elapsedMs: nonNegativeInteger(body.elapsedMs ?? body.durationMs),
    done,
  };
}

function readLogEvent(payload: unknown): LogEventPayload {
  const body = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  return {
    jobId: typeof body.jobId === "string" ? body.jobId : undefined,
    level: normaliseLogLevel(body.level),
    message: typeof body.message === "string" ? body.message : String(payload ?? "Pipeline log"),
    stage: normaliseStage(body.stage),
    phase: typeof body.phase === "string" && body.phase ? body.phase : undefined,
    timestampMs: timestampMs(body.timestampMs ?? body.timestamp) ?? Date.now(),
    completed: nonNegativeInteger(body.completed),
    total: nonNegativeInteger(body.total),
    currentItem: basename(body.currentItem),
  };
}

function parseDoctor(value: unknown, fallback: DoctorReport): DoctorReport {
  if (!value || typeof value !== "object") return fallback;
  const body = value as Record<string, unknown>;
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const accelerators = Array.isArray(body.accelerators) ? body.accelerators : [];
  // Diagnostics are persisted in state as raw backend values. This matters
  // for a locale switch while the settings sheet is open: no old translation
  // should survive in the report object.
  const warnings = Array.isArray(body.warnings) ? body.warnings.map((warning) => String(warning)) : [];
  const itemText = (entry: unknown) => {
    if (typeof entry === "string") return entry;
    if (entry && typeof entry === "object") {
      const record = entry as Record<string, unknown>;
      return String(record.version ?? record.name ?? record.path ?? record.detail ?? "Detected");
    }
    return "";
  };
  const available = (entry: unknown) => {
    if (entry && typeof entry === "object") {
      const record = entry as Record<string, unknown>;
      if (typeof record.available === "boolean") return record.available;
      if (typeof record.ready === "boolean") return record.ready;
      if (typeof record.status === "string") return !/(missing|failed|error|unavailable)/i.test(record.status);
    }
    return typeof entry === "string" ? !/(missing|failed|error|unavailable)/i.test(entry) : true;
  };
  const entryName = (entry: unknown) => entry && typeof entry === "object" ? String((entry as Record<string, unknown>).name ?? "") : String(entry ?? "");
  const entryKind = (entry: unknown) => entry && typeof entry === "object" ? String((entry as Record<string, unknown>).kind ?? "") : "";
  const entryPath = (entry: unknown) => entry && typeof entry === "object" && typeof (entry as Record<string, unknown>).path === "string" ? String((entry as Record<string, unknown>).path) : "";
  const entryNote = (entry: unknown) => entry && typeof entry === "object" && typeof (entry as Record<string, unknown>).note === "string" ? String((entry as Record<string, unknown>).note) : "";
  const ffmpeg = tools.find((entry) => entryName(entry).toLowerCase() === "ffmpeg");
  const ffprobe = tools.find((entry) => entryName(entry).toLowerCase() === "ffprobe");
  const colmap = tools.find((entry) => entryName(entry).toLowerCase() === "colmap");
  const ffmpegAccelerators = accelerators.filter((entry) => {
    if (!ffmpeg) return false;
    const text = `${entryKind(entry)} ${entryName(entry)} ${itemText(entry)}`;
    return /(videotoolbox|video\s*toolbox|ffmpeg)/i.test(text)
      || (/(cuda|cuvid)/i.test(text) && !/colmap/i.test(text));
  });
  const colmapCapabilities = body.colmapCapabilities && typeof body.colmapCapabilities === "object"
    ? body.colmapCapabilities as Record<string, unknown>
    : undefined;
  type CapabilityState = { known: boolean; available: boolean; text: string };
  const capabilityState = (value: unknown): CapabilityState => {
    if (typeof value === "boolean") return { known: true, available: value, text: value ? "Supported" : "Unsupported" };
    if (typeof value === "number" && Number.isFinite(value)) return { known: true, available: value !== 0, text: value !== 0 ? "Supported" : "Unsupported" };
    if (typeof value === "string") {
      const text = value.trim();
      const lower = text.toLowerCase();
      if (!text) return { known: false, available: false, text: "Not reported" };
      if (/^(false|no|none|unsupported|unavailable|missing|failed|disabled|off|0)$/.test(lower) || /(not\s+supported|without|unavailable|missing|failed|disabled)/i.test(lower)) {
        return { known: true, available: false, text: "Unsupported" };
      }
      if (/^(true|yes|supported|available|ready|enabled|on|1)$/.test(lower) || /(cuda|gpu|supported|available|ready|enabled)/i.test(lower)) {
        return { known: true, available: true, text: "Supported" };
      }
      return { known: true, available: true, text };
    }
    if (value && typeof value === "object") {
      const record = value as Record<string, unknown>;
      const stateKey = ["available", "supported", "enabled", "ready", "detected"].find((key) => key in record);
      if (stateKey) {
        const state = capabilityState(record[stateKey]);
        const detail = typeof record.detail === "string" ? record.detail : typeof record.note === "string" ? record.note : state.text;
        return { ...state, text: detail || state.text };
      }
      if (typeof record.status === "string") return capabilityState(record.status);
      if (typeof record.version === "string") return { known: true, available: true, text: record.version };
    }
    return { known: false, available: false, text: "Not reported" };
  };
  const colmapCuda = capabilityState(colmapCapabilities?.cudaBuild);
  const featureExtractionGpu = capabilityState(colmapCapabilities?.featureExtractionGpu);
  const featureMatchingGpu = capabilityState(colmapCapabilities?.featureMatchingGpu);
  const mapperBaGpu = capabilityState(colmapCapabilities?.mapperBaGpu ?? colmapCapabilities?.ceresGpu);
  const globalMapper = capabilityState(colmapCapabilities?.globalMapper);
  const hasColmapCapabilities = Boolean(colmapCapabilities && Object.keys(colmapCapabilities).length > 0);
  const gpuStages = [featureExtractionGpu, featureMatchingGpu, mapperBaGpu];
  const gpuStagesKnown = gpuStages.every((stage) => stage.known);
  const gpuStagesAvailable = gpuStages.every((stage) => stage.available);
  const gpuAvailable = typeof body.gpuAvailable === "boolean"
    ? body.gpuAvailable
    : colmapCuda.known ? colmapCuda.available : undefined;
  const colmapCudaAccelerator = accelerators.find((entry) => /colmap\s+cuda/i.test(`${entryName(entry)} ${entryKind(entry)}`));
  const colmapCudaDetails = hasColmapCapabilities
    ? [
      colmapCudaAccelerator ? entryNote(colmapCudaAccelerator) : "",
      `CUDA build: ${colmapCuda.text}`,
      `SIFT extraction: ${featureExtractionGpu.text}`,
      `SIFT matching: ${featureMatchingGpu.text}`,
      `Ceres BA: ${mapperBaGpu.known ? mapperBaGpu.available ? "May be available (runtime CUDA/cuDSS check required)" : "CPU only" : "Not reported"}`,
      globalMapper.known ? `Global Mapper: ${globalMapper.text}` : "",
    ].filter(Boolean)
    : ["The legacy diagnostic did not report the COLMAP build; FFmpeg CUDA/VideoToolbox does not imply COLMAP CUDA"];
  const colmapCudaStatus: DiagnosticStatus = hasColmapCapabilities && colmapCuda.known
    ? gpuAvailable && gpuStagesKnown && gpuStagesAvailable ? "ready" : "warning"
    : "unknown";
  const colmapCudaValue = hasColmapCapabilities && colmapCuda.known
    ? gpuAvailable
      ? gpuStagesKnown && gpuStagesAvailable ? "CUDA acceleration available" : "CUDA acceleration partially available"
      : "No usable CUDA GPU detected"
    : "CUDA status not confirmed";
  const capabilityLabels: Record<string, string> = { extract: "Frame extraction", mask: "Masking", align: "Alignment" };
  const pipelineCapabilities = body.capabilities && typeof body.capabilities === "object" ? body.capabilities as Record<string, unknown> : undefined;
  const capabilityValue = pipelineCapabilities ? Object.entries(pipelineCapabilities).filter(([, state]) => Boolean(state)).map(([key]) => capabilityLabels[key] ?? key).join(" · ") : "";
  const alignCapability = capabilityState(pipelineCapabilities?.align);
  const platform = platformLabel(typeof body.platform === "string" ? body.platform : typeof body.os === "string" ? body.os : fallback.platform);
  const rawSystemInfo = body.systemInfo && typeof body.systemInfo === "object"
    ? body.systemInfo as Record<string, unknown>
    : {};
  const stringList = (value: unknown) => Array.isArray(value)
    ? Array.from(new Set(value.filter((entry): entry is string => typeof entry === "string" && Boolean(entry.trim())).map((entry) => entry.trim())))
    : [];
  const systemInfo: SystemInfo = {
    osName: typeof rawSystemInfo.osName === "string" && rawSystemInfo.osName.trim() ? rawSystemInfo.osName.trim() : platform,
    osVersion: typeof rawSystemInfo.osVersion === "string" && rawSystemInfo.osVersion.trim() ? rawSystemInfo.osVersion.trim() : "Not detected",
    architecture: typeof rawSystemInfo.architecture === "string" && rawSystemInfo.architecture.trim()
      ? rawSystemInfo.architecture.trim()
      : typeof body.arch === "string" && body.arch.trim() ? body.arch.trim() : "Not detected",
    processors: stringList(rawSystemInfo.processors),
    graphicsAdapters: stringList(rawSystemInfo.graphicsAdapters),
  };
  const gpuDevices = Array.isArray(body.gpuDevices)
    ? body.gpuDevices.flatMap((entry) => {
      if (!entry || typeof entry !== "object") return [];
      const device = entry as Record<string, unknown>;
      if (typeof device.index !== "number" || !Number.isInteger(device.index) || device.index < 0) return [];
      if (typeof device.name !== "string" || !device.name.trim()) return [];
      return [{ index: device.index, name: device.name.trim() }];
    })
    : [];
  const ffmpegAccelerationValue = ffmpegAccelerators
    .map((entry) => `${entryName(entry) || entryKind(entry) || itemText(entry)}: ${available(entry) ? "enabled in build" : "Unsupported"}`)
    .join(" · ");
  const colmapReady = Boolean(colmap && available(colmap));
  const colmapWorkflowReady = colmapReady && alignCapability.known && alignCapability.available;
  const ffmpegReady = Boolean(ffmpeg && available(ffmpeg) && ffprobe && available(ffprobe));
  const hardwareAccelerationReady = ffmpegAccelerators.some((entry) => available(entry));
  const items: DiagnosticItem[] = [
    {
      label: "COLMAP",
      value: colmapWorkflowReady ? itemText(colmap) : colmapReady ? "COLMAP alignment capability not confirmed" : "COLMAP not detected",
      detail: colmapWorkflowReady ? "Native dual-fisheye camera-rig alignment is available" : colmapReady ? "The executable was found, but the complete alignment workflow was not confirmed" : entryNote(colmap) || "The alignment stage will remain pending",
      details: colmapReady ? [entryPath(colmap) ? `Executable: ${entryPath(colmap)}` : "Executable: system PATH", `Alignment workflow: ${alignCapability.text}`] : undefined,
      status: colmapWorkflowReady ? "ready" : "warning",
    },
    {
      label: COLMAP_CUDA_DIAGNOSTIC_LABEL,
      value: colmapCudaValue,
      detail: colmapCudaAccelerator ? entryNote(colmapCudaAccelerator) || "COLMAP CUDA capabilities were checked" : "COLMAP CUDA capability check result",
      details: colmapCudaDetails,
      status: colmapCudaStatus,
    },
    {
      label: "FFmpeg",
      value: ffmpegReady ? itemText(ffmpeg) : "FFmpeg tools incomplete",
      detail: ffmpegReady ? "FFmpeg and ffprobe are both available" : "Frame extraction requires FFmpeg and ffprobe",
      details: [
        `FFmpeg: ${ffmpeg && available(ffmpeg) ? entryPath(ffmpeg) || itemText(ffmpeg) || "Available" : "Not detected"}`,
        `ffprobe: ${ffprobe && available(ffprobe) ? entryPath(ffprobe) || itemText(ffprobe) || "Available" : "Not detected"}`,
      ],
      status: ffmpegReady ? "ready" : "warning",
    },
    {
      label: HARDWARE_ACCELERATION_LABEL,
      value: ffmpegAccelerators.length ? hardwareAccelerationReady ? "FFmpeg hardware acceleration supported" : "FFmpeg hardware acceleration not enabled" : "Hardware decoding status not reported",
      detail: ffmpegAccelerationValue || `${platform} · FFmpeg hardware decoding capability not reported`,
      details: ffmpegAccelerators.map((entry) => entryNote(entry) || `${entryName(entry) || entryKind(entry) || itemText(entry)}: ${available(entry) ? "enabled in build" : "Unsupported"}`),
      status: ffmpegAccelerators.length ? hardwareAccelerationReady ? "ready" : "warning" : "unknown",
    },
  ];
  return {
    platform,
    systemInfo,
    summary: typeof body.summary === "string" ? body.summary : capabilityValue || fallback.summary,
    // Store a stable timestamp; format it with the active locale in the view.
    checkedAt: new Date().toISOString(),
    items,
    warnings,
    colmapCapabilities,
    gpuAvailable,
    gpuDevices,
  };
}

async function invokeSafely<T>(command: string, args?: Record<string, unknown>) {
  if (!IS_TAURI_RUNTIME) return null;
  try {
    return await invoke<T>(command, args);
  } catch (error) {
    console.error(`[SphereAlign] ${command}`, error);
    return null;
  }
}

function deriveOutputPath(path: string) {
  if (!path) return "";
  const separator = path.includes("\\") ? "\\" : "/";
  const parent = path.includes(separator) ? path.slice(0, path.lastIndexOf(separator)) : path;
  const name = path.slice(path.lastIndexOf(separator) + 1).replace(/\.[^.]+$/, "") || "capture";
  return `${parent}${separator}colmap-${name}`;
}

function sourceFromPath(path: string, index: number): OsvSource {
  const label = path.split(/[\\/]/).filter(Boolean).pop() || `OSV ${index + 1}`;
  return { id: `${index}-${path}`, path, label: `OSV ${String(index + 1).padStart(2, "0")}`, detail: label };
}

function normaliseColorInspection(value: unknown): ColorInspection | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const nestedCandidates = [body.colorInspection, body.colorProfile, body.colorDetection, body.detection]
    .filter((candidate): candidate is Record<string, unknown> => Boolean(candidate) && typeof candidate === "object");
  const nested = nestedCandidates[0];
  const shouldApplyValue = body.shouldApply
    ?? body.should_apply
    ?? nested?.shouldApply
    ?? nested?.should_apply;
  return typeof shouldApplyValue === "boolean" ? { shouldApply: shouldApplyValue } : null;
}

function normaliseColorInspectionSummary(value: unknown): ColorInspectionSummary | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const rawSources = Array.isArray(body.sources)
    ? body.sources
    : Array.isArray(body.files)
      ? body.files
      : [];
  const files = rawSources.flatMap((source) => {
    const inspection = normaliseColorInspection(source);
    return inspection ? [inspection] : [];
  });
  const aggregateCandidates = [body.colorInspection, body.colorProfileDetection, body.colorDetection, body.colorProfile]
    .filter((candidate) => candidate && typeof candidate === "object");
  const aggregate = aggregateCandidates.map((candidate) => normaliseColorInspection(candidate)).find(Boolean)
    ?? normaliseColorInspection(body);
  if (!files.length && !aggregate) return null;
  const fileRecommendations = files.flatMap((file) => file.shouldApply === undefined ? [] : [file.shouldApply]);
  return {
    files,
    shouldApply: aggregate?.shouldApply ?? (fileRecommendations.length === files.length && fileRecommendations.length > 0 && fileRecommendations.every(Boolean)
      ? true
      : fileRecommendations.length === files.length && fileRecommendations.every((item) => !item) ? false : undefined),
  };
}

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

function SourceThumbnail({
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

function SupportedFormatCard({ icon: Icon, title, detail }: { icon: LucideIcon; title: ReactNode; detail: ReactNode }) {
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

function SourceListItem({
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

function DetailSectionHeading({ title, meta }: { title: ReactNode; meta: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="text-sm font-semibold text-foreground">{title}</h2>
      <span className="text-sm text-muted-foreground">{meta}</span>
    </div>
  );
}

function DetailMetric({
  label,
  value,
  icon: Icon,
  fullWidth = false,
}: {
  label: ReactNode;
  value: ReactNode;
  icon?: LucideIcon;
  fullWidth?: boolean;
}) {
  return (
    <div className={cn(
      "flex min-w-0 flex-col gap-1 border-b py-2.5 last:border-b-0 odd:border-r odd:pr-3.5 even:pl-3.5",
      fullWidth && "col-span-full border-r-0 px-0 odd:pr-0 even:pl-0",
    )}>
      <dt className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
        {Icon && <Icon className="size-3.5" aria-hidden="true" />}
        {label}
      </dt>
      <dd className={cn("min-w-0 text-sm tabular-nums text-foreground", fullWidth && "truncate font-mono")}>{value}</dd>
    </div>
  );
}

function AppNotice({ message, onClose, avoidBottomAction = false }: { message: string; onClose: () => void; avoidBottomAction?: boolean }) {
  return (
    <m.div
      initial={{ y: 8, opacity: 0, scale: 0.98 }}
      animate={{ y: 0, opacity: 1, scale: 1 }}
      exit={{ y: 6, opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.18, ease: APP_NOTICE_EASE }}
      className={cn("fixed bottom-5 left-5 z-60 flex w-[min(360px,calc(100vw-40px))] items-start gap-2.5 rounded-xl border bg-popover/95 p-3 text-sm text-popover-foreground shadow-md backdrop-blur-md", avoidBottomAction && "max-[760px]:bottom-18")}
      role="status"
      aria-atomic="true"
    >
      <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
        <Info className="size-4" aria-hidden="true" />
      </span>
      <span className="min-w-0 flex-1 self-center leading-relaxed break-words">{message}</span>
      <Button className="-mt-1 -mr-1 shrink-0" variant="ghost" size="icon-xs" onClick={onClose} aria-label={t`Close notification`}>
        <X />
      </Button>
    </m.div>
  );
}

function WindowsWindowControls() {
  if (!IS_WINDOWS_RUNTIME) return null;

  const appWindow = getCurrentWindow();
  const runWindowCommand = (command: () => Promise<void>) => {
    void command().catch((error) => console.error("[SphereAlign] Window control", error));
  };

  return (
    <div className="flex self-stretch" aria-label={t`Window controls`}>
      <Button
        className="h-full w-11 rounded-none"
        variant="ghost"
        aria-label={t`Minimize window`}
        title={t`Minimize window`}
        onClick={() => runWindowCommand(() => appWindow.minimize())}
      >
        <Minus />
      </Button>
      <Button
        className="h-full w-11 rounded-none"
        variant="ghost"
        aria-label={t`Maximize or restore window`}
        title={t`Maximize or restore window`}
        onClick={() => runWindowCommand(() => appWindow.toggleMaximize())}
      >
        <Square />
      </Button>
      <Button
        className="h-full w-11 rounded-none hover:bg-destructive hover:text-destructive-foreground"
        variant="ghost"
        aria-label={t`Close window`}
        title={t`Close window`}
        onClick={() => runWindowCommand(() => appWindow.close())}
      >
        <X />
      </Button>
    </div>
  );
}

function iconForDiagnostic(label: string) {
  if (label.includes("GPU") || label.includes("CUDA") || label.includes("Hardware acceleration")) return Gpu;
  if (label.includes("COLMAP")) return ScanSearch;
  if (label.includes("FFmpeg")) return FileVideoCamera;
  return MonitorCog;
}

function warningAffectsProcessingSpeed(warning: string) {
  return /(CUDA|GPU|CPU|hardware|acceleration|VideoToolbox|decode|decoder|feature extraction|matching|Ceres|cuDSS|performance|speed|slow|fallback|硬體|硬件|加速|解碼|解码|特徵|特征|配對|匹配|影格擷取|顯示卡|显卡|推論回退|推理回退|效能|性能|速度)/i.test(warning);
}

function diagnosticStatusLabel(status: DiagnosticStatus) {
  if (status === "ready") return t`Available`;
  if (status === "warning") return t`Needs attention`;
  return t`Not checked`;
}

// Match absolute paths independently of the language used for the surrounding
// diagnostic label. The line-oriented path alternatives intentionally consume
// spaces so a Windows path such as `C:\\Program Files\\COLMAP` cannot leak its
// second segment through a whitespace-bounded token match.
const DIAGNOSTIC_PATH_PATTERN = /(^|[\s("'=:\uFF1A])((?:\/(?!\/)[^\r\n"'<>]+|[A-Z]:[\\/][^\r\n"'<>]+|\\\\[^\r\n"'<>]+))/gim;
const DIAGNOSTIC_PATH_DETECTION_PATTERN = /(?:^|[\s("'=:\uFF1A])(?:\/(?!\/)[^\r\n"'<>]+|[A-Z]:[\\/][^\r\n"'<>]+|\\\\[^\r\n"'<>]+)/im;

function redactDiagnosticText(value: string) {
  // Redact raw paths before any UI translation. The callback keeps the
  // familiar placeholders for common locations while covering arbitrary
  // absolute paths (for example /usr/local/bin or a Windows path outside
  // `C:\\Users`).
  return value.replace(DIAGNOSTIC_PATH_PATTERN, (_match, prefix: string, path: string) => {
    if (/^\/Users\//i.test(path)) return `${prefix}/Users/<user>`;
    if (/^\/home\//i.test(path)) return `${prefix}/home/<user>`;
    if (/^\/Applications(?:\/|$)/i.test(path)) return `${prefix}/Applications/<app>`;
    if (/^[A-Z]:[\\/]/i.test(path)) return `${prefix}<windows path>`;
    if (/^\\\\/i.test(path)) return `${prefix}<network path>`;
    return `${prefix}<path>`;
  });
}

function containsDiagnosticPath(value: string) {
  return DIAGNOSTIC_PATH_DETECTION_PATTERN.test(value);
}

function englishDiagnosticMessage(value: string, englishI18n: I18n): string {
  const exact = USER_MESSAGE_TRANSLATIONS[value]
    ?? STAGE_SUMMARY_TRANSLATIONS[value]
    ?? APP_MESSAGE_TRANSLATIONS[value];
  if (exact) return englishI18n._(exact);

  const translated = value
    .replace(/^未偵測到$/g, "Not detected")
    .replace(/^(.+) 顯示卡(?: \((.+)\))?$/g, (_match, vendor: string, metadata?: string) => `${vendor} graphics adapter${metadata ? ` (${metadata})` : ""}`)
    .replace(/FFmpeg 硬體加速/g, "FFmpeg hardware acceleration")
    .replace(/指定的 COLMAP 未在 version\/help banner 標示 CUDA；將使用 CPU 特徵擷取與配對/g, "The selected COLMAP did not report CUDA in its version/help banner; CPU feature extraction and matching will be used")
    .replace(/未偵測到可供 COLMAP 使用的 NVIDIA GPU；將使用 CPU/g, "No NVIDIA GPU usable by COLMAP was detected; the CPU will be used")
    .replace(/指定的 COLMAP 建置支援 CUDA，但未確認 Ceres GPU 求解器；Bundle Adjustment 會使用 CPU/g, "The selected COLMAP build supports CUDA, but the Ceres GPU solver was not confirmed; Bundle Adjustment will use the CPU")
    .replace(/^已偵測到 (.+)；COLMAP 可請求 Ceres GPU，實際 CUDA\/cuDSS 支援會在執行時確認$/g, (_match, devices: string) => `Detected ${devices.replace(/、/g, ", ")}; COLMAP can request Ceres GPU, and actual CUDA/cuDSS support will be confirmed at runtime`)
    .replace(/^此 FFmpeg build 已啟用 (.+)；實際可用性仍取決於顯示卡、驅動程式與影片格式$/g, (_match, methods: string) => `This FFmpeg build enables ${methods.replace(/、/g, ", ")}; actual availability still depends on the graphics adapter, driver, and video format`);

  // Backend diagnostics have a bounded English mapping. If a future backend
  // message is added without one, never leak untranslated UI text into the
  // portable report.
  return containsCjk(translated) ? "An untranslated diagnostic message was omitted" : translated;
}

function englishDiagnosticDetail(value: string, englishI18n: I18n): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  if (containsDiagnosticPath(trimmed)) return "Executable detected (full path hidden)";
  return englishDiagnosticMessage(trimmed, englishI18n);
}

function formatEnglishDoctorCheckedAt(value: string) {
  if (value === "Not checked yet") return value;
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit" }).format(parsed);
}

async function doctorReportText(doctor: DoctorReport) {
  const englishI18n = await getEnglishI18n();
  const lines = [
    "SphereAlign diagnostics",
    `Platform: ${englishDiagnosticMessage(doctor.platform, englishI18n)}`,
    `Last checked: ${formatEnglishDoctorCheckedAt(doctor.checkedAt)}`,
    `Summary: ${englishDiagnosticMessage(doctor.summary, englishI18n)}`,
    "",
    "System information",
    `- Operating system: ${englishDiagnosticMessage(doctor.systemInfo.osName, englishI18n)} ${englishDiagnosticMessage(doctor.systemInfo.osVersion, englishI18n)}`,
    `- Architecture: ${englishDiagnosticMessage(doctor.systemInfo.architecture, englishI18n)}`,
    "- Processors:",
    ...(doctor.systemInfo.processors.length > 0 ? doctor.systemInfo.processors.map((processor) => `  - ${englishDiagnosticMessage(processor, englishI18n)}`) : ["  - Not detected"]),
    "- Graphics adapters:",
    ...(doctor.systemInfo.graphicsAdapters.length > 0 ? doctor.systemInfo.graphicsAdapters.map((adapter) => `  - ${englishDiagnosticMessage(adapter, englishI18n)}`) : ["  - Not detected"]),
    "",
    "Environment checks",
  ];
  doctor.items.forEach((item) => {
    const status = item.status === "ready" ? "Available" : item.status === "warning" ? "Needs attention" : "Not checked";
    lines.push(`- ${item.label} [${status}]`);
    lines.push(`  Result: ${englishDiagnosticMessage(item.value, englishI18n)}`);
    lines.push(`  Details: ${englishDiagnosticDetail(item.detail, englishI18n)}`);
    item.details?.forEach((detail) => lines.push(`  - ${englishDiagnosticDetail(detail, englishI18n)}`));
  });
  lines.push("", "Warnings");
  if (doctor.warnings.length > 0) doctor.warnings.forEach((warning) => lines.push(`- ${englishDiagnosticMessage(warning, englishI18n)}`));
  else lines.push("- None");
  return redactDiagnosticText(lines.join("\n"));
}

function stageAction(status: StageStatus) {
  if (status === "running") return t`Cancel`;
  if (status === "cancelled") return t`Resume`;
  if (status === "failed") return t`Retry`;
  if (status === "completed") return t`Run again`;
  return t`Run`;
}

function stageStatusLabel(status: StageStatus) {
  if (status === "running") return t`Running`;
  if (status === "cancelled") return t`Cancelled`;
  if (status === "failed") return t`Failed`;
  if (status === "completed") return t`Completed`;
  return t`Pending`;
}

function StageStatusBadge({ status }: { status: StageStatus }) {
  return (
    <Badge
      className={cn(
        status === "running" && "border-primary/30 bg-primary/10 text-primary [&_[data-icon=inline-start]]:animate-spin [&_[data-icon=inline-start]]:[animation-duration:900ms]",
        status === "completed" && "border-emerald-600/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
        status === "cancelled" && "bg-muted text-muted-foreground",
      )}
      variant={status === "failed" ? "destructive" : "outline"}
    >
      {status === "running"
        ? <LoaderCircle data-icon="inline-start" aria-hidden="true" />
        : <span className={cn(
          "block size-1.25 rounded-full bg-current",
          status === "completed" && "text-emerald-600",
          status === "failed" && "text-destructive",
          status === "cancelled" && "text-amber-600",
          status === "pending" && "text-muted-foreground",
        )} />}
      {stageStatusLabel(status)}
    </Badge>
  );
}

function taskStageDuration(stage: StageState, nowMs: number) {
  if (stage.status === "running" && stage.startedAtMs !== undefined) return Math.max(0, nowMs - stage.startedAtMs);
  if (stage.durationMs !== undefined) return stage.durationMs;
  if (stage.startedAtMs === undefined) return undefined;
  if (stage.finishedAtMs !== undefined) return Math.max(0, stage.finishedAtMs - stage.startedAtMs);
  return undefined;
}

function estimatedRemainingMs(stage: StageState, nowMs: number) {
  if (stage.status !== "running") return undefined;
  const elapsed = taskStageDuration(stage, nowMs);
  if (elapsed === undefined || elapsed <= 0) return undefined;
  if (stage.total !== undefined && stage.completed !== undefined && stage.completed > 0 && stage.total >= stage.completed) {
    return Math.max(0, (elapsed / stage.completed) * (stage.total - stage.completed));
  }
  const progress = stage.progress / 100;
  if (progress < 0.05) return undefined;
  return Math.max(0, elapsed * (1 - progress) / progress);
}

function formatEta(value?: number) {
  if (value === undefined) return t`Estimating`;
  if (value < 60_000) return t`About ${formatDuration(value)}`;
  const minutes = Math.max(1, Math.round(value / 60_000));
  return t`About ${minutes} min`;
}

function processingRateLabel(completed: number | undefined, startedAtMs: number | undefined, nowMs: number) {
  if (completed === undefined || completed <= 0 || startedAtMs === undefined) return t`Estimating`;
  const elapsed = Math.max(0, nowMs - startedAtMs);
  if (elapsed < 1000) return t`Estimating`;
  const rate = completed / (elapsed / 1000);
  return t`${rate >= 10 ? rate.toFixed(1) : rate.toFixed(2)} items/sec`;
}

function logLevelForStatus(status?: StageStatus): TaskLogLevel {
  if (status === "failed") return "error";
  if (status === "cancelled") return "warning";
  return "info";
}

function mergeProgressLog(
  logs: TaskLog[],
  taskId: string,
  stageKey: StageKey,
  previousStage: StageState,
  payload: ProgressEventPayload,
  status: StageStatus,
  eventTime: number,
) {
  const jobId = payload.jobId || previousStage.jobId;
  const phase = payload.phase || previousStage.phase || (status === "running" ? "processing" : status);
  const jobKey = jobId || taskId;
  const activeIndex = logs.reduce((found, log, index) => (
    log.kind === "progress" && log.stage === stageKey && (!jobId || !log.jobId || log.jobId === jobId) && log.finishedAtMs === undefined ? index : found
  ), -1);
  const matchingIndex = logs.findIndex((log) => log.kind === "progress" && log.stage === stageKey && log.jobId === jobId && log.phase === phase);
  let nextLogs = logs.slice();
  if (activeIndex >= 0 && matchingIndex < 0 && nextLogs[activeIndex].phase !== phase) {
    const active = nextLogs[activeIndex];
    nextLogs[activeIndex] = {
      ...active,
      finishedAtMs: eventTime,
      durationMs: active.startedAtMs === undefined ? active.durationMs : Math.max(0, eventTime - active.startedAtMs),
    };
  }
  const index = matchingIndex >= 0 ? matchingIndex : activeIndex >= 0 && nextLogs[activeIndex].phase === phase ? activeIndex : -1;
  const current = index >= 0 ? nextLogs[index] : undefined;
  const startedAtMs = current?.startedAtMs ?? (status === "running" ? eventTime : previousStage.startedAtMs ?? eventTime);
  const finished = payload.done || status !== "running";
  const finishedAtMs = finished ? eventTime : undefined;
  const durationMs = payload.elapsedMs ?? (finished && startedAtMs !== undefined ? Math.max(0, eventTime - startedAtMs) : current?.durationMs);
  const next: TaskLog = {
    id: current?.id ?? `progress:${jobKey}:${stageKey}:${phase}`,
    kind: "progress",
    stage: stageKey,
    phase,
    jobId,
    level: logLevelForStatus(status),
    message: payload.message || current?.message || previousStage.message,
    timestampMs: eventTime,
    startedAtMs,
    finishedAtMs,
    durationMs,
    completed: payload.completed ?? current?.completed,
    total: payload.total ?? current?.total,
    currentItem: payload.currentItem ?? current?.currentItem,
  };
  if (index >= 0) nextLogs[index] = next;
  else nextLogs.push(next);
  return nextLogs.slice(-100);
}

function appendMessageLog(logs: TaskLog[], taskId: string, payload: LogEventPayload, stage?: StageKey, phase?: string) {
  const sequence = `${payload.timestampMs}-${logs.length}`;
  return [...logs, {
    id: `message:${payload.jobId || taskId}:${sequence}`,
    kind: "message" as const,
    stage: payload.stage || stage,
    phase: payload.phase || phase,
    jobId: payload.jobId,
    level: payload.level,
    message: payload.message,
    timestampMs: payload.timestampMs,
    completed: payload.completed,
    total: payload.total,
    currentItem: payload.currentItem,
  }].slice(-100);
}

function App() {
  // Subscribe the whole screen to Lingui's locale-change event. Most of the
  // app uses the macro helpers directly, but raw backend messages are rendered
  // through localiseUserMessage and need the same rerender trigger.
  const { i18n: lingui } = useLingui();
  const { theme, setTheme } = useTheme();
  const shouldReduceMotion = useReducedMotion();
  void lingui.locale;
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);
  const [editingTaskId, setEditingTaskId] = useState<string | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [taskDetailTab, setTaskDetailTab] = useState<"summary" | "records">("summary");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [taskDetailUsesSplitView, setTaskDetailUsesSplitView] = useState(() => window.matchMedia("(min-width: 921px)").matches);
  const taskNameInputRef = useRef<HTMLInputElement>(null);
  const taskDetailTriggerRef = useRef<HTMLButtonElement | null>(null);
  const closeTaskDetail = useCallback(() => setTaskDetailOpen(false), []);
  const [nameDraft, setNameDraft] = useState("");
  const [sourcePaths, setSourcePaths] = useState<string[]>([]);
  const [outputDraft, setOutputDraft] = useState("");
  const [settingsDraft, setSettingsDraft] = useState<PipelineSettings>(DEFAULT_SETTINGS);
  const [colmapPath, setColmapPath] = useState(() => {
    try {
      return window.localStorage.getItem(COLMAP_PATH_STORAGE_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [sourceInspection, setSourceInspection] = useState<string>("");
  const [sourceColorInspection, setSourceColorInspection] = useState<ColorInspectionSummary | null>(null);
  const [doctor, setDoctor] = useState<DoctorReport>(() => emptyDoctor());
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [clockMs, setClockMs] = useState(() => Date.now());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeJobIds = useRef<Record<string, string>>({});
  const jobTaskIds = useRef<Record<string, string>>({});
  const terminalJobIds = useRef<Set<string>>(new Set());
  const ignoredJobIds = useRef<Set<string>>(new Set());
  const pendingStageStarts = useRef<Record<string, StageKey>>({});

  useEffect(() => {
    if (!toast) return;
    const timeoutId = window.setTimeout(() => setToast(null), 5000);
    return () => window.clearTimeout(timeoutId);
  }, [toast]);
  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 921px)");
    const updateSplitView = (event: MediaQueryListEvent) => setTaskDetailUsesSplitView(event.matches);
    mediaQuery.addEventListener("change", updateSplitView);
    return () => mediaQuery.removeEventListener("change", updateSplitView);
  }, []);
  const pendingLogsByJobId = useRef<Record<string, TaskLog[]>>({});
  const taskSnapshot = useRef<Task[]>([]);
  const logSequence = useRef(0);
  const doctorRunId = useRef(0);
  const gpuPreferenceTouched = useRef(false);
  const autoPipelineRuns = useRef<Record<string, AutoPipelineRun>>({});
  const pumpAutoPipelineRef = useRef<() => void>(() => undefined);

  const selectedSources = useMemo(() => sourcePaths.map(sourceFromPath), [sourcePaths]);
  const selectedTask = useMemo(() => tasks.find((task) => task.projectId === selectedTaskId), [selectedTaskId, tasks]);
  const selectedTaskSources = useMemo(() => selectedTask?.inputPaths.map(sourceFromPath) ?? [], [selectedTask]);
  const selectedTaskLogs = useMemo(() => selectedTask ? selectedTask.logs.slice().sort((left, right) => right.timestampMs - left.timestampMs) : [], [selectedTask]);
  const selectedStageDefinition = selectedTask ? taskCurrentStage(selectedTask) : undefined;
  const selectedStage = selectedTask && selectedStageDefinition ? selectedTask.stages[selectedStageDefinition.key] : undefined;
  const selectedRunningStageDefinition = selectedTask ? STAGES.find(({ key }) => selectedTask.stages[key].status === "running") : undefined;
  const selectedActiveProgressLog = selectedStageDefinition ? selectedTaskLogs.find((log) => log.kind === "progress" && log.stage === selectedStageDefinition.key && log.finishedAtMs === undefined) : undefined;

  const isWindowsPlatform = doctor.platform === "Windows";
  const doctorEssentialReady = ["COLMAP", "FFmpeg"].every((label) => doctor.items.find((item) => item.label === label)?.status === "ready");
  const uniqueDoctorWarnings = Array.from(new Set(doctor.warnings));
  const performanceWarnings = uniqueDoctorWarnings.filter(warningAffectsProcessingSpeed);
  const generalDoctorWarnings = uniqueDoctorWarnings.filter((warning) => !warningAffectsProcessingSpeed(warning));
  const gpuDiagnostic = doctor.items.find((item) => item.label === COLMAP_CUDA_DIAGNOSTIC_LABEL);
  const hardwareDiagnostic = doctor.items.find((item) => item.label === HARDWARE_ACCELERATION_LABEL);
  const performanceFallback = gpuDiagnostic?.details?.find((detail) => /(CPU|Unsupported|Not confirmed|unavailable)/i.test(detail))
    || hardwareDiagnostic?.details?.find((detail) => /(CPU|Unsupported|Not confirmed|unavailable)/i.test(detail))
    || t`Some CUDA or hardware acceleration capabilities are unavailable; affected stages will use the CPU.`;
  const performanceStatus: DiagnosticStatus = performanceWarnings.length > 0 || gpuDiagnostic?.status === "warning" || hardwareDiagnostic?.status === "warning"
    ? "warning"
    : gpuDiagnostic?.status === "unknown" || hardwareDiagnostic?.status === "unknown"
      ? "unknown"
      : "ready";
  const orderedTasks = useMemo(() => tasks
    .map((task, index) => ({ task, index }))
    // New tasks are prepended to state, so reverse the original index for
    // manifests created within the backend's same timestamp resolution.
    .sort((left, right) => taskCreatedAtMs(left.task) - taskCreatedAtMs(right.task) || right.index - left.index)
    .map(({ task }) => task), [tasks]);
  const activeTasks = useMemo(() => orderedTasks.filter((task) => !taskIsCompleted(task)), [orderedTasks]);
  const completedTasks = useMemo(() => orderedTasks.filter(taskIsCompleted), [orderedTasks]);
  const hasRunningStage = useMemo(() => tasks.some((task) => STAGES.some(({ key }) => task.stages[key].status === "running")), [tasks]);
  const taskbarProgress = useMemo(() => {
    const runningTasks = tasks.filter((task) => STAGES.some(({ key }) => task.stages[key].status === "running"));
    if (runningTasks.length === 0) return null;
    return Math.round(runningTasks.reduce((total, task) => total + taskProgress(task), 0) / runningTasks.length);
  }, [tasks]);

  useEffect(() => {
    taskSnapshot.current = tasks;
  }, [tasks]);

  useEffect(() => {
    try {
      const path = colmapPath.trim();
      if (path) window.localStorage.setItem(COLMAP_PATH_STORAGE_KEY, path);
      else window.localStorage.removeItem(COLMAP_PATH_STORAGE_KEY);
    } catch (error) {
      console.info("[SphereAlign] COLMAP path preference", error);
    }
  }, [colmapPath]);

  useEffect(() => {
    if (!hasRunningStage) return undefined;
    const interval = window.setInterval(() => setClockMs(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [hasRunningStage]);

  useEffect(() => {
    if (!IS_WINDOWS_RUNTIME) return;
    const state = taskbarProgress === null
      ? { status: ProgressBarStatus.None }
      : taskbarProgress === 0
        ? { status: ProgressBarStatus.Indeterminate }
        : { status: ProgressBarStatus.Normal, progress: taskbarProgress };
    void getCurrentWindow().setProgressBar(state).catch((error) => {
      console.error("[SphereAlign] Windows taskbar progress", error);
    });
  }, [taskbarProgress]);

  const appendTaskLog = useCallback((taskId: string, payload: LogEventPayload, stage?: StageKey, phase?: string) => {
    setTasks((current) => current.map((task) => task.projectId === taskId
      ? { ...task, logs: appendMessageLog(task.logs, taskId, payload, stage, phase) }
      : task));
  }, []);

  const bindJobToTask = useCallback((taskId: string, jobId: string) => {
    if (!jobId) return;
    activeJobIds.current[taskId] = jobId;
    jobTaskIds.current[jobId] = taskId;
    const pending = pendingLogsByJobId.current[jobId];
    if (!pending?.length) return;
    delete pendingLogsByJobId.current[jobId];
    setTasks((current) => current.map((task) => task.projectId === taskId
      ? { ...task, logs: [...task.logs, ...pending].slice(-100) }
      : task));
  }, []);

  const addTaskMessage = useCallback((taskId: string, message: string, level: TaskLogLevel = "info") => {
    logSequence.current += 1;
    appendTaskLog(taskId, {
      level,
      // Keep task logs in the source language. The log row translates this
      // value at render time, so changing locale also updates existing logs.
      message,
      timestampMs: Date.now() + logSequence.current / 1000,
    });
  }, [appendTaskLog]);

  const loadProjectPath = useCallback(async (path: string) => {
    const manifest = manifestFromUnknown(await invokeSafely("load_project", { path }));
    if (!manifest) {
      setToast("No loadable project information was found");
      return false;
    }
    setTasks((current) => [manifest, ...current.filter((task) => task.projectId !== manifest.projectId)]);
    addTaskMessage(manifest.projectId, `Opened ${manifest.name}`);
    setToast(manifest.warnings.length ? `Loaded unfinished project with ${manifest.warnings.length} warnings` : `Loaded unfinished project: ${manifest.name}`);
    return true;
  }, [addTaskMessage]);

  const inspectSourcePaths = useCallback(async (paths: string[]) => {
    if (!paths.length) return;
    const result = await invokeSafely<{
      kind?: string;
      sources?: Array<{
        path?: string;
        name?: string;
        duration?: number;
        fps?: number;
        warnings?: string[];
        colorProfile?: string | Record<string, unknown>;
        shouldApply?: boolean;
      }>;
      colorInspection?: unknown;
      colorProfileDetection?: unknown;
      project?: { path?: string; status?: string; hasManifest?: boolean };
      suggestedOutputPath?: string;
    }>("inspect_paths", { paths });
    if (IS_TAURI_RUNTIME && result?.project && (result.kind === "project" || result.project.status === "partial" || result.project.hasManifest)) {
      const projectPath = result.project.path || paths[0];
      setTaskDialogOpen(false);
      setSourcePaths([]);
      setSourceInspection("");
      setSourceColorInspection(null);
      await loadProjectPath(projectPath);
      return;
    }
    if (result?.sources?.length) {
      const inspectedPaths = result.sources.flatMap((source) => source.path ? [source.path] : []);
      if (inspectedPaths.length) setSourcePaths(inspectedPaths);
      const valid = result.sources.filter((source) => !source.warnings?.length).length;
      setSourceInspection(`${result.sources.length} sources · ${valid} passed inspection`);
      setSourceColorInspection(normaliseColorInspectionSummary(result));
    } else if (result?.suggestedOutputPath) {
      setOutputDraft(result.suggestedOutputPath);
      setSourceInspection("Sources found; you can create a new reconstruction task");
      setSourceColorInspection(normaliseColorInspectionSummary(result));
    } else if (!IS_TAURI_RUNTIME) {
      setSourceInspection(`${paths.length} sources · browser preview`);
      setSourceColorInspection(normaliseColorInspectionSummary(result));
    } else {
      setSourceInspection("Source inspection results are not available yet");
      setSourceColorInspection(normaliseColorInspectionSummary(result));
    }
  }, [loadProjectPath]);

  const applySourcePaths = useCallback((paths: string[], openDialogAfter = true) => {
    const actual = paths.filter(Boolean);
    if (!actual.length) return;
    setSourcePaths(actual);
    setSourceColorInspection(null);
    if (!editingTaskId) {
      setOutputDraft(deriveOutputPath(actual[0]));
      setNameDraft(actual[0].split(/[\\/]/).filter(Boolean).pop()?.replace(/[-_]+/g, " ") || "New reconstruction task");
    }
    if (openDialogAfter) setTaskDialogOpen(true);
    void inspectSourcePaths(actual);
  }, [editingTaskId, inspectSourcePaths]);

  const openNewTaskDialog = useCallback(() => {
    setEditingTaskId(null);
    setNameDraft("");
    setSourcePaths([]);
    setOutputDraft("");
    setSourceInspection("");
    setSourceColorInspection(null);
    gpuPreferenceTouched.current = false;
    setSettingsDraft(selectAvailableGpu({
      ...DEFAULT_SETTINGS,
      align: { ...DEFAULT_SETTINGS.align, useGpu: doctor.gpuAvailable !== false },
    }, doctor.gpuDevices));
    setDragOver(false);
    setTaskDialogOpen(true);
  }, [doctor.gpuAvailable, doctor.gpuDevices]);

  const canChangeQueuedTask = useCallback((task: Task) => {
    const run = autoPipelineRuns.current[task.projectId];
    return taskHasNotStarted(task)
      && !activeJobIds.current[task.projectId]
      && !pendingStageStarts.current[task.projectId]
      && (!run || (!run.stage && !run.jobId));
  }, []);

  const openEditTaskDialog = useCallback((task: Task) => {
    if (!canChangeQueuedTask(task)) {
      setToast("This task has started and can no longer be edited");
      return;
    }
    const run = autoPipelineRuns.current[task.projectId];
    if (run) run.paused = true;
    setEditingTaskId(task.projectId);
    setNameDraft(task.name);
    setSourcePaths(task.inputPaths);
    setOutputDraft(task.outputPath);
    setSettingsDraft(selectAvailableGpu(normalisePipelineSettings(task.settings), doctor.gpuDevices));
      setSourceInspection(`${task.inputPaths.length} sources`);
    setSourceColorInspection(null);
    setDragOver(false);
    setTaskDialogOpen(true);
    void inspectSourcePaths(task.inputPaths);
  }, [canChangeQueuedTask, doctor.gpuDevices, inspectSourcePaths]);

  const handleBrowserFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const paths = Array.from(files).map((file) => {
      const candidate = file as File & { path?: string };
      return candidate.path || file.name;
    });
    applySourcePaths(paths);
  }, [applySourcePaths]);

  const openSourcePicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const result = await openDialog({ directory: false, multiple: true, filters: [{ name: t`OSV / dual-fisheye video`, extensions: ["osv", "mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "ts"] }] });
      const paths = result === null ? [] : Array.isArray(result) ? result : [result];
      applySourcePaths(paths);
    } catch (error) {
      console.info("[SphereAlign] picker fallback", error);
      fileInputRef.current?.click();
    }
  }, [applySourcePaths]);

  const openOutputPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("Browser preview keeps the custom output path");
      return;
    }
    try {
      const result = await openDialog({ directory: true, multiple: false });
      if (typeof result === "string") setOutputDraft(result);
    } catch (error) {
      console.info("[SphereAlign] output picker", error);
    }
  }, []);

  const openProject = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("Browser preview does not read local project information");
      return;
    }
    try {
      const result = await openDialog({ directory: true, multiple: false });
      if (typeof result !== "string") return;
      await loadProjectPath(result);
    } catch (error) {
      console.info("[SphereAlign] load project", error);
      setToast("Failed to open project");
    }
  }, [loadProjectPath]);

  const runDoctor = useCallback(async (customColmapPath: string) => {
    const runId = ++doctorRunId.current;
    setDoctorLoading(true);
    const result = await invokeSafely("doctor", { colmapPath: customColmapPath.trim() || null });
    if (runId !== doctorRunId.current) return;
    if (result) {
      const parsed = parseDoctor(result, emptyDoctor());
      setDoctor(parsed);
      setSettingsDraft((current) => {
        const selected = selectAvailableGpu(current, parsed.gpuDevices);
        const shouldFollowDetection = parsed.gpuAvailable === false
          || (parsed.gpuAvailable === true && !gpuPreferenceTouched.current);
        return shouldFollowDetection
          ? { ...selected, align: { ...selected.align, useGpu: parsed.gpuAvailable === true } }
          : selected;
      });
    }
    else if (!IS_TAURI_RUNTIME) setDoctor({ ...emptyDoctor(), summary: "Browser preview is not connected to the local runtime" });
    setDoctorLoading(false);
  }, []);

  const copyDoctorReport = useCallback(async () => {
    try {
      const report = await doctorReportText(doctor);
      if (IS_TAURI_RUNTIME) await writeClipboardText(report);
      else if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(report);
      else throw new Error("clipboard unavailable");
      setToast("Diagnostics copied; you can paste them into a bug report");
    } catch (error) {
      console.info("[SphereAlign] copy diagnostics", error);
      setToast("Unable to copy diagnostics; check clipboard permissions");
    }
  }, [doctor]);

  const initialColmapPath = useRef(colmapPath);
  useEffect(() => { void runDoctor(initialColmapPath.current); }, [runDoctor]);

  const openColmapPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("The COLMAP path is used by the local Windows runtime");
      return;
    }
    try {
      const result = await openDialog({
        directory: false,
        multiple: false,
        filters: [{ name: t`COLMAP launcher`, extensions: ["bat", "exe", "cmd"] }],
      });
      if (typeof result === "string") {
        setColmapPath(result);
        void runDoctor(result);
      }
    } catch (error) {
      console.info("[SphereAlign] COLMAP picker", error);
    }
  }, [runDoctor]);

  const openLutPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("Browser preview does not read local LUT files; paste a .cube path directly");
      return;
    }
    try {
      const result = await openDialog({
        directory: false,
        multiple: false,
        filters: [{ name: t({ message: "3D LUT (Cube)", context: "file picker filter", comment: "Filter label for a 3D lookup table file using the .cube format." }), extensions: ["cube"] }],
      });
      if (typeof result === "string") {
        setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, lutPath: result } }));
      }
    } catch (error) {
      console.info("[SphereAlign] LUT picker", error);
    }
  }, []);

  useEffect(() => {
    if (!IS_TAURI_RUNTIME) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;
    const register = async () => {
      try {
        const stop = await getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === "enter" || event.payload.type === "over") setDragOver(true);
          if (event.payload.type === "leave") setDragOver(false);
          if (event.payload.type === "drop") { setDragOver(false); applySourcePaths(event.payload.paths); }
        });
        if (disposed) stop(); else unlisten = stop;
      } catch (error) { console.info("[SphereAlign] drag-drop", error); }
    };
    void register();
    return () => { disposed = true; unlisten?.(); };
  }, [applySourcePaths]);

  const updateTaskStage = useCallback((taskId: string, stageKey: StageKey, patch: Partial<StageState>) => {
    setTasks((current) => current.map((task) => task.projectId === taskId ? { ...task, stages: { ...task.stages, [stageKey]: { ...task.stages[stageKey], ...patch } } } : task));
  }, []);

  const resolveTaskForJob = useCallback((jobId?: string, stageKey?: StageKey) => {
    if (jobId && jobTaskIds.current[jobId]) return jobTaskIds.current[jobId];
    if (jobId) {
      const active = Object.entries(activeJobIds.current).find(([, value]) => value === jobId)?.[0];
      if (active) return active;
      const staged = taskSnapshot.current.find((task) => STAGES.some(({ key }) => (!stageKey || key === stageKey) && task.stages[key].jobId === jobId));
      if (staged) return staged.projectId;
    }
    const pending = Object.entries(pendingStageStarts.current).filter(([, key]) => !stageKey || key === stageKey).map(([taskId]) => taskId);
    const auto = Object.entries(autoPipelineRuns.current).filter(([, run]) => run.stage && (!stageKey || run.stage === stageKey) && !run.jobId).map(([taskId]) => taskId);
    const candidates = Array.from(new Set([...pending, ...auto]));
    if (candidates.length === 1) return candidates[0];
    if (!stageKey) {
      const running = taskSnapshot.current.filter((task) => STAGES.some(({ key }) => task.stages[key].status === "running"));
      if (running.length === 1) return running[0].projectId;
    }
    return undefined;
  }, []);

  const applyProgressEvent = useCallback((taskId: string, payload: ProgressEventPayload) => {
    const stageKey = payload.stage;
    if (!stageKey) return;
    const eventTime = payload.timestampMs ?? Date.now();
    setTasks((current) => current.map((task) => {
      if (task.projectId !== taskId) return task;
      const previous = task.stages[stageKey];
      const replacingPendingJob = pendingStageStarts.current[taskId] === stageKey;
      if (payload.jobId && previous.jobId && payload.jobId !== previous.jobId && !replacingPendingJob) return task;
      const status: StageStatus = payload.done
        ? payload.status ?? "completed"
        : payload.status ?? "running";
      const terminal = status === "completed" || status === "cancelled" || status === "failed";
      const startedAtMs = status === "running"
        ? previous.status === "running" ? previous.startedAtMs : eventTime
        : previous.startedAtMs;
      const finishedAtMs = terminal ? eventTime : undefined;
      const durationMs = payload.elapsedMs
        ?? (terminal && startedAtMs !== undefined ? Math.max(0, eventTime - startedAtMs) : undefined);
      const progress = status === "completed"
        ? payload.progress ?? 100
        : terminal
          ? previous.progress
          : payload.progress ?? previous.progress;
      const nextStage: StageState = {
        ...previous,
        progress,
        status,
        message: payload.message || previous.message,
        jobId: payload.jobId || previous.jobId,
        phase: payload.phase || previous.phase,
        completed: payload.completed ?? previous.completed,
        total: payload.total ?? previous.total,
        currentItem: payload.currentItem ?? previous.currentItem,
        timestampMs: eventTime,
        startedAtMs,
        finishedAtMs,
        durationMs,
      };
      return {
        ...task,
        stages: { ...task.stages, [stageKey]: nextStage },
        logs: mergeProgressLog(task.logs, task.projectId, stageKey, previous, payload, status, eventTime),
      };
    }));
  }, []);

  const startAutoStage = useCallback(async (taskId: string, stageKey: StageKey) => {
    const run = autoPipelineRuns.current[taskId];
    if (!run || run.stage || run.jobId || activeJobIds.current[taskId]) return;
    run.stage = stageKey;
    pendingStageStarts.current[taskId] = stageKey;
    const result = await invokeSafely<{ jobId?: string }>("start_stage", {
      request: {
        projectPath: run.task.rootPath || run.task.outputPath,
        stage: stageKey,
        settings: run.task.settings,
        colmapPath: run.colmapPath || null,
      },
    });
    const currentRun = autoPipelineRuns.current[taskId];
    if (!result?.jobId) {
      if (currentRun === run) {
        delete autoPipelineRuns.current[taskId];
        delete pendingStageStarts.current[taskId];
        setToast("Unable to start the stage automatically; check the runtime message");
        queueMicrotask(() => pumpAutoPipelineRef.current());
      }
      return;
    }
    if (terminalJobIds.current.delete(result.jobId)) {
      delete pendingStageStarts.current[taskId];
      return;
    }
    if (currentRun !== run || run.stage !== stageKey) {
      // A user cancellation can race with the command response. Do not leave
      // a backend job running after its auto-pipeline session was stopped.
      delete pendingStageStarts.current[taskId];
      await invokeSafely("cancel_job", { jobId: result.jobId });
      return;
    }
    const progressAlreadyReceived = run.jobId === result.jobId;
    run.jobId = result.jobId;
    delete pendingStageStarts.current[taskId];
    bindJobToTask(taskId, result.jobId);
    const startedAtMs = progressAlreadyReceived ? undefined : Date.now();
    updateTaskStage(taskId, stageKey, progressAlreadyReceived
      ? { status: "running", jobId: result.jobId }
      : { status: "running", progress: 0, message: PREPARING_WORK_SOURCE, jobId: result.jobId, phase: "starting", startedAtMs, finishedAtMs: undefined, durationMs: undefined, completed: undefined, total: undefined, currentItem: undefined });
  }, [bindJobToTask, updateTaskStage]);

  const pumpAutoPipeline = useCallback(() => {
    if (Object.keys(activeJobIds.current).length || Object.keys(pendingStageStarts.current).length) return;
    if (Object.values(autoPipelineRuns.current).some((run) => run.stage || run.jobId)) return;
    const queued = Object.entries(autoPipelineRuns.current).find(([, run]) => !run.paused && !run.stage && !run.jobId);
    if (!queued) return;
    const [taskId, run] = queued;
    void startAutoStage(taskId, run.nextStage);
  }, [startAutoStage]);
  pumpAutoPipelineRef.current = pumpAutoPipeline;

  const startAutoPipeline = useCallback((task: Task) => {
    if (!IS_TAURI_RUNTIME || task.previewOnly) return;
    if (autoPipelineRuns.current[task.projectId] || activeJobIds.current[task.projectId]) return;
    const firstStage = STAGES.find(({ key }) => task.stages[key].status !== "completed");
    if (!firstStage) return;
    autoPipelineRuns.current[task.projectId] = {
      task: { rootPath: task.rootPath, outputPath: task.outputPath, settings: normalisePipelineSettings(task.settings) },
      colmapPath: colmapPath.trim(),
      nextStage: firstStage.key,
    };
    pumpAutoPipelineRef.current();
  }, [colmapPath]);

  const createTask = useCallback(async () => {
    if (!sourcePaths.length) { setToast("Select at least one OSV or dual-fisheye source first"); return; }
    if (customLutPathIsInvalid(settingsDraft.extract.lutPath)) {
      setToast("The custom LUT must be a .cube file");
      return;
    }
    const request = { inputPaths: sourcePaths, outputPath: outputDraft || undefined, name: nameDraft || undefined, settings: { ...settingsDraft } };
    const result = await invokeSafely("create_project", { request });
    const manifest = manifestFromUnknown(result);
    let createdTask: Task | null = null;
    if (manifest) {
      createdTask = manifest;
      const logPayload: LogEventPayload = { level: "info", message: `Created ${manifest.name}`, timestampMs: Date.now() };
      setTasks((current) => [{ ...manifest, logs: appendMessageLog(manifest.logs, manifest.projectId, logPayload) }, ...current]);
    } else if (!IS_TAURI_RUNTIME) {
      const preview: Task = { projectId: `preview-${Date.now()}`, name: nameDraft || BROWSER_PREVIEW_TASK_SOURCE, rootPath: outputDraft, inputPaths: sourcePaths, outputPath: outputDraft, settings: request.settings, stages: cloneStages({}), logs: [], warnings: [BROWSER_PREVIEW_NOT_CONNECTED_SOURCE], createdAt: new Date().toISOString(), previewOnly: true };
      createdTask = preview;
      const logPayload: LogEventPayload = { level: "info", message: `Preview task added: ${preview.name}`, timestampMs: Date.now() };
      setTasks((current) => [{ ...preview, logs: appendMessageLog(preview.logs, preview.projectId, logPayload) }, ...current]);
    } else {
      setToast("Failed to create the task; check the runtime message");
      return;
    }
    setTaskDialogOpen(false);
    setSourcePaths([]);
    setSourceInspection("");
    setSourceColorInspection(null);
    if (createdTask) startAutoPipeline(createdTask);
  }, [nameDraft, outputDraft, settingsDraft, sourcePaths, startAutoPipeline]);

  const saveEditedTask = useCallback(async () => {
    const task = taskSnapshot.current.find((item) => item.projectId === editingTaskId);
    if (!task || !canChangeQueuedTask(task)) {
      setToast("This task has started and can no longer be edited");
      return;
    }
    if (!sourcePaths.length) { setToast("Keep at least one source"); return; }
    if (customLutPathIsInvalid(settingsDraft.extract.lutPath)) {
      setToast("The custom LUT must be a .cube file");
      return;
    }
    const settings = normalisePipelineSettings(settingsDraft);
    if (task.previewOnly) {
      setTasks((current) => current.map((item) => item.projectId === task.projectId
        ? { ...item, name: nameDraft || item.name, inputPaths: sourcePaths, settings }
        : item));
      setTaskDialogOpen(false);
      setEditingTaskId(null);
      setToast("Preview task updated");
      return;
    }
    let result: unknown;
    try {
      result = await invoke("update_queued_project", {
        request: { projectPath: task.rootPath || task.outputPath, name: nameDraft || task.name, inputPaths: sourcePaths, settings },
      });
    } catch (error) {
      console.error("[SphereAlign] update_queued_project", error);
      setToast(backendErrorMessage(error));
      return;
    }
    const manifest = manifestFromUnknown(result);
    if (!manifest) { setToast("Failed to save task changes; check the runtime message"); return; }
    setTasks((current) => current.map((item) => item.projectId === task.projectId ? { ...manifest, logs: item.logs } : item));
    const run = autoPipelineRuns.current[task.projectId];
    if (run) {
      run.task = { rootPath: manifest.rootPath, outputPath: manifest.outputPath, settings: manifest.settings };
      run.paused = false;
    }
    setTaskDialogOpen(false);
    setEditingTaskId(null);
    setToast("Queued task updated");
    queueMicrotask(() => pumpAutoPipelineRef.current());
  }, [canChangeQueuedTask, editingTaskId, nameDraft, settingsDraft, sourcePaths]);

  const deleteQueuedTask = useCallback(() => {
    const task = taskSnapshot.current.find((item) => item.projectId === deletingTaskId);
    if (!task || !canChangeQueuedTask(task)) {
      setDeletingTaskId(null);
      setToast("This task has started and cannot be removed");
      return;
    }
    delete autoPipelineRuns.current[task.projectId];
    delete pendingStageStarts.current[task.projectId];
    delete activeJobIds.current[task.projectId];
    setTasks((current) => current.filter((item) => item.projectId !== task.projectId));
    if (selectedTaskId === task.projectId) {
      setTaskDetailOpen(false);
    }
    setDeletingTaskId(null);
    setToast("Task removed from the queue; the output folder is kept");
    queueMicrotask(() => pumpAutoPipelineRef.current());
  }, [canChangeQueuedTask, deletingTaskId, selectedTaskId]);

  const enqueueQueuedTask = useCallback((task: Task) => {
    if (!taskHasNotStarted(task)) {
      setToast("This task has already started");
      return;
    }
    startAutoPipeline(task);
    setToast("Task added to the run queue");
  }, [startAutoPipeline]);

  const startStage = useCallback(async (task: Task, stageKey: StageKey, mode: "start" | "resume" | "retry") => {
    if (!IS_TAURI_RUNTIME) { setToast("Browser preview does not run backend work"); return; }
    if (activeJobIds.current[task.projectId] || autoPipelineRuns.current[task.projectId]) {
      setToast("A processing stage is already running for this task; please wait");
      return;
    }
    pendingStageStarts.current[task.projectId] = stageKey;
    const result = await invokeSafely<{ jobId?: string }>("start_stage", { request: { projectPath: task.rootPath || task.outputPath, stage: stageKey, mode, settings: normalisePipelineSettings(task.settings || settingsDraft), colmapPath: colmapPath.trim() || null } });
    if (result?.jobId) {
      delete pendingStageStarts.current[task.projectId];
      if (terminalJobIds.current.delete(result.jobId)) return;
      bindJobToTask(task.projectId, result.jobId);
      const receivedEarlyProgress = jobTaskIds.current[result.jobId] === task.projectId
        || taskSnapshot.current.some((currentTask) => currentTask.projectId === task.projectId && currentTask.stages[stageKey].jobId === result.jobId);
      updateTaskStage(task.projectId, stageKey, receivedEarlyProgress
        ? { status: "running", jobId: result.jobId }
        : { status: "running", progress: task.stages[stageKey].progress, message: PREPARING_WORK_SOURCE, phase: "starting", jobId: result.jobId, startedAtMs: Date.now(), finishedAtMs: undefined, durationMs: undefined, completed: undefined, total: undefined, currentItem: undefined });
    } else {
      delete pendingStageStarts.current[task.projectId];
      setToast("Unable to start the stage; check the runtime message");
      queueMicrotask(() => pumpAutoPipelineRef.current());
    }
  }, [bindJobToTask, colmapPath, settingsDraft, updateTaskStage]);

  const cancelStage = useCallback(async (task: Task, stageKey: StageKey) => {
    if (!IS_TAURI_RUNTIME) { setToast("Browser preview does not cancel backend work"); return; }
    const autoRun = autoPipelineRuns.current[task.projectId];
    if (autoRun?.stage === stageKey) delete autoPipelineRuns.current[task.projectId];
    delete pendingStageStarts.current[task.projectId];
    const jobId = task.stages[stageKey].jobId || activeJobIds.current[task.projectId];
    if (!jobId) {
      queueMicrotask(() => pumpAutoPipelineRef.current());
      return;
    }
    const cancelled = await invokeSafely<boolean>("cancel_job", { jobId });
    if (cancelled === true) {
      ignoredJobIds.current.add(jobId);
      delete jobTaskIds.current[jobId];
      if (activeJobIds.current[task.projectId] === jobId) delete activeJobIds.current[task.projectId];
      const finishedAtMs = Date.now();
      updateTaskStage(task.projectId, stageKey, { status: "cancelled", message: "Cancelled; you can resume later", jobId: undefined, finishedAtMs, durationMs: taskStageDuration(task.stages[stageKey], finishedAtMs) });
    }
    queueMicrotask(() => pumpAutoPipelineRef.current());
  }, [updateTaskStage]);

  useEffect(() => {
    let disposed = false;
    let unlistenProgress: (() => void) | undefined;
    let unlistenLog: (() => void) | undefined;
    const register = async () => {
      const [progressStop, logStop] = await Promise.all([
        listen<unknown>("pipeline-progress", (event) => {
          const payload = readProgress(event.payload);
          const stageKey = payload.stage;
          if (!stageKey) return;
          if (payload.jobId && (ignoredJobIds.current.has(payload.jobId) || terminalJobIds.current.has(payload.jobId))) return;
          if (
            payload.jobId
            && (payload.done || payload.status === "completed" || payload.status === "failed" || payload.status === "cancelled")
          ) {
            terminalJobIds.current.add(payload.jobId);
          }
          const targetProjectId = resolveTaskForJob(payload.jobId, stageKey);
          if (targetProjectId && payload.jobId) {
            bindJobToTask(targetProjectId, payload.jobId);
            const pendingRun = autoPipelineRuns.current[targetProjectId];
            if (pendingRun && pendingRun.stage === stageKey && !pendingRun.jobId) pendingRun.jobId = payload.jobId;
          }
          if (!targetProjectId) return;
          applyProgressEvent(targetProjectId, payload);
          if (payload.status === "failed" || payload.status === "cancelled") {
            delete autoPipelineRuns.current[targetProjectId];
            delete pendingStageStarts.current[targetProjectId];
            if (payload.jobId && activeJobIds.current[targetProjectId] === payload.jobId) delete activeJobIds.current[targetProjectId];
            queueMicrotask(() => pumpAutoPipelineRef.current());
          }
          if (!payload.done) return;
          delete pendingStageStarts.current[targetProjectId];
          if (payload.jobId && activeJobIds.current[targetProjectId] === payload.jobId) delete activeJobIds.current[targetProjectId];
          const run = autoPipelineRuns.current[targetProjectId];
          if (!run || run.stage !== stageKey || (run.jobId && run.jobId !== payload.jobId)) {
            queueMicrotask(() => pumpAutoPipelineRef.current());
            return;
          }
          run.jobId = undefined;
          if ((payload.status ?? "completed") !== "completed") {
            delete autoPipelineRuns.current[targetProjectId];
            queueMicrotask(() => pumpAutoPipelineRef.current());
            return;
          }
          const currentIndex = STAGES.findIndex(({ key }) => key === stageKey);
          const nextStage = currentIndex >= 0 ? STAGES[currentIndex + 1] : undefined;
          if (!nextStage) {
            delete autoPipelineRuns.current[targetProjectId];
            addTaskMessage(targetProjectId, "Automatic pipeline completed frame extraction, masking, and alignment");
            queueMicrotask(() => pumpAutoPipelineRef.current());
            return;
          }
          run.nextStage = nextStage.key;
          run.stage = undefined;
          queueMicrotask(() => pumpAutoPipelineRef.current());
        }),
        listen<unknown>("pipeline-log", (event) => {
          const payload = readLogEvent(event.payload);
          const targetProjectId = resolveTaskForJob(payload.jobId, payload.stage);
          if (!targetProjectId) {
            if (payload.jobId) {
              const pending = pendingLogsByJobId.current[payload.jobId] ?? [];
              pendingLogsByJobId.current[payload.jobId] = appendMessageLog(pending, payload.jobId, payload, payload.stage, payload.phase);
            }
            return;
          }
          if (payload.jobId) bindJobToTask(targetProjectId, payload.jobId);
          const targetTask = taskSnapshot.current.find((task) => task.projectId === targetProjectId);
          const inferredStage = payload.stage || STAGES.find(({ key }) => targetTask?.stages[key].jobId === payload.jobId)?.key || STAGES.find(({ key }) => targetTask?.stages[key].status === "running")?.key;
          const inferredPhase = payload.phase || (inferredStage ? targetTask?.stages[inferredStage].phase : undefined);
          appendTaskLog(targetProjectId, payload, inferredStage, inferredPhase);
        }),
      ]);
      if (disposed) { progressStop(); logStop(); } else { unlistenProgress = progressStop; unlistenLog = logStop; }
    };
    if (IS_TAURI_RUNTIME) void register();
    return () => { disposed = true; unlistenProgress?.(); unlistenLog?.(); };
  }, [appendTaskLog, applyProgressEvent, bindJobToTask, resolveTaskForJob, startAutoStage, addTaskMessage]);

  const handleStageAction = useCallback((task: Task, stageKey: StageKey) => {
    const status = task.stages[stageKey].status;
    if (status === "running") {
      void cancelStage(task, stageKey);
      return;
    }
    const prerequisiteKey = stagePrerequisiteKey(task, stageKey);
    if (prerequisiteKey) {
      // Keep the state value locale-independent; localiseUserMessage resolves
      // this small internal key when the toast is rendered.
      setToast(`Complete prerequisite: ${prerequisiteKey}`);
      return;
    }
    if (Object.keys(activeJobIds.current).length || Object.keys(pendingStageStarts.current).length) {
      setToast("A processing stage is already running; please wait");
      return;
    }
    void startStage(task, stageKey, status === "cancelled" ? "resume" : status === "failed" || status === "completed" ? "retry" : "start");
  }, [cancelStage, startStage]);

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    const files = Array.from(event.dataTransfer.files ?? []);
    const paths = files.map((file) => { const candidate = file as File & { path?: string }; return candidate.path || file.name; });
    const text = event.dataTransfer.getData("text/plain");
    if (!paths.length && text) paths.push(text);
    applySourcePaths(paths);
  };

  const renderSettingsFields = () => {
    const candidateMultiplier = candidateMultiplierFor(settingsDraft.extract);
    const candidateFps = settingsDraft.extract.baseFps * candidateMultiplier;
    const colorMode = settingsDraft.extract.colorMode;
    const detectedDlog = sourceColorInspection?.shouldApply === true
      || sourceColorInspection?.files.some((file) => file.shouldApply === true) === true;
    const restoreDlog = colorMode === "dlogMRec709" || (colorMode === "auto" && detectedDlog);
    const lutPath = settingsDraft.extract.lutPath?.trim() ?? "";
    const lutPathInvalid = customLutPathIsInvalid(lutPath);
    return (
      <section className="min-h-0 overflow-hidden border-l max-[760px]:overflow-visible max-[760px]:border-t max-[760px]:border-l-0" aria-labelledby="task-processing-settings-title">
        <div className="scroll-fade-y scroll-fade-8 h-full overflow-y-auto overscroll-contain px-7 py-6 [scrollbar-gutter:stable] max-[920px]:px-5 max-[760px]:h-auto max-[760px]:overflow-visible max-[760px]:p-5 max-[760px]:[--scroll-fade-mask:none]">
        <h2 id="task-processing-settings-title" className="mb-4 text-lg font-semibold text-foreground"><Trans>Processing settings</Trans></h2>
        <FieldGroup className="gap-4 [&>[data-slot=field]]:rounded-lg [&>[data-slot=field]]:border [&>[data-slot=field]]:bg-card [&>[data-slot=field]]:p-3">
          <Field>
            <FieldLabel><Trans context="settings section" comment="Pipeline stage settings for extracting frames.">Frame extraction</Trans></FieldLabel>
            <FieldContent>
              <Field className="min-h-7 border-0 bg-transparent px-0 py-0.5">
                <FieldLabel htmlFor="base-fps"><Trans comment="Base frames-per-second setting for source media extraction.">Base frame rate (FPS)</Trans></FieldLabel>
                <Input
                  id="base-fps"
                  type="number"
                  min={1}
                  max={30}
                  step={1}
                  value={settingsDraft.extract.baseFps}
                  onChange={(event) => {
                    const baseFps = Math.min(30, Math.max(1, Number(event.currentTarget.value) || 1));
                    setSettingsDraft((current) => {
                      const multiplier = candidateMultiplierFor(current.extract);
                      return { ...current, extract: { ...current.extract, baseFps, denseFps: baseFps * multiplier } };
                    });
                  }}
                />
              </Field>
              <Field orientation="horizontal" className="mt-2.5 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="sharpness-filter"
                  checked={settingsDraft.extract.skipBlurry}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, skipBlurry: checked === true } }))}
                />
                <FieldLabel htmlFor="sharpness-filter"><Trans context="frame extraction setting" comment="Whether blurry candidate frames should be filtered out.">Sharpness filtering</Trans></FieldLabel>
              </Field>
              {settingsDraft.extract.skipBlurry && (
                <Field className="mt-2 min-h-7 border-0 bg-transparent px-0 py-0.5">
                  <div className="flex items-center justify-between">
                    <FieldLabel id="candidate-fps-label"><Trans comment="Frame rate used to sample candidate frames before selecting the sharpest ones.">Candidate frame rate</Trans></FieldLabel>
                    <span className="ml-auto font-mono text-sm text-muted-foreground">{candidateMultiplier}× · {candidateFps} FPS</span>
                  </div>
                  <Slider
                    aria-labelledby="candidate-fps-label"
                    min={MIN_CANDIDATE_MULTIPLIER}
                    max={MAX_CANDIDATE_MULTIPLIER}
                    step={1}
                    value={[candidateMultiplier]}
                    onValueChange={(value) => {
                      const multiplier = Array.isArray(value) ? value[0] : value;
                      if (multiplier === undefined) return;
                      setSettingsDraft((current) => ({
                        ...current,
                        extract: { ...current.extract, denseFps: current.extract.baseFps * multiplier },
                      }));
                    }}
                  />
                  <div className="flex items-center justify-between font-mono text-sm text-muted-foreground" aria-hidden="true"><span>2×</span><span>10×</span></div>
                  <FieldDescription><Trans comment="Candidate frames are sampled at a multiple of the base frame rate, then sharper frames are selected.">Sample candidates at a multiple of the base frame rate, then select sharper frames.</Trans></FieldDescription>
                </Field>
              )}
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel><Trans context="settings section" comment="Lookup table settings for restoring source media color.">LUT settings</Trans></FieldLabel>
            <FieldContent>
              <Field orientation="horizontal" className="min-h-7 gap-2 border-0 bg-transparent px-0 py-0.5">
                <Switch
                  id="extract-color-mode"
                  checked={restoreDlog}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({
                    ...current,
                    extract: { ...current.extract, colorMode: checked ? "dlogMRec709" : "native" },
                  }))}
                />
                <div>
                  <FieldLabel htmlFor="extract-color-mode"><Trans comment="Apply the official or custom lookup table to restore DJI D-Log M color.">Apply the D-Log M restoration LUT</Trans></FieldLabel>
                  {colorMode === "auto" && detectedDlog && (
                    <FieldDescription><Trans comment="The source filename suffix or media metadata indicated D-Log M color.">Detected from the _D filename suffix or media metadata and enabled.</Trans></FieldDescription>
                  )}
                </div>
              </Field>
              <Field className="min-h-7 gap-2 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=alert]]:mt-0.5" data-invalid={lutPathInvalid || undefined}>
                <FieldLabel htmlFor="extract-lut-path"><Trans comment="Optional user-provided 3D LUT file path.">Custom LUT (optional)</Trans></FieldLabel>
                <div className="flex items-center gap-2 [&_input]:flex-1">
                  <Input
                    id="extract-lut-path"
                    value={lutPath}
                    placeholder={t`Leave blank to use the official D-Log M → Rec.709 LUT`}
                    aria-invalid={lutPathInvalid || undefined}
                    onChange={(event) => setSettingsDraft((current) => ({
                      ...current,
                      extract: { ...current.extract, lutPath: event.currentTarget.value || undefined },
                    }))}
                  />
                  <Button type="button" variant="outline" size="sm" onClick={() => void openLutPicker()}>{t`Choose .cube`}</Button>
                  {lutPath && <Button type="button" variant="ghost" size="icon-xs" aria-label={t`Clear custom LUT`} onClick={() => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, lutPath: undefined } }))}><X /></Button>}
                </div>
                {lutPathInvalid
                  ? <Alert variant="destructive"><AlertTriangle /><AlertTitle>{t`Invalid LUT file format`}</AlertTitle><AlertDescription>{t`Choose a 3D LUT with the .cube extension.`}</AlertDescription></Alert>
                  : colorMode === "dlogMRec709"
                    ? <FieldDescription>{t`When no custom file is specified, the runtime uses the official LUT; otherwise it uses this .cube file.`}</FieldDescription>
                    : <FieldDescription>{t`Specify a .cube file only when you need to override the official LUT.`}</FieldDescription>}
              </Field>
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel><Trans context="settings section" comment="Pipeline stage settings for creating masks.">Masking</Trans></FieldLabel>
            <FieldContent>
              <Field orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="mask-yolo"
                  checked={settingsDraft.mask.yoloEnabled}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({
                    ...current,
                    mask: {
                      ...current.mask,
                      yoloEnabled: checked === true,
                      classes: checked === true && current.mask.classes.length === 0 ? [...MASK_CLASSES] : current.mask.classes,
                    },
                  }))}
                />
                <FieldLabel htmlFor="mask-yolo"><Trans comment="Enable object detection masks from YOLO11.">YOLO object filtering</Trans></FieldLabel>
              </Field>
              {settingsDraft.mask.yoloEnabled && (
                <FieldGroup className="gap-2 px-0 pt-1 pb-2">
                  <FieldSet className="gap-2">
                    <FieldLegend variant="label"><Trans comment="Select one or more object classes to exclude from reconstruction.">Objects to mask (multiple selection)</Trans></FieldLegend>
                    <FieldGroup data-slot="checkbox-group" className="grid grid-cols-2 gap-x-3 gap-y-2">
                      {MASK_CLASSES.map((maskClass) => {
                        const checkboxId = `mask-class-${maskClass}`;
                        return (
                          <Field key={maskClass} orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                            <Checkbox
                              id={checkboxId}
                              checked={settingsDraft.mask.classes.includes(maskClass)}
                              onCheckedChange={(checked) => setSettingsDraft((current) => {
                                const classes = checked === true
                                  ? Array.from(new Set([...current.mask.classes, maskClass]))
                                  : current.mask.classes.filter((value) => value !== maskClass);
                                return { ...current, mask: { ...current.mask, classes, yoloEnabled: classes.length > 0 } };
                              })}
                            />
                            <FieldLabel htmlFor={checkboxId}>{translate(MASK_CLASS_LABELS[maskClass])}</FieldLabel>
                          </Field>
                        );
                      })}
                    </FieldGroup>
                  </FieldSet>
                </FieldGroup>
              )}
              <Field orientation="horizontal" className="mt-2 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal">
                <Checkbox
                  id="mask-sky"
                  checked={settingsDraft.mask.maskSky}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, maskSky: checked === true } }))}
                />
                <FieldLabel htmlFor="mask-sky"><Trans comment="Enable SkySeg sky masks.">Sky filtering</Trans></FieldLabel>
              </Field>
              {settingsDraft.mask.maskSky && <FieldDescription>{t`Use SkySeg to generate sky masks.`}</FieldDescription>}
              {!settingsDraft.mask.yoloEnabled && !settingsDraft.mask.maskSky && (
                <FieldDescription>{t`Masking is disabled; alignment starts after frame extraction.`}</FieldDescription>
              )}
            </FieldContent>
          </Field>
        <Field>
          <FieldLabel><Trans context="settings section" comment="Pipeline stage settings for aligning source images and camera rigs.">Alignment</Trans></FieldLabel>
          <FieldContent>
            <div className="flex flex-col gap-2">
              <Field orientation="horizontal" className="mt-2.5 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal" data-disabled={doctor.gpuAvailable === false || undefined}>
                <Switch
                  id="use-gpu"
                  size="sm"
                  disabled={doctor.gpuAvailable === false}
                  checked={settingsDraft.align.useGpu}
                  onCheckedChange={(checked) => {
                    gpuPreferenceTouched.current = true;
                    setSettingsDraft((current) => ({ ...current, align: { ...current.align, useGpu: checked } }));
                  }}
                />
                <FieldContent>
                  <FieldLabel htmlFor="use-gpu"><Trans comment="Use CUDA acceleration for the COLMAP alignment stage.">Use CUDA acceleration for alignment</Trans></FieldLabel>
                  <FieldDescription>{doctor.gpuAvailable === false ? t`No usable COLMAP CUDA acceleration was detected, so the CPU will be used.` : t`Enabled by default when a CUDA-capable NVIDIA GPU is detected; falls back to the CPU if execution fails.`}</FieldDescription>
                </FieldContent>
              </Field>
              {doctor.gpuAvailable === true && doctor.gpuDevices.length > 1 && (
                <Field data-disabled={!settingsDraft.align.useGpu || undefined}>
                  <FieldLabel htmlFor="gpu-index"><Trans comment="Select which detected GPU should run alignment.">Select GPU</Trans></FieldLabel>
                  <Select
                    items={doctor.gpuDevices.map((device) => ({ value: String(device.index), label: gpuDeviceLabel(device, doctor.gpuDevices) }))}
                    value={settingsDraft.align.gpuIndex}
                    onValueChange={(gpuIndex) => setSettingsDraft((current) => ({ ...current, align: { ...current.align, gpuIndex: gpuIndex ?? String(doctor.gpuDevices[0].index) } }))}
                    disabled={!settingsDraft.align.useGpu}
                  >
                    <SelectTrigger id="gpu-index" className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {doctor.gpuDevices.map((device) => <SelectItem key={device.index} value={String(device.index)}>{gpuDeviceLabel(device, doctor.gpuDevices)}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </Field>
              )}
            </div>
          </FieldContent>
        </Field>
        </FieldGroup>
        </div>
      </section>
    );
  };

  return (
    <div className="flex size-full flex-col bg-background">
      <main className="flex min-h-0 flex-1 flex-col overflow-auto">
        <input ref={fileInputRef} type="file" multiple accept=".osv,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mts,.m2ts,.ts" hidden onChange={(event) => handleBrowserFiles(event.currentTarget.files)} />
        <header className={cn(
          "sticky top-0 z-40 flex min-h-13 shrink-0 items-center border-b bg-background/95 px-4 py-2 backdrop-blur-sm select-none max-[760px]:px-3.5 max-[760px]:py-3",
          IS_MACOS_RUNTIME && "pl-[86px]",
          IS_WINDOWS_RUNTIME && "pr-0 py-0",
        )}>
          <div className="flex min-h-9 w-full shrink-0 items-center self-stretch" data-tauri-drag-region={IS_TAURI_RUNTIME ? "" : undefined}>
            <h1 className="shrink-0 text-base font-semibold tracking-tight text-foreground">SphereAlign</h1>
            <div className="ml-auto flex items-center gap-2">
              <Button size="sm" onClick={openNewTaskDialog}><Plus data-icon="inline-start" /><Trans context="task action" comment="Create a new reconstruction task.">New reconstruction task</Trans></Button>
              <Button size="sm" variant="outline" onClick={() => void openProject()}><FolderOpen data-icon="inline-start" /><Trans context="project action" comment="Open an existing resumable project.">Open project</Trans></Button>
              <div className="mx-1 flex h-5">
                <Separator orientation="vertical" />
              </div>
              <Button size="sm" variant="outline" onClick={() => setSettingsOpen(true)}><Settings2 data-icon="inline-start" /><Trans>Settings</Trans></Button>
            </div>
            <WindowsWindowControls />
          </div>
        </header>
        {tasks.length === 0 ? (
          <section className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 pt-14 pb-18 text-center" onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
            <div className={cn("mb-5.5 grid size-17.5 place-items-center rounded-[18px] border bg-card text-muted-foreground transition-[border-color,color,transform,background] duration-200 [&_svg]:size-7", dragOver && "scale-[1.03] border-primary/50 bg-primary/10 text-primary")} aria-hidden="true"><FileStack /></div>
            <h2 className="text-[28px] font-semibold tracking-[-0.045em] text-foreground"><Trans context="empty state" comment="Empty task list heading.">No tasks yet</Trans></h2>
            <p className="mt-3 text-base leading-relaxed text-muted-foreground"><Trans comment="Drop source media here or choose files below. Existing projects are opened separately.">Drop OSV or dual-fisheye media here,<br />or use Open project to resume an existing project.</Trans></p>
            <div className="mt-6 flex items-center gap-2.5 max-[760px]:w-[min(280px,100%)] max-[760px]:flex-col max-[760px]:items-stretch"><Button size="lg" onClick={() => void openSourcePicker()}><Upload data-icon="inline-start" /><Trans context="file picker action" comment="Button opens a file picker for source media.">Choose files</Trans></Button><Button size="lg" variant="outline" onClick={() => void openProject()}><FolderOpen data-icon="inline-start" /><Trans context="project action" comment="Open an existing resumable project.">Open project</Trans></Button></div>
            <section className="mt-7 w-[min(520px,100%)] text-left max-[760px]:w-[min(320px,100%)]" aria-labelledby="supported-formats-title">
              <h2 id="supported-formats-title" className="mb-2.5 text-xs font-semibold tracking-wide text-muted-foreground"><Trans>Supported inputs</Trans></h2>
              <div className="grid grid-cols-2 gap-2.5 max-[760px]:grid-cols-1">
                <SupportedFormatCard icon={Film} title={<Trans comment="DJI Osmo 360 source media files.">Osmo 360 source files</Trans>} detail="OSV" />
                <SupportedFormatCard icon={Folder} title={<Trans comment="A project folder containing a resumable reconstruction.">Project folder</Trans>} detail={<Trans>Resume an unfinished reconstruction task</Trans>} />
              </div>
            </section>
          </section>
        ) : (
          <m.div
            className="mr-auto w-full"
            initial={false}
            animate={{ width: taskDetailOpen && selectedTask && taskDetailUsesSplitView ? "calc(100% - 460px)" : "100%" }}
            transition={shouldReduceMotion ? { duration: 0 } : TASK_DETAIL_DRAWER_TRANSITION}
          >
          <section className="mx-auto w-full max-w-[1440px] px-8 pt-6.5 pb-14 max-[760px]:px-3.5 max-[760px]:pt-5.5 max-[760px]:pb-11.5">
            <div className="grid gap-7">
              {([
                { key: "active", title: t`In progress`, items: activeTasks },
                { key: "completed", title: t`Completed`, items: completedTasks },
              ] as const).filter((group) => group.key === "completed" || group.items.length > 0).map((group) => (
                <section key={group.key}>
                  <div className="mb-3 flex items-center px-0.5"><div className="flex items-center gap-2.5"><h2 className="text-lg font-semibold">{group.title}</h2><Badge variant="secondary">{group.items.length}</Badge></div></div>
                  {group.items.length > 0 ? <div className="flex flex-col gap-3.5 overflow-visible">
              {group.items.map((task) => {
                const overall = taskProgress(task);
                const queued = taskHasNotStarted(task);
                const editableQueued = queued && canChangeQueuedTask(task);
                const waitingForEnqueue = queued && !task.previewOnly && !autoPipelineRuns.current[task.projectId];
                const currentStageDefinition = taskCurrentStage(task);
                const currentStage = task.stages[currentStageDefinition.key];
                const currentElapsed = taskStageDuration(currentStage, clockMs);
                const currentEta = estimatedRemainingMs(currentStage, clockMs);
                const currentCount = logCountLabel(currentStage.completed, currentStage.total);
                const primarySource = task.inputPaths.length > 0 ? sourceFromPath(task.inputPaths[0], 0) : undefined;
                return (
                  <article className={cn("rounded-xl border bg-card px-7.5 pt-6.5 pb-7.5 shadow-sm max-[760px]:px-4.5 max-[760px]:pt-5 max-[760px]:pb-6", queued && "bg-primary/[0.02]")} key={task.projectId}>
                    <div className="flex items-center justify-between gap-4.5">
                      <div className="flex min-w-0 items-center gap-3">{primarySource ? <SourceThumbnail source={primarySource} /> : <span className="grid size-12 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary [&_svg]:size-5.5"><FileStack /></span>}<div className="min-w-0"><div className="flex items-center gap-2"><h2 className="truncate text-[17px] font-semibold text-foreground">{task.name}</h2>{queued && <Badge variant="outline">{editableQueued ? t`Waiting to run` : t`Preparing`}</Badge>}{task.previewOnly && <Badge variant="outline">{t`Preview`}</Badge>}</div><p className="mt-1 max-w-160 truncate font-mono text-sm text-muted-foreground" title={task.outputPath}>{task.outputPath || t`Output not specified`}</p></div></div>
                      <div className="flex shrink-0 items-center gap-1">
                        {waitingForEnqueue && <Button size="sm" onClick={() => enqueueQueuedTask(task)}><Play data-icon="inline-start" /><Trans context="queue action" comment="Add a queued task to the automatic execution queue.">Add to queue</Trans></Button>}
                        {editableQueued && <><Button variant="outline" size="sm" onClick={() => openEditTaskDialog(task)}><Pencil data-icon="inline-start" /><Trans context="task action" comment="Edit a task that has not started.">Edit</Trans></Button><Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => setDeletingTaskId(task.projectId)}><Trash2 data-icon="inline-start" /><Trans context="task action" comment="Remove a queued task without deleting its output folder.">Remove</Trans></Button></>}
                        <Button variant="ghost" size="icon-sm" aria-label={t`View details for ${task.name}`} aria-haspopup="dialog" aria-expanded={taskDetailOpen && selectedTaskId === task.projectId} onClick={(event) => { taskDetailTriggerRef.current = event.currentTarget; setTaskDetailTab("summary"); setSelectedTaskId(task.projectId); setTaskDetailOpen(true); }}><Info /></Button>
                      </div>
                    </div>
                    {queued ? <div className="mt-4 ml-15 flex items-center justify-between text-sm text-muted-foreground max-[760px]:ml-0"><span><Trans comment="Queued tasks run automatically in creation order.">The queue runs automatically in creation order</Trans></span><small><Plural value={task.inputPaths.length} one="# source" other="# sources" /></small></div> : <><div className="mt-6.5 mb-4 ml-15 max-[760px]:ml-0 [&_[data-slot=progress-track]]:h-1.75 [&_[data-slot=progress-value]]:hidden">
                      <div className="mb-2 flex items-center gap-2 text-sm"><span className="font-medium text-foreground" title={t`Weighted by three observed run durations: frame extraction 22%, masking 4%, alignment 74%`}><Trans comment="Overall progress weighted by observed stage durations.">Overall progress</Trans></span><small className="text-muted-foreground">{taskProgressSummary(task)}</small><strong className="ml-auto font-mono font-medium text-foreground">{overall}%</strong></div>
                      <Progress value={overall} aria-label={t`${task.name} overall time progress`}><ProgressValue /></Progress>
                      <div className="mt-3.5 grid grid-cols-[minmax(180px,1fr)_minmax(330px,auto)] items-end gap-4.5 max-[920px]:grid-cols-1 max-[920px]:items-start">
                        <span className="flex min-w-0 flex-col gap-1"><strong className="text-sm font-medium text-foreground">{t`Current stage: ${stageLabel(currentStageDefinition)}`}</strong><small className="truncate text-sm text-muted-foreground">{currentStage.phase ? phaseLabel(currentStage.phase) : stageStatusLabel(currentStage.status)}</small></span>
                        <dl className="grid grid-cols-3 gap-5 max-[760px]:grid-cols-1 max-[760px]:gap-2.5">
                          <div className="flex flex-col gap-1"><dt className="text-sm text-muted-foreground"><Trans>Processed</Trans></dt><dd className="text-sm tabular-nums text-foreground">{currentCount || t`Not reported yet`}</dd></div>
                          <div className="flex flex-col gap-1"><dt className="text-sm text-muted-foreground"><Trans>Elapsed</Trans></dt><dd className="text-sm tabular-nums text-foreground">{currentElapsed !== undefined ? formatDuration(currentElapsed) : t`Not started`}</dd></div>
                          <div className="flex flex-col gap-1"><dt className="text-sm text-muted-foreground"><Trans>Estimated remaining</Trans></dt><dd className="text-sm tabular-nums text-foreground">{currentStage.status === "running" ? formatEta(currentEta) : "—"}</dd></div>
                        </dl>
                      </div>
                    </div>
                    <div className="ml-15 grid grid-cols-1 border-t max-[760px]:ml-0" role="list" aria-label={t`Reconstruction pipeline`}>
                      {STAGES.map((stage, stageIndex) => {
                        const current = task.stages[stage.key];
                        const stageProgress = Math.round(current.progress);
                        const Icon = stage.icon;
                        const action = stageActionState(task, stage.key, hasRunningStage);
                        return (
                          <div className="relative flex min-w-0 items-center gap-2 border-t py-4 first:border-t-0 max-[760px]:flex-wrap max-[760px]:items-start max-[760px]:py-3" data-status={current.status} key={stage.key} role="listitem" aria-label={t`Stage ${stageIndex + 1} of ${STAGES.length}: ${stageLabel(stage)}`}>
                            <span className="relative grid w-6 shrink-0 items-start justify-items-center self-stretch" aria-hidden="true"><span className={cn("relative z-10 grid size-5 place-items-center rounded-full border bg-card text-[0.72rem] font-semibold text-muted-foreground [&_svg]:size-3", current.status === "running" && "border-primary text-primary ring-3 ring-primary/10", current.status === "completed" && "border-emerald-600 bg-emerald-500/10 text-emerald-600")}>{current.status === "completed" ? <CheckCircle2 /> : stageIndex + 1}</span>{stageIndex < STAGES.length - 1 && <span className="absolute top-5 -bottom-3 left-1/2 w-px -translate-x-1/2 bg-border" />}</span>
                            <div className="flex min-w-0 flex-1 items-center gap-2">
                              <Icon className={cn("size-4 shrink-0 text-muted-foreground", current.status === "running" && "text-primary")} />
                              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                                <strong className="text-sm font-medium text-foreground">{stageLabel(stage)}</strong>
                                <small className="truncate text-sm text-muted-foreground">{action.prerequisite ? t`Waiting for ${action.prerequisite} to finish` : current.message ? localiseUserMessage(current.message) : stageDescription(stage)}</small>
                                {current.status === "running" && (
                                  <div className={cn("mt-1.5 flex w-[min(360px,100%)] items-center gap-2 [&_[data-slot=progress]]:min-w-20 [&_[data-slot=progress]]:flex-1 [&_[data-slot=progress-track]]:h-0.75 [&_[data-slot=progress-value]]:hidden [&>span]:w-7.5 [&>span]:text-right [&>span]:font-mono [&>span]:text-sm [&>span]:text-muted-foreground", stageProgress <= 0 && "[&_[data-slot=progress-indicator]]:!w-[30%] [&_[data-slot=progress-indicator]]:animate-[stage-progress-waiting_1.2s_ease-in-out_infinite]")}>
                                    <Progress value={stageProgress} aria-label={t`${stageLabel(stage)} progress`}><ProgressValue /></Progress>
                                    <span>{stageProgress}%</span>
                                  </div>
                                )}
                              </div>
                            </div>
                            <StageStatusBadge status={current.status} />
                            <Button variant={current.status === "running" ? "destructive" : "ghost"} size="sm" disabled={current.status !== "running" && action.blocked} onClick={() => handleStageAction(task, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{action.label}</Button>
                          </div>
                        );
                      })}
                    </div></>}
                  </article>
                );
              })}
                  </div> : <Empty className="min-h-39 rounded-xl border border-dashed bg-card/70 text-muted-foreground [&_[data-slot=empty-description]]:text-muted-foreground [&_[data-slot=empty-title]]:text-muted-foreground">
                    <EmptyHeader>
                      <EmptyMedia variant="icon"><CheckCircle2 /></EmptyMedia>
                      <EmptyTitle><Trans>No completed tasks yet</Trans></EmptyTitle>
                      <EmptyDescription><Trans>Completed reconstruction tasks will appear here.</Trans></EmptyDescription>
                    </EmptyHeader>
                  </Empty>}
                </section>
              ))}
            </div>
          </section>
          </m.div>
        )}
      </main>

      <AnimatePresence initial={false}>
        {toast && <AppNotice key={toast} message={localiseUserMessage(toast)} onClose={() => setToast(null)} avoidBottomAction={taskDetailOpen} />}
      </AnimatePresence>

      <Dialog open={taskDialogOpen} onOpenChange={(open) => {
        setTaskDialogOpen(open);
        if (!open && editingTaskId) {
          const run = autoPipelineRuns.current[editingTaskId];
          if (run) run.paused = false;
          setEditingTaskId(null);
          queueMicrotask(() => pumpAutoPipelineRef.current());
        }
      }}>
        <DialogContent className="grid h-[min(880px,calc(100vh-32px))] max-h-[min(880px,calc(100vh-32px))] w-[min(960px,calc(100vw-32px))] grid-rows-[auto_minmax(0,1fr)_auto] gap-0 overflow-hidden rounded-xl p-0 sm:max-w-[960px] max-[760px]:h-[calc(100vh-24px)] max-[760px]:max-h-[calc(100vh-24px)] max-[760px]:w-[calc(100vw-24px)]" showCloseButton={false} initialFocus={taskNameInputRef}>
          <DialogHeader className="relative border-b px-7 py-5 max-[760px]:px-5 max-[760px]:py-4">
            <DialogTitle>{editingTaskId ? <Trans context="queued task dialog" comment="Dialog for editing a task before it starts.">Edit queued task</Trans> : <Trans context="new task dialog" comment="Dialog for creating a new reconstruction task.">New reconstruction task</Trans>}</DialogTitle>
            <DialogDescription>{editingTaskId ? <Trans>Adjust the task name, sources, and processing settings before it starts.</Trans> : <Trans comment="Only add media captured in one scene; separate media from different scenes into separate tasks.">Add only OSV or dual-fisheye media captured in the same scene. A scene may include multiple sources. Create separate reconstruction tasks for different scenes.</Trans>}</DialogDescription>
            <Popover>
              <PopoverTrigger render={<Button type="button" variant="ghost" size="icon-sm" className="absolute top-4 right-5" aria-label={t`Task creation help`} />}><CircleHelp /></PopoverTrigger>
              <PopoverContent className="max-w-80 [&>p]:mt-2 [&>p]:text-sm [&>p]:leading-relaxed [&>p]:text-muted-foreground" side="bottom" sideOffset={8}>
                <PopoverTitle>{editingTaskId ? <Trans context="queued task dialog" comment="Dialog for editing a task before it starts.">Edit queued task</Trans> : <Trans context="new task dialog" comment="Dialog for creating a new reconstruction task.">New reconstruction task</Trans>}</PopoverTitle>
                <p>{editingTaskId ? <Trans>Adjust the task name, sources, and processing settings before it starts.</Trans> : <Trans comment="Only add media captured in one scene; separate media from different scenes into separate tasks.">Add only OSV or dual-fisheye media captured in the same scene. A scene may include multiple sources. Create separate reconstruction tasks for different scenes.</Trans>}</p>
              </PopoverContent>
            </Popover>
          </DialogHeader>
          <div className="min-h-0 overflow-hidden max-[760px]:overflow-y-auto">
            <div className="grid size-full min-h-0 grid-cols-2 max-[760px]:h-auto max-[760px]:min-h-full max-[760px]:grid-cols-1">
              <section className="min-h-0 overflow-hidden max-[760px]:overflow-visible" aria-labelledby="task-information-title">
                <div className="scroll-fade-y scroll-fade-8 h-full overflow-y-auto overscroll-contain px-7 py-6 [scrollbar-gutter:stable] max-[920px]:px-5 max-[760px]:h-auto max-[760px]:overflow-visible max-[760px]:p-5 max-[760px]:[--scroll-fade-mask:none]">
                <h2 id="task-information-title" className="mb-4 text-lg font-semibold text-foreground"><Trans>Task information</Trans></h2>
                <FieldGroup>
                <Field><FieldLabel htmlFor="task-name"><Trans>Task name</Trans></FieldLabel><FieldContent><Input ref={taskNameInputRef} id="task-name" value={nameDraft} placeholder={t`For example: mountain route / 2026-08`} onChange={(event) => setNameDraft(event.currentTarget.value)} /></FieldContent></Field>
                <Field><FieldLabel><Trans context="source input" comment="Input media for a reconstruction task.">Sources</Trans></FieldLabel><FieldContent><div className={cn("flex min-h-18 items-center gap-3 rounded-lg border border-dashed px-4 py-3 text-muted-foreground transition-colors hover:border-primary/50 hover:bg-primary/5 hover:text-primary", dragOver && "border-primary/50 bg-primary/10 text-primary")} onDragOver={(event) => event.preventDefault()} onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}><FileStack className="size-4.5 shrink-0" /><span className="flex-1 text-sm"><Trans>Drop OSV or dual-fisheye media</Trans></span><Button type="button" variant="outline" size="sm" onClick={() => void openSourcePicker()}>{t`Choose sources`}</Button></div>{selectedSources.length > 0 && <div className="mt-2 overflow-hidden rounded-lg border">{selectedSources.map((source) => <SourceListItem key={source.id} source={source} title={source.label} detail={source.detail} removeLabel={t`Remove ${source.label}`} onRemove={() => { setSourcePaths((current) => current.filter((path) => path !== source.path)); setSourceColorInspection(null); }} />)}</div>}<p className="mt-2 text-sm text-muted-foreground">{sourceInspection ? localiseUserMessage(sourceInspection) : t`Choose multiple OSV or dual-fisheye media files.`}</p></FieldContent></Field>
                <Field><FieldLabel htmlFor="output-path"><Trans context="output destination" comment="Folder where project metadata and reconstruction output are saved.">Output folder</Trans></FieldLabel><FieldContent><div className="flex items-center gap-2"><Input className="flex-1" id="output-path" value={outputDraft} disabled={Boolean(editingTaskId)} placeholder={t`Defaults beside the first source: colmap-file-name`} onChange={(event) => setOutputDraft(event.currentTarget.value)} />{!editingTaskId && <Button type="button" variant="outline" size="sm" onClick={() => void openOutputPicker()}><Trans context="output folder picker action" comment="Button opens a folder picker to choose a different output location.">Choose another</Trans></Button>}</div><FieldDescription>{editingTaskId ? t`Saving a new task name also renames the output folder; unsupported filename characters become hyphens.` : t`After creation, project information is saved in the output folder so the task can resume after an interruption.`}</FieldDescription></FieldContent></Field>
                </FieldGroup>
                </div>
              </section>
              {renderSettingsFields()}
            </div>
          </div>
          <DialogFooter className="border-t px-5 py-4"><DialogClose render={<Button variant="outline" />}><Trans>Cancel</Trans></DialogClose><Button onClick={() => void (editingTaskId ? saveEditedTask() : createTask())} disabled={!sourcePaths.length || customLutPathIsInvalid(settingsDraft.extract.lutPath)}>{editingTaskId ? <Trans context="task action" comment="Save changes to a queued task.">Save changes</Trans> : <Trans context="task action" comment="Create the reconstruction task.">Create task</Trans>}</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(deletingTaskId)} onOpenChange={(open) => { if (!open) setDeletingTaskId(null); }}>
        <DialogContent showCloseButton={false}>
          <DialogHeader><DialogTitle><Trans context="remove task confirmation" comment="Confirmation dialog for removing a queued task.">Remove task from queue?</Trans></DialogTitle><DialogDescription><Trans>Only the task and queue state are removed; the existing output folder is not deleted.</Trans></DialogDescription></DialogHeader>
          <DialogFooter><DialogClose render={<Button variant="ghost" />}><Trans>Cancel</Trans></DialogClose><Button variant="destructive" onClick={deleteQueuedTask}><Trash2 /><Trans context="task action" comment="Remove the queued task, while keeping its output folder.">Remove task</Trans></Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <TaskDetailPanel
        open={taskDetailOpen && Boolean(selectedTask)}
        title={selectedTask?.name ?? ""}
        description={selectedTask ? <span title={selectedTask.outputPath}>{selectedTask.outputPath || t`Output not specified`}</span> : null}
        leading={selectedTask ? (selectedTaskSources[0] ? <SourceThumbnail source={selectedTaskSources[0]} previewSide="left" size="compact" /> : <span className="grid size-10.5 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary [&_svg]:size-5"><FileStack /></span>) : null}
        activeTab={taskDetailTab}
        onTabChange={setTaskDetailTab}
        summary={selectedTask ? (
          <>
            {selectedStage && selectedStageDefinition && <section className="border-b py-5">
              <div className="flex items-center justify-between gap-3">
                <span className="flex min-w-0 flex-col gap-1"><small className="text-sm text-muted-foreground"><Trans>Current work</Trans></small><strong className="text-base font-semibold text-foreground">{stageLabel(selectedStageDefinition)}</strong></span>
                <StageStatusBadge status={selectedStage.status} />
              </div>
              <div className="mt-4 flex flex-col gap-1 text-sm">
                <strong className="font-semibold text-foreground">{phaseLabel(selectedStage.phase)}</strong>
                <p className="leading-relaxed break-anywhere text-muted-foreground">{selectedStage.message ? localiseUserMessage(selectedStage.message) : stageDescription(selectedStageDefinition)}</p>
                {selectedStage.currentItem && <small className="leading-relaxed break-anywhere text-muted-foreground">{t`Current item: ${selectedStage.currentItem}`}</small>}
              </div>
              {(selectedStage.status === "running" || selectedStage.progress > 0) && <div className="mt-4 flex flex-col gap-2 [&_[data-slot=progress-track]]:h-1.25 [&_[data-slot=progress-value]]:hidden">
                <div className="flex items-center justify-between gap-3 text-sm"><span className="text-muted-foreground">{t`${stageLabel(selectedStageDefinition)} progress`}</span><strong className="font-mono text-primary">{Math.round(selectedStage.progress)}%</strong></div>
                <Progress value={selectedStage.progress} aria-label={t`${stageLabel(selectedStageDefinition)} progress`}><ProgressValue /></Progress>
              </div>}
              <dl className="mt-4.5 grid grid-cols-2 border-t">
                <DetailMetric label={<Trans>Processed</Trans>} value={logCountLabel(selectedStage.completed, selectedStage.total) || t`Not reported yet`} />
                <DetailMetric label={<Trans>Elapsed</Trans>} value={formatDuration(taskStageDuration(selectedStage, clockMs))} />
                <DetailMetric label={<Trans>Estimated remaining</Trans>} value={selectedStage.status === "running" ? formatEta(estimatedRemainingMs(selectedStage, clockMs)) : "—"} />
                <DetailMetric label={<Trans comment="Estimated processing throughput, not network speed.">Rate (estimated)</Trans>} value={selectedStage.status === "running" ? processingRateLabel(selectedActiveProgressLog?.completed, selectedActiveProgressLog?.startedAtMs, clockMs) : "—"} />
                <DetailMetric fullWidth label={<Trans context="output metric" comment="Output folder or artifact location.">Output</Trans>} value={selectedTask.outputPath || t`Not specified`} />
              </dl>
            </section>}

            <section className="border-b py-5">
              <DetailSectionHeading title={<Trans context="source section" comment="Source media included in this reconstruction task.">Sources</Trans>} meta={<Plural value={selectedTask.inputPaths.length} one="# file" other="# files" />} />
              {selectedTaskSources.length > 0 ? <div className="overflow-hidden border-t">{selectedTaskSources.map((source) => <SourceListItem key={source.id} source={source} title={source.detail} detail={source.path} previewSide="left" />)}</div> : <p className="text-sm text-muted-foreground"><Trans>This task has no recorded source files.</Trans></p>}
            </section>

            {selectedTask.warnings.length > 0 && (
              <section className="border-b py-5">
                <DetailSectionHeading title={<Trans>Warnings</Trans>} meta={<Badge variant="destructive">{selectedTask.warnings.length}</Badge>} />
                <div className="flex flex-col border-t">{selectedTask.warnings.map((warning, index) => <div className="flex min-w-0 items-start gap-2 border-t py-2.5 first:border-t-0" key={`${index}-${warning}`}><AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-amber-600" /><span className="min-w-0 text-sm leading-relaxed text-amber-700 dark:text-amber-400">{localiseUserMessage(warning)}</span></div>)}</div>
              </section>
            )}
          </>
        ) : null}
        records={selectedTask ? (
          <section className="pt-5">
            <DetailSectionHeading title={<Trans>Processing records</Trans>} meta={<Plural value={selectedTaskLogs.length} one="# entry" other="# entries" />} />
            {selectedTaskLogs.length > 0 ? <ol className="overflow-hidden rounded-md border bg-muted/20" aria-label={t`Processing records`}>{selectedTaskLogs.map((log) => {
              const count = logCountLabel(log.completed, log.total);
              const scope = `${taskStageLabel(log.stage)}${log.phase ? `/${phaseLabel(log.phase)}` : ""}`;
              return <li className="grid min-w-0 grid-cols-[auto_auto_minmax(0,1fr)] items-baseline gap-x-2 border-t px-3 py-2 font-mono text-xs first:border-t-0" key={log.id}>
                <time className="shrink-0 text-muted-foreground" dateTime={timestampDateTime(log.timestampMs)}>{formatTimestamp(log.timestampMs, true)}</time>
                <strong className={cn("font-semibold text-primary", log.level === "warning" && "text-amber-600", log.level === "error" && "text-destructive")}>{log.level === "warning" ? "WARN" : log.level.toUpperCase()}</strong>
                <p className="min-w-0 leading-relaxed break-anywhere whitespace-pre-wrap text-foreground"><span className="text-muted-foreground">[{scope}]</span>{" "}{localiseUserMessage(log.message)}</p>
                {(count || log.currentItem || log.durationMs !== undefined) && <div className="col-start-3 flex min-w-0 flex-wrap gap-x-3 text-muted-foreground">{count && <span>{count}</span>}{log.currentItem && <span className="break-anywhere">{log.currentItem}</span>}{log.durationMs !== undefined && <span>{t`Duration ${formatDuration(log.durationMs)}`}</span>}</div>}
              </li>;
            })}</ol> : <p className="text-sm text-muted-foreground"><Trans>There are no processing records yet; each stage and its current position will appear here after execution starts.</Trans></p>}
          </section>
        ) : null}
        footer={selectedTask ? (
          <>
            {selectedRunningStageDefinition && <Button variant="destructive" onClick={() => handleStageAction(selectedTask, selectedRunningStageDefinition.key)}><Square data-icon="inline-start" /><Trans context="task action" comment="Cancel the currently running stage for the whole selected task.">Cancel entire task</Trans></Button>}
            <Button variant="outline" onClick={closeTaskDetail}><Trans>Close</Trans></Button>
          </>
        ) : null}
        onClose={closeTaskDetail}
        restoreFocusRef={taskDetailTriggerRef}
        escapeBlocked={taskDialogOpen || Boolean(deletingTaskId) || settingsOpen}
        onExitComplete={() => {
          if (taskDetailOpen) return;
          setSelectedTaskId(null);
          taskDetailTriggerRef.current = null;
        }}
      />

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent className="w-[min(420px,100vw)] gap-0 border-l bg-card p-0 data-[side=right]:sm:max-w-[420px]" side="right">
          <SheetHeader className="border-b px-6 pt-6 pb-4"><SheetTitle className="text-base font-semibold"><Trans>Settings</Trans></SheetTitle><SheetDescription className="mt-2 text-sm leading-relaxed"><Trans>Choose English, Simplified Chinese, Traditional Chinese, or Japanese.</Trans></SheetDescription></SheetHeader>
          <div className="scroll-fade-y scroll-fade-8 flex-1 overflow-y-auto px-6">
            <section className="border-b py-4 last:border-b-0">
              <FieldSet>
                <FieldLegend variant="label"><Trans context="language setting" comment="Select the language used by the interface.">Language</Trans></FieldLegend>
                <Select
                  items={LANGUAGE_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
                  value={getLocale()}
                  onValueChange={(value) => { if (value) void setLocale(value); }}
                >
                  <SelectTrigger className="w-full" aria-label={t`Interface language`}><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {LANGUAGE_OPTIONS.map((option) => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </FieldSet>
            </section>
            <section className="border-b py-4 last:border-b-0">
              <FieldSet className="gap-2.5">
                <FieldLegend variant="label"><Trans context="theme setting" comment="Select the visual theme of the interface.">Interface theme</Trans></FieldLegend>
                <FieldDescription><Trans>Choose light, dark, or follow the system appearance automatically.</Trans></FieldDescription>
                <ToggleGroup
                  className="grid w-full grid-cols-3 [&_[data-slot=toggle-group-item]]:w-full [&_[data-slot=toggle-group-item]]:min-w-0"
                  variant="outline"
                  size="sm"
                  spacing={0}
                  value={[theme]}
                  onValueChange={(values) => {
                    const nextTheme = values[0] as Theme | undefined;
                    if (nextTheme) setTheme(nextTheme);
                  }}
                  aria-label={t`Interface theme`}
                >
                  <ToggleGroupItem value="system"><Trans>System</Trans></ToggleGroupItem>
                  <ToggleGroupItem value="light"><Trans>Light</Trans></ToggleGroupItem>
                  <ToggleGroupItem value="dark"><Trans>Dark</Trans></ToggleGroupItem>
                </ToggleGroup>
              </FieldSet>
            </section>
            <section className="border-b py-4 last:border-b-0">
              <div className="mb-3 flex flex-col items-stretch gap-3">
                <div className="flex min-w-0 flex-1 flex-col gap-1"><h2 className="text-base font-semibold text-foreground"><Trans>Runtime environment</Trans></h2><span className="text-sm text-muted-foreground">{t`Last checked: ${formatDoctorCheckedAt(doctor.checkedAt)}`}</span></div>
                <div className="grid w-full grid-cols-2 gap-2 [&_[data-slot=button]]:w-full [&_[data-slot=button]]:min-w-0" role="group" aria-label={t`Diagnostic actions`}>
                  <Button type="button" variant="outline" size="sm" disabled={doctorLoading || doctor.checkedAt === "Not checked yet"} onClick={() => void copyDoctorReport()}><Copy data-icon="inline-start" /><Trans>Copy diagnostics</Trans></Button>
                  <Button type="button" size="sm" className={doctorLoading ? "[&_svg]:animate-spin [&_svg]:[animation-duration:750ms]" : undefined} disabled={doctorLoading} onClick={() => void runDoctor(colmapPath)}><RefreshCw data-icon="inline-start" />{doctorLoading ? <Trans>Checking</Trans> : <Trans>Check again</Trans>}</Button>
                </div>
              </div>
              <div className="mb-4 flex flex-col gap-2.5">
                <Alert className={doctorEssentialReady ? "border-emerald-600/30 bg-emerald-500/10 text-emerald-700 [&_[data-slot=alert-description]]:text-muted-foreground dark:text-emerald-400" : undefined} variant={doctorEssentialReady ? "default" : "destructive"} role={doctorEssentialReady ? "status" : "alert"}>
                  {doctorEssentialReady ? <CheckCircle2 /> : <AlertTriangle />}
                  <AlertTitle>{doctorEssentialReady ? <Trans>All required capabilities are available</Trans> : <Trans>Required capabilities need attention</Trans>}</AlertTitle>
                  <AlertDescription>{doctorEssentialReady ? <Trans>Basic reconstruction can run. CUDA and hardware acceleration are optional; processing still works without them but frame extraction, feature matching, and reconstruction will be slower.</Trans> : <Trans>Missing required tools will block some stages. Address the items marked “Needs attention” below first.</Trans>}</AlertDescription>
                </Alert>
                {performanceStatus !== "ready" && <Alert className={cn("[&_[data-slot=alert-description]]:leading-relaxed [&_[data-slot=alert-description]]:text-muted-foreground", performanceStatus === "warning" ? "border-amber-600/40 bg-amber-500/10 text-amber-700 dark:text-amber-400" : "bg-muted text-muted-foreground")} role={performanceStatus === "warning" ? "alert" : "status"}>
                  <Gauge />
                  <AlertTitle>{performanceStatus === "warning" ? <Trans>Performance will be affected</Trans> : <Trans>Acceleration capabilities not confirmed</Trans>}</AlertTitle>
                  <AlertDescription>
                    {performanceWarnings.length > 0
                      ? performanceWarnings.map((warning) => <p key={warning}>{localiseUserMessage(warning)}</p>)
                      : performanceStatus === "warning"
                        ? <p>{localiseUserMessage(performanceFallback)}</p>
                        : <p><Trans>Run the environment check to confirm whether CUDA, hardware decoding, or the CPU will be used.</Trans></p>}
                    {performanceStatus === "warning" && <p><Trans>Stages that fall back to the CPU can still run, but processing may take significantly longer.</Trans></p>}
                  </AlertDescription>
                </Alert>}
              </div>
              {isWindowsPlatform && <Field><FieldLabel htmlFor="colmap-path"><Trans comment="Path to the COLMAP executable used by the Windows runtime.">COLMAP executable</Trans></FieldLabel><FieldContent><div className="flex items-center gap-2 [&_input]:flex-1"><Input id="colmap-path" value={colmapPath} placeholder={t`Leave blank to detect from PATH`} onChange={(event) => setColmapPath(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openColmapPicker()}>{t`Change path`}</Button></div><FieldDescription><Trans>For the official Windows portable build, select COLMAP.bat in the root folder; you can also specify a self-built colmap.exe.</Trans></FieldDescription></FieldContent></Field>}
              <div className="flex items-start gap-2.5 py-2.5"><MonitorCog className="mt-0.5 size-4 shrink-0 text-primary" strokeWidth={1.75} /><span className="flex min-w-0 flex-1 flex-col gap-1"><strong className="text-base font-medium text-foreground">{localiseUserMessage(doctor.platform)}</strong><small className="text-sm leading-snug break-anywhere text-muted-foreground">{localiseUserMessage(doctor.summary)}</small></span></div>
              <div className="border-t">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <article className="flex min-w-0 items-start gap-2.5 border-t py-2.5 first:border-t-0" key={item.label}><Icon className={cn("mt-0.75 size-4 shrink-0 text-muted-foreground", item.status === "ready" && "text-emerald-600", item.status === "warning" && "text-destructive")} strokeWidth={1.75} /><div className="flex min-w-0 flex-1 flex-col gap-1"><div className="flex min-w-0 items-start gap-2"><span className="flex min-w-0 flex-1 flex-col gap-1"><small className="text-sm leading-snug break-anywhere text-muted-foreground">{diagnosticItemLabel(item.label)}</small><strong className="text-base font-medium text-foreground">{localiseUserMessage(item.value)}</strong></span><Badge className="shrink-0 text-sm" variant={item.status === "warning" ? "destructive" : "outline"}>{diagnosticStatusLabel(item.status)}</Badge></div><p className="mt-0.5 text-sm leading-relaxed break-anywhere text-muted-foreground">{localiseUserMessage(item.detail)}</p>{item.details && item.details.length > 0 && <Accordion className="mt-1"><AccordionItem value={`${item.label}-details`}><AccordionTrigger className="w-fit py-1 text-sm font-medium text-primary"><Trans>View details</Trans></AccordionTrigger><AccordionContent><ul className="mt-2 mb-0.5 flex list-none flex-col gap-1.5 text-sm leading-relaxed text-muted-foreground">{item.details.map((detail) => <li className="border-l-2 pl-3 break-anywhere" key={detail}>{localiseUserMessage(detail)}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}</div></article>; })}</div>
              {generalDoctorWarnings.length > 0 && <Alert variant="destructive"><AlertTriangle /><AlertTitle><Trans>Needs attention</Trans></AlertTitle><AlertDescription>{generalDoctorWarnings.map((warning) => <p key={warning}>{localiseUserMessage(warning)}</p>)}</AlertDescription></Alert>}
            </section>
          </div>
          <SheetFooter className="border-t px-6 pt-4 pb-5.5"><Button variant="outline" onClick={() => setSettingsOpen(false)}><Trans>Close</Trans></Button></SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default App;

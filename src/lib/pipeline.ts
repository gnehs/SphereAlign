import {
  CircleDashed,
  FileVideoCamera,
  Gpu,
  MonitorCog,
  ScanLine,
  ScanSearch,
  Workflow,
  type LucideIcon,
} from "lucide-react";
import { i18n, type I18n, type MessageDescriptor } from "@lingui/core";
import { msg, plural, t } from "@lingui/core/macro";
import { invoke } from "@tauri-apps/api/core";
import { getEnglishI18n, getLocale, localeLabels, supportedLocales } from "@/i18n";

export const LANGUAGE_OPTIONS = supportedLocales.map((value) => ({ value, label: localeLabels[value] }));
export const APP_NOTICE_EASE = [0.22, 1, 0.36, 1] as const;

type StageKey = "extract" | "mask" | "align";
type StageStatus = "pending" | "running" | "completed" | "cancelled" | "failed";
type DiagnosticStatus = "ready" | "warning" | "unknown";
type ExtractColorMode = "auto" | "dlogMRec709" | "native";

function translate(descriptor: MessageDescriptor) {
  return i18n._(descriptor);
}

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

type SourceIssueSeverity = "warning" | "error";

interface SourceIssue {
  code: string;
  severity: SourceIssueSeverity;
  message: string;
  detail: string;
  impacts: string[];
}

interface SourceInspection {
  path: string;
  name?: string;
  size?: number;
  valid?: boolean;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  lensCount?: number;
  colorProfile?: string | Record<string, unknown>;
  warnings: string[];
  issues: SourceIssue[];
}

interface SourceMedia {
  id: string;
  path: string;
  label: string;
  detail: string;
  status?: "ready" | "warning" | "unknown";
  issues?: SourceIssue[];
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
  /** Per-source inspection results are kept in the client task state so the
   * task detail view can explain issues after the editor closes. The backend
   * manifest may not contain this optional field on older projects. */
  sourceInspections?: Record<string, SourceInspection>;
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
    label: msg({ message: "Alignment", context: "pipeline stage label", comment: "The stage that aligns multiple panoramic sources and camera rigs." }),
    description: msg({ message: "Multi-source panoramic camera-rig alignment", context: "pipeline stage description", comment: "Technical description of the alignment stage for multiple panoramic source formats." }),
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
const COLMAP_PATH_STORAGE_KEY = "spherealign.colmapPath";

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
  "Select at least one panoramic source first": msg({ message: "Select at least one panoramic source first" }),
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
  "Unable to cancel the stage; it may still be running": msg({ message: "Unable to cancel the stage; it may still be running" }),
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
      message: `${source} does not contain the video streams required by its source adapter`,
      context: "source validation error",
      comment: "A source adapter must expose the video streams required by the panoramic workflow.",
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
    .replace(/COLMAP\.bat 無法直接使用；請選擇官方可攜版 bin\\COLMAP\.exe/g, () => t`COLMAP.bat cannot be used directly; select bin/COLMAP.exe from the official portable build`)
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

function finiteNumber(value: unknown) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function nonNegativeNumber(value: unknown) {
  const number = finiteNumber(value);
  return number !== undefined && number >= 0 ? number : undefined;
}

function normaliseSourceIssue(value: unknown, fallbackCode = "source-warning"): SourceIssue | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const message = typeof body.message === "string" && body.message.trim()
    ? body.message.trim()
    : typeof body.title === "string" && body.title.trim()
      ? body.title.trim()
      : "Source inspection reported a problem";
  const detail = typeof body.detail === "string" && body.detail.trim()
    ? body.detail.trim()
    : typeof body.description === "string" && body.description.trim()
      ? body.description.trim()
      : message;
  const severity = String(body.severity ?? "warning").trim().toLowerCase() === "error"
    ? "error"
    : "warning";
  const impacts = Array.isArray(body.impacts)
    ? body.impacts.filter((impact): impact is string => typeof impact === "string" && impact.trim().length > 0).map((impact) => impact.trim())
    : [];
  return {
    code: typeof body.code === "string" && body.code.trim() ? body.code.trim() : fallbackCode,
    severity,
    message,
    detail,
    impacts,
  };
}

function normaliseSourceInspection(value: unknown, fallbackPath = ""): SourceInspection | null {
  if (!value || typeof value !== "object") return null;
  const body = value as Record<string, unknown>;
  const path = typeof body.path === "string" && body.path.trim() ? body.path : fallbackPath;
  if (!path) return null;
  const warnings = Array.isArray(body.warnings)
    ? body.warnings.filter((warning): warning is string => typeof warning === "string" && warning.trim().length > 0).map((warning) => warning.trim())
    : [];
  const issues = Array.isArray(body.issues)
    ? body.issues.flatMap((issue, index) => {
        const normalised = normaliseSourceIssue(issue, `source-issue-${index + 1}`);
        return normalised ? [normalised] : [];
      })
    : [];
  const issueMessages = new Set(issues.map((issue) => issue.message));
  warnings.forEach((warning, index) => {
    if (issueMessages.has(warning)) return;
    issues.push({
      code: `legacy-warning-${index + 1}`,
      severity: "warning",
      message: warning,
      detail: warning,
      impacts: [],
    });
  });
  if (body.valid === false && issues.length === 0) {
    issues.push({
      code: "invalid-source",
      severity: "error",
      message: "This source cannot be used",
      detail: "The source did not pass inspection.",
      impacts: [],
    });
  }
  const colorProfile = typeof body.colorProfile === "string"
    ? body.colorProfile
    : body.colorProfile && typeof body.colorProfile === "object"
      ? body.colorProfile as Record<string, unknown>
      : undefined;
  return {
    path,
    name: typeof body.name === "string" ? body.name : undefined,
    size: nonNegativeNumber(body.size),
    valid: typeof body.valid === "boolean" ? body.valid : undefined,
    duration: nonNegativeNumber(body.duration),
    fps: nonNegativeNumber(body.fps),
    width: nonNegativeNumber(body.width),
    height: nonNegativeNumber(body.height),
    lensCount: nonNegativeNumber(body.lensCount ?? body.lens_count),
    colorProfile,
    warnings,
    issues,
  };
}

function normaliseSourceInspectionMap(value: unknown): Record<string, SourceInspection> {
  if (!value || typeof value !== "object") return {};
  return Object.entries(value as Record<string, unknown>).reduce<Record<string, SourceInspection>>((result, [path, inspection]) => {
    const normalised = normaliseSourceInspection(inspection, path);
    if (normalised) result[path] = normalised;
    return result;
  }, {});
}

function sourceStatusForInspection(inspection?: SourceInspection): SourceMedia["status"] {
  if (!inspection) return "unknown";
  if (inspection.issues.some((issue) => issue.severity === "error") || inspection.valid === false) return "warning";
  if (inspection.issues.length > 0) return "warning";
  return "ready";
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
  const sourceInspections = normaliseSourceInspectionMap(body.sourceInspections ?? body.source_inspections);
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
    sourceInspections: Object.keys(sourceInspections).length > 0 ? sourceInspections : undefined,
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

function sourceFromPath(path: string, index: number, inspection?: SourceInspection): SourceMedia {
  const label = path.split(/[\\/]/).filter(Boolean).pop() || `Source ${index + 1}`;
  return {
    id: `${index}-${path}`,
    path,
    label: `Source ${String(index + 1).padStart(2, "0")}`,
    detail: label,
    status: sourceStatusForInspection(inspection),
    issues: inspection?.issues,
  };
}

function sourceInspectionForPath(path: string, inspections?: Record<string, SourceInspection>) {
  if (!inspections) return undefined;
  return inspections[path]
    ?? inspections[path.split("\\").join("/")]
    ?? inspections[path.split("/").join("\\")];
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
export type {
  StageKey,
  StageStatus,
  DiagnosticStatus,
  ExtractColorMode,
  ColorInspection,
  ColorInspectionSummary,
  SourceIssueSeverity,
  SourceIssue,
  SourceInspection,
  StageState,
  PipelineSettings,
  SourceMedia,
  ProjectManifest,
  Task,
  DiagnosticItem,
  SystemInfo,
  TaskLogKind,
  TaskLogLevel,
  TaskLog,
  DoctorReport,
  ProgressEventPayload,
  LogEventPayload,
  AutoPipelineRun,
  StageDefinition,
};

export {
  translate,
  STAGES,
  stageLabel,
  stageDescription,
  STAGE_OBSERVED_DURATION_MS,
  TOTAL_OBSERVED_DURATION_MS,
  MASK_CLASSES,
  MASK_CLASS_LABELS,
  MIN_CANDIDATE_MULTIPLIER,
  MAX_CANDIDATE_MULTIPLIER,
  DEFAULT_CANDIDATE_MULTIPLIER,
  DEFAULT_SETTINGS,
  COLMAP_PATH_STORAGE_KEY,
  normaliseExtractColorMode,
  customLutPathIsInvalid,
  candidateMultiplierFor,
  normalisePipelineSettings,
  selectAvailableGpu,
  gpuDeviceLabel,
  COLMAP_CUDA_DIAGNOSTIC_LABEL,
  HARDWARE_ACCELERATION_LABEL,
  emptyDoctor,
  diagnosticItemLabel,
  IS_TAURI_RUNTIME,
  IS_WINDOWS_RUNTIME,
  IS_MACOS_RUNTIME,
  PREPARING_WORK_SOURCE,
  BROWSER_PREVIEW_TASK_SOURCE,
  BROWSER_PREVIEW_NOT_CONNECTED_SOURCE,
  UNKNOWN_BACKEND_ERROR,
  HIDDEN_TECHNICAL_ERROR_DETAILS,
  HIDDEN_BACKEND_STATUS,
  containsCjk,
  localiseTechnicalErrorDetail,
  isLikelyBackendError,
  localiseUserMessage,
  backendErrorMessage,
  platformLabel,
  timestampMs,
  nonNegativeInteger,
  basename,
  normaliseLogLevel,
  PHASE_LABELS,
  phaseLabel,
  formatDuration,
  formatTimestamp,
  timestampDateTime,
  formatDoctorCheckedAt,
  stageSummaryLogs,
  parseTaskLog,
  parseTaskLogs,
  taskStageLabel,
  logCountLabel,
  taskProgress,
  taskHasNotStarted,
  taskIsCompleted,
  taskProgressSummary,
  taskCurrentStage,
  stagePrerequisiteKey,
  stagePrerequisiteLabel,
  taskHasRunningStage,
  stageActionState,
  normaliseStageStatus,
  normaliseStage,
  toProgress,
  cloneStages,
  manifestFromUnknown,
  taskCreatedAtMs,
  readProgress,
  readLogEvent,
  parseDoctor,
  invokeSafely,
  deriveOutputPath,
  sourceFromPath,
  sourceInspectionForPath,
  normaliseSourceInspection,
  normaliseSourceInspectionMap,
  normaliseColorInspection,
  normaliseColorInspectionSummary,
  MAX_SOURCE_PREVIEW_CACHE_ENTRIES,
  loadSourcePreview,
  iconForDiagnostic,
  warningAffectsProcessingSpeed,
  diagnosticStatusLabel,
  redactDiagnosticText,
  containsDiagnosticPath,
  englishDiagnosticMessage,
  englishDiagnosticDetail,
  formatEnglishDoctorCheckedAt,
  doctorReportText,
  stageAction,
  stageStatusLabel,
  taskStageDuration,
  estimatedRemainingMs,
  formatEta,
  processingRateLabel,
  logLevelForStatus,
  mergeProgressLog,
  appendMessageLog,
};

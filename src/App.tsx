import {
  AlertTriangle,
  CircleDashed,
  Clock3,
  Cpu,
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
  MemoryStick,
  CheckCircle2,
  Copy,
  Trash2,
  Upload,
  Video,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { writeText as writeClipboardText } from "@tauri-apps/plugin-clipboard-manager";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
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
import { useTheme, type Theme } from "@/components/theme-provider";
import "./App.css";

type StageKey = "extract" | "mask" | "align";
type StageStatus = "pending" | "running" | "completed" | "cancelled" | "failed";
type DiagnosticStatus = "ready" | "warning" | "unknown";

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

const STAGES: Array<{ key: StageKey; label: string; description: string; icon: LucideIcon }> = [
  { key: "extract", label: "影格擷取", description: "雙魚眼影格、內參與 IMU", icon: ScanLine },
  { key: "mask", label: "遮罩", description: "動態物件與天空遮罩", icon: CircleDashed },
  { key: "align", label: "對齊", description: "多組 OSV／相機組對齊", icon: Workflow },
];

const MASK_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];
const MASK_CLASS_LABELS: Record<string, string> = {
  person: "人員",
  bicycle: "腳踏車",
  car: "汽車",
  motorcycle: "機車",
  bus: "公車",
  truck: "卡車",
};
const MIN_CANDIDATE_MULTIPLIER = 2;
const MAX_CANDIDATE_MULTIPLIER = 10;
const DEFAULT_CANDIDATE_MULTIPLIER = 4;
const DEFAULT_SETTINGS: PipelineSettings = {
  extract: {
    baseFps: 3,
    denseFps: 12,
    skipBlurry: true,
  },
  mask: { yoloEnabled: false, classes: [], maskSky: false, modelDir: "" },
  align: {
    useGpu: true,
    gpuIndex: "-1",
  },
};
const COLMAP_PATH_STORAGE_KEY = "gs360studio.colmapPath";

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
  return {
    extract: {
      baseFps,
      denseFps: finiteNumber(extract.denseFps, baseFps * DEFAULT_CANDIDATE_MULTIPLIER, baseFps * MIN_CANDIDATE_MULTIPLIER, baseFps * MAX_CANDIDATE_MULTIPLIER),
      skipBlurry: typeof extract.skipBlurry === "boolean" ? extract.skipBlurry : DEFAULT_SETTINGS.extract.skipBlurry,
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

const COLMAP_CUDA_DIAGNOSTIC_LABEL = "CUDA 加速";

const EMPTY_DOCTOR: DoctorReport = {
  platform: "尚未檢查平台",
  systemInfo: {
    osName: "尚未檢查",
    osVersion: "尚未檢查",
    architecture: "尚未檢查",
    processors: [],
    graphicsAdapters: [],
  },
  summary: "執行環境診斷以確認可用能力",
  checkedAt: "尚未檢查",
  items: [
    { label: "COLMAP", value: "尚未檢查", detail: "檢查 COLMAP 執行檔與版本", status: "unknown" },
    { label: COLMAP_CUDA_DIAGNOSTIC_LABEL, value: "尚未檢查", detail: "檢查 COLMAP 的 CUDA 建置與 NVIDIA GPU 是否可用", status: "unknown" },
    { label: "FFmpeg", value: "尚未檢查", detail: "確認系統 PATH 中的 FFmpeg", status: "unknown" },
    { label: "硬體加速", value: "尚未檢查", detail: "確認 FFmpeg 硬體解碼能力", status: "unknown" },
  ],
  warnings: [],
};

const IS_TAURI_RUNTIME = typeof window !== "undefined" && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);

const USER_MESSAGE_TRANSLATIONS: Record<string, string> = {
  "scanning paired fisheye candidates": "正在掃描雙魚眼配對候選影格",
  "cancelled before interval": "已在處理下一個區間前取消",
  "cancelled while scoring candidates": "已在評分候選影格時取消",
  "selected pair already exists; skipped": "選定的配對影格已存在，已略過",
  "cancelled before output commit": "已在寫入輸出前取消",
  "copying selected pair": "正在複製選定的配對影格",
  "selected pair committed": "已寫入選定的配對影格",
  "extraction cancelled": "影格擷取已取消",
  "scanning native fisheye images": "正在掃描原生雙魚眼影像",
  "loading YOLO11/skyseg models": "正在載入 YOLO11／SkySeg 模型",
  "masking cancelled": "遮罩處理已取消",
  "verified mask exists; skipped": "已確認遮罩存在，已略過",
  "running YOLO11/skyseg": "正在執行 YOLO11／SkySeg 推論",
  "masking cancelled before output commit": "已在寫入遮罩前取消",
  "committing mask files": "正在寫入遮罩檔案",
  "mask completed": "遮罩處理完成",
  "Stage started": "處理階段已開始",
  "Stage cancelled; committed artifacts are resumable": "處理階段已取消，已寫入的結果可繼續使用",
  "Stage completed": "處理階段已完成",
  "Existing dual-fisheye frames were discovered": "已找到現有的雙魚眼影格",
  "Existing masks were discovered": "已找到現有遮罩",
  "Existing COLMAP reconstruction was discovered": "已找到現有的 COLMAP 重建結果",
  "Previous run was interrupted; this stage can be resumed": "上次處理中斷，此階段可繼續執行",
  "This project manifest was recovered from existing artifacts": "已依現有處理結果復原專案資訊",
  "Extract requires both system ffmpeg and ffprobe": "影格擷取需要系統已安裝 FFmpeg 與 ffprobe",
  "COLMAP is unavailable; alignment will remain in a resumable pending state": "找不到 COLMAP；對齊階段會維持可繼續的待執行狀態",
  "FFmpeg was found without VideoToolbox support; extraction will use the CPU decoder": "FFmpeg 不支援 VideoToolbox；影格擷取將使用 CPU 解碼",
};

function localiseUserMessage(value: string) {
  const exact = USER_MESSAGE_TRANSLATIONS[value];
  if (exact) return exact;
  return value
    .replace(/cancelled before interval (\d+)/g, "已在第 $1 個區間前取消")
    .replace(/scoring (\d+) paired candidates/g, "正在評分 $1 組配對候選影格")
    .replace(/(\d+) masks failed; see pipeline log/g, "$1 個遮罩處理失敗，請查看處理紀錄")
    .replace(/System ffmpeg was not found on PATH/g, "在系統 PATH 中找不到 FFmpeg")
    .replace(/System ffprobe was not found on PATH/g, "在系統 PATH 中找不到 ffprobe")
    .replace(/COLMAP was not found on PATH/g, "在系統 PATH 中找不到 COLMAP")
    .replace(/COLMAP bootstrap did not produce sparse\/0/g, "COLMAP 初始建模未產生 sparse/0")
    .replace(/^invalid extraction input: /, "影格擷取輸入無效：")
    .replace(/^extraction image error: /, "影格擷取影像錯誤：")
    .replace(/^extraction I\/O error: /, "影格擷取檔案錯誤：")
    .replace(/^invalid mask input: /, "遮罩輸入無效：")
    .replace(/^mask model error: /, "遮罩模型錯誤：")
    .replace(/^mask inference error: /, "遮罩推論錯誤：")
    .replace(/^mask image error: /, "遮罩影像錯誤：")
    .replace(/^mask I\/O error: /, "遮罩檔案錯誤：")
    .replace(/^mask operation cancelled$/, "遮罩處理已取消");
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

const PHASE_LABELS: Record<string, string> = {
  starting: "準備",
  scanning: "掃描",
  scoring: "評分候選",
  "selecting-in-memory": "記憶體候選評分",
  "decoding-full-resolution": "原始解析度解碼",
  committing: "寫入輸出",
  masking: "遮罩推論",
  matching: "影像配對",
  "feature-extraction": "特徵擷取",
  bootstrap: "初始建模",
  "final-mapping": "最終重建",
  rig: "相機組估計",
  completed: "完成",
  cancelled: "已取消",
  failed: "失敗",
  summary: "階段摘要",
};

function phaseLabel(value?: string) {
  if (!value) return "處理中";
  return PHASE_LABELS[value] ?? value.replace(/[-_]+/g, " ");
}

function formatDuration(value?: number) {
  if (!Number.isFinite(value) || value === undefined || value < 0) return "尚未計時";
  const totalSeconds = Math.max(0, Math.floor(value / 1000));
  if (totalSeconds < 1) return "不到 1 秒";
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60) % 60;
  const hours = Math.floor(totalSeconds / 3600);
  if (hours > 0) return `${hours} 小時 ${String(minutes).padStart(2, "0")} 分`;
  if (minutes > 0) return `${minutes} 分 ${String(seconds).padStart(2, "0")} 秒`;
  return `${seconds} 秒`;
}

function formatTimestamp(value?: number, includeDate = false) {
  if (!value) return "尚未記錄";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "尚未記錄";
  return date.toLocaleString("zh-TW", includeDate
    ? { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }
    : { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function stageLogStatus(status: StageStatus) {
  if (status === "completed") return "已完成";
  if (status === "cancelled") return "已取消";
  if (status === "failed") return "失敗";
  if (status === "running") return "執行中";
  return "待執行";
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
      message: `${label}：${stageLogStatus(stage.status)}`,
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
  const message = typeof body.message === "string" && body.message ? localiseUserMessage(body.message) : "處理紀錄";
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
  return STAGES.find((item) => item.key === stage)?.label ?? "系統";
}

function logCountLabel(completed?: number, total?: number) {
  const hasCompleted = completed !== undefined && completed > 0;
  const hasTotal = total !== undefined && total > 0;
  if (completed !== undefined && hasTotal) return `${completed.toLocaleString("zh-TW")} / ${total.toLocaleString("zh-TW")}`;
  if (hasTotal) return `總計 ${total.toLocaleString("zh-TW")}`;
  if (hasCompleted) return `已處理 ${completed.toLocaleString("zh-TW")}`;
  return undefined;
}

function taskProgress(task: Task) {
  return Math.round(Object.values(task.stages).reduce((sum, stage) => sum + (stage.status === "completed" ? 100 : stage.progress), 0) / STAGES.length);
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

function taskProgressSummary(task: Task) {
  const runningIndex = STAGES.findIndex(({ key }) => task.stages[key].status === "running");
  if (runningIndex >= 0) return `第 ${runningIndex + 1} / ${STAGES.length} 階段 · ${STAGES[runningIndex].label}`;
  const interruptedIndex = STAGES.findIndex(({ key }) => ["failed", "cancelled"].includes(task.stages[key].status));
  if (interruptedIndex >= 0) return `停在第 ${interruptedIndex + 1} / ${STAGES.length} 階段 · ${STAGES[interruptedIndex].label}`;
  const nextIndex = STAGES.findIndex(({ key }) => task.stages[key].status !== "completed");
  return nextIndex >= 0 ? `等待第 ${nextIndex + 1} / ${STAGES.length} 階段 · ${STAGES[nextIndex].label}` : `${STAGES.length} / ${STAGES.length} 階段完成`;
}

function taskCurrentStage(task: Task) {
  const running = STAGES.find(({ key }) => task.stages[key].status === "running");
  if (running) return running;
  const interrupted = STAGES.find(({ key }) => ["failed", "cancelled"].includes(task.stages[key].status));
  if (interrupted) return interrupted;
  const next = STAGES.find(({ key }) => task.stages[key].status !== "completed");
  return next ?? STAGES[STAGES.length - 1];
}

function stagePrerequisiteLabel(task: Task, stageKey: StageKey) {
  const stageIndex = STAGES.findIndex(({ key }) => key === stageKey);
  if (stageIndex <= 0) return undefined;
  const prerequisite = STAGES.slice(0, stageIndex).find(({ key }) => task.stages[key].status !== "completed");
  return prerequisite?.label;
}

function taskHasRunningStage(task: Task, except?: StageKey) {
  return STAGES.some(({ key }) => key !== except && task.stages[key].status === "running");
}

function stageActionState(task: Task, stageKey: StageKey, globallyRunning: boolean) {
  const current = task.stages[stageKey];
  const prerequisite = stagePrerequisiteLabel(task, stageKey);
  const blocked = Boolean(prerequisite) || taskHasRunningStage(task, stageKey) || (globallyRunning && current.status !== "running");
  const label = current.status === "running"
    ? `停止${STAGES.find((stage) => stage.key === stageKey)?.label ?? "階段"}`
    : prerequisite
      ? `等待${prerequisite}完成`
      : blocked
        ? "等待目前階段完成"
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
      message: typeof item.message === "string" && item.message ? localiseUserMessage(item.message) : "尚未執行",
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
    name: typeof body.name === "string" && body.name ? body.name : outputPath.split(/[\\/]/).filter(Boolean).pop() || "未命名重建",
    rootPath: typeof body.rootPath === "string" ? body.rootPath : outputPath,
    inputPaths,
    outputPath,
    settings: normalisePipelineSettings(body.settings),
    stages,
    logs: parseTaskLogs(body.logs ?? body.pipelineLogs, stages),
    warnings: Array.isArray(body.warnings) ? body.warnings.map((warning) => localiseUserMessage(String(warning))) : [],
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
    message: typeof body.message === "string" ? localiseUserMessage(body.message) : undefined,
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
    message: typeof body.message === "string" ? localiseUserMessage(body.message) : String(payload ?? "處理紀錄"),
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
  const warnings = Array.isArray(body.warnings) ? body.warnings.map((warning) => localiseUserMessage(String(warning))) : [];
  const itemText = (entry: unknown) => {
    if (typeof entry === "string") return entry;
    if (entry && typeof entry === "object") {
      const record = entry as Record<string, unknown>;
      return String(record.version ?? record.name ?? record.path ?? record.detail ?? "已偵測");
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
  const entryNote = (entry: unknown) => entry && typeof entry === "object" && typeof (entry as Record<string, unknown>).note === "string" ? localiseUserMessage(String((entry as Record<string, unknown>).note)) : "";
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
    if (typeof value === "boolean") return { known: true, available: value, text: value ? "已支援" : "未支援" };
    if (typeof value === "number" && Number.isFinite(value)) return { known: true, available: value !== 0, text: value !== 0 ? "已支援" : "未支援" };
    if (typeof value === "string") {
      const text = value.trim();
      const lower = text.toLowerCase();
      if (!text) return { known: false, available: false, text: "未回報" };
      if (/^(false|no|none|unsupported|unavailable|missing|failed|disabled|off|0)$/.test(lower) || /(not\s+supported|without|unavailable|missing|failed|disabled)/i.test(lower)) {
        return { known: true, available: false, text: "未支援" };
      }
      if (/^(true|yes|supported|available|ready|enabled|on|1)$/.test(lower) || /(cuda|gpu|supported|available|ready|enabled)/i.test(lower)) {
        return { known: true, available: true, text: "已支援" };
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
    return { known: false, available: false, text: "未回報" };
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
      `CUDA build：${colmapCuda.text}`,
      `SIFT 擷取：${featureExtractionGpu.text}`,
      `SIFT 配對：${featureMatchingGpu.text}`,
      `Ceres BA：${mapperBaGpu.known ? mapperBaGpu.available ? "可嘗試（執行期確認 CUDA／cuDSS）" : "僅 CPU" : "未回報"}`,
      globalMapper.known ? `Global Mapper：${globalMapper.text}` : "",
    ].filter(Boolean)
    : ["舊版診斷未回報 COLMAP build；FFmpeg CUDA／VideoToolbox 不代表 COLMAP CUDA"];
  const colmapCudaStatus: DiagnosticStatus = hasColmapCapabilities && colmapCuda.known
    ? gpuAvailable && gpuStagesKnown && gpuStagesAvailable ? "ready" : "warning"
    : "unknown";
  const colmapCudaValue = hasColmapCapabilities && colmapCuda.known
    ? gpuAvailable
      ? gpuStagesKnown && gpuStagesAvailable ? "CUDA 加速可用" : "CUDA 加速部分可用"
      : "未偵測到可用的 CUDA GPU"
    : "CUDA 狀態未確認";
  const capabilityLabels: Record<string, string> = { extract: "影格擷取", mask: "遮罩", align: "對齊" };
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
    osVersion: typeof rawSystemInfo.osVersion === "string" && rawSystemInfo.osVersion.trim() ? rawSystemInfo.osVersion.trim() : "未偵測到",
    architecture: typeof rawSystemInfo.architecture === "string" && rawSystemInfo.architecture.trim()
      ? rawSystemInfo.architecture.trim()
      : typeof body.arch === "string" && body.arch.trim() ? body.arch.trim() : "未偵測到",
    processors: stringList(rawSystemInfo.processors),
    graphicsAdapters: stringList(rawSystemInfo.graphicsAdapters),
  };
  const ffmpegAccelerationValue = ffmpegAccelerators
    .map((entry) => `${entryName(entry) || entryKind(entry) || itemText(entry)}：${available(entry) ? "build 已啟用" : "未支援"}`)
    .join(" · ");
  const colmapReady = Boolean(colmap && available(colmap));
  const colmapWorkflowReady = colmapReady && alignCapability.known && alignCapability.available;
  const ffmpegReady = Boolean(ffmpeg && available(ffmpeg) && ffprobe && available(ffprobe));
  const hardwareAccelerationReady = ffmpegAccelerators.some((entry) => available(entry));
  const items: DiagnosticItem[] = [
    {
      label: "COLMAP",
      value: colmapWorkflowReady ? itemText(colmap) : colmapReady ? "COLMAP 對齊能力未確認" : "未偵測到 COLMAP",
      detail: colmapWorkflowReady ? entryPath(colmap) || "原生雙魚眼相機組對齊流程可用" : colmapReady ? "已找到執行檔，但診斷未確認完整對齊流程" : entryNote(colmap) || "對齊階段會維持待執行",
      details: colmapReady ? [entryPath(colmap) ? `執行檔：${entryPath(colmap)}` : "執行檔：系統 PATH", `對齊流程：${alignCapability.text}`] : undefined,
      status: colmapWorkflowReady ? "ready" : "warning",
    },
    {
      label: COLMAP_CUDA_DIAGNOSTIC_LABEL,
      value: colmapCudaValue,
      detail: colmapCudaAccelerator ? entryNote(colmapCudaAccelerator) || "COLMAP CUDA 能力已完成檢查" : "COLMAP 的 CUDA 能力檢查結果",
      details: colmapCudaDetails,
      status: colmapCudaStatus,
    },
    {
      label: "FFmpeg",
      value: ffmpegReady ? itemText(ffmpeg) : "FFmpeg 工具不完整",
      detail: ffmpegReady ? entryPath(ffmpeg) || "FFmpeg 與 ffprobe 皆可用" : "影格擷取需要 FFmpeg 與 ffprobe",
      details: [
        `FFmpeg：${ffmpeg && available(ffmpeg) ? entryPath(ffmpeg) || itemText(ffmpeg) || "可用" : "未偵測到"}`,
        `ffprobe：${ffprobe && available(ffprobe) ? entryPath(ffprobe) || itemText(ffprobe) || "可用" : "未偵測到"}`,
      ],
      status: ffmpegReady ? "ready" : "warning",
    },
    {
      label: "硬體加速",
      value: ffmpegAccelerators.length ? hardwareAccelerationReady ? "FFmpeg 支援硬體加速" : "FFmpeg 未啟用硬體加速" : "硬體解碼狀態未回報",
      detail: ffmpegAccelerationValue || `${platform} · FFmpeg 未回報硬體解碼能力`,
      details: ffmpegAccelerators.map((entry) => entryNote(entry) || `${entryName(entry) || entryKind(entry) || itemText(entry)}：${available(entry) ? "build 已啟用" : "未支援"}`),
      status: ffmpegAccelerators.length ? hardwareAccelerationReady ? "ready" : "warning" : "unknown",
    },
  ];
  return {
    platform,
    systemInfo,
    summary: typeof body.summary === "string" ? localiseUserMessage(body.summary) : capabilityValue || fallback.summary,
    checkedAt: new Date().toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" }),
    items,
    warnings,
    colmapCapabilities,
    gpuAvailable,
  };
}

async function invokeSafely<T>(command: string, args?: Record<string, unknown>) {
  if (!IS_TAURI_RUNTIME) return null;
  try {
    return await invoke<T>(command, args);
  } catch (error) {
    console.info(`[GS360] ${command}`, error);
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

function SourceThumbnail({ source }: { source: OsvSource }) {
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
    void invoke<ArrayBuffer>("source_preview", { path: source.path })
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

  const alt = `${source.detail} 第一個鏡頭的第一幀預覽`;

  if (!previewUrl) {
    return (
      <div className={`source-thumbnail${failed ? " source-thumbnail--failed" : ""}`} title={failed ? "無法產生第一幀預覽" : undefined}>
        {failed ? <Video aria-hidden="true" /> : <CircleDashed aria-hidden="true" />}
      </div>
    );
  }

  return (
    <HoverCard>
      <HoverCardTrigger
        delay={120}
        closeDelay={80}
        render={<button type="button" className="source-thumbnail source-thumbnail--interactive" aria-label={`預覽 ${alt}`} />}
      >
        <img src={previewUrl} alt={alt} />
      </HoverCardTrigger>
      <HoverCardContent className="source-preview-card" side="right" sideOffset={12} aria-label={`${source.detail} 魚眼快照`}>
        <img className="source-preview-image" src={previewUrl} alt="" />
      </HoverCardContent>
    </HoverCard>
  );
}

function iconForDiagnostic(label: string) {
  if (label.includes("GPU") || label.includes("CUDA") || label.includes("硬體加速")) return Gpu;
  if (label.includes("COLMAP")) return ScanSearch;
  if (label.includes("FFmpeg")) return FileVideoCamera;
  return MonitorCog;
}

function warningAffectsProcessingSpeed(warning: string) {
  return /(CUDA|GPU|CPU|硬體|加速|VideoToolbox|解碼|特徵擷取|配對|Ceres|cuDSS)/i.test(warning);
}

function diagnosticStatusLabel(status: DiagnosticStatus) {
  if (status === "ready") return "可用";
  if (status === "warning") return "需檢查";
  return "未檢查";
}

function redactDiagnosticText(value: string) {
  return value
    .replace(/\/Users\/[^/\\\r\n]+/gi, "/Users/<user>")
    .replace(/\/home\/[^/\\\r\n]+/gi, "/home/<user>")
    .replace(/([A-Z]:\\Users\\)[^\\\r\n]+/gi, "$1<user>");
}

function safeDiagnosticDetail(value: string) {
  const trimmed = value.trim();
  const absolutePath = /^(?:\/|[A-Z]:[\\/]|\\\\)/i;
  if (absolutePath.test(trimmed)) return "已偵測到執行檔（完整路徑已隱藏）";
  return trimmed.replace(
    /^(執行檔|FFmpeg|ffprobe)：\s*(?:\/|[A-Z]:[\\/]|\\\\).+$/i,
    "$1：已偵測（完整路徑已隱藏）",
  );
}

function doctorReportText(doctor: DoctorReport) {
  const lines = [
    "GS360 Studio 診斷資訊",
    `平台：${doctor.platform}`,
    `最後檢查：${doctor.checkedAt}`,
    `摘要：${doctor.summary}`,
    "",
    "系統資訊",
    `- 作業系統：${doctor.systemInfo.osName} ${doctor.systemInfo.osVersion}`,
    `- 架構：${doctor.systemInfo.architecture}`,
    "- 處理器：",
    ...(doctor.systemInfo.processors.length > 0 ? doctor.systemInfo.processors.map((processor) => `  - ${processor}`) : ["  - 未偵測到"]),
    "- 顯示卡：",
    ...(doctor.systemInfo.graphicsAdapters.length > 0 ? doctor.systemInfo.graphicsAdapters.map((adapter) => `  - ${adapter}`) : ["  - 未偵測到"]),
    "",
    "環境項目",
  ];
  doctor.items.forEach((item) => {
    lines.push(`- ${item.label} [${diagnosticStatusLabel(item.status)}]`);
    lines.push(`  結果：${item.value}`);
    lines.push(`  說明：${safeDiagnosticDetail(item.detail)}`);
    item.details?.forEach((detail) => lines.push(`  - ${safeDiagnosticDetail(detail)}`));
  });
  lines.push("", "警告");
  if (doctor.warnings.length > 0) doctor.warnings.forEach((warning) => lines.push(`- ${warning}`));
  else lines.push("- 無");
  return redactDiagnosticText(lines.join("\n"));
}

function stageAction(status: StageStatus) {
  if (status === "running") return "取消";
  if (status === "cancelled") return "繼續";
  if (status === "failed") return "重試";
  if (status === "completed") return "重跑";
  return "執行";
}

function stageStatusLabel(status: StageStatus) {
  if (status === "running") return "執行中";
  if (status === "cancelled") return "已取消";
  if (status === "failed") return "失敗";
  if (status === "completed") return "完成";
  return "待執行";
}

function StageStatusBadge({ status }: { status: StageStatus }) {
  return (
    <Badge data-status={status} variant={status === "failed" ? "destructive" : "outline"}>
      {status === "running"
        ? <LoaderCircle data-icon="inline-start" aria-hidden="true" />
        : <span className={`status-dot status-dot--${status}`} />}
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
  if (value === undefined) return "估算中";
  if (value < 60_000) return `約 ${formatDuration(value)}`;
  const minutes = Math.max(1, Math.round(value / 60_000));
  return `約 ${minutes} 分鐘`;
}

function processingRateLabel(completed: number | undefined, startedAtMs: number | undefined, nowMs: number) {
  if (completed === undefined || completed <= 0 || startedAtMs === undefined) return "估算中";
  const elapsed = Math.max(0, nowMs - startedAtMs);
  if (elapsed < 1000) return "估算中";
  const rate = completed / (elapsed / 1000);
  return `${rate >= 10 ? rate.toFixed(1) : rate.toFixed(2)} 項目/秒`;
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
  const { theme, setTheme } = useTheme();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);
  const [editingTaskId, setEditingTaskId] = useState<string | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
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
  const [doctor, setDoctor] = useState<DoctorReport>(EMPTY_DOCTOR);
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
  const pendingLogsByJobId = useRef<Record<string, TaskLog[]>>({});
  const taskSnapshot = useRef<Task[]>([]);
  const logSequence = useRef(0);
  const doctorRunId = useRef(0);
  const gpuPreferenceTouched = useRef(false);
  const autoPipelineRuns = useRef<Record<string, AutoPipelineRun>>({});
  const pumpAutoPipelineRef = useRef<() => void>(() => undefined);

  const selectedSources = useMemo(() => sourcePaths.map(sourceFromPath), [sourcePaths]);
  const selectedTask = useMemo(() => tasks.find((task) => task.projectId === selectedTaskId), [selectedTaskId, tasks]);
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
  const hardwareDiagnostic = doctor.items.find((item) => item.label === "硬體加速");
  const performanceFallback = gpuDiagnostic?.details?.find((detail) => /(CPU|未支援|未確認|不可用)/i.test(detail))
    || hardwareDiagnostic?.details?.find((detail) => /(CPU|未支援|未確認|不可用)/i.test(detail))
    || "部分 CUDA 或硬體加速能力不可用，相關階段將改用 CPU。";
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
  const queuedTasks = useMemo(() => orderedTasks.filter(taskHasNotStarted), [orderedTasks]);
  const startedTasks = useMemo(() => orderedTasks.filter((task) => !taskHasNotStarted(task)), [orderedTasks]);
  const hasRunningStage = useMemo(() => tasks.some((task) => STAGES.some(({ key }) => task.stages[key].status === "running")), [tasks]);

  useEffect(() => {
    taskSnapshot.current = tasks;
  }, [tasks]);

  useEffect(() => {
    try {
      const path = colmapPath.trim();
      if (path) window.localStorage.setItem(COLMAP_PATH_STORAGE_KEY, path);
      else window.localStorage.removeItem(COLMAP_PATH_STORAGE_KEY);
    } catch (error) {
      console.info("[GS360] COLMAP path preference", error);
    }
  }, [colmapPath]);

  useEffect(() => {
    if (!hasRunningStage) return undefined;
    const interval = window.setInterval(() => setClockMs(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [hasRunningStage]);

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
      message: localiseUserMessage(message),
      timestampMs: Date.now() + logSequence.current / 1000,
    });
  }, [appendTaskLog]);

  const inspectSourcePaths = useCallback(async (paths: string[]) => {
    if (!paths.length) return;
    const result = await invokeSafely<{ kind?: string; sources?: Array<{ path?: string; name?: string; duration?: number; fps?: number; warnings?: string[] }>; project?: { path?: string; status?: string; hasManifest?: boolean }; suggestedOutputPath?: string }>("inspect_paths", { paths });
    if (IS_TAURI_RUNTIME && result?.project && (result.kind === "project" || result.project.status === "partial" || result.project.hasManifest)) {
      const projectPath = result.project.path || paths[0];
      const manifest = manifestFromUnknown(await invokeSafely("load_project", { path: projectPath }));
      if (manifest) {
        setTasks((current) => [manifest, ...current.filter((task) => task.projectId !== manifest.projectId)]);
        setTaskDialogOpen(false);
        setSourcePaths([]);
        setSourceInspection("");
        addTaskMessage(manifest.projectId, `已載入可續作專案 ${manifest.name}`);
        setToast(manifest.warnings.length ? `已載入未完成專案：${manifest.warnings.length} 項警告` : `已載入未完成專案：${manifest.name}`);
        return;
      }
    }
    if (result?.sources?.length) {
      const inspectedPaths = result.sources.flatMap((source) => source.path ? [source.path] : []);
      if (inspectedPaths.length) setSourcePaths(inspectedPaths);
      const valid = result.sources.filter((source) => !source.warnings?.length).length;
      setSourceInspection(`${result.sources.length} 個來源 · ${valid} 個通過檢查`);
    } else if (result?.suggestedOutputPath) {
      setOutputDraft(result.suggestedOutputPath);
      setSourceInspection("已找到來源，可建立新的重建任務");
    } else if (!IS_TAURI_RUNTIME) {
      setSourceInspection(`${paths.length} 個來源 · 瀏覽器預覽`);
    } else {
      setSourceInspection("尚未取得來源檢查結果");
    }
  }, [addTaskMessage]);

  const applySourcePaths = useCallback((paths: string[], openDialogAfter = true) => {
    const actual = paths.filter(Boolean);
    if (!actual.length) return;
    setSourcePaths(actual);
    if (!editingTaskId) {
      setOutputDraft(deriveOutputPath(actual[0]));
      setNameDraft(actual[0].split(/[\\/]/).filter(Boolean).pop()?.replace(/[-_]+/g, " ") || "新重建任務");
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
    gpuPreferenceTouched.current = false;
    setSettingsDraft({
      ...DEFAULT_SETTINGS,
      align: { ...DEFAULT_SETTINGS.align, useGpu: doctor.gpuAvailable !== false },
    });
    setDragOver(false);
    setTaskDialogOpen(true);
  }, [doctor.gpuAvailable]);

  const canChangeQueuedTask = useCallback((task: Task) => {
    const run = autoPipelineRuns.current[task.projectId];
    return taskHasNotStarted(task)
      && !activeJobIds.current[task.projectId]
      && !pendingStageStarts.current[task.projectId]
      && (!run || (!run.stage && !run.jobId));
  }, []);

  const openEditTaskDialog = useCallback((task: Task) => {
    if (!canChangeQueuedTask(task)) {
      setToast("任務已開始，無法再修改");
      return;
    }
    const run = autoPipelineRuns.current[task.projectId];
    if (run) run.paused = true;
    setEditingTaskId(task.projectId);
    setNameDraft(task.name);
    setSourcePaths(task.inputPaths);
    setOutputDraft(task.outputPath);
    setSettingsDraft(normalisePipelineSettings(task.settings));
    setSourceInspection(`${task.inputPaths.length} 個來源`);
    setDragOver(false);
    setTaskDialogOpen(true);
  }, [canChangeQueuedTask]);

  const handleBrowserFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const paths = Array.from(files).map((file) => {
      const candidate = file as File & { path?: string };
      return candidate.path || file.name;
    });
    applySourcePaths(paths);
  }, [applySourcePaths]);

  const openSourcePicker = useCallback(async (mode: "files" | "directories") => {
    if (!IS_TAURI_RUNTIME) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const result = await openDialog(mode === "directories" ? { directory: true, multiple: true } : { directory: false, multiple: true, filters: [{ name: "OSV / 雙魚眼影片", extensions: ["osv", "mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "ts"] }] });
      const paths = result === null ? [] : Array.isArray(result) ? result : [result];
      applySourcePaths(paths);
    } catch (error) {
      console.info("[GS360] picker fallback", error);
      fileInputRef.current?.click();
    }
  }, [applySourcePaths]);

  const openOutputPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("瀏覽器預覽會保留自訂輸出路徑");
      return;
    }
    try {
      const result = await openDialog({ directory: true, multiple: false });
      if (typeof result === "string") setOutputDraft(result);
    } catch (error) {
      console.info("[GS360] output picker", error);
    }
  }, []);

  const openProject = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("瀏覽器預覽不會讀取本機專案資訊");
      return;
    }
    try {
      const result = await openDialog({ directory: true, multiple: false });
      if (typeof result !== "string") return;
      const manifest = manifestFromUnknown(await invokeSafely("load_project", { path: result }));
      if (manifest) {
        setTasks((current) => [manifest, ...current]);
        addTaskMessage(manifest.projectId, `已開啟 ${manifest.name}`);
      } else {
        setToast("找不到可載入的專案資訊");
      }
    } catch (error) {
      console.info("[GS360] load project", error);
      setToast("開啟專案失敗");
    }
  }, [addTaskMessage]);

  const runDoctor = useCallback(async (customColmapPath: string) => {
    const runId = ++doctorRunId.current;
    setDoctorLoading(true);
    const result = await invokeSafely("doctor", { colmapPath: customColmapPath.trim() || null });
    if (runId !== doctorRunId.current) return;
    if (result) {
      const parsed = parseDoctor(result, EMPTY_DOCTOR);
      setDoctor(parsed);
      if (parsed.gpuAvailable === false || (parsed.gpuAvailable === true && !gpuPreferenceTouched.current)) {
        setSettingsDraft((current) => ({
          ...current,
          align: { ...current.align, useGpu: parsed.gpuAvailable === true },
        }));
      }
    }
    else if (!IS_TAURI_RUNTIME) setDoctor({ ...EMPTY_DOCTOR, summary: "瀏覽器預覽未連接本機執行環境" });
    setDoctorLoading(false);
  }, []);

  const copyDoctorReport = useCallback(async () => {
    const report = doctorReportText(doctor);
    try {
      if (IS_TAURI_RUNTIME) await writeClipboardText(report);
      else if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(report);
      else throw new Error("clipboard unavailable");
      setToast("診斷資訊已複製，可直接貼到除錯回報");
    } catch (error) {
      console.info("[GS360] copy diagnostics", error);
      setToast("無法複製診斷資訊，請檢查剪貼簿權限");
    }
  }, [doctor]);

  const initialColmapPath = useRef(colmapPath);
  useEffect(() => { void runDoctor(initialColmapPath.current); }, [runDoctor]);

  const openColmapPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("COLMAP 路徑會由 Windows 本機執行環境使用");
      return;
    }
    try {
      const result = await openDialog({
        directory: false,
        multiple: false,
        filters: [{ name: "COLMAP 啟動程式", extensions: ["bat", "exe", "cmd"] }],
      });
      if (typeof result === "string") {
        setColmapPath(result);
        void runDoctor(result);
      }
    } catch (error) {
      console.info("[GS360] COLMAP picker", error);
    }
  }, [runDoctor]);

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
      } catch (error) { console.info("[GS360] drag-drop", error); }
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
        setToast("無法自動啟動階段，請查看執行環境訊息");
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
      : { status: "running", progress: 0, message: "正在準備工作", jobId: result.jobId, phase: "starting", startedAtMs, finishedAtMs: undefined, durationMs: undefined, completed: undefined, total: undefined, currentItem: undefined });
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
    if (!sourcePaths.length) { setToast("請先選擇至少一個 OSV 或雙魚眼來源"); return; }
    const request = { inputPaths: sourcePaths, outputPath: outputDraft || undefined, name: nameDraft || undefined, settings: { ...settingsDraft } };
    const result = await invokeSafely("create_project", { request });
    const manifest = manifestFromUnknown(result);
    let createdTask: Task | null = null;
    if (manifest) {
      createdTask = manifest;
      const logPayload: LogEventPayload = { level: "info", message: `已建立 ${manifest.name}`, timestampMs: Date.now() };
      setTasks((current) => [{ ...manifest, logs: appendMessageLog(manifest.logs, manifest.projectId, logPayload) }, ...current]);
    } else if (!IS_TAURI_RUNTIME) {
      const preview: Task = { projectId: `preview-${Date.now()}`, name: nameDraft || "瀏覽器預覽任務", rootPath: outputDraft, inputPaths: sourcePaths, outputPath: outputDraft, settings: request.settings, stages: cloneStages({}), logs: [], warnings: ["瀏覽器預覽：尚未連接本機執行環境"], createdAt: new Date().toISOString(), previewOnly: true };
      createdTask = preview;
      const logPayload: LogEventPayload = { level: "info", message: `預覽任務已加入 ${preview.name}`, timestampMs: Date.now() };
      setTasks((current) => [{ ...preview, logs: appendMessageLog(preview.logs, preview.projectId, logPayload) }, ...current]);
    } else {
      setToast("建立任務失敗，請查看執行環境訊息");
      return;
    }
    setTaskDialogOpen(false);
    setSourcePaths([]);
    setSourceInspection("");
    if (createdTask) startAutoPipeline(createdTask);
  }, [nameDraft, outputDraft, settingsDraft, sourcePaths, startAutoPipeline]);

  const saveEditedTask = useCallback(async () => {
    const task = taskSnapshot.current.find((item) => item.projectId === editingTaskId);
    if (!task || !canChangeQueuedTask(task)) {
      setToast("任務已開始，無法再修改");
      return;
    }
    if (!sourcePaths.length) { setToast("請保留至少一個來源"); return; }
    const settings = normalisePipelineSettings(settingsDraft);
    if (task.previewOnly) {
      setTasks((current) => current.map((item) => item.projectId === task.projectId
        ? { ...item, name: nameDraft || item.name, inputPaths: sourcePaths, settings }
        : item));
      setTaskDialogOpen(false);
      setEditingTaskId(null);
      setToast("已更新預覽任務");
      return;
    }
    const result = await invokeSafely("update_queued_project", {
      request: { projectPath: task.rootPath || task.outputPath, name: nameDraft || task.name, inputPaths: sourcePaths, settings },
    });
    const manifest = manifestFromUnknown(result);
    if (!manifest) { setToast("儲存任務修改失敗，請查看執行環境訊息"); return; }
    setTasks((current) => current.map((item) => item.projectId === task.projectId ? { ...manifest, logs: item.logs } : item));
    const run = autoPipelineRuns.current[task.projectId];
    if (run) {
      run.task = { rootPath: manifest.rootPath, outputPath: manifest.outputPath, settings: manifest.settings };
      run.paused = false;
    }
    setTaskDialogOpen(false);
    setEditingTaskId(null);
    setToast("已更新排隊中的任務");
    queueMicrotask(() => pumpAutoPipelineRef.current());
  }, [canChangeQueuedTask, editingTaskId, nameDraft, settingsDraft, sourcePaths]);

  const deleteQueuedTask = useCallback(() => {
    const task = taskSnapshot.current.find((item) => item.projectId === deletingTaskId);
    if (!task || !canChangeQueuedTask(task)) {
      setDeletingTaskId(null);
      setToast("任務已開始，無法刪除");
      return;
    }
    delete autoPipelineRuns.current[task.projectId];
    delete pendingStageStarts.current[task.projectId];
    delete activeJobIds.current[task.projectId];
    setTasks((current) => current.filter((item) => item.projectId !== task.projectId));
    if (selectedTaskId === task.projectId) setSelectedTaskId(null);
    setDeletingTaskId(null);
    setToast("已從佇列移除任務；輸出資料夾仍保留");
    queueMicrotask(() => pumpAutoPipelineRef.current());
  }, [canChangeQueuedTask, deletingTaskId, selectedTaskId]);

  const enqueueQueuedTask = useCallback((task: Task) => {
    if (!taskHasNotStarted(task)) {
      setToast("任務已經開始");
      return;
    }
    startAutoPipeline(task);
    setToast("已將任務加入執行佇列");
  }, [startAutoPipeline]);

  const startStage = useCallback(async (task: Task, stageKey: StageKey, mode: "start" | "resume" | "retry") => {
    if (!IS_TAURI_RUNTIME) { setToast("瀏覽器預覽不會執行後端工作"); return; }
    if (activeJobIds.current[task.projectId] || autoPipelineRuns.current[task.projectId]) {
      setToast("此任務已有處理階段執行中，請稍候");
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
        : { status: "running", progress: task.stages[stageKey].progress, message: "正在準備工作", phase: "starting", jobId: result.jobId, startedAtMs: Date.now(), finishedAtMs: undefined, durationMs: undefined, completed: undefined, total: undefined, currentItem: undefined });
    } else {
      delete pendingStageStarts.current[task.projectId];
      setToast("無法啟動階段，請查看執行環境訊息");
      queueMicrotask(() => pumpAutoPipelineRef.current());
    }
  }, [bindJobToTask, colmapPath, settingsDraft, updateTaskStage]);

  const cancelStage = useCallback(async (task: Task, stageKey: StageKey) => {
    if (!IS_TAURI_RUNTIME) { setToast("瀏覽器預覽不會取消後端工作"); return; }
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
      updateTaskStage(task.projectId, stageKey, { status: "cancelled", message: "已取消，可稍後繼續", jobId: undefined, finishedAtMs, durationMs: taskStageDuration(task.stages[stageKey], finishedAtMs) });
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
            addTaskMessage(targetProjectId, "自動管線已完成影格擷取、遮罩與對齊");
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
    const prerequisite = stagePrerequisiteLabel(task, stageKey);
    if (prerequisite) {
      setToast(`請先完成${prerequisite}`);
      return;
    }
    if (Object.keys(activeJobIds.current).length || Object.keys(pendingStageStarts.current).length) {
      setToast("目前已有處理階段執行中，請稍候");
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
    return (
      <div className="settings-form">
        <FieldGroup>
          <Field>
            <FieldLabel>影格擷取</FieldLabel>
            <FieldContent>
              <Field className="extract-base-fps-field">
                <FieldLabel htmlFor="base-fps">截取影格率（FPS）</FieldLabel>
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
              <Field orientation="horizontal" className="extract-filter-option">
                <Checkbox
                  id="sharpness-filter"
                  checked={settingsDraft.extract.skipBlurry}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, skipBlurry: checked === true } }))}
                />
                <FieldLabel htmlFor="sharpness-filter">清晰度過濾</FieldLabel>
              </Field>
              {settingsDraft.extract.skipBlurry && (
                <Field className="extract-candidate-field">
                  <div className="slider-heading">
                    <FieldLabel id="candidate-fps-label">候選影格率</FieldLabel>
                    <span className="range-label">{candidateMultiplier}× · {candidateFps} FPS</span>
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
                  <div className="range-scale" aria-hidden="true"><span>2×</span><span>10×</span></div>
                  <FieldDescription>以截取影格率的倍率取樣候選，再挑選較清晰的影格。</FieldDescription>
                </Field>
              )}
              <div className="rig-note"><ScanLine /><span><strong>智慧抽幀</strong><small>固定使用 IMU 相對旋轉與低解析畫面變化，減少重複影格</small></span></div>
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>遮罩</FieldLabel>
            <FieldContent>
              <Field orientation="horizontal" className="mask-feature-option">
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
                <FieldLabel htmlFor="mask-yolo">YOLO 物件過濾</FieldLabel>
              </Field>
              {settingsDraft.mask.yoloEnabled && (
                <FieldGroup className="mask-feature-settings">
                  <FieldSet className="mask-object-options">
                    <FieldLegend variant="label">要遮蔽的物件（可複選）</FieldLegend>
                    <FieldGroup data-slot="checkbox-group" className="mask-checkbox-list">
                      {MASK_CLASSES.map((maskClass) => {
                        const checkboxId = `mask-class-${maskClass}`;
                        return (
                          <Field key={maskClass} orientation="horizontal" className="mask-checkbox-option">
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
                            <FieldLabel htmlFor={checkboxId}>{MASK_CLASS_LABELS[maskClass]}</FieldLabel>
                          </Field>
                        );
                      })}
                    </FieldGroup>
                  </FieldSet>
                </FieldGroup>
              )}
              <Field orientation="horizontal" className="mask-feature-option">
                <Checkbox
                  id="mask-sky"
                  checked={settingsDraft.mask.maskSky}
                  onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, maskSky: checked === true } }))}
                />
                <FieldLabel htmlFor="mask-sky">天空過濾</FieldLabel>
              </Field>
              {settingsDraft.mask.maskSky && <FieldDescription>使用 SkySeg 產生天空遮罩。</FieldDescription>}
              {!settingsDraft.mask.yoloEnabled && !settingsDraft.mask.maskSky && (
                <FieldDescription>未啟用遮罩；影格擷取完成後會直接進入對齊。</FieldDescription>
              )}
            </FieldContent>
          </Field>
        <Field>
          <FieldLabel>對齊</FieldLabel>
          <FieldContent>
            <div className="settings-stack">
              <div className="rig-note"><Workflow /><span><strong>穩定重建</strong><small>固定使用跨檔視覺檢索與 incremental mapper</small></span></div>
              <Field orientation="horizontal" className="extract-filter-option" data-disabled={doctor.gpuAvailable === false || undefined}>
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
                  <FieldLabel htmlFor="use-gpu">對齊使用 CUDA 加速</FieldLabel>
                  <FieldDescription>{doctor.gpuAvailable === false ? "目前未偵測到可用的 COLMAP CUDA 加速，因此會使用 CPU。" : "偵測到支援 CUDA 的 NVIDIA GPU 時預設開啟；若執行失敗會自動改用 CPU。"}</FieldDescription>
                </FieldContent>
              </Field>
              <Field data-disabled={doctor.gpuAvailable === false || !settingsDraft.align.useGpu || undefined}>
                <FieldLabel htmlFor="gpu-index">選擇 GPU（進階）</FieldLabel>
                <Input id="gpu-index" className="w-20" type="text" inputMode="numeric" disabled={doctor.gpuAvailable === false || !settingsDraft.align.useGpu} value={settingsDraft.align.gpuIndex ?? DEFAULT_SETTINGS.align.gpuIndex} onChange={(event) => { const value = event.currentTarget.value; setSettingsDraft((current) => ({ ...current, align: { ...current.align, gpuIndex: value } })); }} />
                <FieldDescription>保持 -1 會自動選擇。多張 GPU 可輸入 0,1；部分處理只會使用清單中的第一張。</FieldDescription>
              </Field>
              <div className="rig-note"><Workflow /><span><strong>實體雙鏡頭相機組流程</strong><small>未知外參先建模校正；已校正外參則直接沿用</small></span></div>
            </div>
          </FieldContent>
        </Field>
        </FieldGroup>
      </div>
    );
  };

  return (
    <div className="studio-app">
      <header className="window-bar">
        {!IS_TAURI_RUNTIME && <div className="traffic-lights" aria-hidden="true"><span className="traffic-red" /><span className="traffic-yellow" /><span className="traffic-green" /></div>}
        <span className="window-title">GS360 Studio</span>
        <div className="window-actions">{!IS_TAURI_RUNTIME && <Badge variant="outline" className="runtime-badge">瀏覽器預覽</Badge>}<Button variant="ghost" size="icon-sm" aria-label="開啟設定" onClick={() => setSettingsOpen(true)}><Settings2 /></Button></div>
      </header>

      <main className="studio-main">
        <input ref={fileInputRef} type="file" multiple accept=".osv,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mts,.m2ts,.ts" hidden onChange={(event) => handleBrowserFiles(event.currentTarget.files)} />
        {tasks.length === 0 ? (
          <section className="empty-state" onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
            <div className={`empty-icon ${dragOver ? "is-dragging" : ""}`} aria-hidden="true"><FileStack /></div>
            <h1>尚無任務</h1>
            <p className="empty-description">將 OSV 素材或尚未完成的專案資料夾拖放到這裡，<br />也可以從下方選擇檔案或資料夾。</p>
            <div className="empty-actions"><Button size="lg" onClick={() => void openSourcePicker("files")}><Upload data-icon="inline-start" />選擇檔案</Button><Button size="lg" variant="outline" onClick={() => void openSourcePicker("directories")}><FolderOpen data-icon="inline-start" />選擇資料夾</Button></div>
            <section className="supported-formats" aria-labelledby="supported-formats-title">
              <h2 id="supported-formats-title">目前支援</h2>
              <div className="supported-format-list">
                <article className="supported-format-card"><Film aria-hidden="true" /><span><strong>Osmo 360 原始檔案</strong><small>OSV</small></span></article>
                <article className="supported-format-card"><Folder aria-hidden="true" /><span><strong>專案資料夾</strong><small>繼續未完成的重建任務</small></span></article>
              </div>
            </section>
          </section>
        ) : (
          <section className="tasks-view">
            <header className="content-header"><div><h1>重建任務</h1><p>新增任務後會依序執行影格擷取、遮罩與對齊；各階段仍可獨立取消或重試。</p></div><div className="header-actions"><Button variant="outline" onClick={() => void openProject()}><FolderOpen />開啟專案</Button><Button onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div></header>
            <div className="task-groups">
              {([
                { key: "queued", title: "排隊中", description: "尚未開始，可修改或移除。", items: queuedTasks },
                { key: "started", title: "處理中與已結束", description: "已開始的任務會保留處理階段與紀錄。", items: startedTasks },
              ] as const).filter((group) => group.items.length > 0).map((group) => (
                <section className="task-group" key={group.key}>
                  <div className="task-group-heading"><div><h2>{group.title}</h2><p>{group.description}</p></div><Badge variant="outline">{group.items.length}</Badge></div>
                  <div className="task-list">
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
                return (
                  <article className="task-row" data-queued={queued || undefined} key={task.projectId}>
                    <div className="task-row-top">
                      <div className="task-identity"><span className="task-mark"><FileStack /></span><div><div className="task-name-line"><h2>{task.name}</h2>{queued && <Badge variant="outline">{editableQueued ? "等待執行" : "正在準備"}</Badge>}{task.previewOnly && <Badge variant="outline">預覽</Badge>}</div><p title={task.outputPath}>{task.outputPath || "尚未指定輸出"}</p></div></div>
                      <div className="task-row-actions">
                        {waitingForEnqueue && <Button size="sm" onClick={() => enqueueQueuedTask(task)}><Play data-icon="inline-start" />加入佇列</Button>}
                        {editableQueued && <><Button variant="outline" size="sm" onClick={() => openEditTaskDialog(task)}><Pencil data-icon="inline-start" />修改</Button><Button variant="ghost" size="sm" className="task-delete-button" onClick={() => setDeletingTaskId(task.projectId)}><Trash2 data-icon="inline-start" />刪除</Button></>}
                        <Button variant="ghost" size="icon-sm" aria-label={`查看 ${task.name} 的詳細資料`} aria-haspopup="dialog" aria-expanded={selectedTaskId === task.projectId} onClick={() => setSelectedTaskId(task.projectId)}><Info /></Button>
                      </div>
                    </div>
                    {queued ? <div className="queued-task-summary"><span>佇列會依建立順序自動執行</span><small>{task.inputPaths.length} 個來源</small></div> : <><div className="task-progress-block">
                      <div className="task-progress-summary"><span title="三個處理階段採等權平均">總進度（階段平均）</span><small>{taskProgressSummary(task)}</small><strong>{overall}%</strong></div>
                      <Progress value={overall} aria-label={`${task.name} 整體進度`}><ProgressValue /></Progress>
                      <div className="task-live-summary">
                        <span><strong>目前階段：{currentStageDefinition.label}</strong><small>{currentStage.phase ? phaseLabel(currentStage.phase) : stageStatusLabel(currentStage.status)}</small></span>
                        <dl>
                          <div><dt>處理量</dt><dd>{currentCount || "尚未回報"}</dd></div>
                          <div><dt>已執行</dt><dd>{currentElapsed !== undefined ? formatDuration(currentElapsed) : "尚未開始"}</dd></div>
                          <div><dt>預估剩餘</dt><dd>{currentStage.status === "running" ? formatEta(currentEta) : "—"}</dd></div>
                        </dl>
                      </div>
                    </div>
                    <div className="stage-row-list" role="list" aria-label="重建處理流程">
                      {STAGES.map((stage, stageIndex) => {
                        const current = task.stages[stage.key];
                        const stageProgress = Math.round(current.progress);
                        const Icon = stage.icon;
                        const action = stageActionState(task, stage.key, hasRunningStage);
                        return (
                          <div className="task-stage" data-status={current.status} key={stage.key} role="listitem" aria-label={`第 ${stageIndex + 1} / ${STAGES.length} 階段：${stage.label}`}>
                            <span className="task-stage-step" aria-hidden="true"><span>{current.status === "completed" ? <CheckCircle2 /> : stageIndex + 1}</span></span>
                            <div className="task-stage-label">
                              <Icon />
                              <div className="task-stage-copy">
                                <strong>{stage.label}</strong>
                                <small>{action.prerequisite ? `等待${action.prerequisite}完成` : current.message || stage.description}</small>
                                {current.status === "running" && (
                                  <div className={`task-stage-progress${stageProgress <= 0 ? " is-waiting" : ""}`}>
                                    <Progress value={stageProgress} aria-label={`${stage.label}進度`}><ProgressValue /></Progress>
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
                  </div>
                </section>
              ))}
            </div>
          </section>
        )}
      </main>

      {toast && <div className="toast" role="status"><Info /><span>{toast}</span><Button variant="ghost" size="icon-xs" onClick={() => setToast(null)} aria-label="關閉通知"><X /></Button></div>}

      <Dialog open={taskDialogOpen} onOpenChange={(open) => {
        setTaskDialogOpen(open);
        if (!open && editingTaskId) {
          const run = autoPipelineRuns.current[editingTaskId];
          if (run) run.paused = false;
          setEditingTaskId(null);
          queueMicrotask(() => pumpAutoPipelineRef.current());
        }
      }}>
        <DialogContent className="task-dialog" showCloseButton>
          <DialogHeader><DialogTitle>{editingTaskId ? "修改排隊任務" : "新增重建任務"}</DialogTitle><DialogDescription>{editingTaskId ? "可在開始前調整任務名稱、來源與處理設定。" : "選擇多組 OSV 或雙魚眼素材，所有來源都會保存在同一份專案資訊中。"}</DialogDescription></DialogHeader>
          <div className="dialog-scroll">
            <div className="dialog-columns">
              <FieldGroup className="dialog-source-column">
                <Field><FieldLabel htmlFor="task-name">任務名稱</FieldLabel><FieldContent><Input id="task-name" value={nameDraft} placeholder="例如：山區路線／2026-08" onChange={(event) => setNameDraft(event.currentTarget.value)} /></FieldContent></Field>
                <Field><FieldLabel>來源</FieldLabel><FieldContent><div className={`source-drop ${dragOver ? "is-dragging" : ""}`} onDragOver={(event) => event.preventDefault()} onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}><FileStack /><span>拖放 OSV 或尚未完成的專案資料夾</span><Button type="button" variant="outline" size="sm" onClick={() => void openSourcePicker("files")}>選擇來源</Button></div>{selectedSources.length > 0 && <div className="source-list">{selectedSources.map((source) => <div className="source-item" key={source.id}><SourceThumbnail source={source} /><span><strong>{source.label}</strong><small>{source.detail}</small></span><Button type="button" variant="ghost" size="icon-xs" aria-label={`移除 ${source.label}`} onClick={() => setSourcePaths((current) => current.filter((path) => path !== source.path))}><X /></Button></div>)}</div>}<p className="inspection-note">{sourceInspection || "可選擇多個檔案，或直接拖入尚未完成的專案資料夾。"}</p></FieldContent></Field>
                <Field><FieldLabel htmlFor="output-path">輸出資料夾</FieldLabel><FieldContent><div className="input-with-button"><Input id="output-path" value={outputDraft} disabled={Boolean(editingTaskId)} placeholder="預設與第一個來源並列：colmap-檔案名稱" onChange={(event) => setOutputDraft(event.currentTarget.value)} />{!editingTaskId && <Button type="button" variant="outline" size="sm" onClick={() => void openOutputPicker()}>另選</Button>}</div><FieldDescription>{editingTaskId ? "專案已建立，為避免移動既有資料，輸出資料夾不能修改。" : "建立後會在輸出資料夾保存專案資訊，之後可從中斷處繼續。"}</FieldDescription></FieldContent></Field>
              </FieldGroup>
              {renderSettingsFields()}
            </div>
          </div>
          <DialogFooter><DialogClose render={<Button variant="ghost" />}>取消</DialogClose><Button onClick={() => void (editingTaskId ? saveEditedTask() : createTask())} disabled={!sourcePaths.length}>{editingTaskId ? <Pencil /> : <Plus />}{editingTaskId ? "儲存修改" : "新增任務"}</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(deletingTaskId)} onOpenChange={(open) => { if (!open) setDeletingTaskId(null); }}>
        <DialogContent showCloseButton={false}>
          <DialogHeader><DialogTitle>從佇列移除任務？</DialogTitle><DialogDescription>只會移除任務與排隊狀態，不會刪除已建立的輸出資料夾。</DialogDescription></DialogHeader>
          <DialogFooter><DialogClose render={<Button variant="ghost" />}>取消</DialogClose><Button variant="destructive" onClick={deleteQueuedTask}><Trash2 />移除任務</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      {selectedTask && (
        <Sheet open onOpenChange={(open) => { if (!open) setSelectedTaskId(null); }}>
          <SheetContent className="task-detail-sheet" side="right">
            <SheetHeader>
              <SheetTitle>{selectedTask.name}</SheetTitle>
              <SheetDescription>查看目前工作、處理指標與完整處理紀錄；效能監測尚未接通時會明確標示。</SheetDescription>
            </SheetHeader>
            <div className="task-detail-scroll">
              {selectedStage && selectedStageDefinition && <section className="task-detail-overview">
                <div className="task-detail-current-heading">
                  <span><small>目前工作</small><strong>{selectedStageDefinition.label}</strong></span>
                  <StageStatusBadge status={selectedStage.status} />
                </div>
                <div className="task-detail-current-message">
                  <strong>{phaseLabel(selectedStage.phase)}</strong>
                  <p>{selectedStage.message || selectedStageDefinition.description}</p>
                  {selectedStage.currentItem && <small>目前項目：{selectedStage.currentItem}</small>}
                </div>
                {(selectedStage.status === "running" || selectedStage.progress > 0) && <div className="task-detail-current-progress">
                  <div><span>{selectedStageDefinition.label}進度</span><strong>{Math.round(selectedStage.progress)}%</strong></div>
                  <Progress value={selectedStage.progress} aria-label={`${selectedStageDefinition.label}進度`}><ProgressValue /></Progress>
                </div>}
                <dl className="task-detail-metrics">
                  <div><dt>處理量</dt><dd>{logCountLabel(selectedStage.completed, selectedStage.total) || "尚未回報"}</dd></div>
                  <div><dt>已執行</dt><dd>{formatDuration(taskStageDuration(selectedStage, clockMs))}</dd></div>
                  <div><dt>預估剩餘</dt><dd>{selectedStage.status === "running" ? formatEta(estimatedRemainingMs(selectedStage, clockMs)) : "—"}</dd></div>
                  <div><dt>速度（估算）</dt><dd>{selectedStage.status === "running" ? processingRateLabel(selectedActiveProgressLog?.completed, selectedActiveProgressLog?.startedAtMs, clockMs) : "—"}</dd></div>
                  <div><dt><Cpu />CPU</dt><dd>尚未回報</dd></div>
                  <div><dt><Gauge />GPU</dt><dd>尚未回報</dd></div>
                  <div><dt><MemoryStick />記憶體</dt><dd>尚未回報</dd></div>
                  <div className="task-detail-metric-output"><dt>輸出</dt><dd title={selectedTask.outputPath}>{selectedTask.outputPath || "尚未指定"}</dd></div>
                </dl>
              </section>}

              <Accordion>
                <AccordionItem className="task-detail-stage-details" value="pipeline-details">
                  <AccordionTrigger>查看 Pipeline 詳細資訊</AccordionTrigger>
                  <AccordionContent><div className="task-detail-stages">
                  {STAGES.map((stage) => { const current = selectedTask.stages[stage.key]; const Icon = stage.icon; const action = stageActionState(selectedTask, stage.key, hasRunningStage); return (
                    <div className="task-detail-stage" key={stage.key}>
                      <div className="task-detail-stage-main"><Icon /><span><strong>{stage.label}</strong><small>{action.prerequisite ? `等待${action.prerequisite}完成` : current.phase ? phaseLabel(current.phase) : current.message || stage.description}</small></span><StageStatusBadge status={current.status} /></div>
                      <div className="task-detail-stage-footer">
                        <div className="task-detail-stage-time"><span><Clock3 />{taskStageDuration(current, clockMs) !== undefined ? `耗時 ${formatDuration(taskStageDuration(current, clockMs))}` : "尚未開始"}</span></div>
                        <Button variant={current.status === "running" ? "destructive" : "outline"} size="sm" disabled={current.status !== "running" && action.blocked} onClick={() => handleStageAction(selectedTask, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{action.label}</Button>
                      </div>
                    </div>
                  ); })}
                  </div></AccordionContent>
                </AccordionItem>
              </Accordion>

              <section className="task-detail-section">
                <div className="task-detail-section-title"><h2>處理紀錄</h2><span>{selectedTaskLogs.length} 筆</span></div>
                {selectedTaskLogs.length > 0 ? <ol className="task-detail-log-list">{selectedTaskLogs.map((log) => {
                  const count = logCountLabel(log.completed, log.total);
                  return <li className={`task-detail-log task-detail-log--${log.level}`} key={log.id}>
                    <span className="task-detail-log-marker" aria-hidden="true" />
                    <div className="task-detail-log-main"><div className="task-detail-log-heading"><span>{formatTimestamp(log.timestampMs, true)}</span><strong>{taskStageLabel(log.stage)}{log.phase ? ` · ${phaseLabel(log.phase)}` : ""}</strong></div><p>{log.message}</p><div className="task-detail-log-meta">{count && <span>{count}</span>}{log.currentItem && <span>{log.currentItem}</span>}{log.durationMs !== undefined && <span>耗時 {formatDuration(log.durationMs)}</span>}</div></div>
                  </li>;
                })}</ol> : <p className="task-detail-empty">尚無處理紀錄；開始執行後會在這裡顯示每個階段與目前位置。</p>}
              </section>

              <section className="task-detail-section">
                <div className="task-detail-section-title"><h2>來源</h2><span>{selectedTask.inputPaths.length} 個檔案</span></div>
                {selectedTask.inputPaths.length > 0 ? <div className="task-detail-sources">{selectedTask.inputPaths.map((path, index) => <div key={`${index}-${path}`} title={path}><Video /><span>{path}</span></div>)}</div> : <p className="task-detail-empty">此任務沒有記錄來源檔案。</p>}
              </section>

              {selectedTask.warnings.length > 0 && (
                <section className="task-detail-section">
                  <div className="task-detail-section-title"><h2>警告</h2><Badge variant="destructive">{selectedTask.warnings.length}</Badge></div>
                  <div className="task-detail-warnings">{selectedTask.warnings.map((warning, index) => <div key={`${index}-${warning}`}><AlertTriangle /><span>{warning}</span></div>)}</div>
                </section>
              )}
            </div>
            <SheetFooter>{selectedRunningStageDefinition && <Button variant="destructive" onClick={() => handleStageAction(selectedTask, selectedRunningStageDefinition.key)}><Square data-icon="inline-start" />取消整個任務</Button>}<Button variant="outline" onClick={() => setSelectedTaskId(null)}>關閉</Button></SheetFooter>
          </SheetContent>
        </Sheet>
      )}

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent className="settings-sheet" side="right">
          <SheetHeader><SheetTitle>設定</SheetTitle><SheetDescription>調整介面主題，並檢查 COLMAP、CUDA、FFmpeg 與硬體加速能力。</SheetDescription></SheetHeader>
          <div className="settings-sheet-scroll">
            <section className="settings-section">
              <FieldSet className="appearance-fieldset">
                <FieldLegend variant="label">介面主題</FieldLegend>
                <FieldDescription>選擇亮色、暗色，或自動跟隨系統外觀。</FieldDescription>
                <ToggleGroup
                  className="theme-toggle-group"
                  variant="outline"
                  size="sm"
                  spacing={0}
                  value={[theme]}
                  onValueChange={(values) => {
                    const nextTheme = values[0] as Theme | undefined;
                    if (nextTheme) setTheme(nextTheme);
                  }}
                  aria-label="介面主題"
                >
                  <ToggleGroupItem value="system">跟隨系統</ToggleGroupItem>
                  <ToggleGroupItem value="light">亮色</ToggleGroupItem>
                  <ToggleGroupItem value="dark">暗色</ToggleGroupItem>
                </ToggleGroup>
              </FieldSet>
            </section>
            <section className="settings-section">
              <div className="settings-section-heading">
                <div className="settings-section-title"><h2>執行環境</h2><span>最後檢查：{doctor.checkedAt}</span></div>
                <div className="settings-section-actions" role="group" aria-label="診斷操作">
                  <Button type="button" variant="outline" size="sm" disabled={doctorLoading || doctor.checkedAt === "尚未檢查"} onClick={() => void copyDoctorReport()}><Copy data-icon="inline-start" />複製診斷資訊</Button>
                  <Button type="button" size="sm" className={doctorLoading ? "is-spinning" : ""} disabled={doctorLoading} onClick={() => void runDoctor(colmapPath)}><RefreshCw data-icon="inline-start" />{doctorLoading ? "正在檢查" : "重新檢查環境"}</Button>
                </div>
              </div>
              <div className="environment-alert-stack">
                <Alert data-status={doctorEssentialReady ? "ready" : "warning"} role={doctorEssentialReady ? "status" : "alert"}>
                  {doctorEssentialReady ? <CheckCircle2 /> : <AlertTriangle />}
                  <AlertTitle>{doctorEssentialReady ? "所有必要功能皆可使用" : "有必要功能需要處理"}</AlertTitle>
                  <AlertDescription>{doctorEssentialReady ? "基本重建流程可以執行。CUDA 與硬體加速屬於選用能力；不可用時仍能處理，但會降低影格擷取、特徵配對與重建速度。" : "缺少必要工具會阻止部分處理階段，請先處理下方標示為「需檢查」的項目。"}</AlertDescription>
                </Alert>
                {performanceStatus !== "ready" && <Alert data-status={performanceStatus === "warning" ? "performance-warning" : "unknown"} role={performanceStatus === "warning" ? "alert" : "status"}>
                  <Gauge />
                  <AlertTitle>{performanceStatus === "warning" ? "效能會受到影響" : "尚未確認加速能力"}</AlertTitle>
                  <AlertDescription>
                    {performanceWarnings.length > 0
                      ? performanceWarnings.map((warning) => <p key={warning}>{warning}</p>)
                      : performanceStatus === "warning"
                        ? <p>{performanceFallback}</p>
                        : <p>完成環境檢查後，才能確認目前會使用 CUDA、硬體解碼或 CPU。</p>}
                    {performanceStatus === "warning" && <p>改用 CPU 的階段仍可執行，但處理時間可能明顯增加。</p>}
                  </AlertDescription>
                </Alert>}
              </div>
              {isWindowsPlatform && <Field><FieldLabel htmlFor="colmap-path">COLMAP 執行檔</FieldLabel><FieldContent><div className="input-with-button"><Input id="colmap-path" value={colmapPath} placeholder="留白時從 PATH 自動偵測" onChange={(event) => setColmapPath(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openColmapPicker()}>變更路徑</Button></div><FieldDescription>Windows 官方免安裝版請選根目錄的 COLMAP.bat；也可指定自行編譯的 colmap.exe。</FieldDescription></FieldContent></Field>}
              <div className="doctor-summary"><MonitorCog /><span><strong>{doctor.platform}</strong><small>{doctor.summary}</small></span></div>
              <div className="doctor-list">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <article className="doctor-row" key={item.label} data-status={item.status}><Icon /><div className="doctor-row-content"><div className="doctor-row-heading"><span><small>{item.label}</small><strong>{item.value}</strong></span><Badge variant={item.status === "warning" ? "destructive" : "outline"}>{item.status === "ready" ? "可用" : item.status === "warning" ? "需檢查" : "未檢查"}</Badge></div><p>{item.detail}</p>{item.details && item.details.length > 0 && <Accordion className="doctor-details"><AccordionItem value={`${item.label}-details`}><AccordionTrigger>查看詳細資料</AccordionTrigger><AccordionContent><ul>{item.details.map((detail) => <li key={detail}>{detail}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}</div></article>; })}</div>
              {generalDoctorWarnings.length > 0 && <Alert variant="destructive"><AlertTriangle /><AlertTitle>需要處理</AlertTitle><AlertDescription>{generalDoctorWarnings.map((warning) => <p key={warning}>{warning}</p>)}</AlertDescription></Alert>}
            </section>
          </div>
          <SheetFooter><Button variant="outline" onClick={() => setSettingsOpen(false)}>關閉</Button></SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default App;

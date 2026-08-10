import {
  AlertTriangle,
  CircleDashed,
  Clock3,
  Cpu,
  FileStack,
  FolderOpen,
  HardDrive,
  Info,
  MonitorCog,
  MoreHorizontal,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  ScanLine,
  Settings2,
  Square,
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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
  extract: { baseFps: number; denseFps: number; skipBlurry: boolean };
  mask: { yoloEnabled: boolean; classes: string[]; maskSky: boolean; confidence: number; confidenceVersion: number; modelDir: string };
  align: { useGpu: boolean; gpuIndex: string };
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
  settings: Record<string, unknown>;
  stages: Record<StageKey, StageState>;
  logs: TaskLog[];
  warnings: string[];
  updatedAt?: string;
}

interface Task extends ProjectManifest {
  previewOnly?: boolean;
}

interface DiagnosticItem {
  label: string;
  value: string;
  detail: string;
  status: DiagnosticStatus;
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
  summary: string;
  checkedAt: string;
  items: DiagnosticItem[];
  warnings: string[];
  colmapCapabilities?: Record<string, unknown>;
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
  extract: { baseFps: 3, denseFps: 12, skipBlurry: true },
  mask: { yoloEnabled: true, classes: ["person", "bicycle", "car", "motorcycle", "bus", "truck"], maskSky: true, confidence: 25, confidenceVersion: 2, modelDir: "" },
  align: { useGpu: false, gpuIndex: "-1" },
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

function normalisePipelineSettings(value: unknown): Record<string, unknown> {
  const source = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const mask = source.mask && typeof source.mask === "object" ? source.mask as Record<string, unknown> : {};
  const classes = Array.isArray(mask.classes) ? mask.classes.filter((item): item is string => typeof item === "string" && Boolean(item.trim())) : [];
  const yoloEnabled = (typeof mask.yoloEnabled === "boolean" ? mask.yoloEnabled : classes.length > 0) && classes.length > 0;
  const align = source.align && typeof source.align === "object" ? source.align as Record<string, unknown> : {};
  const rawGpuIndex = align.gpuIndex;
  const gpuIndex = typeof rawGpuIndex === "string"
    ? rawGpuIndex
    : typeof rawGpuIndex === "number" && Number.isFinite(rawGpuIndex)
      ? String(rawGpuIndex)
      : DEFAULT_SETTINGS.align.gpuIndex;
  return {
    ...source,
    mask: { ...mask, classes, yoloEnabled },
    align: { ...align, gpuIndex },
  };
}

const EMPTY_DOCTOR: DoctorReport = {
  platform: "尚未檢查平台",
  summary: "執行環境診斷以確認可用能力",
  checkedAt: "尚未檢查",
  items: [
    { label: "COLMAP CUDA", value: "尚未檢查", detail: "不預設 COLMAP build 或 FFmpeg／VideoToolbox 加速能力", status: "unknown" },
    { label: "FFmpeg", value: "尚未檢查", detail: "確認系統 PATH 中的 FFmpeg", status: "unknown" },
    { label: "執行環境", value: "尚未檢查", detail: "確認作業系統與執行環境", status: "unknown" },
    { label: "儲存空間", value: "尚未檢查", detail: "確認輸出磁碟可用空間", status: "unknown" },
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

function formatUpdatedAt(value?: string) {
  if (!value) return "尚無更新時間";
  const parsed = timestampMs(value);
  if (!parsed) return value;
  return new Date(parsed).toLocaleString("zh-TW", { dateStyle: "medium", timeStyle: "short" });
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

function taskProgressSummary(task: Task) {
  const runningIndex = STAGES.findIndex(({ key }) => task.stages[key].status === "running");
  if (runningIndex >= 0) return `第 ${runningIndex + 1} / ${STAGES.length} 階段 · ${STAGES[runningIndex].label}`;
  const interruptedIndex = STAGES.findIndex(({ key }) => ["failed", "cancelled"].includes(task.stages[key].status));
  if (interruptedIndex >= 0) return `停在第 ${interruptedIndex + 1} / ${STAGES.length} 階段 · ${STAGES[interruptedIndex].label}`;
  const nextIndex = STAGES.findIndex(({ key }) => task.stages[key].status !== "completed");
  return nextIndex >= 0 ? `等待第 ${nextIndex + 1} / ${STAGES.length} 階段 · ${STAGES[nextIndex].label}` : `${STAGES.length} / ${STAGES.length} 階段完成`;
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
    updatedAt: typeof body.updatedAt === "string" ? body.updatedAt : undefined,
  };
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
  const colmapCudaDetail = hasColmapCapabilities
    ? [
      `CUDA build：${colmapCuda.text}`,
      `SIFT 擷取：${featureExtractionGpu.text}`,
      `SIFT 配對：${featureMatchingGpu.text}`,
      `Ceres BA：${mapperBaGpu.known ? mapperBaGpu.available ? "可嘗試（執行期確認 CUDA／cuDSS）" : "僅 CPU" : "未回報"}`,
      globalMapper.known ? `Global Mapper：${globalMapper.text}` : "",
    ].filter(Boolean).join(" · ")
    : "舊版診斷未回報 COLMAP build；FFmpeg CUDA／VideoToolbox 不代表 COLMAP CUDA";
  const colmapCudaStatus: DiagnosticStatus = hasColmapCapabilities && colmapCuda.known
    ? colmapCuda.available && gpuStagesKnown && gpuStagesAvailable ? "ready" : "warning"
    : "unknown";
  const colmapCudaValue = hasColmapCapabilities && colmapCuda.known
    ? colmapCuda.available
      ? gpuStagesKnown && gpuStagesAvailable ? "完整支援" : "部分支援"
      : "未支援"
    : "未確認";
  const capabilityLabels: Record<string, string> = { extract: "影格擷取", mask: "遮罩", align: "對齊" };
  const capabilityValue = body.capabilities && typeof body.capabilities === "object" ? Object.entries(body.capabilities as Record<string, unknown>).filter(([, state]) => Boolean(state)).map(([key]) => capabilityLabels[key] ?? key).join(" · ") : "";
  const platform = platformLabel(typeof body.platform === "string" ? body.platform : typeof body.os === "string" ? body.os : fallback.platform);
  const ffmpegAccelerationValue = ffmpegAccelerators
    .map((entry) => `${entryName(entry) || entryKind(entry) || itemText(entry)}：${available(entry) ? "可用" : "未支援"}`)
    .join(" · ");
  const items: DiagnosticItem[] = [
    { label: "COLMAP CUDA", value: colmapCudaValue, detail: colmapCudaDetail, status: colmapCudaStatus },
    { label: "FFmpeg", value: ffmpeg && available(ffmpeg) ? itemText(ffmpeg) : "未偵測到", detail: ffmpeg && available(ffmpeg) ? ["系統 PATH 可用", ffmpegAccelerationValue ? `FFmpeg／VideoToolbox：${ffmpegAccelerationValue}` : ""].filter(Boolean).join(" · ") : "請安裝或加入 PATH", status: ffmpeg && available(ffmpeg) ? "ready" : "warning" },
    { label: "COLMAP", value: colmap && available(colmap) ? itemText(colmap) : "未偵測到", detail: colmap && available(colmap) ? entryPath(colmap) || "可執行原生雙魚眼相機組對齊" : entryNote(colmap) || "對齊階段會維持待執行", status: colmap && available(colmap) ? "ready" : "warning" },
    { label: "執行環境", value: platform, detail: typeof body.arch === "string" ? body.arch : "Tauri 執行環境", status: "ready" },
  ];
  return { platform, summary: typeof body.summary === "string" ? localiseUserMessage(body.summary) : capabilityValue || fallback.summary, checkedAt: new Date().toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" }), items, warnings, colmapCapabilities };
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

function iconForDiagnostic(label: string) {
  if (label.includes("GPU") || label.includes("CUDA")) return Cpu;
  if (label.includes("FFmpeg")) return Video;
  if (label.includes("儲存空間")) return HardDrive;
  return MonitorCog;
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
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);
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
  const pendingStageStarts = useRef<Record<string, StageKey>>({});
  const pendingLogsByJobId = useRef<Record<string, TaskLog[]>>({});
  const taskSnapshot = useRef<Task[]>([]);
  const logSequence = useRef(0);
  const doctorRunId = useRef(0);
  const autoPipelineRuns = useRef<Record<string, AutoPipelineRun>>({});

  const selectedSources = useMemo(() => sourcePaths.map(sourceFromPath), [sourcePaths]);
  const selectedTask = useMemo(() => tasks.find((task) => task.projectId === selectedTaskId), [selectedTaskId, tasks]);
  const selectedTaskLogs = useMemo(() => selectedTask ? selectedTask.logs.slice().sort((left, right) => right.timestampMs - left.timestampMs) : [], [selectedTask]);
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
    const result = await invokeSafely<{ kind?: string; sources?: Array<{ name?: string; duration?: number; fps?: number; warnings?: string[] }>; project?: { path?: string; status?: string; hasManifest?: boolean }; suggestedOutputPath?: string }>("inspect_paths", { paths });
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
    setOutputDraft(deriveOutputPath(actual[0]));
    setNameDraft(actual[0].split(/[\\/]/).filter(Boolean).pop()?.replace(/[-_]+/g, " ") || "新重建任務");
    if (openDialogAfter) setTaskDialogOpen(true);
    void inspectSourcePaths(actual);
  }, [inspectSourcePaths]);

  const openNewTaskDialog = useCallback(() => {
    setNameDraft("");
    setSourcePaths([]);
    setOutputDraft("");
    setSourceInspection("");
    setSettingsDraft(DEFAULT_SETTINGS);
    setDragOver(false);
    setTaskDialogOpen(true);
  }, []);

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

  const openModelPicker = useCallback(async () => {
    if (!IS_TAURI_RUNTIME) {
      setToast("模型資料夾會由本機執行環境使用");
      return;
    }
    try {
      const result = await openDialog({ directory: true, multiple: false });
      if (typeof result === "string") {
        setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, modelDir: result } }));
      }
    } catch (error) {
      console.info("[GS360] model picker", error);
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
    if (result) setDoctor(parseDoctor(result, EMPTY_DOCTOR));
    else if (!IS_TAURI_RUNTIME) setDoctor({ ...EMPTY_DOCTOR, summary: "瀏覽器預覽未連接本機執行環境" });
    setDoctorLoading(false);
  }, []);

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

  const startAutoPipeline = useCallback((task: Task) => {
    if (!IS_TAURI_RUNTIME || task.previewOnly) return;
    if (autoPipelineRuns.current[task.projectId] || activeJobIds.current[task.projectId]) return;
    const firstStage = STAGES.find(({ key }) => task.stages[key].status !== "completed");
    if (!firstStage) return;
    autoPipelineRuns.current[task.projectId] = {
      task: { rootPath: task.rootPath, outputPath: task.outputPath, settings: normalisePipelineSettings(task.settings) },
      colmapPath: colmapPath.trim(),
    };
    void startAutoStage(task.projectId, firstStage.key);
  }, [colmapPath, startAutoStage]);

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
      const preview: Task = { projectId: `preview-${Date.now()}`, name: nameDraft || "瀏覽器預覽任務", rootPath: outputDraft, inputPaths: sourcePaths, outputPath: outputDraft, settings: request.settings, stages: cloneStages({}), logs: [], warnings: ["瀏覽器預覽：尚未連接本機執行環境"], previewOnly: true };
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
    }
  }, [bindJobToTask, colmapPath, settingsDraft, updateTaskStage]);

  const cancelStage = useCallback(async (task: Task, stageKey: StageKey) => {
    if (!IS_TAURI_RUNTIME) { setToast("瀏覽器預覽不會取消後端工作"); return; }
    const autoRun = autoPipelineRuns.current[task.projectId];
    if (autoRun?.stage === stageKey) delete autoPipelineRuns.current[task.projectId];
    delete pendingStageStarts.current[task.projectId];
    const jobId = task.stages[stageKey].jobId || activeJobIds.current[task.projectId];
    if (!jobId) return;
    const cancelled = await invokeSafely<boolean>("cancel_job", { jobId });
    if (cancelled === true) {
      if (activeJobIds.current[task.projectId] === jobId) delete activeJobIds.current[task.projectId];
      const finishedAtMs = Date.now();
      updateTaskStage(task.projectId, stageKey, { status: "cancelled", message: "已取消，可稍後繼續", finishedAtMs, durationMs: taskStageDuration(task.stages[stageKey], finishedAtMs) });
    }
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
          }
          if (!payload.done) return;
          delete pendingStageStarts.current[targetProjectId];
          if (payload.jobId && activeJobIds.current[targetProjectId] === payload.jobId) delete activeJobIds.current[targetProjectId];
          const run = autoPipelineRuns.current[targetProjectId];
          if (!run || run.stage !== stageKey || (run.jobId && run.jobId !== payload.jobId)) return;
          run.jobId = undefined;
          if ((payload.status ?? "completed") !== "completed") {
            delete autoPipelineRuns.current[targetProjectId];
            return;
          }
          const currentIndex = STAGES.findIndex(({ key }) => key === stageKey);
          const nextStage = currentIndex >= 0 ? STAGES[currentIndex + 1] : undefined;
          if (!nextStage) {
            delete autoPipelineRuns.current[targetProjectId];
            addTaskMessage(targetProjectId, "自動管線已完成影格擷取、遮罩與對齊");
            return;
          }
          run.stage = undefined;
          void startAutoStage(targetProjectId, nextStage.key);
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
    if (status === "running") void cancelStage(task, stageKey);
    else void startStage(task, stageKey, status === "cancelled" ? "resume" : status === "failed" || status === "completed" ? "retry" : "start");
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
                  <Field className="mask-confidence-field">
                    <div className="slider-heading">
                      <FieldLabel htmlFor="mask-confidence">YOLO 信心度</FieldLabel>
                      <span className="range-label">{settingsDraft.mask.confidence}%</span>
                    </div>
                    <input id="mask-confidence" className="range-input" type="range" min={10} max={98} value={settingsDraft.mask.confidence} onChange={(event) => { const value = Number(event.currentTarget.value); setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, confidence: value } })); }} />
                  </Field>
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
              {(settingsDraft.mask.yoloEnabled || settingsDraft.mask.maskSky) ? (
                <div className="input-with-button model-dir-input"><Input value={settingsDraft.mask.modelDir} placeholder="模型資料夾（未指定時自動探索）" onChange={(event) => { const value = event.currentTarget.value; setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, modelDir: value } })); }} /><Button type="button" variant="outline" size="sm" onClick={() => void openModelPicker()}>選擇</Button></div>
              ) : (
                <FieldDescription>未啟用遮罩；影格擷取完成後會直接進入對齊。</FieldDescription>
              )}
            </FieldContent>
          </Field>
        <Field>
          <FieldLabel>對齊</FieldLabel>
          <FieldContent>
            <div className="settings-stack">
              <label className="control-line">
                <Switch size="sm" checked={settingsDraft.align.useGpu} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, align: { ...current.align, useGpu: checked } }))} />
                <span>使用 COLMAP GPU（CUDA：SIFT＋Ceres BA）</span>
              </label>
              <Field>
                <FieldLabel htmlFor="gpu-index">GPU index</FieldLabel>
                <Input id="gpu-index" className="w-20" type="text" inputMode="numeric" value={settingsDraft.align.gpuIndex ?? DEFAULT_SETTINGS.align.gpuIndex} onChange={(event) => { const value = event.currentTarget.value; setSettingsDraft((current) => ({ ...current, align: { ...current.align, gpuIndex: value } })); }} />
                <FieldDescription>-1 代表自動選擇；0,1 可讓 SIFT 使用多張 GPU，Ceres BA 會使用清單中的第一張。僅在 COLMAP build 確認支援 CUDA 時套用。</FieldDescription>
              </Field>
              <small>只有指定的 COLMAP build 確認支援 CUDA 才啟用；執行失敗會以 CPU 重試。</small>
              <div className="rig-note"><Workflow /><span><strong>雙階段相機組固定流程</strong><small>先建立初始模型，再固定相機組進行重建</small></span></div>
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
            <p className="empty-description">將 OSV 素材或尚未完成的專案資料夾拖放到這裡，<br />也可以先建立新的重建任務。</p>
            <div className="empty-actions"><Button size="lg" onClick={() => void openSourcePicker("files")}><Upload />選擇檔案</Button><Button size="lg" variant="outline" onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div>
            <p className="drop-hint">支援同一空間的多個檔案 · 雙魚眼素材 · 專案資料夾</p>
          </section>
        ) : (
          <section className="tasks-view">
            <header className="content-header"><div><h1>重建任務</h1><p>新增任務後會依序執行影格擷取、遮罩與對齊；各階段仍可獨立取消或重試。</p></div><div className="header-actions"><Button variant="outline" onClick={() => void openProject()}><FolderOpen />開啟專案</Button><Button onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div></header>
            <div className="task-list">
              {tasks.map((task) => {
                const overall = taskProgress(task);
                return (
                  <article className="task-row" key={task.projectId}>
                    <div className="task-row-top">
                      <div className="task-identity"><span className="task-mark"><FileStack /></span><div><div className="task-name-line"><h2>{task.name}</h2>{task.previewOnly && <Badge variant="outline">預覽</Badge>}</div><p title={task.outputPath}>{task.outputPath || "尚未指定輸出"}</p></div></div>
                      <Button variant="ghost" size="icon-sm" aria-label={`查看 ${task.name} 的詳細資料`} aria-haspopup="dialog" aria-expanded={selectedTaskId === task.projectId} onClick={() => setSelectedTaskId(task.projectId)}><MoreHorizontal /></Button>
                    </div>
                    <div className="task-progress-block">
                      <div className="task-progress-summary"><span>整體進度</span><small>{taskProgressSummary(task)}</small><strong>{overall}%</strong></div>
                      <Progress value={overall} aria-label={`${task.name} 整體進度`}><ProgressValue /></Progress>
                    </div>
                    <div className="stage-row-list">
                      {STAGES.map((stage) => {
                        const current = task.stages[stage.key];
                        const stageProgress = Math.round(current.progress);
                        const Icon = stage.icon;
                        return (
                          <div className="task-stage" data-status={current.status} key={stage.key}>
                            <div className="task-stage-label">
                              <Icon />
                              <div className="task-stage-copy">
                                <strong>{stage.label}</strong>
                                <small>{current.message || stage.description}</small>
                                {current.status === "running" && (
                                  <div className={`task-stage-progress${stageProgress <= 0 ? " is-waiting" : ""}`}>
                                    <Progress value={stageProgress} aria-label={`${stage.label}進度`}><ProgressValue /></Progress>
                                    <span>{stageProgress}%</span>
                                  </div>
                                )}
                              </div>
                            </div>
                            <Badge variant={current.status === "completed" ? "secondary" : current.status === "failed" ? "destructive" : current.status === "running" ? "default" : "outline"}><span className={`status-dot status-dot--${current.status}`} />{stageStatusLabel(current.status)}</Badge>
                            <Button variant={current.status === "running" ? "destructive" : "ghost"} size="sm" onClick={() => handleStageAction(task, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{stageAction(current.status)}</Button>
                          </div>
                        );
                      })}
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        )}
      </main>

      {toast && <div className="toast" role="status"><Info /><span>{toast}</span><Button variant="ghost" size="icon-xs" onClick={() => setToast(null)} aria-label="關閉通知"><X /></Button></div>}

      <Dialog open={taskDialogOpen} onOpenChange={setTaskDialogOpen}>
        <DialogContent className="task-dialog" showCloseButton>
          <DialogHeader><DialogTitle>新增重建任務</DialogTitle><DialogDescription>選擇多組 OSV 或雙魚眼素材，所有來源都會保存在同一份專案資訊中。</DialogDescription></DialogHeader>
          <div className="dialog-scroll">
            <div className="dialog-columns">
              <FieldGroup className="dialog-source-column">
                <Field><FieldLabel htmlFor="task-name">任務名稱</FieldLabel><FieldContent><Input id="task-name" value={nameDraft} placeholder="例如：山區路線／2026-08" onChange={(event) => setNameDraft(event.currentTarget.value)} /></FieldContent></Field>
                <Field><FieldLabel>來源</FieldLabel><FieldContent><div className={`source-drop ${dragOver ? "is-dragging" : ""}`} onDragOver={(event) => event.preventDefault()} onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}><FileStack /><span>拖放 OSV 或尚未完成的專案資料夾</span><Button type="button" variant="outline" size="sm" onClick={() => void openSourcePicker("files")}>選擇來源</Button></div>{selectedSources.length > 0 && <div className="source-list">{selectedSources.map((source) => <div className="source-item" key={source.id}><span><strong>{source.label}</strong><small>{source.detail}</small></span><Button type="button" variant="ghost" size="icon-xs" aria-label={`移除 ${source.label}`} onClick={() => setSourcePaths((current) => current.filter((path) => path !== source.path))}><X /></Button></div>)}</div>}<p className="inspection-note">{sourceInspection || "可選擇多個檔案，或直接拖入尚未完成的專案資料夾。"}</p></FieldContent></Field>
                <Field><FieldLabel htmlFor="output-path">輸出資料夾</FieldLabel><FieldContent><div className="input-with-button"><Input id="output-path" value={outputDraft} placeholder="預設與第一個來源並列：colmap-檔案名稱" onChange={(event) => setOutputDraft(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openOutputPicker()}>另選</Button></div><FieldDescription>建立後會在輸出資料夾保存專案資訊，之後可從中斷處繼續。</FieldDescription></FieldContent></Field>
              </FieldGroup>
              {renderSettingsFields()}
            </div>
          </div>
          <DialogFooter><DialogClose render={<Button variant="ghost" />}>取消</DialogClose><Button onClick={() => void createTask()} disabled={!sourcePaths.length}><Plus />新增任務</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      {selectedTask && (
        <Sheet open onOpenChange={(open) => { if (!open) setSelectedTaskId(null); }}>
          <SheetContent className="task-detail-sheet" side="right">
            <SheetHeader>
              <SheetTitle>{selectedTask.name}</SheetTitle>
              <SheetDescription>查看整體進度、各階段狀態、耗時與處理紀錄。</SheetDescription>
            </SheetHeader>
            <div className="task-detail-scroll">
              <section className="task-detail-overview">
                <div className="task-detail-heading"><span>整體進度</span><strong>{taskProgress(selectedTask)}%</strong></div>
                <Progress value={taskProgress(selectedTask)} aria-label={`${selectedTask.name} 整體進度`}><ProgressValue /></Progress>
                <dl className="task-detail-meta">
                  <div><dt>輸出資料夾</dt><dd title={selectedTask.outputPath}>{selectedTask.outputPath || "尚未指定"}</dd></div>
                  <div><dt>來源數量</dt><dd>{selectedTask.inputPaths.length} 個</dd></div>
                  <div><dt>最後更新</dt><dd>{formatUpdatedAt(selectedTask.updatedAt)}</dd></div>
                </dl>
              </section>

              <section className="task-detail-section">
                <div className="task-detail-section-title"><h2>處理階段</h2><span>{STAGES.length} 個階段</span></div>
                <div className="task-detail-stages">
                  {STAGES.map((stage) => { const current = selectedTask.stages[stage.key]; const Icon = stage.icon; const elapsed = taskStageDuration(current, clockMs); const eta = estimatedRemainingMs(current, clockMs); const showProgress = current.status === "running" || (["cancelled", "failed"].includes(current.status) && current.progress > 0); const count = logCountLabel(current.completed, current.total); return (
                    <div className="task-detail-stage" key={stage.key}>
                      <div className="task-detail-stage-main"><Icon /><span><strong>{stage.label}</strong><small>{current.phase ? `${phaseLabel(current.phase)} · ` : ""}{current.message || stage.description}{current.currentItem ? ` · ${current.currentItem}` : ""}</small></span><Badge variant={current.status === "completed" ? "secondary" : current.status === "failed" ? "destructive" : current.status === "running" ? "default" : "outline"}>{stageStatusLabel(current.status)}</Badge></div>
                      {showProgress && <div className="task-detail-stage-progress"><Progress value={current.progress} aria-label={`${stage.label}進度`}><ProgressValue /></Progress><span>{count ? `${count} · ` : ""}{current.progress}%</span></div>}
                      <div className="task-detail-stage-footer">
                        <div className="task-detail-stage-time"><span><Clock3 />{current.status === "running" ? `經過 ${formatDuration(elapsed)}` : elapsed !== undefined ? `耗時 ${formatDuration(elapsed)}` : "尚未開始"}</span>{current.status === "running" ? <span>剩餘 {formatEta(eta)}</span> : current.finishedAtMs ? <small>結束 {formatTimestamp(current.finishedAtMs, true)}</small> : current.startedAtMs ? <small>開始 {formatTimestamp(current.startedAtMs, true)}</small> : null}</div>
                        <Button variant={current.status === "running" ? "destructive" : "outline"} size="sm" onClick={() => handleStageAction(selectedTask, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{stageAction(current.status)}</Button>
                      </div>
                    </div>
                  ); })}
                </div>
              </section>

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
            <SheetFooter><Button variant="outline" onClick={() => setSelectedTaskId(null)}>完成</Button></SheetFooter>
          </SheetContent>
        </Sheet>
      )}

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent className="settings-sheet" side="right">
          <SheetHeader><SheetTitle>設定</SheetTitle><SheetDescription>以本機執行環境回報為準；不預設 GPU、FFmpeg 或模型已就緒。</SheetDescription></SheetHeader>
          <div className="settings-sheet-scroll">
            <section className="settings-section"><div className="settings-section-heading"><h2>執行環境</h2><Button variant="ghost" size="icon-sm" className={doctorLoading ? "is-spinning" : ""} onClick={() => void runDoctor(colmapPath)} aria-label="重新檢查環境"><RefreshCw /></Button></div><Field><FieldLabel htmlFor="colmap-path">COLMAP 執行檔</FieldLabel><FieldContent><div className="input-with-button"><Input id="colmap-path" value={colmapPath} placeholder="留白時從 PATH 自動偵測" onChange={(event) => setColmapPath(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openColmapPicker()}>選擇</Button></div><FieldDescription>Windows 官方免安裝版請選根目錄的 COLMAP.bat；也可指定自行編譯的 colmap.exe。</FieldDescription></FieldContent></Field><div className="doctor-summary"><MonitorCog /><span><strong>{doctor.platform}</strong><small>{doctor.summary} · {doctor.checkedAt}</small></span></div><div className="doctor-list">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <div className="doctor-row" key={item.label}><Icon /><span><strong>{item.value}</strong><small>{item.label} · {item.detail}</small></span><Badge variant={item.status === "ready" ? "secondary" : item.status === "warning" ? "destructive" : "outline"}>{item.status === "ready" ? "可用" : item.status === "warning" ? "需檢查" : "未檢查"}</Badge></div>; })}</div>{doctor.warnings.length > 0 && <div className="warning-list"><AlertTriangle />{doctor.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}</section>
          </div>
          <SheetFooter><Button variant="outline" onClick={() => setSettingsOpen(false)}>完成</Button></SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default App;

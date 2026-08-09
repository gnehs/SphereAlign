import {
  AlertTriangle,
  CircleDashed,
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
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress, ProgressValue } from "@/components/ui/progress";
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
}

interface PipelineSettings {
  extract: { baseFps: number; denseFps: number; skipBlurry: boolean };
  mask: { classes: string[]; maskSky: boolean; confidence: number; confidenceVersion: number; modelDir: string };
  align: { useGpu: boolean };
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

interface DoctorReport {
  platform: string;
  summary: string;
  checkedAt: string;
  items: DiagnosticItem[];
  warnings: string[];
}

interface ProgressEventPayload {
  stage?: StageKey;
  progress: number;
  status?: StageStatus;
  message?: string;
  jobId?: string;
  done?: boolean;
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
const DEFAULT_SETTINGS: PipelineSettings = {
  extract: { baseFps: 2, denseFps: 8, skipBlurry: true },
  mask: { classes: ["person", "bicycle", "car", "motorcycle", "bus", "truck"], maskSky: true, confidence: 25, confidenceVersion: 2, modelDir: "" },
  align: { useGpu: false },
};

const EMPTY_DOCTOR: DoctorReport = {
  platform: "尚未檢查平台",
  summary: "執行環境診斷以確認可用能力",
  checkedAt: "尚未檢查",
  items: [
    { label: "GPU／加速器", value: "尚未檢查", detail: "不預設任何硬體加速能力", status: "unknown" },
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
  "COLMAP 3.x is supported for incremental alignment, but gravity/global mapper capabilities are not claimed without COLMAP 4.x": "COLMAP 3.x 可用於增量對齊；若未安裝 COLMAP 4.x，將不啟用重力與全域對齊功能",
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

function formatUpdatedAt(value?: string) {
  if (!value) return "尚無更新時間";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-TW", { dateStyle: "medium", timeStyle: "short" });
}

function taskProgress(task: Task) {
  return Math.round(Object.values(task.stages).reduce((sum, stage) => sum + stage.progress, 0) / STAGES.length);
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
    result[stage.key] = {
      status: normaliseStageStatus(item.status),
      progress: toProgress(item.progress),
      message: typeof item.message === "string" && item.message ? localiseUserMessage(item.message) : "尚未執行",
      jobId: typeof item.jobId === "string" ? item.jobId : undefined,
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
  return {
    projectId: typeof body.projectId === "string" ? body.projectId : `project-${Date.now()}`,
    name: typeof body.name === "string" && body.name ? body.name : outputPath.split(/[\\/]/).filter(Boolean).pop() || "未命名重建",
    rootPath: typeof body.rootPath === "string" ? body.rootPath : outputPath,
    inputPaths,
    outputPath,
    settings: body.settings && typeof body.settings === "object" ? (body.settings as Record<string, unknown>) : {},
    stages: cloneStages(body.stages),
    warnings: Array.isArray(body.warnings) ? body.warnings.map((warning) => localiseUserMessage(String(warning))) : [],
    updatedAt: typeof body.updatedAt === "string" ? body.updatedAt : undefined,
  };
}

function readProgress(payload: unknown): ProgressEventPayload {
  const body = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  return {
    stage: normaliseStage(body.stage ?? body.name),
    progress: toProgress(body.progress ?? body.percent),
    status: normaliseStageStatus(body.status ?? body.state),
    message: typeof body.message === "string" ? localiseUserMessage(body.message) : undefined,
    jobId: typeof body.jobId === "string" ? body.jobId : undefined,
    done: Boolean(body.done ?? body.completed),
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
  const ffmpeg = tools.find((entry) => entryName(entry).toLowerCase() === "ffmpeg");
  const colmap = tools.find((entry) => entryName(entry).toLowerCase() === "colmap");
  const acceleratorCandidates = accelerators.filter((entry) => /(cuda|metal|videotoolbox|gpu|nvidia|apple)/i.test(`${entryName(entry)} ${itemText(entry)}`));
  const accelerator = acceleratorCandidates.find(available) ?? acceleratorCandidates[0];
  const capabilityLabels: Record<string, string> = { extract: "影格擷取", mask: "遮罩", align: "對齊" };
  const capabilityValue = body.capabilities && typeof body.capabilities === "object" ? Object.entries(body.capabilities as Record<string, unknown>).filter(([, state]) => Boolean(state)).map(([key]) => capabilityLabels[key] ?? key).join(" · ") : "";
  const platform = platformLabel(typeof body.platform === "string" ? body.platform : typeof body.os === "string" ? body.os : fallback.platform);
  const items: DiagnosticItem[] = [
    { label: "GPU／加速器", value: accelerator && available(accelerator) ? itemText(accelerator) : "未偵測到可用加速", detail: capabilityValue || "CUDA／VideoToolbox 狀態由環境診斷回報", status: accelerator && available(accelerator) ? "ready" : "warning" },
    { label: "FFmpeg", value: ffmpeg && available(ffmpeg) ? itemText(ffmpeg) : "未偵測到", detail: ffmpeg && available(ffmpeg) ? "系統 PATH 可用" : "請安裝或加入 PATH", status: ffmpeg && available(ffmpeg) ? "ready" : "warning" },
    { label: "COLMAP", value: colmap && available(colmap) ? itemText(colmap) : "未偵測到", detail: colmap && available(colmap) ? "可執行原生雙魚眼相機組對齊" : "對齊階段會維持待執行", status: colmap && available(colmap) ? "ready" : "warning" },
    { label: "執行環境", value: platform, detail: typeof body.arch === "string" ? body.arch : "Tauri 執行環境", status: "ready" },
  ];
  return { platform, summary: typeof body.summary === "string" ? localiseUserMessage(body.summary) : capabilityValue || fallback.summary, checkedAt: new Date().toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" }), items, warnings };
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
  if (label.includes("GPU")) return Cpu;
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

function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [sourcePaths, setSourcePaths] = useState<string[]>([]);
  const [outputDraft, setOutputDraft] = useState("");
  const [settingsDraft, setSettingsDraft] = useState<PipelineSettings>(DEFAULT_SETTINGS);
  const [sourceInspection, setSourceInspection] = useState<string>("");
  const [doctor, setDoctor] = useState<DoctorReport>(EMPTY_DOCTOR);
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [, setLogs] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeJobIds = useRef<Record<string, string>>({});

  const selectedSources = useMemo(() => sourcePaths.map(sourceFromPath), [sourcePaths]);
  const selectedTask = useMemo(() => tasks.find((task) => task.projectId === selectedTaskId), [selectedTaskId, tasks]);

  const addLog = useCallback((message: string) => {
    setLogs((current) => [message, ...current].slice(0, 10));
  }, []);

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
        addLog(`已載入可續作專案 ${manifest.name}`);
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
  }, [addLog]);

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
        addLog(`已開啟 ${manifest.name}`);
      } else {
        setToast("找不到可載入的專案資訊");
      }
    } catch (error) {
      console.info("[GS360] load project", error);
      setToast("開啟專案失敗");
    }
  }, [addLog]);

  const runDoctor = useCallback(async () => {
    setDoctorLoading(true);
    const result = await invokeSafely("doctor");
    if (result) setDoctor(parseDoctor(result, EMPTY_DOCTOR));
    else if (!IS_TAURI_RUNTIME) setDoctor({ ...EMPTY_DOCTOR, summary: "瀏覽器預覽未連接本機執行環境" });
    setDoctorLoading(false);
  }, []);

  useEffect(() => { void runDoctor(); }, [runDoctor]);

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
          const targetProjectId = payload.jobId
            ? Object.entries(activeJobIds.current).find(([, jobId]) => jobId === payload.jobId)?.[0]
            : undefined;
          setTasks((current) => current.map((task) => {
            if (!targetProjectId || task.projectId !== targetProjectId) return task;
            return { ...task, stages: { ...task.stages, [stageKey]: { ...task.stages[stageKey], progress: payload.progress, status: payload.done ? (payload.status === "failed" ? "failed" : payload.status === "cancelled" ? "cancelled" : "completed") : payload.status || "running", message: payload.message || task.stages[stageKey].message, jobId: payload.jobId || task.stages[stageKey].jobId } } };
          }));
          if (payload.done && targetProjectId) delete activeJobIds.current[targetProjectId];
        }),
        listen<unknown>("pipeline-log", (event) => {
          const body = event.payload && typeof event.payload === "object" ? (event.payload as Record<string, unknown>) : {};
          addLog(typeof body.message === "string" ? body.message : String(event.payload));
        }),
      ]);
      if (disposed) { progressStop(); logStop(); } else { unlistenProgress = progressStop; unlistenLog = logStop; }
    };
    if (IS_TAURI_RUNTIME) void register();
    return () => { disposed = true; unlistenProgress?.(); unlistenLog?.(); };
  }, [addLog]);

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

  const createTask = useCallback(async () => {
    if (!sourcePaths.length) { setToast("請先選擇至少一個 OSV 或雙魚眼來源"); return; }
    const request = { inputPaths: sourcePaths, outputPath: outputDraft || undefined, name: nameDraft || undefined, settings: { ...settingsDraft } };
    const result = await invokeSafely("create_project", { request });
    const manifest = manifestFromUnknown(result);
    if (manifest) {
      setTasks((current) => [manifest, ...current]);
      addLog(`已建立 ${manifest.name}`);
    } else if (!IS_TAURI_RUNTIME) {
      const preview: Task = { projectId: `preview-${Date.now()}`, name: nameDraft || "瀏覽器預覽任務", rootPath: outputDraft, inputPaths: sourcePaths, outputPath: outputDraft, settings: request.settings, stages: cloneStages({}), warnings: ["瀏覽器預覽：尚未連接本機執行環境"], previewOnly: true };
      setTasks((current) => [preview, ...current]);
      addLog(`預覽任務已加入 ${preview.name}`);
    } else {
      setToast("建立任務失敗，請查看執行環境訊息");
      return;
    }
    setTaskDialogOpen(false);
    setSourcePaths([]);
    setSourceInspection("");
  }, [addLog, nameDraft, outputDraft, settingsDraft, sourcePaths]);

  const updateTaskStage = useCallback((taskId: string, stageKey: StageKey, patch: Partial<StageState>) => {
    setTasks((current) => current.map((task) => task.projectId === taskId ? { ...task, stages: { ...task.stages, [stageKey]: { ...task.stages[stageKey], ...patch } } } : task));
  }, []);

  const startStage = useCallback(async (task: Task, stageKey: StageKey, mode: "start" | "resume" | "retry") => {
    if (!IS_TAURI_RUNTIME) { setToast("瀏覽器預覽不會執行後端工作"); return; }
    const result = await invokeSafely<{ jobId?: string }>("start_stage", { request: { projectPath: task.rootPath || task.outputPath, stage: stageKey, mode, settings: task.settings || settingsDraft } });
    if (result?.jobId) {
      activeJobIds.current[task.projectId] = result.jobId;
      updateTaskStage(task.projectId, stageKey, { status: "running", progress: task.stages[stageKey].progress, message: "正在準備工作", jobId: result.jobId });
    } else setToast("無法啟動階段，請查看執行環境訊息");
  }, [settingsDraft, updateTaskStage]);

  const cancelStage = useCallback(async (task: Task, stageKey: StageKey) => {
    if (!IS_TAURI_RUNTIME) { setToast("瀏覽器預覽不會取消後端工作"); return; }
    const jobId = task.stages[stageKey].jobId || activeJobIds.current[task.projectId];
    if (!jobId) return;
    const cancelled = await invokeSafely<boolean>("cancel_job", { jobId });
    if (cancelled !== null) updateTaskStage(task.projectId, stageKey, { status: "cancelled", message: "已取消，可稍後繼續" });
  }, [updateTaskStage]);

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

  const renderSettingsFields = () => (
    <div className="settings-form">
      <FieldGroup>
        <Field><FieldLabel>影格擷取</FieldLabel><FieldContent><div className="field-pair"><Field><FieldLabel htmlFor="base-fps">基本影格率</FieldLabel><Input id="base-fps" type="number" min={1} max={30} value={settingsDraft.extract.baseFps} onChange={(event) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, baseFps: Number(event.currentTarget.value) || 1 } }))} /></Field><Field><FieldLabel htmlFor="dense-fps">候選影格率</FieldLabel><Input id="dense-fps" type="number" min={1} max={60} value={settingsDraft.extract.denseFps} onChange={(event) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, denseFps: Number(event.currentTarget.value) || 1 } }))} /></Field></div><FieldDescription>設定雙魚眼影格取樣頻率，以及模糊篩選時的候選密度。</FieldDescription><div className="control-line"><Checkbox checked={settingsDraft.extract.skipBlurry} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, skipBlurry: checked === true } }))} /><span>跳過模糊影格</span></div></FieldContent></Field>
        <Field><FieldLabel>遮罩／物件與天空</FieldLabel><FieldContent><div className="mask-chip-list">{MASK_CLASSES.map((maskClass) => { const selected = settingsDraft.mask.classes.includes(maskClass); return <Button key={maskClass} type="button" variant={selected ? "secondary" : "outline"} size="sm" aria-pressed={selected} onClick={() => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, classes: selected ? current.mask.classes.filter((value) => value !== maskClass) : [...current.mask.classes, maskClass] } }))}>{MASK_CLASS_LABELS[maskClass]}</Button>; })}</div><div className="control-line"><Checkbox checked={settingsDraft.mask.maskSky} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, maskSky: checked === true } }))} /><span>遮蔽天空</span><span className="range-label">信心度 {settingsDraft.mask.confidence}%</span></div><input className="range-input" type="range" min={10} max={98} aria-label="遮罩信心度" value={settingsDraft.mask.confidence} onChange={(event) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, confidence: Number(event.currentTarget.value) } }))} /><div className="input-with-button model-dir-input"><Input value={settingsDraft.mask.modelDir} placeholder="模型資料夾（未指定時自動探索）" onChange={(event) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, modelDir: event.currentTarget.value } }))} /><Button type="button" variant="outline" size="sm" onClick={() => void openModelPicker()}>選擇</Button></div></FieldContent></Field>
        <Field><FieldLabel>對齊</FieldLabel><FieldContent><div className="settings-stack"><label className="control-line"><Switch size="sm" checked={settingsDraft.align.useGpu} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, align: { ...current.align, useGpu: checked } }))} /><span>使用 COLMAP GPU（CUDA）</span><small>無法使用時會自動改用 CPU</small></label><div className="rig-note"><Workflow /><span><strong>雙階段相機組固定流程</strong><small>先建立初始模型，再固定相機組進行重建</small></span></div></div></FieldContent></Field>
      </FieldGroup>
    </div>
  );

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
            <header className="content-header"><div><h1>重建任務</h1><p>每個階段都能獨立執行、取消或繼續。</p></div><div className="header-actions"><Button variant="outline" onClick={() => void openProject()}><FolderOpen />開啟專案</Button><Button onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div></header>
            <div className="task-list">{tasks.map((task) => { const overall = taskProgress(task); return <article className="task-row" key={task.projectId}><div className="task-row-top"><div className="task-identity"><span className="task-mark"><FileStack /></span><div><div className="task-name-line"><h2>{task.name}</h2>{task.previewOnly && <Badge variant="outline">預覽</Badge>}</div><p title={task.outputPath}>{task.outputPath || "尚未指定輸出"}</p></div></div><Button variant="ghost" size="icon-sm" aria-label={`查看 ${task.name} 的詳細資料`} aria-haspopup="dialog" aria-expanded={selectedTaskId === task.projectId} onClick={() => setSelectedTaskId(task.projectId)}><MoreHorizontal /></Button></div><div className="task-progress-line"><Progress value={overall}><ProgressValue /></Progress><span>{overall}%</span></div><div className="stage-row-list">{STAGES.map((stage) => { const current = task.stages[stage.key]; const Icon = stage.icon; return <div className="task-stage" key={stage.key}><div className="task-stage-label"><Icon /><span><strong>{stage.label}</strong><small>{current.message || stage.description}</small></span></div><Badge variant={current.status === "completed" ? "secondary" : current.status === "failed" ? "destructive" : current.status === "running" ? "default" : "outline"}><span className={`status-dot status-dot--${current.status}`} />{stageStatusLabel(current.status)}</Badge><Button variant={current.status === "running" ? "destructive" : "ghost"} size="sm" onClick={() => handleStageAction(task, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{stageAction(current.status)}</Button></div>; })}</div></article>; })}</div>
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
              <SheetDescription>查看任務進度、來源與各處理階段的最新狀態。</SheetDescription>
            </SheetHeader>
            <div className="task-detail-scroll">
              <section className="task-detail-overview">
                <div className="task-detail-heading"><span>整體進度</span><strong>{taskProgress(selectedTask)}%</strong></div>
                <Progress value={taskProgress(selectedTask)}><ProgressValue /></Progress>
                <dl className="task-detail-meta">
                  <div><dt>輸出資料夾</dt><dd title={selectedTask.outputPath}>{selectedTask.outputPath || "尚未指定"}</dd></div>
                  <div><dt>來源數量</dt><dd>{selectedTask.inputPaths.length} 個</dd></div>
                  <div><dt>最後更新</dt><dd>{formatUpdatedAt(selectedTask.updatedAt)}</dd></div>
                </dl>
              </section>

              <section className="task-detail-section">
                <div className="task-detail-section-title"><h2>處理階段</h2><span>{STAGES.length} 個階段</span></div>
                <div className="task-detail-stages">
                  {STAGES.map((stage) => { const current = selectedTask.stages[stage.key]; const Icon = stage.icon; return (
                    <div className="task-detail-stage" key={stage.key}>
                      <div className="task-detail-stage-main"><Icon /><span><strong>{stage.label}</strong><small>{current.message || stage.description}</small></span><Badge variant={current.status === "completed" ? "secondary" : current.status === "failed" ? "destructive" : current.status === "running" ? "default" : "outline"}>{stageStatusLabel(current.status)}</Badge></div>
                      <div className="task-detail-stage-progress"><Progress value={current.progress}><ProgressValue /></Progress><span>{current.progress}%</span></div>
                      <Button variant={current.status === "running" ? "destructive" : "outline"} size="sm" onClick={() => handleStageAction(selectedTask, stage.key)}>{current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}{stageAction(current.status)}</Button>
                    </div>
                  ); })}
                </div>
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
            <section className="settings-section"><div className="settings-section-heading"><h2>執行環境</h2><Button variant="ghost" size="icon-sm" className={doctorLoading ? "is-spinning" : ""} onClick={() => void runDoctor()} aria-label="重新檢查環境"><RefreshCw /></Button></div><div className="doctor-summary"><MonitorCog /><span><strong>{doctor.platform}</strong><small>{doctor.summary} · {doctor.checkedAt}</small></span></div><div className="doctor-list">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <div className="doctor-row" key={item.label}><Icon /><span><strong>{item.value}</strong><small>{item.label} · {item.detail}</small></span><Badge variant={item.status === "ready" ? "secondary" : item.status === "warning" ? "destructive" : "outline"}>{item.status === "ready" ? "可用" : item.status === "warning" ? "需檢查" : "未檢查"}</Badge></div>; })}</div>{doctor.warnings.length > 0 && <div className="warning-list"><AlertTriangle />{doctor.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}</section>
          </div>
          <SheetFooter><Button variant="outline" onClick={() => setSettingsOpen(false)}>完成</Button></SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default App;

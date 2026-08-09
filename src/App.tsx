import {
  AlertTriangle,
  ArrowUpRight,
  CircleDashed,
  CircleHelp,
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
  Terminal,
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
  mask: { classes: string[]; maskSky: boolean; confidence: number; modelDir: string };
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
  { key: "extract", label: "Extract", description: "雙魚眼影格、內參與 IMU", icon: ScanLine },
  { key: "mask", label: "Mask", description: "動態物件與天空遮罩", icon: CircleDashed },
  { key: "align", label: "Align", description: "多組 OSV / rig 對齊", icon: Workflow },
];

const MASK_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];
const DEFAULT_SETTINGS: PipelineSettings = {
  extract: { baseFps: 2, denseFps: 8, skipBlurry: true },
  mask: { classes: ["person", "bicycle", "car", "motorcycle", "bus", "truck"], maskSky: true, confidence: 72, modelDir: "" },
  align: { useGpu: false },
};

const EMPTY_DOCTOR: DoctorReport = {
  platform: "尚未檢查平台",
  summary: "執行環境診斷以確認可用能力",
  checkedAt: "尚未檢查",
  items: [
    { label: "GPU / accelerator", value: "尚未檢查", detail: "不預設任何硬體能力", status: "unknown" },
    { label: "FFmpeg", value: "尚未檢查", detail: "確認系統 PATH 中的 FFmpeg", status: "unknown" },
    { label: "Runtime", value: "尚未檢查", detail: "確認作業系統與執行環境", status: "unknown" },
    { label: "Storage", value: "尚未檢查", detail: "確認輸出磁碟可用空間", status: "unknown" },
  ],
  warnings: [],
};

const IS_TAURI_RUNTIME = typeof window !== "undefined" && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);

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
      message: typeof item.message === "string" && item.message ? item.message : "尚未執行",
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
    warnings: Array.isArray(body.warnings) ? body.warnings.map(String) : [],
    updatedAt: typeof body.updatedAt === "string" ? body.updatedAt : undefined,
  };
}

function readProgress(payload: unknown): ProgressEventPayload {
  const body = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  return {
    stage: normaliseStage(body.stage ?? body.name),
    progress: toProgress(body.progress ?? body.percent),
    status: normaliseStageStatus(body.status ?? body.state),
    message: typeof body.message === "string" ? body.message : undefined,
    jobId: typeof body.jobId === "string" ? body.jobId : undefined,
    done: Boolean(body.done ?? body.completed),
  };
}

function parseDoctor(value: unknown, fallback: DoctorReport): DoctorReport {
  if (!value || typeof value !== "object") return fallback;
  const body = value as Record<string, unknown>;
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const accelerators = Array.isArray(body.accelerators) ? body.accelerators : [];
  const warnings = Array.isArray(body.warnings) ? body.warnings.map(String) : [];
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
  const capabilityValue = body.capabilities && typeof body.capabilities === "object" ? Object.entries(body.capabilities as Record<string, unknown>).filter(([, state]) => Boolean(state)).map(([key]) => key).join(" · ") : "";
  const platform = typeof body.platform === "string" ? body.platform : typeof body.os === "string" ? body.os : fallback.platform;
  const items: DiagnosticItem[] = [
    { label: "GPU / accelerator", value: accelerator && available(accelerator) ? itemText(accelerator) : "未偵測到可用加速", detail: capabilityValue || "CUDA / VideoToolbox 狀態由 doctor 回報", status: accelerator && available(accelerator) ? "ready" : "warning" },
    { label: "FFmpeg", value: ffmpeg && available(ffmpeg) ? itemText(ffmpeg) : "未偵測到", detail: ffmpeg && available(ffmpeg) ? "系統 PATH 可用" : "請安裝或加入 PATH", status: ffmpeg && available(ffmpeg) ? "ready" : "warning" },
    { label: "COLMAP", value: colmap && available(colmap) ? itemText(colmap) : "未偵測到", detail: colmap && available(colmap) ? "可執行 native fisheye rig 對齊" : "Align 會保持待執行", status: colmap && available(colmap) ? "ready" : "warning" },
    { label: "Runtime", value: platform, detail: typeof body.arch === "string" ? body.arch : "Tauri runtime", status: "ready" },
  ];
  return { platform, summary: typeof body.summary === "string" ? body.summary : capabilityValue || fallback.summary, checkedAt: new Date().toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" }), items, warnings };
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
  if (label.includes("Storage")) return HardDrive;
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
        setToast(manifest.warnings.length ? `已載入半成品：${manifest.warnings.length} 項警告` : `已載入半成品：${manifest.name}`);
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
      setToast("模型資料夾會在本機 runtime 中使用");
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
      setToast("瀏覽器預覽不會讀取本機 manifest");
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
        setToast("找不到可載入的 manifest");
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
    else if (!IS_TAURI_RUNTIME) setDoctor({ ...EMPTY_DOCTOR, summary: "瀏覽器預覽未連接本機 runtime" });
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
      const preview: Task = { projectId: `preview-${Date.now()}`, name: nameDraft || "瀏覽器預覽任務", rootPath: outputDraft, inputPaths: sourcePaths, outputPath: outputDraft, settings: request.settings, stages: cloneStages({}), warnings: ["瀏覽器預覽：尚未連接本機 runtime"], previewOnly: true };
      setTasks((current) => [preview, ...current]);
      addLog(`預覽任務已加入 ${preview.name}`);
    } else {
      setToast("建立任務失敗，請查看 runtime 訊息");
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
    } else setToast("無法啟動階段，請查看 runtime 訊息");
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
        <Field><FieldLabel>Extract / 影格</FieldLabel><FieldContent><div className="field-pair"><Field><FieldLabel htmlFor="base-fps">baseFps</FieldLabel><Input id="base-fps" type="number" min={1} max={30} value={settingsDraft.extract.baseFps} onChange={(event) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, baseFps: Number(event.currentTarget.value) || 1 } }))} /></Field><Field><FieldLabel htmlFor="dense-fps">denseFps</FieldLabel><Input id="dense-fps" type="number" min={1} max={60} value={settingsDraft.extract.denseFps} onChange={(event) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, denseFps: Number(event.currentTarget.value) || 1 } }))} /></Field></div><FieldDescription>雙魚眼影格取樣與模糊篩選候選密度。</FieldDescription><div className="control-line"><Checkbox checked={settingsDraft.extract.skipBlurry} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, extract: { ...current.extract, skipBlurry: checked === true } }))} /><span>跳過模糊影格</span></div></FieldContent></Field>
        <Field><FieldLabel>Mask / 類別與天空</FieldLabel><FieldContent><div className="mask-chip-list">{MASK_CLASSES.map((maskClass) => { const selected = settingsDraft.mask.classes.includes(maskClass); return <Button key={maskClass} type="button" variant={selected ? "secondary" : "outline"} size="sm" aria-pressed={selected} onClick={() => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, classes: selected ? current.mask.classes.filter((value) => value !== maskClass) : [...current.mask.classes, maskClass] } }))}>{maskClass}</Button>; })}</div><div className="control-line"><Checkbox checked={settingsDraft.mask.maskSky} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, maskSky: checked === true } }))} /><span>遮天空</span><span className="range-label">confidence {settingsDraft.mask.confidence}%</span></div><input className="range-input" type="range" min={40} max={98} value={settingsDraft.mask.confidence} onChange={(event) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, confidence: Number(event.currentTarget.value) } }))} /><div className="input-with-button model-dir-input"><Input value={settingsDraft.mask.modelDir} placeholder="模型資料夾（未指定時自動探索）" onChange={(event) => setSettingsDraft((current) => ({ ...current, mask: { ...current.mask, modelDir: event.currentTarget.value } }))} /><Button type="button" variant="outline" size="sm" onClick={() => void openModelPicker()}>選擇</Button></div></FieldContent></Field>
        <Field><FieldLabel>Align / 對齊</FieldLabel><FieldContent><div className="settings-stack"><label className="control-line"><Switch size="sm" checked={settingsDraft.align.useGpu} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, align: { ...current.align, useGpu: checked } }))} /><span>COLMAP GPU（CUDA）</span><small>未偵測到時自動使用 CPU</small></label><div className="rig-note"><Workflow /><span><strong>two-pass rig 固定流程</strong><small>先 bootstrap，再固定 rig 重建</small></span></div></div></FieldContent></Field>
      </FieldGroup>
    </div>
  );

  return (
    <div className="studio-app">
      <header className="window-bar">
        {!IS_TAURI_RUNTIME && <div className="traffic-lights" aria-hidden="true"><span className="traffic-red" /><span className="traffic-yellow" /><span className="traffic-green" /></div>}
        <span className="window-title">GS360 Studio</span>
        <div className="window-actions"><Badge variant="outline" className="runtime-badge">{IS_TAURI_RUNTIME ? "本機 runtime" : "瀏覽器預覽"}</Badge><Button variant="ghost" size="icon-sm" aria-label="開啟設定" onClick={() => setSettingsOpen(true)}><Settings2 /></Button></div>
      </header>

      <main className="studio-main">
        <input ref={fileInputRef} type="file" multiple accept=".osv,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mts,.m2ts,.ts" hidden onChange={(event) => handleBrowserFiles(event.currentTarget.files)} />
        {tasks.length === 0 ? (
          <section className="empty-state" onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
            <div className={`empty-icon ${dragOver ? "is-dragging" : ""}`} aria-hidden="true"><FileStack /></div>
            <p className="eyebrow">WORKSPACE / LOCAL</p>
            <h1>尚無任務</h1>
            <p className="empty-description">拖放 OSV 或處理到一半的資料夾到這裡，<br />或先建立一個可續作的重建任務。</p>
            <div className="empty-actions"><Button size="lg" onClick={() => void openSourcePicker("files")}><Upload />選擇檔案</Button><Button size="lg" variant="outline" onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div>
            <p className="drop-hint">支援多檔同一空間 · 雙魚眼素材 · project folder</p>
          </section>
        ) : (
          <section className="tasks-view">
            <header className="content-header"><div><p className="eyebrow">WORKSPACE / LOCAL</p><h1>重建任務</h1><p>每一步都可獨立執行、取消與續作。</p></div><div className="header-actions"><Button variant="outline" onClick={() => void openProject()}><FolderOpen />開啟 project</Button><Button onClick={openNewTaskDialog}><Plus />新增重建任務</Button></div></header>
            <div className="task-list">{tasks.map((task) => { const overall = Math.round(Object.values(task.stages).reduce((sum, stage) => sum + stage.progress, 0) / 3); return <article className="task-row" key={task.projectId}><div className="task-row-top"><div className="task-identity"><span className="task-mark"><FileStack /></span><div><div className="task-name-line"><h2>{task.name}</h2>{task.previewOnly && <Badge variant="outline">預覽</Badge>}</div><p title={task.outputPath}>{task.outputPath || "尚未指定輸出"}</p></div></div><Button variant="ghost" size="icon-sm" aria-label={`${task.name} 更多選項`}><MoreHorizontal /></Button></div><div className="task-progress-line"><Progress value={overall}><ProgressValue /></Progress><span>{overall}%</span></div><div className="stage-row-list">{STAGES.map((stage) => { const current = task.stages[stage.key]; const Icon = stage.icon; return <div className="task-stage" key={stage.key}><div className="task-stage-label"><Icon /><span><strong>{stage.label}</strong><small>{current.message || stage.description}</small></span></div><Badge variant={current.status === "completed" ? "secondary" : current.status === "failed" ? "destructive" : current.status === "running" ? "default" : "outline"}><span className={`status-dot status-dot--${current.status}`} />{stageStatusLabel(current.status)}</Badge><Button variant={current.status === "running" ? "destructive" : "ghost"} size="sm" onClick={() => handleStageAction(task, stage.key)}>{current.status === "running" ? <Square /> : current.status === "completed" ? <RotateCcw /> : <Play />}{stageAction(current.status)}</Button></div>; })}</div></article>; })}</div>
          </section>
        )}
      </main>

      <footer className="studio-footer"><span><CircleHelp />拖放素材或選擇資料夾即可開始</span><span><Terminal />本機資料不會離開這台電腦</span><Button variant="link" size="sm" onClick={() => setSettingsOpen(true)}>設定與環境 <ArrowUpRight /></Button></footer>

      {toast && <div className="toast" role="status"><Info /><span>{toast}</span><Button variant="ghost" size="icon-xs" onClick={() => setToast(null)} aria-label="關閉通知"><X /></Button></div>}

      <Dialog open={taskDialogOpen} onOpenChange={setTaskDialogOpen}>
        <DialogContent className="task-dialog" showCloseButton>
          <DialogHeader><DialogTitle>新增重建任務</DialogTitle><DialogDescription>選擇多組 OSV 或雙魚眼素材，它們會在同一個 project manifest 中保留。</DialogDescription></DialogHeader>
          <div className="dialog-scroll">
            <div className="dialog-columns">
              <FieldGroup className="dialog-source-column">
                <Field><FieldLabel htmlFor="task-name">任務名稱</FieldLabel><FieldContent><Input id="task-name" value={nameDraft} placeholder="例如：Mountain pass / 2026-08" onChange={(event) => setNameDraft(event.currentTarget.value)} /></FieldContent></Field>
                <Field><FieldLabel>來源</FieldLabel><FieldContent><div className={`source-drop ${dragOver ? "is-dragging" : ""}`} onDragOver={(event) => event.preventDefault()} onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop}><FileStack /><span>拖放 OSV 或處理到一半的資料夾</span><Button type="button" variant="outline" size="sm" onClick={() => void openSourcePicker("files")}>選擇來源</Button></div>{selectedSources.length > 0 && <div className="source-list">{selectedSources.map((source) => <div className="source-item" key={source.id}><span><strong>{source.label}</strong><small>{source.detail}</small></span><Button type="button" variant="ghost" size="icon-xs" aria-label={`移除 ${source.label}`} onClick={() => setSourcePaths((current) => current.filter((path) => path !== source.path))}><X /></Button></div>)}</div>}<p className="inspection-note">{sourceInspection || "可選擇多個檔案，或直接拖入處理到一半的資料夾。"}</p></FieldContent></Field>
                <Field><FieldLabel htmlFor="output-path">輸出資料夾</FieldLabel><FieldContent><div className="input-with-button"><Input id="output-path" value={outputDraft} placeholder="預設與第一個來源並列：colmap-{filename}" onChange={(event) => setOutputDraft(event.currentTarget.value)} /><Button type="button" variant="outline" size="sm" onClick={() => void openOutputPicker()}>另選</Button></div><FieldDescription>建立後會在輸出資料夾保存 manifest，可從中斷處續作。</FieldDescription></FieldContent></Field>
              </FieldGroup>
              {renderSettingsFields()}
            </div>
          </div>
          <DialogFooter><DialogClose render={<Button variant="ghost" />}>取消</DialogClose><Button onClick={() => void createTask()} disabled={!sourcePaths.length}><Plus />新增任務</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent className="settings-sheet" side="right">
          <SheetHeader><SheetTitle>設定</SheetTitle><SheetDescription>以本機 runtime 回報為準；不預設 GPU、FFmpeg 或模型已就緒。</SheetDescription></SheetHeader>
          <div className="settings-sheet-scroll">
            <section className="settings-section"><div className="settings-section-heading"><h2>介面</h2><span>本機</span></div><div className="settings-item"><span><strong>介面語言</strong><small>目前版本採台灣繁體中文</small></span><Badge variant="outline">繁體中文</Badge></div><div className="settings-item"><span><strong>任務啟動方式</strong><small>所有 stage 都由使用者手動執行</small></span><Badge variant="outline">不自動執行</Badge></div></section>
            <section className="settings-section"><div className="settings-section-heading"><h2>Runtime</h2><Button variant="ghost" size="icon-sm" className={doctorLoading ? "is-spinning" : ""} onClick={() => void runDoctor()} aria-label="重新檢查環境"><RefreshCw /></Button></div><div className="doctor-summary"><MonitorCog /><span><strong>{doctor.platform}</strong><small>{doctor.summary} · {doctor.checkedAt}</small></span></div><div className="doctor-list">{doctor.items.map((item) => { const Icon = iconForDiagnostic(item.label); return <div className="doctor-row" key={item.label}><Icon /><span><strong>{item.value}</strong><small>{item.label} · {item.detail}</small></span><Badge variant={item.status === "ready" ? "secondary" : item.status === "warning" ? "destructive" : "outline"}>{item.status === "ready" ? "可用" : item.status === "warning" ? "需檢查" : "未檢查"}</Badge></div>; })}</div>{doctor.warnings.length > 0 && <div className="warning-list"><AlertTriangle />{doctor.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}</section>
            <section className="settings-section"><div className="settings-section-heading"><h2>模型</h2><span>Mask</span></div><div className="settings-item"><span><strong>來源模型</strong><small title={settingsDraft.mask.modelDir || undefined}>{settingsDraft.mask.modelDir || "依 models/、.models/ 或 GS360_MODEL_DIR 自動探索"}</small></span><Badge variant="outline">{settingsDraft.mask.modelDir ? "已指定" : "自動探索"}</Badge></div><div className="settings-item"><span><strong>系統 FFmpeg</strong><small>不下載、不內嵌額外執行檔</small></span><Badge variant="outline">依 doctor</Badge></div></section>
          </div>
          <SheetFooter><Button variant="outline" onClick={() => setSettingsOpen(false)}>完成</Button></SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default App;

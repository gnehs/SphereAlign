import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence } from "motion/react";
import { t } from "@lingui/core/macro";
import { useLingui } from "@lingui/react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow, ProgressBarStatus } from "@tauri-apps/api/window";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { writeText as writeClipboardText } from "@tauri-apps/plugin-clipboard-manager";
import { AppHeader, AppNotice } from "@/components/app-chrome";
import { SettingsSheet } from "@/components/settings-sheet";
import { TaskDetail } from "@/components/task-detail";
import { TaskEditorDialog, RemoveTaskDialog } from "@/components/task-editor-dialog";
import { TaskWorkspace } from "@/components/task-list";
import { useTheme } from "@/components/theme-provider";
import {
  BROWSER_PREVIEW_NOT_CONNECTED_SOURCE,
  BROWSER_PREVIEW_TASK_SOURCE,
  COLMAP_CUDA_DIAGNOSTIC_LABEL,
  COLMAP_PATH_STORAGE_KEY,
  DEFAULT_SETTINGS,
  HARDWARE_ACCELERATION_LABEL,
  IS_TAURI_RUNTIME,
  IS_WINDOWS_RUNTIME,
  PREPARING_WORK_SOURCE,
  STAGES,
  appendMessageLog,
  backendErrorMessage,
  cloneStages,
  customLutPathIsInvalid,
  deriveOutputPath,
  doctorReportText,
  emptyDoctor,
  invokeSafely,
  localiseUserMessage,
  manifestFromUnknown,
  mergeProgressLog,
  normaliseColorInspectionSummary,
  normalisePipelineSettings,
  parseDoctor,
  readLogEvent,
  readProgress,
  selectAvailableGpu,
  sourceFromPath,
  stagePrerequisiteKey,
  taskCreatedAtMs,
  taskCurrentStage,
  taskHasNotStarted,
  taskIsCompleted,
  taskProgress,
  taskStageDuration,
  warningAffectsProcessingSpeed,
  type AutoPipelineRun,
  type ColorInspectionSummary,
  type DiagnosticStatus,
  type DoctorReport,
  type LogEventPayload,
  type PipelineSettings,
  type ProgressEventPayload,
  type StageKey,
  type StageState,
  type StageStatus,
  type Task,
  type TaskLog,
  type TaskLogLevel,
} from "@/lib/pipeline";
import "./App.css";

 function App() {
  // Subscribe the whole screen to Lingui's locale-change event. Most of the
  // app uses the macro helpers directly, but raw backend messages are rendered
  // through localiseUserMessage and need the same rerender trigger.
  const { i18n: lingui } = useLingui();
  const { theme, setTheme } = useTheme();
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

  const taskGroups = [
    { key: "active", title: t`In progress`, items: activeTasks },
    { key: "completed", title: t`Completed`, items: completedTasks },
  ] as const;

  const handleTaskDialogOpenChange = (open: boolean) => {
    setTaskDialogOpen(open);
    if (!open && editingTaskId) {
      const run = autoPipelineRuns.current[editingTaskId];
      if (run) run.paused = false;
      setEditingTaskId(null);
      queueMicrotask(() => pumpAutoPipelineRef.current());
    }
  };

  return (
    <div className="flex size-full flex-col bg-background">
      <main className="flex min-h-0 flex-1 flex-col overflow-auto">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".osv,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mts,.m2ts,.ts"
          hidden
          onChange={(event) => handleBrowserFiles(event.currentTarget.files)}
        />
        <AppHeader
          onNewTask={openNewTaskDialog}
          onOpenProject={openProject}
          onOpenSettings={() => setSettingsOpen(true)}
        />
        <TaskWorkspace
          groups={taskGroups}
          clockMs={clockMs}
          hasRunningStage={hasRunningStage}
          taskDetailOpen={taskDetailOpen}
          selectedTask={selectedTask}
          taskDetailUsesSplitView={taskDetailUsesSplitView}
          dragOver={dragOver}
          canChangeQueuedTask={canChangeQueuedTask}
          isWaitingForEnqueue={(task) => !autoPipelineRuns.current[task.projectId]}
          localiseUserMessage={localiseUserMessage}
          onOpenSourcePicker={openSourcePicker}
          onOpenProject={openProject}
          onDragEnter={() => setDragOver(true)}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onEnqueueTask={enqueueQueuedTask}
          onEditTask={openEditTaskDialog}
          onRemoveTask={(task) => setDeletingTaskId(task.projectId)}
          onOpenTaskDetail={(task, trigger) => {
            taskDetailTriggerRef.current = trigger;
            setTaskDetailTab("summary");
            setSelectedTaskId(task.projectId);
            setTaskDetailOpen(true);
          }}
          onStageAction={handleStageAction}
        />
      </main>

      <AnimatePresence initial={false}>
        {toast && (
          <AppNotice
            key={toast}
            message={localiseUserMessage(toast)}
            onClose={() => setToast(null)}
            avoidBottomAction={taskDetailOpen}
          />
        )}
      </AnimatePresence>

      <TaskEditorDialog
        open={taskDialogOpen}
        editingTaskId={editingTaskId}
        onOpenChange={handleTaskDialogOpenChange}
        nameDraft={nameDraft}
        setNameDraft={setNameDraft}
        outputDraft={outputDraft}
        setOutputDraft={setOutputDraft}
        settingsDraft={settingsDraft}
        setSettingsDraft={setSettingsDraft}
        selectedSources={selectedSources}
        sourceInspection={sourceInspection}
        sourceColorInspection={sourceColorInspection}
        dragOver={dragOver}
        setDragOver={setDragOver}
        onDrop={handleDrop}
        onRemoveSource={(path) => {
          setSourcePaths((current) => current.filter((sourcePath) => sourcePath !== path));
          setSourceColorInspection(null);
        }}
        onSourcePicker={openSourcePicker}
        onOutputPicker={openOutputPicker}
        onLutPicker={openLutPicker}
        onGpuPreferenceTouched={() => { gpuPreferenceTouched.current = true; }}
        onSubmit={editingTaskId ? saveEditedTask : createTask}
        doctor={doctor}
      />

      <RemoveTaskDialog
        open={Boolean(deletingTaskId)}
        onOpenChange={(open) => { if (!open) setDeletingTaskId(null); }}
        onConfirm={deleteQueuedTask}
      />

      <TaskDetail
        open={taskDetailOpen && Boolean(selectedTask)}
        selectedTask={selectedTask}
        selectedTaskSources={selectedTaskSources}
        selectedTaskLogs={selectedTaskLogs}
        selectedStageDefinition={selectedStageDefinition}
        selectedStage={selectedStage}
        selectedRunningStageDefinition={selectedRunningStageDefinition}
        selectedActiveProgressLog={selectedActiveProgressLog}
        clockMs={clockMs}
        activeTab={taskDetailTab}
        onTabChange={setTaskDetailTab}
        onStageAction={handleStageAction}
        onClose={closeTaskDetail}
        restoreFocusRef={taskDetailTriggerRef}
        escapeBlocked={taskDialogOpen || Boolean(deletingTaskId) || settingsOpen}
        onExitComplete={() => {
          if (taskDetailOpen) return;
          setSelectedTaskId(null);
          taskDetailTriggerRef.current = null;
        }}
      />

      <SettingsSheet
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        theme={theme}
        setTheme={setTheme}
        doctor={doctor}
        doctorEssentialReady={doctorEssentialReady}
        performanceWarnings={performanceWarnings}
        generalDoctorWarnings={generalDoctorWarnings}
        performanceFallback={performanceFallback}
        performanceStatus={performanceStatus}
        isWindowsPlatform={isWindowsPlatform}
        colmapPath={colmapPath}
        setColmapPath={setColmapPath}
        doctorLoading={doctorLoading}
        runDoctor={runDoctor}
        copyDoctorReport={copyDoctorReport}
        openColmapPicker={openColmapPicker}
      />
    </div>
  );
}

export default App;

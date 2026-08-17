import { create } from "zustand";
import {
  COLMAP_PATH_STORAGE_KEY,
  DEFAULT_SETTINGS,
  emptyDoctor,
  normalisePipelineSettings,
  type ColorInspectionSummary,
  type DoctorReport,
  type PipelineSettings,
  type StageKey,
  type StageState,
  type Task,
} from "@/lib/pipeline";

export type TaskDetailTab = "summary" | "records";

type StateUpdater<T> = T | ((current: T) => T);

interface AppState {
  tasks: Task[];
  taskDialogOpen: boolean;
  editingTaskId: string | null;
  deletingTaskId: string | null;
  selectedTaskId: string | null;
  taskDetailOpen: boolean;
  taskDetailTab: TaskDetailTab;
  settingsOpen: boolean;
  nameDraft: string;
  sourcePaths: string[];
  outputDraft: string;
  settingsDraft: PipelineSettings;
  colmapPath: string;
  sourceInspection: string;
  sourceColorInspection: ColorInspectionSummary | null;
  doctor: DoctorReport;
  doctorLoading: boolean;
  toast: string | null;
}

interface AppActions {
  upsertTask: (task: Task) => void;
  updateTask: (taskId: string, update: (task: Task) => Task) => void;
  updateTaskStage: (taskId: string, stageKey: StageKey, patch: Partial<StageState>) => void;
  removeTask: (taskId: string) => void;
  setTaskDialogOpen: (open: boolean) => void;
  setEditingTaskId: (taskId: string | null) => void;
  setDeletingTaskId: (taskId: string | null) => void;
  setSelectedTaskId: (taskId: string | null) => void;
  setTaskDetailOpen: (open: boolean) => void;
  setTaskDetailTab: (tab: TaskDetailTab) => void;
  setSettingsOpen: (open: boolean) => void;
  setNameDraft: (update: StateUpdater<string>) => void;
  setSourcePaths: (update: StateUpdater<string[]>) => void;
  setOutputDraft: (update: StateUpdater<string>) => void;
  setSettingsDraft: (update: StateUpdater<PipelineSettings>) => void;
  setColmapPath: (path: string) => void;
  setSourceInspection: (inspection: string) => void;
  setSourceColorInspection: (inspection: ColorInspectionSummary | null) => void;
  setDoctor: (report: DoctorReport) => void;
  setDoctorLoading: (loading: boolean) => void;
  setToast: (message: string | null) => void;
}

export type AppStore = AppState & AppActions;

function resolveUpdate<T>(update: StateUpdater<T>, current: T): T {
  return typeof update === "function"
    ? (update as (value: T) => T)(current)
    : update;
}

function readStoredColmapPath() {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(COLMAP_PATH_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export const useAppStore = create<AppStore>()((set) => ({
  tasks: [],
  taskDialogOpen: false,
  editingTaskId: null,
  deletingTaskId: null,
  selectedTaskId: null,
  taskDetailOpen: false,
  taskDetailTab: "summary",
  settingsOpen: false,
  nameDraft: "",
  sourcePaths: [],
  outputDraft: "",
  settingsDraft: normalisePipelineSettings(DEFAULT_SETTINGS),
  colmapPath: readStoredColmapPath(),
  sourceInspection: "",
  sourceColorInspection: null,
  doctor: emptyDoctor(),
  doctorLoading: false,
  toast: null,

  upsertTask: (task) => set((state) => ({
    tasks: [task, ...state.tasks.filter((item) => item.projectId !== task.projectId)],
  })),
  updateTask: (taskId, update) => set((state) => ({
    tasks: state.tasks.map((task) => task.projectId === taskId ? update(task) : task),
  })),
  updateTaskStage: (taskId, stageKey, patch) => set((state) => ({
    tasks: state.tasks.map((task) => task.projectId === taskId
      ? {
          ...task,
          stages: {
            ...task.stages,
            [stageKey]: { ...task.stages[stageKey], ...patch },
          },
        }
      : task),
  })),
  removeTask: (taskId) => set((state) => ({
    tasks: state.tasks.filter((task) => task.projectId !== taskId),
  })),
  setTaskDialogOpen: (taskDialogOpen) => set({ taskDialogOpen }),
  setEditingTaskId: (editingTaskId) => set({ editingTaskId }),
  setDeletingTaskId: (deletingTaskId) => set({ deletingTaskId }),
  setSelectedTaskId: (selectedTaskId) => set({ selectedTaskId }),
  setTaskDetailOpen: (taskDetailOpen) => set({ taskDetailOpen }),
  setTaskDetailTab: (taskDetailTab) => set({ taskDetailTab }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setNameDraft: (update) => set((state) => ({ nameDraft: resolveUpdate(update, state.nameDraft) })),
  setSourcePaths: (update) => set((state) => ({ sourcePaths: resolveUpdate(update, state.sourcePaths) })),
  setOutputDraft: (update) => set((state) => ({ outputDraft: resolveUpdate(update, state.outputDraft) })),
  setSettingsDraft: (update) => set((state) => ({ settingsDraft: resolveUpdate(update, state.settingsDraft) })),
  setColmapPath: (colmapPath) => set({ colmapPath }),
  setSourceInspection: (sourceInspection) => set({ sourceInspection }),
  setSourceColorInspection: (sourceColorInspection) => set({ sourceColorInspection }),
  setDoctor: (doctor) => set({ doctor }),
  setDoctorLoading: (doctorLoading) => set({ doctorLoading }),
  setToast: (toast) => set({ toast }),
}));

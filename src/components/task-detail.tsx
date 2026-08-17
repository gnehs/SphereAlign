import { AlertTriangle, FileStack, LoaderCircle, Square, type LucideIcon } from "lucide-react";
import { t } from "@lingui/core/macro";
import { Plural, Trans } from "@lingui/react/macro";
import { type ReactNode, type RefObject } from "react";
import { useShallow } from "zustand/react/shallow";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress, ProgressValue } from "@/components/ui/progress";
import { SourceListItem, SourceThumbnail } from "@/components/source-media";
import {
  STAGES,
  estimatedRemainingMs,
  formatDuration,
  formatEta,
  formatTimestamp,
  localiseUserMessage,
  logCountLabel,
  phaseLabel,
  processingRateLabel,
  sourceFromPath,
  stageDescription,
  stageLabel,
  stageStatusLabel,
  taskStageDuration,
  taskStageLabel,
  taskCurrentStage,
  timestampDateTime,
  type OsvSource,
  type StageDefinition,
  type StageKey,
  type StageState,
  type Task,
  type TaskLog,
} from "@/lib/pipeline";
import { TaskDetailPanel, type TaskDetailTab } from "@/components/task-detail-panel";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app-store";

export interface TaskDetailProps {
  clockMs: number;
  onStageAction: (task: Task, stageKey: StageKey) => void;
  restoreFocusRef?: RefObject<HTMLElement | null>;
  onExitComplete?: () => void;
}

interface TaskSummaryProps {
  selectedTask?: Task;
  selectedTaskSources: OsvSource[];
  selectedStageDefinition?: StageDefinition;
  selectedStage?: StageState;
  selectedActiveProgressLog?: TaskLog;
  clockMs: number;
}

interface TaskRecordsProps {
  selectedTask?: Task;
  selectedTaskLogs: TaskLog[];
}

function StageStatusBadge({ status }: { status: StageState["status"] }) {
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

function TaskSummary({
  selectedTask,
  selectedTaskSources,
  selectedStageDefinition,
  selectedStage,
  selectedActiveProgressLog,
  clockMs,
}: TaskSummaryProps) {
  if (!selectedTask) return null;

  return (
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
  );
}

function TaskRecords({ selectedTask, selectedTaskLogs }: TaskRecordsProps) {
  if (!selectedTask) return null;

  return (
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
  );
}

export function TaskDetail({
  clockMs,
  onStageAction,
  restoreFocusRef,
  onExitComplete,
}: TaskDetailProps) {
  const selectedTask = useAppStore((state) => state.tasks.find((task) => task.projectId === state.selectedTaskId));
  const {
    taskDetailOpen,
    activeTab,
    setTaskDetailOpen,
    setTaskDetailTab,
    taskDialogOpen,
    deletingTaskId,
    settingsOpen,
  } = useAppStore(useShallow((state) => ({
    taskDetailOpen: state.taskDetailOpen,
    activeTab: state.taskDetailTab,
    setTaskDetailOpen: state.setTaskDetailOpen,
    setTaskDetailTab: state.setTaskDetailTab,
    taskDialogOpen: state.taskDialogOpen,
    deletingTaskId: state.deletingTaskId,
    settingsOpen: state.settingsOpen,
  })));
  const selectedTaskSources = selectedTask?.inputPaths.map(sourceFromPath) ?? [];
  const selectedTaskLogs = selectedTask
    ? selectedTask.logs.slice().sort((left, right) => right.timestampMs - left.timestampMs)
    : [];
  const selectedStageDefinition = selectedTask ? taskCurrentStage(selectedTask) : undefined;
  const selectedStage = selectedTask && selectedStageDefinition
    ? selectedTask.stages[selectedStageDefinition.key]
    : undefined;
  const selectedRunningStageDefinition = selectedTask
    ? STAGES.find(({ key }) => selectedTask.stages[key].status === "running")
    : undefined;
  const selectedActiveProgressLog = selectedStageDefinition
    ? selectedTaskLogs.find((log) => log.kind === "progress" && log.stage === selectedStageDefinition.key && log.finishedAtMs === undefined)
    : undefined;
  const onClose = () => setTaskDetailOpen(false);

  return (
    <TaskDetailPanel
      open={taskDetailOpen && Boolean(selectedTask)}
      title={selectedTask?.name ?? ""}
      description={selectedTask ? <span title={selectedTask.outputPath}>{selectedTask.outputPath || t`Output not specified`}</span> : null}
      leading={selectedTask ? (selectedTaskSources[0] ? <SourceThumbnail source={selectedTaskSources[0]} previewSide="left" size="compact" /> : <span className="grid size-10.5 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary [&_svg]:size-5"><FileStack /></span>) : null}
      activeTab={activeTab}
      onTabChange={setTaskDetailTab}
      summary={<TaskSummary
        selectedTask={selectedTask}
        selectedTaskSources={selectedTaskSources}
        selectedStageDefinition={selectedStageDefinition}
        selectedStage={selectedStage}
        selectedActiveProgressLog={selectedActiveProgressLog}
        clockMs={clockMs}
      />}
      records={<TaskRecords selectedTask={selectedTask} selectedTaskLogs={selectedTaskLogs} />}
      footer={selectedTask ? (
        <>
          {selectedRunningStageDefinition && <Button variant="destructive" onClick={() => onStageAction(selectedTask, selectedRunningStageDefinition.key)}><Square data-icon="inline-start" /><Trans context="task action" comment="Cancel the currently running stage for the whole selected task.">Cancel entire task</Trans></Button>}
          <Button variant="outline" onClick={onClose}><Trans>Close</Trans></Button>
        </>
      ) : null}
      onClose={onClose}
      restoreFocusRef={restoreFocusRef}
      escapeBlocked={taskDialogOpen || Boolean(deletingTaskId) || settingsOpen}
      onExitComplete={onExitComplete}
    />
  );
}

export type { TaskDetailTab };

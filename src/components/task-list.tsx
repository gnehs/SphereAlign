import {
  CheckCircle2,
  FileStack,
  Film,
  Folder,
  FolderOpen,
  Info,
  LoaderCircle,
  Pencil,
  Play,
  RotateCcw,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { type DragEvent, type ReactNode } from "react";
import { t } from "@lingui/core/macro";
import { Plural, Trans } from "@lingui/react/macro";
import { useReducedMotion } from "motion/react";
import * as m from "motion/react-m";
import {
  STAGES,
  estimatedRemainingMs,
  formatDuration,
  formatEta,
  logCountLabel,
  phaseLabel,
  stageActionState,
  stageDescription,
  stageLabel,
  stageStatusLabel,
  sourceFromPath,
  taskCurrentStage,
  taskHasNotStarted,
  taskProgress,
  taskProgressSummary,
  taskStageDuration,
  type StageKey,
  type StageStatus,
  type Task,
} from "@/lib/pipeline";
import { SourceThumbnail, SupportedFormatCard } from "@/components/source-media";
import { TASK_DETAIL_DRAWER_TRANSITION } from "@/components/task-detail-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Progress, ProgressValue } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export interface TaskGroup {
  key: "active" | "completed";
  title: ReactNode;
  items: readonly Task[];
}

export interface TaskWorkspaceProps {
  /** The active and completed task groups, in display order. */
  groups: readonly TaskGroup[];
  clockMs: number;
  hasRunningStage: boolean;
  taskDetailOpen: boolean;
  selectedTask?: Task;
  taskDetailUsesSplitView: boolean;
  dragOver: boolean;
  canChangeQueuedTask: (task: Task) => boolean;
  isWaitingForEnqueue: (task: Task) => boolean;
  localiseUserMessage: (value: string) => string;
  onOpenSourcePicker: () => void | Promise<void>;
  onOpenProject: () => void | Promise<void>;
  onDragEnter: () => void;
  onDragLeave: () => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onEnqueueTask: (task: Task) => void;
  onEditTask: (task: Task) => void;
  onRemoveTask: (task: Task) => void;
  onOpenTaskDetail: (task: Task, trigger: HTMLButtonElement) => void;
  onStageAction: (task: Task, stageKey: StageKey) => void;
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

function TaskCard({
  task,
  clockMs,
  hasRunningStage,
  canChangeQueuedTask,
  isWaitingForEnqueue,
  localiseUserMessage,
  taskDetailOpen,
  selectedTask,
  onEnqueueTask,
  onEditTask,
  onRemoveTask,
  onOpenTaskDetail,
  onStageAction,
}: {
  task: Task;
  clockMs: number;
  hasRunningStage: boolean;
  taskDetailOpen: boolean;
  selectedTask?: Task;
  canChangeQueuedTask: (task: Task) => boolean;
  isWaitingForEnqueue: (task: Task) => boolean;
  localiseUserMessage: (value: string) => string;
  onEnqueueTask: (task: Task) => void;
  onEditTask: (task: Task) => void;
  onRemoveTask: (task: Task) => void;
  onOpenTaskDetail: (task: Task, trigger: HTMLButtonElement) => void;
  onStageAction: (task: Task, stageKey: StageKey) => void;
}) {
  const overall = taskProgress(task);
  const queued = taskHasNotStarted(task);
  const editableQueued = queued && canChangeQueuedTask(task);
  const waitingForEnqueue = queued && !task.previewOnly && isWaitingForEnqueue(task);
  const currentStageDefinition = taskCurrentStage(task);
  const currentStage = task.stages[currentStageDefinition.key];
  const currentElapsed = taskStageDuration(currentStage, clockMs);
  const currentEta = estimatedRemainingMs(currentStage, clockMs);
  const currentCount = logCountLabel(currentStage.completed, currentStage.total);
  const primarySource = task.inputPaths.length > 0 ? sourceFromPath(task.inputPaths[0], 0) : undefined;

  return (
    <article className={cn("rounded-xl border bg-card px-7.5 pt-6.5 pb-7.5 shadow-sm max-[760px]:px-4.5 max-[760px]:pt-5 max-[760px]:pb-6", queued && "bg-primary/[0.02]")}>
      <div className="flex items-center justify-between gap-4.5">
        <div className="flex min-w-0 items-center gap-3">
          {primarySource
            ? <SourceThumbnail source={primarySource} />
            : <span className="grid size-12 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary [&_svg]:size-5.5"><FileStack /></span>}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-[17px] font-semibold text-foreground">{task.name}</h2>
              {queued && <Badge variant="outline">{editableQueued ? t`Waiting to run` : t`Preparing`}</Badge>}
              {task.previewOnly && <Badge variant="outline">{t`Preview`}</Badge>}
            </div>
            <p className="mt-1 max-w-160 truncate font-mono text-sm text-muted-foreground" title={task.outputPath}>
              {task.outputPath || t`Output not specified`}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {waitingForEnqueue && (
            <Button size="sm" onClick={() => onEnqueueTask(task)}>
              <Play data-icon="inline-start" />
              <Trans context="queue action" comment="Add a queued task to the automatic execution queue.">Add to queue</Trans>
            </Button>
          )}
          {editableQueued && (
            <>
              <Button variant="outline" size="sm" onClick={() => onEditTask(task)}>
                <Pencil data-icon="inline-start" />
                <Trans context="task action" comment="Edit a task that has not started.">Edit</Trans>
              </Button>
              <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => onRemoveTask(task)}>
                <Trash2 data-icon="inline-start" />
                <Trans context="task action" comment="Remove a queued task without deleting its output folder.">Remove</Trans>
              </Button>
            </>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t`View details for ${task.name}`}
            aria-haspopup="dialog"
            aria-expanded={taskDetailOpen && selectedTask?.projectId === task.projectId}
            onClick={(event) => onOpenTaskDetail(task, event.currentTarget)}
          >
            <Info />
          </Button>
        </div>
      </div>

      {queued ? (
        <div className="mt-4 ml-15 flex items-center justify-between text-sm text-muted-foreground max-[760px]:ml-0">
          <span><Trans comment="Queued tasks run automatically in creation order.">The queue runs automatically in creation order</Trans></span>
          <small><Plural value={task.inputPaths.length} one="# source" other="# sources" /></small>
        </div>
      ) : (
        <>
          <div className="mt-6.5 mb-4 ml-15 max-[760px]:ml-0 [&_[data-slot=progress-track]]:h-1.75 [&_[data-slot=progress-value]]:hidden">
            <div className="mb-2 flex items-center gap-2 text-sm">
              <span className="font-medium text-foreground" title={t`Weighted by three observed run durations: frame extraction 22%, masking 4%, alignment 74%`}>
                <Trans comment="Overall progress weighted by observed stage durations.">Overall progress</Trans>
              </span>
              <small className="text-muted-foreground">{taskProgressSummary(task)}</small>
              <strong className="ml-auto font-mono font-medium text-foreground">{overall}%</strong>
            </div>
            <Progress value={overall} aria-label={t`${task.name} overall time progress`}><ProgressValue /></Progress>
            <div className="mt-3.5 grid grid-cols-[minmax(180px,1fr)_minmax(330px,auto)] items-end gap-4.5 max-[920px]:grid-cols-1 max-[920px]:items-start">
              <span className="flex min-w-0 flex-col gap-1">
                <strong className="text-sm font-medium text-foreground">{t`Current stage: ${stageLabel(currentStageDefinition)}`}</strong>
                <small className="truncate text-sm text-muted-foreground">{currentStage.phase ? phaseLabel(currentStage.phase) : stageStatusLabel(currentStage.status)}</small>
              </span>
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
                  <span className="relative grid w-6 shrink-0 items-start justify-items-center self-stretch" aria-hidden="true">
                    <span className={cn("relative z-10 grid size-5 place-items-center rounded-full border bg-card text-[0.72rem] font-semibold text-muted-foreground [&_svg]:size-3", current.status === "running" && "border-primary text-primary ring-3 ring-primary/10", current.status === "completed" && "border-emerald-600 bg-emerald-500/10 text-emerald-600")}>
                      {current.status === "completed" ? <CheckCircle2 /> : stageIndex + 1}
                    </span>
                    {stageIndex < STAGES.length - 1 && <span className="absolute top-5 -bottom-3 left-1/2 w-px -translate-x-1/2 bg-border" />}
                  </span>
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
                  <Button variant={current.status === "running" ? "destructive" : "ghost"} size="sm" disabled={current.status !== "running" && action.blocked} onClick={() => onStageAction(task, stage.key)}>
                    {current.status === "running" ? <Square data-icon="inline-start" /> : current.status === "completed" ? <RotateCcw data-icon="inline-start" /> : <Play data-icon="inline-start" />}
                    {action.label}
                  </Button>
                </div>
              );
            })}
          </div>
        </>
      )}
    </article>
  );
}

export function TaskWorkspace({
  groups,
  clockMs,
  hasRunningStage,
  taskDetailOpen,
  selectedTask,
  taskDetailUsesSplitView,
  dragOver,
  canChangeQueuedTask,
  isWaitingForEnqueue,
  localiseUserMessage,
  onOpenSourcePicker,
  onOpenProject,
  onDragEnter,
  onDragLeave,
  onDrop,
  onEnqueueTask,
  onEditTask,
  onRemoveTask,
  onOpenTaskDetail,
  onStageAction,
}: TaskWorkspaceProps) {
  const shouldReduceMotion = useReducedMotion();
  const hasTasks = groups.some((group) => group.items.length > 0);

  if (!hasTasks) {
    return (
      <section
        className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 pt-14 pb-18 text-center"
        onDragEnter={(event) => { event.preventDefault(); onDragEnter(); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div className={cn("mb-5.5 grid size-17.5 place-items-center rounded-[18px] border bg-card text-muted-foreground transition-[border-color,color,transform,background] duration-200 [&_svg]:size-7", dragOver && "scale-[1.03] border-primary/50 bg-primary/10 text-primary")} aria-hidden="true"><FileStack /></div>
        <h2 className="text-[28px] font-semibold tracking-[-0.045em] text-foreground"><Trans context="empty state" comment="Empty task list heading.">No tasks yet</Trans></h2>
        <p className="mt-3 text-base leading-relaxed text-muted-foreground"><Trans comment="Drop source media here or choose files below. Existing projects are opened separately.">Drop OSV or dual-fisheye media here,<br />or use Open project to resume an existing project.</Trans></p>
        <div className="mt-6 flex items-center gap-2.5 max-[760px]:w-[min(280px,100%)] max-[760px]:flex-col max-[760px]:items-stretch">
          <Button size="lg" onClick={() => void onOpenSourcePicker()}><Upload data-icon="inline-start" /><Trans context="file picker action" comment="Button opens a file picker for source media.">Choose files</Trans></Button>
          <Button size="lg" variant="outline" onClick={() => void onOpenProject()}><FolderOpen data-icon="inline-start" /><Trans context="project action" comment="Open an existing resumable project.">Open project</Trans></Button>
        </div>
        <section className="mt-7 w-[min(520px,100%)] text-left max-[760px]:w-[min(320px,100%)]" aria-labelledby="supported-formats-title">
          <h2 id="supported-formats-title" className="mb-2.5 text-xs font-semibold tracking-wide text-muted-foreground"><Trans>Supported inputs</Trans></h2>
          <div className="grid grid-cols-2 gap-2.5 max-[760px]:grid-cols-1">
            <SupportedFormatCard icon={Film} title={<Trans comment="DJI Osmo 360 source media files.">Osmo 360 source files</Trans>} detail="OSV" />
            <SupportedFormatCard icon={Folder} title={<Trans comment="A project folder containing a resumable reconstruction.">Project folder</Trans>} detail={<Trans>Resume an unfinished reconstruction task</Trans>} />
          </div>
        </section>
      </section>
    );
  }

  return (
    <m.div
      className="mr-auto w-full"
      initial={false}
      animate={{ width: taskDetailOpen && selectedTask && taskDetailUsesSplitView ? "calc(100% - 460px)" : "100%" }}
      transition={shouldReduceMotion ? { duration: 0 } : TASK_DETAIL_DRAWER_TRANSITION}
    >
      <section className="mx-auto w-full max-w-[1440px] px-8 pt-6.5 pb-14 max-[760px]:px-3.5 max-[760px]:pt-5.5 max-[760px]:pb-11.5">
        <div className="grid gap-7">
          {groups.filter((group) => group.key === "completed" || group.items.length > 0).map((group) => (
            <section key={group.key}>
              <div className="mb-3 flex items-center px-0.5"><div className="flex items-center gap-2.5"><h2 className="text-lg font-semibold">{group.title}</h2><Badge variant="secondary">{group.items.length}</Badge></div></div>
              {group.items.length > 0 ? (
                <div className="flex flex-col gap-3.5 overflow-visible">
                  {group.items.map((task) => (
                    <TaskCard
                      key={task.projectId}
                      task={task}
                      clockMs={clockMs}
                      hasRunningStage={hasRunningStage}
                      taskDetailOpen={taskDetailOpen}
                      selectedTask={selectedTask}
                      canChangeQueuedTask={canChangeQueuedTask}
                      isWaitingForEnqueue={isWaitingForEnqueue}
                      localiseUserMessage={localiseUserMessage}
                      onEnqueueTask={onEnqueueTask}
                      onEditTask={onEditTask}
                      onRemoveTask={onRemoveTask}
                      onOpenTaskDetail={onOpenTaskDetail}
                      onStageAction={onStageAction}
                    />
                  ))}
                </div>
              ) : (
                <Empty className="min-h-39 rounded-xl border border-dashed bg-card/70 text-muted-foreground [&_[data-slot=empty-description]]:text-muted-foreground [&_[data-slot=empty-title]]:text-muted-foreground">
                  <EmptyHeader>
                    <EmptyMedia variant="icon"><CheckCircle2 /></EmptyMedia>
                    <EmptyTitle><Trans>No completed tasks yet</Trans></EmptyTitle>
                    <EmptyDescription><Trans>Completed reconstruction tasks will appear here.</Trans></EmptyDescription>
                  </EmptyHeader>
                </Empty>
              )}
            </section>
          ))}
        </div>
      </section>
    </m.div>
  );
}

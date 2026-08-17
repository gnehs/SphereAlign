import { useRef, type Dispatch, type DragEvent, type SetStateAction } from "react";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { CircleHelp, FileStack, Trash2 } from "lucide-react";
import { useShallow } from "zustand/react/shallow";

import { ProcessingSettingsFields } from "@/components/processing-settings-fields";
import { SourceListItem } from "@/components/source-media";
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
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import {
  customLutPathIsInvalid,
  localiseUserMessage,
  sourceFromPath,
} from "@/lib/pipeline";
import { useAppStore } from "@/stores/app-store";

 export interface TaskEditorDialogProps {
  onOpenChange: (open: boolean) => void;
  dragOver: boolean;
  setDragOver: Dispatch<SetStateAction<boolean>>;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onSourcePicker: () => void | Promise<void>;
  onOutputPicker: () => void | Promise<void>;
  onLutPicker: () => void | Promise<void>;
  onGpuPreferenceTouched: () => void;
  onSubmit: () => void | Promise<void>;
}

export function TaskEditorDialog({
  onOpenChange,
  dragOver,
  setDragOver,
  onDrop,
  onSourcePicker,
  onOutputPicker,
  onLutPicker,
  onGpuPreferenceTouched,
  onSubmit,
}: TaskEditorDialogProps) {
  const {
    open,
    editingTaskId,
    nameDraft,
    setNameDraft,
    outputDraft,
    setOutputDraft,
    settingsDraft,
    setSettingsDraft,
    sourcePaths,
    setSourcePaths,
    sourceInspection,
    sourceColorInspection,
    setSourceColorInspection,
    doctor,
  } = useAppStore(useShallow((state) => ({
    open: state.taskDialogOpen,
    editingTaskId: state.editingTaskId,
    nameDraft: state.nameDraft,
    setNameDraft: state.setNameDraft,
    outputDraft: state.outputDraft,
    setOutputDraft: state.setOutputDraft,
    settingsDraft: state.settingsDraft,
    setSettingsDraft: state.setSettingsDraft,
    sourcePaths: state.sourcePaths,
    setSourcePaths: state.setSourcePaths,
    sourceInspection: state.sourceInspection,
    sourceColorInspection: state.sourceColorInspection,
    setSourceColorInspection: state.setSourceColorInspection,
    doctor: state.doctor,
  })));
  const selectedSources = sourcePaths.map(sourceFromPath);
  const removeSource = (path: string) => {
    setSourcePaths((current) => current.filter((sourcePath) => sourcePath !== path));
    setSourceColorInspection(null);
  };
  const taskNameInputRef = useRef<HTMLInputElement>(null);
  const lutPathInvalid = customLutPathIsInvalid(settingsDraft.extract.lutPath);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
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
                  <Field>
                    <FieldLabel><Trans context="source input" comment="Input media for a reconstruction task.">Sources</Trans></FieldLabel>
                    <FieldContent>
                      <div
                        className={`flex min-h-18 items-center gap-3 rounded-lg border border-dashed px-4 py-3 text-muted-foreground transition-colors hover:border-primary/50 hover:bg-primary/5 hover:text-primary ${dragOver ? "border-primary/50 bg-primary/10 text-primary" : ""}`}
                        onDragOver={(event) => event.preventDefault()}
                        onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={(event) => { setDragOver(false); onDrop(event); }}
                      >
                        <FileStack className="size-4.5 shrink-0" />
                        <span className="flex-1 text-sm"><Trans>Drop OSV or dual-fisheye media</Trans></span>
                        <Button type="button" variant="outline" size="sm" onClick={() => void onSourcePicker()}>{t`Choose sources`}</Button>
                      </div>
                      {selectedSources.length > 0 && <div className="mt-2 overflow-hidden rounded-lg border">{selectedSources.map((source) => <SourceListItem key={source.id} source={source} title={source.label} detail={source.detail} removeLabel={t`Remove ${source.label}`} onRemove={() => removeSource(source.path)} />)}</div>}
                      <p className="mt-2 text-sm text-muted-foreground">{sourceInspection ? localiseUserMessage(sourceInspection) : t`Choose multiple OSV or dual-fisheye media files.`}</p>
                    </FieldContent>
                  </Field>
                  <Field><FieldLabel htmlFor="output-path"><Trans context="output destination" comment="Folder where project metadata and reconstruction output are saved.">Output folder</Trans></FieldLabel><FieldContent><div className="flex items-center gap-2"><Input className="flex-1" id="output-path" value={outputDraft} disabled={Boolean(editingTaskId)} placeholder={t`Defaults beside the first source: colmap-file-name`} onChange={(event) => setOutputDraft(event.currentTarget.value)} />{!editingTaskId && <Button type="button" variant="outline" size="sm" onClick={() => void onOutputPicker()}><Trans context="output folder picker action" comment="Button opens a folder picker to choose a different output location.">Choose another</Trans></Button>}</div><FieldDescription>{editingTaskId ? t`Saving a new task name also renames the output folder; unsupported filename characters become hyphens.` : t`After creation, project information is saved in the output folder so the task can resume after an interruption.`}</FieldDescription></FieldContent></Field>
                </FieldGroup>
              </div>
            </section>

            <ProcessingSettingsFields
              settings={settingsDraft}
              onSettingsChange={setSettingsDraft}
              sourceColorInspection={sourceColorInspection}
              doctor={doctor}
              onChooseLut={onLutPicker}
              onGpuPreferenceTouched={onGpuPreferenceTouched}
            />
          </div>
        </div>

        <DialogFooter className="border-t px-5 py-4">
          <DialogClose render={<Button variant="outline" />}><Trans>Cancel</Trans></DialogClose>
          <Button onClick={() => void onSubmit()} disabled={!selectedSources.length || lutPathInvalid}>{editingTaskId ? <Trans context="task action" comment="Save changes to a queued task.">Save changes</Trans> : <Trans context="task action" comment="Create the reconstruction task.">Create task</Trans>}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export interface RemoveTaskDialogProps {
  onConfirm: () => void;
}

export function RemoveTaskDialog({ onConfirm }: RemoveTaskDialogProps) {
  const deletingTaskId = useAppStore((state) => state.deletingTaskId);
  const setDeletingTaskId = useAppStore((state) => state.setDeletingTaskId);
  return (
    <Dialog open={Boolean(deletingTaskId)} onOpenChange={(open) => { if (!open) setDeletingTaskId(null); }}>
      <DialogContent showCloseButton={false}>
        <DialogHeader><DialogTitle><Trans context="remove task confirmation" comment="Confirmation dialog for removing a queued task.">Remove task from queue?</Trans></DialogTitle><DialogDescription><Trans>Only the task and queue state are removed; the existing output folder is not deleted.</Trans></DialogDescription></DialogHeader>
        <DialogFooter><DialogClose render={<Button variant="ghost" />}><Trans>Cancel</Trans></DialogClose><Button variant="destructive" onClick={onConfirm}><Trash2 /><Trans context="task action" comment="Remove the queued task, while keeping its output folder.">Remove task</Trans></Button></DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

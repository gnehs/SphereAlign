import { useState } from "react";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { LoaderCircle, Settings2 } from "lucide-react";
import { AlignmentSettingsFields } from "@/components/alignment-settings-fields";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { backendErrorMessage, normalisePipelineSettings, type PipelineSettings, type Task } from "@/lib/pipeline";
import { useAppStore } from "@/stores/app-store";

interface TaskAlignmentSettingsProps {
  task: Task;
  disabled: boolean;
  onSave: (task: Task, align: PipelineSettings["align"]) => Promise<void>;
}

export function TaskAlignmentSettings({ task, disabled, onSave }: TaskAlignmentSettingsProps) {
  const doctor = useAppStore((state) => state.doctor);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(() => normalisePipelineSettings(task.settings));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const changeOpen = (next: boolean) => {
    if (saving) return;
    if (next) {
      setDraft(normalisePipelineSettings(task.settings));
      setError("");
    }
    setOpen(next);
  };
  const save = async () => {
    if (disabled || saving) return;
    setSaving(true);
    setError("");
    try {
      await onSave(task, normalisePipelineSettings(draft).align);
      setOpen(false);
    } catch (cause) {
      setError(backendErrorMessage(cause));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogTrigger render={<Button variant="outline" size="sm" disabled={disabled} />}>
        <Settings2 data-icon="inline-start" />
        <Trans>Alignment settings</Trans>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" showCloseButton={!saving}>
        <DialogHeader>
          <DialogTitle><Trans>Alignment settings</Trans></DialogTitle>
          <DialogDescription>
            <Trans>Save settings for the next alignment run. Existing reconstruction outputs are kept until you rerun alignment.</Trans>
          </DialogDescription>
        </DialogHeader>
        <p className="truncate font-medium" title={task.name}>{task.name}</p>
        <fieldset disabled={disabled || saving} className="min-w-0 border-0 p-0 disabled:opacity-60">
          <AlignmentSettingsFields settings={draft} onSettingsChange={setDraft} doctor={doctor} />
        </fieldset>
        {disabled && <p className="text-sm text-muted-foreground"><Trans>Wait for the running stage to finish before saving settings.</Trans></p>}
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        <DialogFooter>
          <Button variant="outline" disabled={saving} onClick={() => changeOpen(false)}><Trans>Cancel</Trans></Button>
          <Button disabled={disabled || saving} onClick={() => void save()}>
            {saving && <LoaderCircle data-icon="inline-start" className="animate-spin" />}
            {saving ? t`Saving…` : t`Save settings`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

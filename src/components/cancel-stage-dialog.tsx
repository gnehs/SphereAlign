import { Trans } from "@lingui/react/macro";
import { Square } from "lucide-react";

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

export interface CancelStageDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void | Promise<void>;
}

export function CancelStageDialog({ open, onOpenChange, onConfirm }: CancelStageDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle><Trans>Cancel the current stage?</Trans></DialogTitle>
          <DialogDescription><Trans>The current stage will stop. Completed output is kept, and you can resume the task later.</Trans></DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose render={<Button variant="ghost" />}><Trans>Keep running</Trans></DialogClose>
          <Button variant="destructive" onClick={() => void onConfirm()}><Square data-icon="inline-start" /><Trans>Cancel stage</Trans></Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

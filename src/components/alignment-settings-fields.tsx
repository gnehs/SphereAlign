import { useId, type Dispatch, type SetStateAction } from "react";
import { t } from "@lingui/core/macro";
import { Trans } from "@lingui/react/macro";
import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Field, FieldContent, FieldDescription, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { gpuDeviceLabel, type DoctorReport, type FeaturePipeline, type MapperMode, type PipelineSettings } from "@/lib/pipeline";

interface AlignmentSettingsFieldsProps {
  settings: PipelineSettings;
  onSettingsChange: Dispatch<SetStateAction<PipelineSettings>>;
  doctor: DoctorReport;
  onGpuPreferenceTouched?: () => void;
}

export function AlignmentSettingsFields({ settings, onSettingsChange, doctor, onGpuPreferenceTouched }: AlignmentSettingsFieldsProps) {
  const idPrefix = useId();
  const featurePipeline = settings.align.featurePipeline;
  const featurePipelineItems: Array<{ value: FeaturePipeline; label: string }> = [
    { value: "sift", label: t`SIFT (fast default)` },
    { value: "aliked-n32-lightglue", label: t`ALIKED-N32 + LightGlue` },
    { value: "aliked-n16rot-lightglue", label: t`ALIKED-N16Rot + LightGlue` },
  ];
  const mapperModeItems: Array<{ value: MapperMode; label: string }> = [
    { value: "incremental", label: t`COLMAP incremental mapper` },
    { value: "auto", label: t`GLOMAP with validated fallback` },
    { value: "global", label: t`GLOMAP only` },
  ];

  return (
    <div className="flex flex-col gap-2">
      <Field>
        <FieldLabel htmlFor={`${idPrefix}-temporal-window`}><Trans>Neighboring frame pairs</Trans></FieldLabel>
        <Input id={`${idPrefix}-temporal-window`} type="number" min={2} max={30} step={1} value={settings.align.temporalWindow}
          onChange={(event) => onSettingsChange((current) => ({ ...current, align: { ...current.align, temporalWindow: Math.round(Math.min(30, Math.max(2, Number(event.target.value) || 2))) } }))} />
        <FieldDescription><Trans>Count selected physical frames within each capture. Try 15 for difficult sections; larger windows add work and need geometric verification.</Trans></FieldDescription>
      </Field>
      <Field className="min-h-7 border-0 bg-transparent px-0 py-0.5">
        <FieldLabel htmlFor={`${idPrefix}-feature-pipeline`}><Trans comment="Select the local feature extractor and matcher used by COLMAP alignment.">Feature matching method</Trans></FieldLabel>
        <Select
          items={featurePipelineItems}
          value={featurePipeline}
          onValueChange={(value) => {
            const nextPipeline = (value ?? "sift") as FeaturePipeline;
            onSettingsChange((current) => ({
              ...current,
              align: {
                ...current.align,
                featurePipeline: nextPipeline,
                ...(nextPipeline !== "sift" && doctor.gpuAvailable === true ? { useGpu: true } : {}),
              },
            }));
          }}
        >
          <SelectTrigger id={`${idPrefix}-feature-pipeline`} className="w-full"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {featurePipelineItems.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}
            </SelectGroup>
          </SelectContent>
        </Select>
        {featurePipeline === "sift" ? (
          <FieldDescription><Trans comment="Explain the default SIFT feature matching option.">Fastest and most mature option for typical scenes.</Trans></FieldDescription>
        ) : (
          <Alert>
            <AlertTriangle />
            <AlertTitle><Trans comment="Heading for the learned feature matching performance tradeoff.">Slower, with a higher matching rate</Trans></AlertTitle>
            <AlertDescription>
              {featurePipeline === "aliked-n32-lightglue"
                ? <Trans comment="Explain the ALIKED-N32 and LightGlue option.">Usually improves matching and camera registration in low-texture or difficult scenes. N32 is the recommended ALIKED option, but alignment takes longer.</Trans>
                : <Trans comment="Explain the rotation-aware ALIKED-N16Rot and LightGlue option.">Usually improves matching under large viewpoint or rotation changes, but alignment takes longer.</Trans>}
            </AlertDescription>
          </Alert>
        )}
      </Field>
      {featurePipeline !== "sift" && (
        <FieldDescription>
          {featurePipeline === "aliked-n32-lightglue"
            ? <Trans>Missing aliked-n32.onnx and aliked-lightglue.onnx models are downloaded automatically into the shared YOLO model folder. The verified accelerated path uses an NVIDIA CUDA GPU.</Trans>
            : <Trans>Missing aliked-n16rot.onnx and aliked-lightglue.onnx models are downloaded automatically into the shared YOLO model folder. The verified accelerated path uses an NVIDIA CUDA GPU.</Trans>}
        </FieldDescription>
      )}
      <Field className="min-h-7 border-0 bg-transparent px-0 py-0.5">
        <FieldLabel htmlFor={`${idPrefix}-mapper-mode`}><Trans comment="Select the mapper that estimates camera poses after feature matching.">Camera pose solver</Trans></FieldLabel>
        <Select
          items={mapperModeItems}
          value={settings.align.mapperMode}
          onValueChange={(value) => {
            const nextMode = (value ?? "incremental") as MapperMode;
            onSettingsChange((current) => ({
              ...current,
              align: { ...current.align, mapperMode: nextMode },
            }));
          }}
        >
          <SelectTrigger id={`${idPrefix}-mapper-mode`} className="w-full"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {mapperModeItems.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}
            </SelectGroup>
          </SelectContent>
        </Select>
        {settings.align.mapperMode === "incremental" && (
          <FieldDescription><Trans>The mature sequential COLMAP mapper; usually slower but tolerant of incomplete global connectivity.</Trans></FieldDescription>
        )}
        {settings.align.mapperMode === "auto" && (
          <FieldDescription><Trans>Builds a safe incremental calibration seed when needed, then keeps the GLOMAP result only if rig coverage and geometry validation pass.</Trans></FieldDescription>
        )}
        {settings.align.mapperMode === "global" && (
          <Alert>
            <AlertTriangle />
            <AlertTitle><Trans>Validated priors required</Trans></AlertTitle>
            <AlertDescription><Trans>Runs GLOMAP directly and stops if this project does not already contain compatible focal and rig priors.</Trans></AlertDescription>
          </Alert>
        )}
      </Field>
      <Field orientation="horizontal" className="min-h-7 border-0 bg-transparent px-0 py-0.5">
        <Switch
          id={`${idPrefix}-use-intra-source-loop-closure`}
          size="sm"
          checked={settings.align.useIntraSourceLoopClosure}
          onCheckedChange={(checked) => onSettingsChange((current) => ({
            ...current,
            align: { ...current.align, useIntraSourceLoopClosure: checked },
          }))}
        />
        <FieldContent>
          <FieldLabel htmlFor={`${idPrefix}-use-intra-source-loop-closure`}><Trans comment="Find long-distance revisits within one source video to help close a reconstruction loop.">Single-video loop closure</Trans></FieldLabel>
          <FieldDescription><Trans comment="Explain when the optional single-video loop-closure setting is useful.">Turn this on if the video passes through the same place again.</Trans></FieldDescription>
          {settings.align.useIntraSourceLoopClosure && (
            <Alert>
              <AlertTriangle />
              <AlertTitle><Trans>Possible incorrect matches</Trans></AlertTitle>
              <AlertDescription><Trans comment="Warn that visually repetitive scenes can cause a false loop closure.">Similar-looking corridors or objects may be mistaken for a revisit, creating incorrect matches.</Trans></AlertDescription>
            </Alert>
          )}
        </FieldContent>
      </Field>
      <Field orientation="horizontal" className="mt-2.5 min-h-7 border-0 bg-transparent px-0 py-0.5 [&_[data-slot=field-label]]:cursor-pointer [&_[data-slot=field-label]]:font-normal" data-disabled={doctor.gpuAvailable === false || undefined}>
        <Switch
          id={`${idPrefix}-use-gpu`}
          size="sm"
          disabled={doctor.gpuAvailable === false}
          checked={settings.align.useGpu}
          onCheckedChange={(checked) => {
            onGpuPreferenceTouched?.();
            onSettingsChange((current) => ({ ...current, align: { ...current.align, useGpu: checked } }));
          }}
        />
        <FieldContent>
          <FieldLabel htmlFor={`${idPrefix}-use-gpu`}><Trans comment="Use CUDA acceleration for the COLMAP alignment stage.">Use CUDA acceleration for alignment</Trans></FieldLabel>
          <FieldDescription>{doctor.gpuAvailable === false ? t`No usable COLMAP CUDA acceleration was detected, so the CPU will be used.` : t`Enabled by default when a CUDA-capable NVIDIA GPU is detected; falls back to the CPU if execution fails.`}</FieldDescription>
        </FieldContent>
      </Field>
      {doctor.gpuAvailable === true && doctor.gpuDevices.length > 1 && (
        <Field data-disabled={!settings.align.useGpu || undefined}>
          <FieldLabel htmlFor={`${idPrefix}-gpu-index`}><Trans comment="Select which detected GPU should run alignment.">Select GPU</Trans></FieldLabel>
          <Select
            items={[{ value: "-1", label: t`Automatic GPU selection` }, ...doctor.gpuDevices.map((device) => ({ value: String(device.index), label: gpuDeviceLabel(device, doctor.gpuDevices) }))]}
            value={settings.align.gpuIndex}
            onValueChange={(gpuIndex) => onSettingsChange((current) => ({ ...current, align: { ...current.align, gpuIndex: gpuIndex ?? String(doctor.gpuDevices[0].index) } }))}
            disabled={!settings.align.useGpu}
          >
            <SelectTrigger id={`${idPrefix}-gpu-index`} className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="-1"><Trans>Automatic GPU selection</Trans></SelectItem>
                {doctor.gpuDevices.map((device) => <SelectItem key={device.index} value={String(device.index)}>{gpuDeviceLabel(device, doctor.gpuDevices)}</SelectItem>)}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>
      )}
    </div>
  );
}

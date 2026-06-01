#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Create a RunPod probe Pod, discover its datacenter, delete it, create a network
volume in that datacenter, then create the final Pod attached to that volume.

Required environment:
  RUNPOD_API_KEY       RunPod API key.
  GPU_TYPE_ID          RunPod GPU id, for example "NVIDIA L40S".
  TEMPLATE_ID          Template id to use. Required unless IMAGE_NAME is set.
  IMAGE_NAME           Docker image to use. Required unless TEMPLATE_ID is set.

Optional environment:
  CUDA_VERSION         Exact allowed CUDA version. Default: 13.0
  CUDA_VERSIONS_JSON   JSON array of allowed CUDA versions. Overrides CUDA_VERSION.
  POD_NAME             Final pod name. Default: cuda-volume-pod
  TEMP_POD_NAME        Probe pod name. Default: ${POD_NAME}-probe
  NETWORK_VOLUME_NAME  Network volume name. Default: ${POD_NAME}-volume-<timestamp>
  NETWORK_VOLUME_GB    Network volume size in GB. Default: 200
  GPU_COUNT            GPU count. Default: 1
  CONTAINER_DISK_GB    Container disk size in GB. Default: 80
  TEMP_VOLUME_GB       Probe pod local volume size in GB. Default: 20
  VOLUME_MOUNT_PATH    Mount path. Default: /workspace
  CLOUD_TYPE           RunPod cloud type. Default: SECURE
  INTERRUPTIBLE        true or false. Default: false
  PORTS_JSON           JSON array, for example '["22/tcp","8888/http"]'. Default: []
  ENV_JSON             JSON object passed to the pod. Default: {}
  DATA_CENTER_IDS_JSON Optional JSON array limiting probe datacenters.
  EXTRA_POD_JSON       Optional JSON object merged into probe and final payloads.
  RUNPOD_API_BASE      API base URL. Default: https://rest.runpod.io/v1

Example:
  RUNPOD_API_KEY=... \
  GPU_TYPE_ID="NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  CUDA_VERSION=13.0 \
  TEMPLATE_ID=obgryfbuad \
  POD_NAME=tofy-cuda13 \
  ./scripts/runpod_cuda_volume_pod.sh
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

json_type() {
  jq -er "$1 | type" >/dev/null
}

api() {
  local method="$1"
  local path="$2"
  local body_file="${3:-}"

  if [[ -n "$body_file" ]]; then
    curl -fsS \
      --request "$method" \
      --url "${RUNPOD_API_BASE}${path}" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      --header "Content-Type: application/json" \
      --data-binary @"$body_file"
  else
    curl -fsS \
      --request "$method" \
      --url "${RUNPOD_API_BASE}${path}" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}"
  fi
}

extract_pod_id() {
  jq -r 'if type == "object" and (.id | type == "string") then .id else empty end' "$1"
}

extract_volume_id() {
  jq -r 'if type == "object" and (.id | type == "string") then .id else empty end' "$1"
}

extract_datacenter_id() {
  jq -r '
    .machine.dataCenterId
    // .machine.datacenterId
    // .machine.data_center_id
    // .machine.dataCenter.id
    // .networkVolume.dataCenterId
    // empty
  ' "$1"
}

summarize_pod() {
  jq '{
    id,
    name,
    desiredStatus,
    image,
    templateId,
    gpu,
    machine: (
      if .machine == null then null else {
        machineId: .machine.machineId,
        dataCenterId: .machine.dataCenterId,
        gpuDisplayName: .machine.gpuDisplayName,
        gpuAvailable: .machine.gpuAvailable,
        secureCloud: .machine.secureCloud
      } end
    ),
    networkVolume
  }' "$1"
}

delete_temp_pod() {
  if [[ -n "${TEMP_POD_ID:-}" && "${TEMP_POD_DELETED:-0}" != "1" ]]; then
    echo "Deleting probe Pod ${TEMP_POD_ID}..."
    api DELETE "/pods/${TEMP_POD_ID}" >/dev/null || true
    TEMP_POD_DELETED=1
  fi
}

cleanup() {
  delete_temp_pod
  if [[ "${KEEP_RUNPOD_TMP:-0}" != "1" && -n "${TMPDIR_RUNPOD:-}" ]]; then
    rm -rf "$TMPDIR_RUNPOD"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd curl
require_cmd jq

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${GPU_TYPE_ID:?GPU_TYPE_ID is required}"

if [[ -z "${TEMPLATE_ID:-}" && -z "${IMAGE_NAME:-}" ]]; then
  die "set TEMPLATE_ID or IMAGE_NAME"
fi

RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
CUDA_VERSION="${CUDA_VERSION:-13.0}"
CUDA_VERSIONS_JSON="${CUDA_VERSIONS_JSON:-}"
POD_NAME="${POD_NAME:-cuda-volume-pod}"
TEMP_POD_NAME="${TEMP_POD_NAME:-${POD_NAME}-probe}"
NETWORK_VOLUME_NAME="${NETWORK_VOLUME_NAME:-${POD_NAME}-volume-$(date +%Y%m%d-%H%M%S)}"
NETWORK_VOLUME_GB="${NETWORK_VOLUME_GB:-200}"
GPU_COUNT="${GPU_COUNT:-1}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-80}"
TEMP_VOLUME_GB="${TEMP_VOLUME_GB:-20}"
VOLUME_MOUNT_PATH="${VOLUME_MOUNT_PATH:-/workspace}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
INTERRUPTIBLE="${INTERRUPTIBLE:-false}"
PORTS_JSON="${PORTS_JSON:-[]}"
ENV_JSON="${ENV_JSON:-{}}"
DATA_CENTER_IDS_JSON="${DATA_CENTER_IDS_JSON:-[]}"
EXTRA_POD_JSON="${EXTRA_POD_JSON:-{}}"

[[ "$INTERRUPTIBLE" == "true" || "$INTERRUPTIBLE" == "false" ]] || die "INTERRUPTIBLE must be true or false"
jq -en --argjson value "$NETWORK_VOLUME_GB" '$value | numbers' >/dev/null || die "NETWORK_VOLUME_GB must be a number"
jq -en --argjson value "$GPU_COUNT" '$value | numbers' >/dev/null || die "GPU_COUNT must be a number"
jq -en --argjson value "$CONTAINER_DISK_GB" '$value | numbers' >/dev/null || die "CONTAINER_DISK_GB must be a number"
jq -en --argjson value "$TEMP_VOLUME_GB" '$value | numbers' >/dev/null || die "TEMP_VOLUME_GB must be a number"

if [[ -z "$CUDA_VERSIONS_JSON" ]]; then
  CUDA_VERSIONS_JSON="$(jq -nc --arg version "$CUDA_VERSION" '[$version]')"
fi

printf '%s' "$CUDA_VERSIONS_JSON" | json_type '. as $v | if ($v | type) == "array" then $v else error("not array") end' || die "CUDA_VERSIONS_JSON must be a JSON array"
printf '%s' "$PORTS_JSON" | json_type '. as $v | if ($v | type) == "array" then $v else error("not array") end' || die "PORTS_JSON must be a JSON array"
printf '%s' "$ENV_JSON" | json_type '. as $v | if ($v | type) == "object" then $v else error("not object") end' || die "ENV_JSON must be a JSON object"
printf '%s' "$DATA_CENTER_IDS_JSON" | json_type '. as $v | if ($v | type) == "array" then $v else error("not array") end' || die "DATA_CENTER_IDS_JSON must be a JSON array"
printf '%s' "$EXTRA_POD_JSON" | json_type '. as $v | if ($v | type) == "object" then $v else error("not object") end' || die "EXTRA_POD_JSON must be a JSON object"

TMPDIR_RUNPOD="$(mktemp -d)"
TEMP_POD_ID=""
TEMP_POD_DELETED=0
trap cleanup EXIT

TEMP_PAYLOAD="${TMPDIR_RUNPOD}/temp-pod.json"
TEMP_RESPONSE="${TMPDIR_RUNPOD}/temp-pod-response.json"
TEMP_GET_RESPONSE="${TMPDIR_RUNPOD}/temp-pod-get-response.json"
VOLUME_PAYLOAD="${TMPDIR_RUNPOD}/network-volume.json"
VOLUME_RESPONSE="${TMPDIR_RUNPOD}/network-volume-response.json"
FINAL_PAYLOAD="${TMPDIR_RUNPOD}/final-pod.json"
FINAL_RESPONSE="${TMPDIR_RUNPOD}/final-pod-response.json"

jq -n \
  --arg name "$TEMP_POD_NAME" \
  --arg cloudType "$CLOUD_TYPE" \
  --arg gpuTypeId "$GPU_TYPE_ID" \
  --arg templateId "${TEMPLATE_ID:-}" \
  --arg imageName "${IMAGE_NAME:-}" \
  --arg mountPath "$VOLUME_MOUNT_PATH" \
  --argjson gpuCount "$GPU_COUNT" \
  --argjson containerDiskInGb "$CONTAINER_DISK_GB" \
  --argjson tempVolumeInGb "$TEMP_VOLUME_GB" \
  --argjson interruptible "$INTERRUPTIBLE" \
  --argjson cudaVersions "$CUDA_VERSIONS_JSON" \
  --argjson ports "$PORTS_JSON" \
  --argjson env "$ENV_JSON" \
  --argjson dataCenterIds "$DATA_CENTER_IDS_JSON" \
  --argjson extra "$EXTRA_POD_JSON" \
  '
  {
    name: $name,
    cloudType: $cloudType,
    computeType: "GPU",
    gpuTypeIds: [$gpuTypeId],
    gpuTypePriority: "availability",
    gpuCount: $gpuCount,
    dataCenterPriority: "availability",
    allowedCudaVersions: $cudaVersions,
    containerDiskInGb: $containerDiskInGb,
    volumeInGb: $tempVolumeInGb,
    volumeMountPath: $mountPath,
    interruptible: $interruptible,
    ports: $ports,
    env: $env
  }
  + (if $templateId != "" then {templateId: $templateId} else {imageName: $imageName} end)
  + (if ($dataCenterIds | length) > 0 then {dataCenterIds: $dataCenterIds} else {} end)
  + $extra
  ' > "$TEMP_PAYLOAD"

echo "Creating probe Pod for GPU '${GPU_TYPE_ID}' and CUDA versions ${CUDA_VERSIONS_JSON}..."
api POST "/pods" "$TEMP_PAYLOAD" > "$TEMP_RESPONSE" || die "probe Pod creation request failed"

TEMP_POD_ID="$(extract_pod_id "$TEMP_RESPONSE")"
if [[ -z "$TEMP_POD_ID" ]]; then
  echo "RunPod did not return a probe Pod id. Response:" >&2
  jq . "$TEMP_RESPONSE" >&2 || cat "$TEMP_RESPONSE" >&2
  exit 1
fi

echo "Probe Pod created: ${TEMP_POD_ID}"
DATA_CENTER_ID="$(extract_datacenter_id "$TEMP_RESPONSE")"

if [[ -z "$DATA_CENTER_ID" ]]; then
  echo "Waiting for probe Pod datacenter..."
  for _ in $(seq 1 30); do
    sleep 2
    if api GET "/pods/${TEMP_POD_ID}" > "$TEMP_GET_RESPONSE"; then
      DATA_CENTER_ID="$(extract_datacenter_id "$TEMP_GET_RESPONSE")"
    fi
    [[ -n "$DATA_CENTER_ID" ]] && break
  done
fi

if [[ -z "$DATA_CENTER_ID" ]]; then
  echo "Could not determine probe Pod datacenter. Last response:" >&2
  if [[ -s "$TEMP_GET_RESPONSE" ]]; then
    jq . "$TEMP_GET_RESPONSE" >&2 || cat "$TEMP_GET_RESPONSE" >&2
  else
    jq . "$TEMP_RESPONSE" >&2 || cat "$TEMP_RESPONSE" >&2
  fi
  exit 1
fi

echo "Probe Pod datacenter: ${DATA_CENTER_ID}"
delete_temp_pod

jq -n \
  --arg dataCenterId "$DATA_CENTER_ID" \
  --arg name "$NETWORK_VOLUME_NAME" \
  --argjson size "$NETWORK_VOLUME_GB" \
  '{dataCenterId: $dataCenterId, name: $name, size: $size}' > "$VOLUME_PAYLOAD"

echo "Creating ${NETWORK_VOLUME_GB} GB network volume '${NETWORK_VOLUME_NAME}' in ${DATA_CENTER_ID}..."
api POST "/networkvolumes" "$VOLUME_PAYLOAD" > "$VOLUME_RESPONSE" || die "network volume creation request failed"

NETWORK_VOLUME_ID="$(extract_volume_id "$VOLUME_RESPONSE")"
if [[ -z "$NETWORK_VOLUME_ID" ]]; then
  echo "RunPod did not return a network volume id. Response:" >&2
  jq . "$VOLUME_RESPONSE" >&2 || cat "$VOLUME_RESPONSE" >&2
  exit 1
fi

echo "Network volume created: ${NETWORK_VOLUME_ID}"

jq \
  --arg name "$POD_NAME" \
  --arg dataCenterId "$DATA_CENTER_ID" \
  --arg networkVolumeId "$NETWORK_VOLUME_ID" \
  '
  .name = $name
  | .dataCenterIds = [$dataCenterId]
  | .networkVolumeId = $networkVolumeId
  | del(.volumeInGb)
  ' "$TEMP_PAYLOAD" > "$FINAL_PAYLOAD"

echo "Creating final Pod '${POD_NAME}' in ${DATA_CENTER_ID} with network volume ${NETWORK_VOLUME_ID}..."
api POST "/pods" "$FINAL_PAYLOAD" > "$FINAL_RESPONSE" || die "final Pod creation request failed; network volume ${NETWORK_VOLUME_ID} remains in ${DATA_CENTER_ID}"

FINAL_POD_ID="$(extract_pod_id "$FINAL_RESPONSE")"
if [[ -z "$FINAL_POD_ID" ]]; then
  echo "RunPod did not return a final Pod id. Network volume ${NETWORK_VOLUME_ID} remains in ${DATA_CENTER_ID}. Response:" >&2
  jq . "$FINAL_RESPONSE" >&2 || cat "$FINAL_RESPONSE" >&2
  exit 1
fi

echo "Final Pod created: ${FINAL_POD_ID}"
echo
echo "Summary:"
jq -n \
  --arg probePodId "$TEMP_POD_ID" \
  --arg dataCenterId "$DATA_CENTER_ID" \
  --arg networkVolumeId "$NETWORK_VOLUME_ID" \
  --arg finalPodId "$FINAL_POD_ID" \
  '{
    probePodId: $probePodId,
    dataCenterId: $dataCenterId,
    networkVolumeId: $networkVolumeId,
    finalPodId: $finalPodId
  }'
echo
echo "Final Pod details:"
summarize_pod "$FINAL_RESPONSE"

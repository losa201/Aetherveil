{{/*
Expand the name of the chart.
*/}}
{{- define "aetherveil-sentinel.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "aetherveil-sentinel.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aetherveil-sentinel.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aetherveil-sentinel.labels" -}}
helm.sh/chart: {{ include "aetherveil-sentinel.chart" . }}
{{ include "aetherveil-sentinel.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aetherveil-sentinel.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aetherveil-sentinel.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "aetherveil-sentinel.serviceAccountName" -}}
{{- if .Values.common.serviceAccount.create }}
{{- default (include "aetherveil-sentinel.fullname" .) .Values.common.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.common.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "aetherveil-sentinel.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.common.image.registry -}}
{{- $repository := .Values.common.image.repository -}}
{{- $tag := .Values.common.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create agent image name
*/}}
{{- define "aetherveil-sentinel.agentImage" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.common.image.registry -}}
{{- $repository := .agent.image.repository -}}
{{- $tag := .agent.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create PostgreSQL connection string
*/}}
{{- define "aetherveil-sentinel.postgresqlConnectionString" -}}
{{- $host := .Values.coordinator.config.database.host -}}
{{- $port := .Values.coordinator.config.database.port -}}
{{- $database := .Values.coordinator.config.database.name -}}
{{- $user := .Values.coordinator.config.database.user -}}
{{- printf "postgresql://%s:${DATABASE_PASSWORD}@%s:%d/%s" $user $host $port $database }}
{{- end }}

{{/*
Create Redis connection string
*/}}
{{- define "aetherveil-sentinel.redisConnectionString" -}}
{{- $host := .Values.coordinator.config.redis.host -}}
{{- $port := .Values.coordinator.config.redis.port -}}
{{- $database := .Values.coordinator.config.redis.database -}}
{{- if .Values.redis.auth.enabled }}
{{- printf "redis://:%s@%s:%d/%d" "${REDIS_PASSWORD}" $host $port $database }}
{{- else }}
{{- printf "redis://%s:%d/%d" $host $port $database }}
{{- end }}
{{- end }}

{{/*
Create TLS certificate volume mounts
*/}}
{{- define "aetherveil-sentinel.tlsVolumeMounts" -}}
{{- if .Values.coordinator.config.security.enableTLS }}
- name: certificates
  mountPath: /app/certs
  readOnly: true
{{- end }}
{{- end }}

{{/*
Create TLS certificate volumes
*/}}
{{- define "aetherveil-sentinel.tlsVolumes" -}}
{{- if .Values.coordinator.config.security.enableTLS }}
- name: certificates
  secret:
    secretName: {{ include "aetherveil-sentinel.fullname" . }}-certificates
{{- end }}
{{- end }}

{{/*
Create common environment variables
*/}}
{{- define "aetherveil-sentinel.commonEnvVars" -}}
- name: LOG_LEVEL
  value: {{ .Values.common.env.LOG_LEVEL }}
- name: PYTHONUNBUFFERED
  value: {{ .Values.common.env.PYTHONUNBUFFERED | quote }}
- name: NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- if .Values.tracing.enabled }}
- name: JAEGER_ENDPOINT
  value: "{{ .Values.tracing.jaeger.collector.endpoint | default "http://jaeger-collector:14268/api/traces" }}"
{{- end }}
{{- end }}

{{/*
Create security context
*/}}
{{- define "aetherveil-sentinel.securityContext" -}}
{{- if .Values.security.enabled }}
securityContext:
  {{- toYaml .Values.common.securityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create pod security context
*/}}
{{- define "aetherveil-sentinel.podSecurityContext" -}}
{{- if .Values.security.enabled }}
securityContext:
  {{- toYaml .Values.common.podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create image pull secrets
*/}}
{{- define "aetherveil-sentinel.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
  {{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Create resource limits
*/}}
{{- define "aetherveil-sentinel.resources" -}}
{{- if .Values.common.resources }}
resources:
  {{- toYaml .Values.common.resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create node selector
*/}}
{{- define "aetherveil-sentinel.nodeSelector" -}}
{{- if .Values.common.nodeSelector }}
nodeSelector:
  {{- toYaml .Values.common.nodeSelector | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create affinity
*/}}
{{- define "aetherveil-sentinel.affinity" -}}
{{- if .Values.common.affinity }}
affinity:
  {{- toYaml .Values.common.affinity | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create tolerations
*/}}
{{- define "aetherveil-sentinel.tolerations" -}}
{{- if .Values.common.tolerations }}
tolerations:
  {{- toYaml .Values.common.tolerations | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create volume mounts for common directories
*/}}
{{- define "aetherveil-sentinel.commonVolumeMounts" -}}
- name: config
  mountPath: /app/config
  readOnly: true
- name: secrets
  mountPath: /app/secrets
  readOnly: true
- name: logs
  mountPath: /app/logs
- name: data
  mountPath: /app/data
- name: tmp
  mountPath: /tmp
{{- include "aetherveil-sentinel.tlsVolumeMounts" . }}
{{- end }}

{{/*
Create volumes for common directories
*/}}
{{- define "aetherveil-sentinel.commonVolumes" -}}
- name: config
  configMap:
    name: {{ include "aetherveil-sentinel.fullname" . }}-config
- name: secrets
  secret:
    secretName: {{ include "aetherveil-sentinel.fullname" . }}-secret
- name: logs
  emptyDir: {}
- name: data
  emptyDir: {}
- name: tmp
  emptyDir: {}
{{- include "aetherveil-sentinel.tlsVolumes" . }}
{{- end }}
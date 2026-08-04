#!/usr/bin/env bash

load_conda_env() {
  local config_path="$1"
  local configured_env

  if [[ ! -f "$config_path" ]]; then
    echo "错误：未找到 Conda 环境配置文件：$config_path" >&2
    return 1
  fi

  configured_env="$(awk '
    /^[[:space:]]*\[defaults\][[:space:]]*(#.*)?$/ {
      in_defaults = 1
      next
    }
    in_defaults && /^[[:space:]]*\[/ {
      exit
    }
    in_defaults && /^[[:space:]]*env_name[[:space:]]*=/ {
      value = $0
      sub(/^[^=]*=[[:space:]]*/, "", value)
      sub(/[[:space:]]*#.*/, "", value)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      if (value !~ /^"[^"]+"$/) {
        print "__INVALID__"
        exit
      }
      sub(/^"/, "", value)
      sub(/"$/, "", value)
      print value
      exit
    }
  ' "$config_path")"

  if [[ -z "$configured_env" || "$configured_env" == "__INVALID__" ]]; then
    echo "错误：$config_path 的 [defaults].env_name 缺失或格式无效。" >&2
    return 1
  fi
  if [[ ! "$configured_env" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "错误：$config_path 的 [defaults].env_name 包含非法字符。" >&2
    return 1
  fi

  ENV_NAME="$configured_env"
  export ENV_NAME
}

#!/usr/bin/env python3
"""
LLM Hardware Detection & Benchmark Pipeline
=============================================
원커맨드로 하드웨어 감지 → 모델 추천 → Ollama 벤치마크 → 리포트 생성까지.

Usage:
    python llm_bench.py                    # 전체 파이프라인 실행
    python llm_bench.py --detect-only      # 하드웨어 감지만
    python llm_bench.py --recommend-only   # 감지 + 추천만
    python llm_bench.py --models "gemma2,llama3.1"  # 특정 모델만 벤치마크
    python llm_bench.py --tasks reasoning,coding     # 특정 태스크만
    python llm_bench.py --top-n 3          # 추천 모델 상위 N개만 테스트
    python llm_bench.py --output report.html         # HTML 리포트 출력
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


def _ps_run(cmd: str, timeout: int = 10) -> str:
    """PowerShell 명령을 UTF-8로 안전하게 실행 (절대 크래시하지 않음)"""
    try:
        full_cmd = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; " + cmd
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-NoLogo", "-Command", full_cmd],
            encoding="utf-8", errors="replace",
            stderr=subprocess.DEVNULL, timeout=timeout
        )
        return out.strip()
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  PowerShell 타임아웃 ({timeout}s)")
        return ""
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        return ""
    except Exception:
        return ""

# ─────────────────────────────────────────────
# 1. DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class GPUInfo:
    name: str = "Unknown"
    vendor: str = "unknown"        # nvidia, amd, intel, none
    vram_mb: int = 0
    driver: str = ""
    compute_backend: str = "cpu"   # cuda, rocm, vulkan, metal, cpu
    is_igpu: bool = False
    pci_id: str = ""

@dataclass
class HardwareProfile:
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_arch: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    swap_gb: float = 0.0
    gpus: list = field(default_factory=list)
    os_name: str = ""
    os_version: str = ""
    ollama_version: str = ""
    has_avx2: bool = False
    has_f16c: bool = False

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        # Prefer discrete GPU over iGPU
        discrete = [g for g in self.gpus if not g.is_igpu]
        return discrete[0] if discrete else self.gpus[0]

    @property
    def tier(self) -> str:
        gpu = self.primary_gpu
        vram = gpu.vram_mb if gpu else 0
        ram = self.ram_total_gb

        # Apple Silicon: Unified Memory → VRAM = RAM의 75%로 취급
        if gpu and gpu.vendor == "apple":
            effective_vram = ram * 1024 * 0.75
            if effective_vram >= 20000 or ram >= 32:
                return "high"
            elif effective_vram >= 6000 or ram >= 16:
                return "mid"
            elif ram >= 8:
                return "low"
            return "minimal"

        if vram >= 20000 or (vram >= 10000 and ram >= 32):
            return "high"
        elif vram >= 6000 or (vram >= 2000 and ram >= 16):
            return "mid"
        elif ram >= 8:
            return "low"
        else:
            return "minimal"

    @property
    def usable_memory_gb(self) -> float:
        """벤치마크에 사용 가능한 메모리 추정치 (VRAM + 시스템 RAM 오프로드)"""
        gpu = self.primary_gpu
        vram = (gpu.vram_mb / 1024) if gpu else 0
        # Apple Silicon Unified Memory: Ollama가 RAM 대부분 활용 가능
        if gpu and gpu.vendor == "apple":
            return self.ram_available_gb * 0.8
        # iGPU는 시스템 RAM을 공유하므로 별도 계산
        if gpu and gpu.is_igpu:
            return self.ram_available_gb * 0.7
        # dGPU: VRAM + 일부 시스템 RAM 오프로드
        if vram > 0:
            return vram + (self.ram_available_gb * 0.3)
        return self.ram_available_gb * 0.7


@dataclass
class ModelSpec:
    name: str                      # Ollama 모델명 (e.g. "llama3.2:1b")
    display_name: str              # 표시명
    param_billions: float          # 파라미터 수 (B)
    quant: str = "Q4_K_M"         # 양자화
    min_ram_gb: float = 2.0       # 최소 RAM
    recommended_vram_gb: float = 0 # 권장 VRAM
    context_length: int = 4096
    strengths: list = field(default_factory=list)  # ["reasoning", "coding", ...]
    supports_korean: bool = False
    priority: int = 50            # 추천 우선순위 (높을수록 우선)

@dataclass
class TaskResult:
    task_name: str
    prompt: str
    response: str = ""
    tokens_generated: int = 0
    time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    first_token_ms: float = 0.0
    score: float = 0.0            # 0-100
    score_reason: str = ""
    error: str = ""

@dataclass
class BenchmarkResult:
    model_name: str
    pull_time_seconds: float = 0.0
    model_size_gb: float = 0.0
    task_results: list = field(default_factory=list)
    avg_tps: float = 0.0
    avg_score: float = 0.0
    overall_rank: int = 0
    score_rank: int = 0
    speed_rank: int = 0
    composite_score: float = 0.0
    error: str = ""


# ─────────────────────────────────────────────
# 2. HARDWARE DETECTION
# ─────────────────────────────────────────────

class HardwareDetector:
    """시스템 하드웨어 감지 모듈"""

    def detect(self) -> HardwareProfile:
        profile = HardwareProfile()

        print("    OS 감지...", end=" ", flush=True)
        try:
            self._detect_os(profile)
            print("✓")
        except Exception as e:
            print(f"⚠️ {e}")

        print("    CPU 감지...", end=" ", flush=True)
        try:
            self._detect_cpu(profile)
            print("✓")
        except Exception as e:
            print(f"⚠️ {e}")
            profile.cpu_model = platform.processor() or "Unknown"
            profile.cpu_cores = os.cpu_count() or 1
            profile.cpu_threads = profile.cpu_cores

        print("    메모리 감지...", end=" ", flush=True)
        try:
            self._detect_memory(profile)
            print("✓")
        except Exception as e:
            print(f"⚠️ {e}")

        print("    GPU 감지...", end=" ", flush=True)
        try:
            self._detect_gpus(profile)
            print("✓")
        except Exception as e:
            print(f"⚠️ {e}")
            profile.gpus = [GPUInfo(name="Detection failed", vendor="none", compute_backend="cpu")]

        print("    Ollama 감지...", end=" ", flush=True)
        try:
            self._detect_ollama(profile)
            print("✓")
        except Exception as e:
            print(f"⚠️ {e}")

        return profile

    def _detect_os(self, p: HardwareProfile):
        p.os_name = platform.system()
        p.os_version = platform.release()

        # macOS: "Darwin" → "macOS", 커널 버전 → macOS 버전
        if p.os_name == "Darwin":
            p.os_name = "macOS"
            mac_ver = platform.mac_ver()[0]  # e.g. "14.1.2"
            if mac_ver:
                p.os_version = mac_ver

        # Windows: "10" → "10" or "11"
        elif p.os_name == "Windows":
            win_ver = platform.version()  # e.g. "10.0.22631"
            build = int(win_ver.split(".")[-1]) if win_ver else 0
            p.os_version = "11" if build >= 22000 else platform.release()

    def _detect_cpu(self, p: HardwareProfile):
        p.cpu_arch = platform.machine()

        if p.os_name == "Linux":
            try:
                out = subprocess.check_output(["lscpu"], encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    if "Model name" in line or "모델명" in line:
                        p.cpu_model = line.split(":", 1)[1].strip()
                    elif "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                        try:
                            p.cpu_cores = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    elif "Thread(s) per core" in line:
                        try:
                            tpc = int(line.split(":", 1)[1].strip())
                            if p.cpu_cores:
                                p.cpu_threads = p.cpu_cores * tpc
                        except ValueError:
                            pass

                # CPU flags
                try:
                    flags_out = subprocess.check_output(
                        ["grep", "-m1", "flags", "/proc/cpuinfo"],
                        encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                    )
                    flags = flags_out.lower()
                    p.has_avx2 = "avx2" in flags
                    p.has_f16c = "f16c" in flags
                except Exception:
                    pass

            except FileNotFoundError:
                p.cpu_model = platform.processor() or "Unknown"

        elif p.os_name == "macOS":
            # Apple Silicon: machdep.cpu.brand_string이 없음
            try:
                p.cpu_model = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                ).strip()
            except Exception:
                # Apple Silicon fallback
                p.cpu_model = platform.processor() or ""
                if not p.cpu_model or p.cpu_model == "arm":
                    try:
                        chip = subprocess.check_output(
                            ["sysctl", "-n", "machdep.cpu.brand"],
                            encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                        ).strip()
                        p.cpu_model = chip if chip else "Apple Silicon"
                    except Exception:
                        p.cpu_model = "Apple Silicon"

            # 코어 수: physicalcpu + logicalcpu 분리
            try:
                p.cpu_cores = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                ).strip())
            except Exception:
                p.cpu_cores = os.cpu_count() or 1
            try:
                p.cpu_threads = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.logicalcpu"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                ).strip())
            except Exception:
                p.cpu_threads = p.cpu_cores

            # Intel Mac AVX2/F16C 감지
            if "arm" not in p.cpu_arch.lower():
                try:
                    features = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.features"],
                        encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                    ).strip().lower()
                    p.has_avx2 = "avx2" in features
                    p.has_f16c = "f16c" in features
                except Exception:
                    pass
            else:
                # Apple Silicon은 NEON 기반 (AVX2 해당 없음, 하지만 성능 우수)
                p.has_avx2 = False
                p.has_f16c = False

        elif p.os_name == "Windows":
            p.cpu_model = platform.processor() or "Unknown"
            p.cpu_cores = os.cpu_count() or 0
            p.cpu_threads = p.cpu_cores

            # PowerShell로 상세 CPU 정보
            try:
                ps_cmd = (
                    "Get-CimInstance Win32_Processor | "
                    "Select-Object Name,NumberOfCores,ThreadCount | "
                    "ForEach-Object { $_.Name + '|' + $_.NumberOfCores + '|' + $_.ThreadCount }"
                )
                out = _ps_run(ps_cmd)
                if "|" in out:
                    parts = out.split("|")
                    p.cpu_model = parts[0].strip()
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        p.cpu_cores = int(parts[1].strip())
                    if len(parts) >= 3 and parts[2].strip().isdigit():
                        p.cpu_threads = int(parts[2].strip())
            except Exception:
                pass

            # Windows에서 AVX2 감지 (레지스트리 또는 환경변수 기반은 불가, PowerShell로 시도)
            try:
                ps_avx = (
                    "$feat = (Get-CimInstance Win32_Processor).Caption; "
                    "$env:PROCESSOR_IDENTIFIER"
                )
                # 간접적으로: 6세대 이상 Intel 또는 Zen+ 이상 AMD면 AVX2 거의 확정
                cpu_lower = p.cpu_model.lower()
                if any(x in cpu_lower for x in ["core i", "xeon", "ryzen", "epyc"]):
                    p.has_avx2 = True
                    p.has_f16c = True
            except Exception:
                pass

        if not p.cpu_threads:
            p.cpu_threads = p.cpu_cores or (os.cpu_count() or 1)
        if not p.cpu_cores:
            p.cpu_cores = os.cpu_count() or 1

    def _detect_memory(self, p: HardwareProfile):
        if p.os_name == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                for line in meminfo.splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(re.findall(r"\d+", line)[0])
                        p.ram_total_gb = round(kb / 1048576, 1)
                    elif line.startswith("MemAvailable:"):
                        kb = int(re.findall(r"\d+", line)[0])
                        p.ram_available_gb = round(kb / 1048576, 1)
                    elif line.startswith("SwapTotal:"):
                        kb = int(re.findall(r"\d+", line)[0])
                        p.swap_gb = round(kb / 1048576, 1)
            except Exception:
                pass

        elif p.os_name == "macOS":
            # 전체 RAM
            try:
                mem = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], encoding="utf-8", errors="replace"
                ).strip())
                p.ram_total_gb = round(mem / (1024**3), 1)
            except Exception:
                pass

            # 가용 RAM: vm_stat에서 free + inactive 페이지로 계산
            try:
                vm_out = subprocess.check_output(
                    ["vm_stat"], encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                ).strip()
                page_size = 16384  # Apple Silicon 기본
                ps_match = re.search(r"page size of (\d+) bytes", vm_out)
                if ps_match:
                    page_size = int(ps_match.group(1))

                free_pages = 0
                inactive_pages = 0
                for line in vm_out.splitlines():
                    if "Pages free:" in line:
                        m = re.search(r"(\d+)", line.split(":")[1])
                        if m:
                            free_pages = int(m.group(1))
                    elif "Pages inactive:" in line:
                        m = re.search(r"(\d+)", line.split(":")[1])
                        if m:
                            inactive_pages = int(m.group(1))

                available_bytes = (free_pages + inactive_pages) * page_size
                p.ram_available_gb = round(available_bytes / (1024**3), 1)
            except Exception:
                p.ram_available_gb = round(p.ram_total_gb * 0.7, 1)  # fallback

            # Swap
            try:
                swap_out = subprocess.check_output(
                    ["sysctl", "-n", "vm.swapusage"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                ).strip()
                # "total = 2048.00M  used = 512.00M  free = 1536.00M"
                total_match = re.search(r"total\s*=\s*([\d.]+)M", swap_out)
                if total_match:
                    p.swap_gb = round(float(total_match.group(1)) / 1024, 1)
            except Exception:
                pass

        elif p.os_name == "Windows":
            # PowerShell로 메모리 정보
            try:
                ps_cmd = (
                    "$os = Get-CimInstance Win32_OperatingSystem; "
                    "$cs = Get-CimInstance Win32_ComputerSystem; "
                    "$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue; "
                    "$swap = if($pf){($pf | Measure-Object -Property AllocatedBaseSize -Sum).Sum}else{0}; "
                    "'{0}|{1}|{2}' -f $cs.TotalPhysicalMemory, $os.FreePhysicalMemory, $swap"
                )
                out = _ps_run(ps_cmd)
                parts = out.split("|")
                if len(parts) >= 2:
                    total_bytes = int(parts[0].strip())
                    free_kb = int(parts[1].strip())
                    p.ram_total_gb = round(total_bytes / (1024**3), 1)
                    p.ram_available_gb = round(free_kb / 1048576, 1)
                    if len(parts) >= 3 and parts[2].strip().isdigit():
                        p.swap_gb = round(int(parts[2].strip()) / 1024, 1)
            except Exception:
                pass

            # fallback: wmic (deprecated but widely available)
            if p.ram_total_gb == 0:
                try:
                    out = subprocess.check_output(
                        ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                        encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL, timeout=10
                    )
                    match = re.search(r"TotalPhysicalMemory=(\d+)", out)
                    if match:
                        p.ram_total_gb = round(int(match.group(1)) / (1024**3), 1)
                        p.ram_available_gb = p.ram_total_gb * 0.7
                except Exception:
                    pass

    def _detect_gpus(self, p: HardwareProfile):
        gpus = []

        # Linux: lspci 기반 GPU 감지
        if p.os_name == "Linux":
            gpus += self._detect_gpus_lspci()
            # AMD iGPU 추가 감지 (ROCm)
            gpus = self._enhance_amd_info(gpus)
            # NVIDIA 추가 감지
            gpus = self._enhance_nvidia_info(gpus)

        # macOS: Apple Silicon + Intel Mac GPU 감지
        elif p.os_name == "macOS":
            gpus += self._detect_gpus_macos(p)
            gpus = self._enhance_nvidia_info(gpus)  # eGPU 가능성

        # Windows: PowerShell + nvidia-smi
        elif p.os_name == "Windows":
            gpus += self._detect_gpus_windows()
            gpus = self._enhance_nvidia_info(gpus)

        if not gpus:
            gpus.append(GPUInfo(name="No GPU detected", vendor="none", compute_backend="cpu"))

        p.gpus = gpus

    def _detect_gpus_windows(self) -> list:
        """Windows: PowerShell로 GPU 감지 (레지스트리에서 정확한 VRAM 읽기)"""
        gpus = []
        try:
            # 1단계: GPU 이름, 드라이버, PNPDeviceID 가져오기
            # 2단계: 레지스트리에서 64비트 VRAM (qwMemorySize) 읽기
            # AdapterRAM은 uint32라 4GB 이상에서 오버플로우됨
            ps_cmd = (
                "$regPath = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';"
                "Get-CimInstance Win32_VideoController | ForEach-Object {"
                "  $name = $_.Name;"
                "  $driver = $_.DriverVersion;"
                "  $adapterRAM = $_.AdapterRAM;"
                "  $pnpId = $_.PNPDeviceID;"
                "  $gpuDevId = ($pnpId -split '&')[0..1] -join '&';"
                "  $qwMem = 0;"
                "  try {"
                "    $regEntries = Get-ItemProperty -Path \"$regPath\\0*\" -ErrorAction SilentlyContinue;"
                "    foreach ($entry in $regEntries) {"
                "      if ($entry.MatchingDeviceId -and $entry.MatchingDeviceId -like \"$gpuDevId*\") {"
                "        $val = $entry.'HardwareInformation.qwMemorySize';"
                "        if ($val -and $val -gt 0) { $qwMem = $val; break }"
                "      }"
                "    }"
                "  } catch {};"
                "  $vram = if ($qwMem -gt 0) { $qwMem } elseif ($adapterRAM -gt 0) { $adapterRAM } else { 0 };"
                "  $name + '|' + $vram + '|' + $driver + '|' + $pnpId"
                "}"
            )
            out = _ps_run(ps_cmd, timeout=20)

            for line in out.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|")
                name = parts[0].strip() if parts else "Unknown GPU"
                name_lower = name.lower()

                gpu = GPUInfo(name=name)

                # VRAM (이제 64비트 값이므로 4GB+ 정확히 감지)
                if len(parts) >= 2 and parts[1].strip().isdigit():
                    vram_bytes = int(parts[1].strip())
                    if vram_bytes > 0:
                        gpu.vram_mb = vram_bytes // (1024 * 1024)

                # Driver
                if len(parts) >= 3:
                    gpu.driver = parts[2].strip()

                # Vendor 판별
                if "nvidia" in name_lower:
                    gpu.vendor = "nvidia"
                    gpu.compute_backend = "cuda"
                elif "amd" in name_lower or "radeon" in name_lower:
                    gpu.vendor = "amd"
                    igpu_markers = [
                        "radeon graphics", "radeon(tm) graphics", "vega",
                        "radeon 780m", "radeon 760m", "radeon 740m",
                        "radeon 680m", "radeon 660m",
                    ]
                    if any(m in name_lower for m in igpu_markers):
                        gpu.is_igpu = True
                        gpu.compute_backend = "rocm"
                    else:
                        gpu.compute_backend = "rocm"
                elif "intel" in name_lower:
                    gpu.vendor = "intel"
                    gpu.is_igpu = True
                    if "arc" in name_lower:
                        gpu.compute_backend = "vulkan"
                        gpu.is_igpu = False
                    else:
                        gpu.compute_backend = "vulkan"
                elif "microsoft" in name_lower and "basic" in name_lower:
                    continue  # Microsoft Basic Display Adapter는 제외

                gpus.append(gpu)

        except Exception:
            pass

        # DXGI로 VRAM 보강 (PowerShell 쿼리)
        for gpu in gpus:
            if gpu.vram_mb == 0 and gpu.vendor == "nvidia":
                # nvidia-smi가 있으면 _enhance_nvidia_info에서 처리
                pass
            elif gpu.vram_mb == 0:
                try:
                    ps_vram = (
                        f"(Get-CimInstance Win32_VideoController | "
                        f"Where-Object {{ $_.Name -like '*{gpu.name.split()[0]}*' }}).AdapterRAM"
                    )
                    # fallback: 직접 DXGI로 대용량 VRAM 쿼리
                    ps_dxgi = (
                        "Add-Type -TypeDefinition @'\n"
                        "using System; using System.Runtime.InteropServices;\n"
                        "public class DXGI {\n"
                        "  [DllImport(\"dxgi.dll\")] public static extern int CreateDXGIFactory(ref Guid riid, out IntPtr ppFactory);\n"
                        "}\n"
                        "'@ -ErrorAction SilentlyContinue"
                    )
                except Exception:
                    pass

        return gpus

    def _detect_gpus_macos(self, profile: HardwareProfile) -> list:
        """macOS: Apple Silicon + Intel Mac GPU 감지 (system_profiler 기반)"""
        gpus = []
        is_apple_silicon = "arm" in profile.cpu_arch.lower()

        # system_profiler로 GPU 정보 가져오기
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-detailLevel", "basic"],
                encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL, timeout=15
            ).strip()

            current_gpu = None
            for line in out.splitlines():
                stripped = line.strip()

                # GPU 이름 (들여쓰기 패턴으로 감지)
                if stripped.endswith(":") and not any(k in stripped.lower() for k in
                    ["displays:", "resolution:", "framebuffer", "metal", "chipset"]):
                    if len(line) - len(line.lstrip()) <= 8:  # 상위 레벨
                        if current_gpu:
                            gpus.append(current_gpu)
                        gpu_name = stripped.rstrip(":")
                        current_gpu = GPUInfo(name=gpu_name)

                elif current_gpu:
                    if "Chipset Model:" in stripped or "Chip:" in stripped:
                        val = stripped.split(":", 1)[1].strip()
                        current_gpu.name = val
                    elif "VRAM" in stripped or "Memory" in stripped:
                        match = re.search(r"(\d+)\s*(MB|GB)", stripped)
                        if match:
                            val = int(match.group(1))
                            unit = match.group(2)
                            current_gpu.vram_mb = val * 1024 if unit == "GB" else val
                    elif "Vendor:" in stripped:
                        vendor_lower = stripped.lower()
                        if "nvidia" in vendor_lower:
                            current_gpu.vendor = "nvidia"
                            current_gpu.compute_backend = "cuda"
                        elif "amd" in vendor_lower or "ati" in vendor_lower:
                            current_gpu.vendor = "amd"
                            current_gpu.compute_backend = "metal"
                        elif "intel" in vendor_lower:
                            current_gpu.vendor = "intel"
                            current_gpu.is_igpu = True
                            current_gpu.compute_backend = "metal"
                        elif "apple" in vendor_lower:
                            current_gpu.vendor = "apple"
                            current_gpu.compute_backend = "metal"
                            current_gpu.is_igpu = True
                    elif "Metal Support:" in stripped or "Metal Family:" in stripped:
                        current_gpu.compute_backend = "metal"

            if current_gpu:
                gpus.append(current_gpu)

        except Exception:
            pass

        # Apple Silicon이면서 system_profiler에서 못 잡았을 때 fallback
        if is_apple_silicon and not gpus:
            # M1/M2/M3/M4 칩 이름 추출
            chip_name = "Apple Silicon"
            cpu_lower = profile.cpu_model.lower()
            for chip in ["m4 ultra", "m4 max", "m4 pro", "m4",
                         "m3 ultra", "m3 max", "m3 pro", "m3",
                         "m2 ultra", "m2 max", "m2 pro", "m2",
                         "m1 ultra", "m1 max", "m1 pro", "m1"]:
                if chip in cpu_lower:
                    chip_name = f"Apple {chip.upper()}"
                    break

            gpu = GPUInfo(
                name=chip_name,
                vendor="apple",
                # Unified memory: Ollama는 전체 RAM을 활용 가능
                vram_mb=int(profile.ram_total_gb * 1024 * 0.75),
                compute_backend="metal",
                is_igpu=True,
            )
            gpus.append(gpu)

        # Apple Silicon GPU에 vendor가 안 잡힌 경우 보정
        for gpu in gpus:
            if gpu.vendor == "unknown" and is_apple_silicon:
                gpu.vendor = "apple"
                gpu.compute_backend = "metal"
                gpu.is_igpu = True
                if gpu.vram_mb == 0:
                    gpu.vram_mb = int(profile.ram_total_gb * 1024 * 0.75)

        return gpus

    def _detect_gpus_lspci(self) -> list:
        gpus = []
        try:
            out = subprocess.check_output(
                ["lspci", "-nn"], encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    gpu = GPUInfo()

                    # PCI ID 추출
                    pci_match = re.search(r"\[([0-9a-f]{4}:[0-9a-f]{4})\]", line)
                    if pci_match:
                        gpu.pci_id = pci_match.group(1)

                    line_lower = line.lower()
                    if "nvidia" in line_lower:
                        gpu.vendor = "nvidia"
                        gpu.compute_backend = "cuda"
                        gpu.name = self._extract_gpu_name(line, "NVIDIA")
                    elif "amd" in line_lower or "ati" in line_lower or "advanced micro" in line_lower:
                        gpu.vendor = "amd"
                        gpu.name = self._extract_gpu_name(line, "AMD")
                        # AMD iGPU 판별 (Radeon Graphics, Vega, Renoir, Cezanne, Phoenix, Rembrandt, etc.)
                        igpu_markers = [
                            "radeon graphics", "vega", "renoir", "cezanne",
                            "barcelo", "phoenix", "rembrandt", "raphael",
                            "hawk point", "strix point", "granite ridge"
                        ]
                        if any(m in line_lower for m in igpu_markers):
                            gpu.is_igpu = True
                            gpu.compute_backend = "rocm"  # ROCm 또는 Vulkan
                        else:
                            gpu.compute_backend = "rocm"
                    elif "intel" in line_lower:
                        gpu.vendor = "intel"
                        gpu.name = self._extract_gpu_name(line, "Intel")
                        gpu.is_igpu = True
                        gpu.compute_backend = "vulkan"

                    gpus.append(gpu)
        except FileNotFoundError:
            pass
        return gpus

    def _extract_gpu_name(self, line: str, vendor: str) -> str:
        # "XX:XX.X VGA compatible controller: NVIDIA Corporation GeForce RTX 3060 [xxxx:xxxx]"
        match = re.search(r":\s*(.+?)(?:\s*\[[0-9a-f:]+\])*$", line.split("controller:")[-1] if "controller:" in line else line)
        return match.group(1).strip() if match else f"{vendor} GPU"

    def _enhance_amd_info(self, gpus: list) -> list:
        """AMD GPU에 대해 rocm-smi / VRAM 정보 보강"""
        for gpu in gpus:
            if gpu.vendor != "amd":
                continue

            # rocm-smi로 VRAM 감지
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    if "Total" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            gpu.vram_mb = int(match.group(1))
                gpu.driver = "ROCm"
                gpu.compute_backend = "rocm"
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # ROCm 없으면 Vulkan 시도
            if not gpu.vram_mb:
                try:
                    out = subprocess.check_output(
                        ["vulkaninfo", "--summary"],
                        encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                    )
                    for line in out.splitlines():
                        if "deviceName" in line and ("amd" in line.lower() or "radeon" in line.lower()):
                            gpu.compute_backend = "vulkan"
                        if "deviceLocalMem" in line or "heap" in line.lower():
                            match = re.search(r"(\d+)", line)
                            if match:
                                val = int(match.group(1))
                                if val > gpu.vram_mb:
                                    gpu.vram_mb = val
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            # iGPU이고 VRAM 감지 안된 경우: 시스템 RAM에서 추정
            if gpu.is_igpu and gpu.vram_mb == 0:
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                total_kb = int(re.findall(r"\d+", line)[0])
                                # 보통 시스템 RAM의 50% (최대 8GB)를 VRAM으로 할당 가능
                                gpu.vram_mb = min(int(total_kb / 1024 * 0.5), 8192)
                                break
                except Exception:
                    gpu.vram_mb = 2048  # 보수적 기본값

            # HSA (Heterogeneous System Architecture) 감지 - AMD APU 최적화
            if gpu.is_igpu:
                try:
                    if os.path.exists("/dev/kfd"):
                        gpu.driver = gpu.driver or "amdgpu (KFD)"
                        gpu.compute_backend = "rocm"
                except Exception:
                    pass

        return gpus

    def _enhance_nvidia_info(self, gpus: list) -> list:
        for gpu in gpus:
            if gpu.vendor != "nvidia":
                continue
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.total,driver_version,name",
                     "--format=csv,noheader,nounits"],
                    encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
                )
                parts = out.strip().split(",")
                if len(parts) >= 3:
                    gpu.vram_mb = int(parts[0].strip())
                    gpu.driver = parts[1].strip()
                    gpu.name = parts[2].strip()
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        return gpus

    def _detect_ollama(self, p: HardwareProfile):
        try:
            out = subprocess.check_output(
                ["ollama", "--version"], encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL
            )
            match = re.search(r"(\d+\.\d+\.\d+)", out)
            p.ollama_version = match.group(1) if match else out.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            p.ollama_version = ""


# ─────────────────────────────────────────────
# 3. MODEL RECOMMENDER
# ─────────────────────────────────────────────

# 모델 카탈로그 (2025.03 기준 최신 + 인기 모델)
MODEL_CATALOG = [
    # ── Tiny (0.5-3B) ────────────────────────
    ModelSpec("qwen3:0.6b", "Qwen 3 0.6B", 0.6, min_ram_gb=1, context_length=32768,
             strengths=["lightweight", "korean"], supports_korean=True, priority=32),
    ModelSpec("qwen2.5:0.5b", "Qwen 2.5 0.5B", 0.5, min_ram_gb=1, context_length=32768,
             strengths=["lightweight", "korean"], supports_korean=True, priority=28),
    ModelSpec("llama3.2:1b", "Llama 3.2 1B", 1.0, min_ram_gb=1.5, context_length=131072,
             strengths=["fast", "reasoning"], priority=35),
    ModelSpec("qwen3:1.7b", "Qwen 3 1.7B", 1.7, min_ram_gb=2, context_length=32768,
             strengths=["reasoning", "korean", "coding"], supports_korean=True, priority=48),
    ModelSpec("qwen2.5:1.5b", "Qwen 2.5 1.5B", 1.5, min_ram_gb=2, context_length=32768,
             strengths=["reasoning", "korean", "coding"], supports_korean=True, priority=42),
    ModelSpec("gemma3:1b", "Gemma 3 1B", 1.0, min_ram_gb=1.5, context_length=32768,
             strengths=["reasoning", "multimodal"], priority=38),
    ModelSpec("gemma4:e2b", "Gemma 4 E2B", 2.3, min_ram_gb=4, context_length=131072,
             strengths=["reasoning", "multimodal", "coding"], priority=50),
    ModelSpec("llama3.2:3b", "Llama 3.2 3B", 3.0, min_ram_gb=3, context_length=131072,
             strengths=["reasoning", "general"], priority=46),
    ModelSpec("phi4-mini:3.8b", "Phi-4 Mini 3.8B", 3.8, min_ram_gb=3, context_length=16384,
             strengths=["reasoning", "coding", "math"], priority=52),

    # ── Small (4-9B) ─────────────────────────
    ModelSpec("qwen3:4b", "Qwen 3 4B", 4.0, min_ram_gb=3.5, recommended_vram_gb=4,
             context_length=32768, strengths=["reasoning", "korean", "coding"],
             supports_korean=True, priority=60),
    ModelSpec("gemma3:4b", "Gemma 3 4B", 4.0, min_ram_gb=3.5, recommended_vram_gb=4,
             context_length=32768, strengths=["reasoning", "multimodal"], priority=55),
    ModelSpec("gemma4:e4b", "Gemma 4 E4B", 4.0, min_ram_gb=6, recommended_vram_gb=8,
             context_length=131072, strengths=["reasoning", "multimodal", "coding"],
             priority=68),
    ModelSpec("phi4-mini", "Phi-4 Mini", 3.8, min_ram_gb=3, recommended_vram_gb=4,
             context_length=16384, strengths=["reasoning", "coding", "math"], priority=54),
    ModelSpec("qwen3:8b", "Qwen 3 8B", 8.0, min_ram_gb=5.5, recommended_vram_gb=6,
             context_length=32768, strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=78),
    ModelSpec("qwen2.5:7b", "Qwen 2.5 7B", 7.0, min_ram_gb=5, recommended_vram_gb=6,
             context_length=32768, strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=72),
    ModelSpec("llama3.1:8b", "Llama 3.1 8B", 8.0, min_ram_gb=5.5, recommended_vram_gb=6,
             context_length=131072, strengths=["reasoning", "coding", "general"], priority=68),
    ModelSpec("gemma3:12b", "Gemma 3 12B", 12.0, min_ram_gb=8, recommended_vram_gb=8,
             context_length=32768, strengths=["reasoning", "multimodal", "general"], priority=74),
    ModelSpec("deepseek-r1:7b", "DeepSeek R1 7B", 7.0, min_ram_gb=5, recommended_vram_gb=6,
             context_length=32768, strengths=["reasoning", "math", "coding"], priority=76),
    ModelSpec("mistral:7b", "Mistral 7B v0.3", 7.0, min_ram_gb=5, recommended_vram_gb=6,
             context_length=32768, strengths=["reasoning", "coding"], priority=58),
    ModelSpec("gemma2:9b", "Gemma 2 9B", 9.0, min_ram_gb=6, recommended_vram_gb=6,
             context_length=8192, strengths=["reasoning", "summarization"], priority=56),
    ModelSpec("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B", 7.0, min_ram_gb=5,
             recommended_vram_gb=6, context_length=32768,
             strengths=["coding"], priority=62),

    # ── Medium (14-30B) ──────────────────────
    ModelSpec("qwen3:14b", "Qwen 3 14B", 14.0, min_ram_gb=10, recommended_vram_gb=10,
             context_length=32768, strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=88),
    ModelSpec("qwen2.5:14b", "Qwen 2.5 14B", 14.0, min_ram_gb=10, recommended_vram_gb=10,
             context_length=32768, strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=82),
    ModelSpec("phi4:14b", "Phi-4 14B", 14.0, min_ram_gb=10, recommended_vram_gb=10,
             context_length=16384, strengths=["reasoning", "math", "coding"], priority=84),
    ModelSpec("deepseek-r1:14b", "DeepSeek R1 14B", 14.0, min_ram_gb=10, recommended_vram_gb=10,
             context_length=32768, strengths=["reasoning", "math", "coding"], priority=86),
    ModelSpec("gemma3:27b", "Gemma 3 27B", 27.0, min_ram_gb=18, recommended_vram_gb=16,
             context_length=32768, strengths=["reasoning", "multimodal"], priority=80),
    ModelSpec("gemma4:26b", "Gemma 4 26B (MoE)", 26.0, min_ram_gb=12, recommended_vram_gb=16,
             context_length=262144, strengths=["reasoning", "multimodal", "coding", "general"],
             priority=89),
    ModelSpec("qwen3:30b-a3b", "Qwen 3 30B-A3B (MoE)", 30.0, min_ram_gb=12,
             recommended_vram_gb=10, context_length=32768,
             strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=87),
    ModelSpec("qwen2.5-coder:14b", "Qwen 2.5 Coder 14B", 14.0, min_ram_gb=10,
             recommended_vram_gb=10, context_length=32768,
             strengths=["coding"], priority=79),
    ModelSpec("llama3.3:70b", "Llama 3.3 70B (Q4)", 70.0, min_ram_gb=40, recommended_vram_gb=40,
             context_length=131072, strengths=["reasoning", "coding", "general"], priority=92),

    # ── Large (32B+) ─────────────────────────
    ModelSpec("gemma4:31b", "Gemma 4 31B", 31.0, min_ram_gb=22, recommended_vram_gb=24,
             context_length=262144, strengths=["reasoning", "multimodal", "coding", "general"],
             priority=91),
    ModelSpec("qwen3:32b", "Qwen 3 32B", 32.0, min_ram_gb=22, recommended_vram_gb=20,
             context_length=32768, strengths=["reasoning", "korean", "coding", "general"],
             supports_korean=True, priority=90),
    ModelSpec("deepseek-r1:32b", "DeepSeek R1 32B", 32.0, min_ram_gb=22,
             recommended_vram_gb=20, context_length=32768,
             strengths=["reasoning", "math", "coding"], priority=89),
    ModelSpec("mixtral:8x7b", "Mixtral 8x7B", 46.7, min_ram_gb=26, recommended_vram_gb=24,
             context_length=32768, strengths=["reasoning", "coding", "general"], priority=75),
    ModelSpec("deepseek-r1:70b", "DeepSeek R1 70B", 70.0, min_ram_gb=42,
             recommended_vram_gb=40, context_length=32768,
             strengths=["reasoning", "math", "coding"], priority=94),
]

# ── 외부 카탈로그 (JSON) 로딩/갱신 ──────────

CATALOG_FILE = "model_catalog.json"

def save_catalog_json(models: list[ModelSpec], path: str = CATALOG_FILE):
    """카탈로그를 JSON으로 저장"""
    data = [asdict(m) for m in models]
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        json.dump({"updated": datetime.now().isoformat(), "models": data}, f,
                  ensure_ascii=False, indent=2)
    print(f"  💾 카탈로그 저장: {path} ({len(models)}개 모델)")

def load_catalog_json(path: str = CATALOG_FILE) -> list[ModelSpec]:
    """JSON에서 카탈로그 로딩"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        models = []
        for m in data.get("models", []):
            models.append(ModelSpec(**{k: v for k, v in m.items()
                                      if k in ModelSpec.__dataclass_fields__}))
        updated = data.get("updated", "unknown")
        print(f"  📂 외부 카탈로그 로드: {path} ({len(models)}개 모델, 갱신: {updated})")
        return models
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"  ⚠️  카탈로그 로드 실패: {e}")
        return []

def fetch_ollama_library() -> list[dict]:
    """ollama.com/library에서 인기 모델 목록 스크래핑"""
    import urllib.request

    print("  🌐 ollama.com/library에서 최신 모델 목록 가져오는 중...")
    models_raw = []

    try:
        req = urllib.request.Request(
            "https://ollama.com/search",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")

        # JSON-LD 또는 script 태그에서 모델 데이터 추출 시도
        # ollama.com은 Next.js 기반이라 __NEXT_DATA__에 데이터가 있을 수 있음
        import re
        next_data_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if next_data_match:
            try:
                next_data = json.loads(next_data_match.group(1))
                # Next.js 데이터 구조에서 모델 추출
                props = next_data.get("props", {}).get("pageProps", {})
                if "models" in props:
                    models_raw = props["models"]
            except json.JSONDecodeError:
                pass

        # fallback: HTML에서 모델명 + 설명 파싱
        if not models_raw:
            # ollama.com/library 페이지에서 모델 링크 추출
            model_links = re.findall(r'href="/library/([^"]+)"', html)
            # 중복 제거 및 상위 추출
            seen = set()
            for name in model_links:
                clean = name.split("?")[0].strip("/")
                if clean and clean not in seen:
                    seen.add(clean)
                    models_raw.append({"name": clean})

    except Exception as e:
        print(f"  ⚠️  웹 스크래핑 실패: {e}")

    return models_raw


def fetch_github_catalog(repo_url: str = "") -> list[ModelSpec]:
    """GitHub 레포에서 최신 카탈로그 JSON을 가져옴 (커뮤니티 유지보수)"""
    import urllib.request

    if not repo_url:
        repo_url = "https://raw.githubusercontent.com/mstory78/llm-bench-pipeline/main/model_catalog.json"

    print(f"  🌐 GitHub에서 최신 카탈로그 가져오는 중...")
    try:
        req = urllib.request.Request(repo_url, headers={"User-Agent": "llm-bench-pipeline"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = []
        for m in data.get("models", []):
            models.append(ModelSpec(**{k: v for k, v in m.items()
                                      if k in ModelSpec.__dataclass_fields__}))
        updated = data.get("updated", "unknown")
        print(f"  ✅ GitHub 카탈로그 로드: {len(models)}개 모델 (갱신: {updated})")
        return models
    except Exception as e:
        print(f"  ⚠️  GitHub 카탈로그 가져오기 실패: {e}")
        return []


def discover_model_spec(runner, model_name: str) -> Optional[ModelSpec]:
    """Ollama API(/api/show)에서 모델 스펙을 자동 감지하여 ModelSpec 생성
    카탈로그에 없는 최신 모델도 자동으로 벤치마크 가능하게 함"""
    info = runner.get_model_info(model_name)
    if not info:
        return None

    details = info.get("details", {})
    param_str = details.get("parameter_size", "")  # e.g. "8.0B", "14B"
    family = details.get("family", "")
    quant = details.get("quantization_level", "Q4_K_M")

    # 파라미터 크기 파싱
    param_b = 0.0
    if param_str:
        match = re.match(r"([\d.]+)\s*[Bb]", param_str)
        if match:
            param_b = float(match.group(1))

    if param_b == 0:
        return None

    # RAM 요구량 추정 (Q4 기준: 파라미터 × 0.6GB + 1GB 오버헤드)
    min_ram = round(param_b * 0.6 + 1, 1)
    rec_vram = round(param_b * 0.6, 0)

    # 패밀리 기반 강점/한국어 추정
    family_lower = family.lower()
    strengths = FAMILY_STRENGTHS.get(family_lower, ["general"])
    korean = family_lower in KOREAN_FAMILIES

    # 컨텍스트 길이
    model_info = info.get("model_info", {})
    context = 4096
    for key, val in model_info.items():
        if "context_length" in key and isinstance(val, int):
            context = val
            break

    spec = ModelSpec(
        name=model_name,
        display_name=f"{family.title()} {param_str}" if family else model_name,
        param_billions=param_b,
        quant=quant,
        min_ram_gb=min_ram,
        recommended_vram_gb=rec_vram,
        context_length=context,
        strengths=strengths,
        supports_korean=korean,
        priority=min(int(param_b * 5 + 30), 95),
    )
    print(f"  🔍 자동 감지: {model_name} → {param_str}, {quant}, RAM≥{min_ram}GB, ctx={context}")
    return spec

# 모델 이름 → 사양 추정 규칙
MODEL_SIZE_HINTS = {
    # model_family: [(tag_pattern, param_billions, min_ram_gb, recommended_vram_gb)]
    "qwen3": [("0.6b", 0.6, 1, 0), ("1.7b", 1.7, 2, 0), ("4b", 4, 3.5, 4),
              ("8b", 8, 5.5, 6), ("14b", 14, 10, 10), ("30b-a3b", 30, 12, 10),
              ("32b", 32, 22, 20)],
    "qwen2.5": [("0.5b", 0.5, 1, 0), ("1.5b", 1.5, 2, 0), ("3b", 3, 2.5, 3),
                ("7b", 7, 5, 6), ("14b", 14, 10, 10), ("32b", 32, 22, 20),
                ("72b", 72, 44, 40)],
    "llama3": [("1b", 1, 1.5, 0), ("3b", 3, 3, 3), ("8b", 8, 5.5, 6), ("70b", 70, 40, 40)],
    "gemma3": [("1b", 1, 1.5, 0), ("4b", 4, 3.5, 4), ("12b", 12, 8, 8), ("27b", 27, 18, 16)],
    "gemma4": [("e2b", 2.3, 4, 0), ("e4b", 4, 6, 8), ("26b", 26, 12, 16), ("31b", 31, 22, 24)],
    "gemma2": [("2b", 2, 2.5, 0), ("9b", 9, 6, 6), ("27b", 27, 18, 16)],
    "phi4": [("3.8b", 3.8, 3, 4), ("14b", 14, 10, 10)],
    "deepseek-r1": [("1.5b", 1.5, 2, 0), ("7b", 7, 5, 6), ("8b", 8, 5.5, 6),
                    ("14b", 14, 10, 10), ("32b", 32, 22, 20), ("70b", 70, 42, 40)],
    "mistral": [("7b", 7, 5, 6)],
    "mixtral": [("8x7b", 46.7, 26, 24)],
}

# 한국어 지원 모델 패밀리
KOREAN_FAMILIES = {"qwen3", "qwen2.5", "qwen2.5-coder", "gemma3", "gemma4", "gemma2", "exaone"}

# 모델 패밀리별 강점
FAMILY_STRENGTHS = {
    "qwen3": ["reasoning", "korean", "coding", "general"],
    "qwen2.5": ["reasoning", "korean", "coding", "general"],
    "qwen2.5-coder": ["coding"],
    "llama3": ["reasoning", "coding", "general"],
    "gemma4": ["reasoning", "multimodal", "coding", "general"],
    "gemma3": ["reasoning", "multimodal", "general"],
    "gemma2": ["reasoning", "summarization"],
    "phi4": ["reasoning", "math", "coding"],
    "deepseek-r1": ["reasoning", "math", "coding"],
    "deepseek-coder": ["coding"],
    "mistral": ["reasoning", "coding"],
    "mixtral": ["reasoning", "coding", "general"],
    "exaone": ["reasoning", "korean", "general"],
    "codellama": ["coding"],
}

def update_catalog_from_web(save_path: str = CATALOG_FILE) -> list[ModelSpec]:
    """카탈로그 갱신: GitHub → ollama.com 스크래핑 → 내장 카탈로그 순서"""

    # 1차: GitHub에서 커뮤니티 유지보수 카탈로그 가져오기
    github_models = fetch_github_catalog()
    if github_models:
        if save_path:
            save_catalog_json(github_models, save_path)
        return github_models

    # 2차: ollama.com 웹 스크래핑
    web_models = fetch_ollama_library()
    if not web_models:
        print("  ⚠️  웹에서 모델을 가져오지 못했습니다. 내장 카탈로그를 사용합니다.")
        return MODEL_CATALOG

    new_models = list(MODEL_CATALOG)
    existing_names = {m.name for m in new_models}

    added = 0
    for raw in web_models:
        name = raw.get("name", "")
        if not name:
            continue

        family = name.split(":")[0].split("/")[-1]

        if family in MODEL_SIZE_HINTS:
            for tag, params, min_ram, rec_vram in MODEL_SIZE_HINTS[family]:
                full_name = f"{family}:{tag}"
                if full_name in existing_names:
                    continue
                strengths = FAMILY_STRENGTHS.get(family, ["general"])
                korean = family in KOREAN_FAMILIES
                priority = min(int(params * 5 + 30), 95)
                spec = ModelSpec(
                    name=full_name,
                    display_name=f"{family.title()} {tag.upper()}",
                    param_billions=params,
                    min_ram_gb=min_ram,
                    recommended_vram_gb=rec_vram,
                    context_length=32768,
                    strengths=strengths,
                    supports_korean=korean,
                    priority=priority
                )
                new_models.append(spec)
                existing_names.add(full_name)
                added += 1

    print(f"  ✅ 카탈로그 갱신 완료: 기존 {len(MODEL_CATALOG)} + 신규 {added} = 총 {len(new_models)}개")

    if save_path:
        save_catalog_json(new_models, save_path)

    return new_models


class ModelRecommender:
    """하드웨어 프로필에 맞는 모델 추천"""

    def __init__(self, hw: HardwareProfile, catalog: list[ModelSpec] = None):
        self.hw = hw
        self.catalog = catalog or MODEL_CATALOG

    def recommend(self, top_n: int = 5) -> list[ModelSpec]:
        usable = self.hw.usable_memory_gb
        candidates = []

        for model in self.catalog:
            if model.min_ram_gb > usable:
                continue

            score = model.priority

            # GPU 가속 가능 시 보너스
            gpu = self.hw.primary_gpu
            if gpu and gpu.vendor != "none":
                if model.recommended_vram_gb and model.recommended_vram_gb <= (gpu.vram_mb / 1024):
                    score += 15  # Full VRAM fit
                elif gpu.is_igpu:
                    score += 5   # iGPU partial offload

            # 한국어 지원 보너스
            if model.supports_korean:
                score += 10

            # RAM 여유 보너스 (넉넉할수록)
            headroom = usable - model.min_ram_gb
            if headroom > 4:
                score += 5

            # AMD iGPU 환경에서 작은 모델 선호 (메모리 공유 이슈)
            if gpu and gpu.is_igpu and model.param_billions > 14:
                score -= 10

            # 최신 모델 보너스 (3세대 > 2.5세대 > 2세대)
            name_lower = model.name.lower()
            if any(g in name_lower for g in ["qwen3", "gemma4", "gemma3", "phi4", "deepseek-r1", "llama3.3"]):
                score += 8  # 최신 세대
            elif any(g in name_lower for g in ["qwen2.5", "llama3.2", "gemma2"]):
                score += 4  # 현세대

            candidates.append((model, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in candidates[:top_n]]


# ─────────────────────────────────────────────
# 4. EVALUATION TASKS
# ─────────────────────────────────────────────

EVALUATION_TASKS = {
    "reasoning": {
        "name": "논리 추론 (Reasoning)",
        "prompt": (
            "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
            "Think step by step and give only the final number."
        ),
        "check": lambda r: "9" in r,
        "weight": 1.0,
    },
    "coding": {
        "name": "코딩 (Coding)",
        "prompt": (
            "Write a Python function called `is_palindrome(s)` that checks if a string is a palindrome, "
            "ignoring spaces and case. Include a brief docstring."
        ),
        "check": lambda r: "def is_palindrome" in r and ("lower" in r or "casefold" in r),
        "weight": 1.0,
    },
    "korean": {
        "name": "한국어 이해 (Korean)",
        "prompt": (
            "다음 문장을 영어로 번역하고, 핵심 키워드 3개를 추출하세요: "
            "'인공지능 기술의 발전으로 음악 저작권 관리 시스템이 더욱 정교해지고 있다.'"
        ),
        "check": lambda r: any(w in r.lower() for w in ["artificial intelligence", "ai", "copyright", "music"]),
        "weight": 1.0,
    },
    "summarization": {
        "name": "요약 (Summarization)",
        "prompt": (
            "Summarize the following in exactly 2 sentences:\n\n"
            "Large Language Models (LLMs) have transformed natural language processing by enabling "
            "machines to generate, understand, and reason about text at unprecedented levels. "
            "These models, trained on vast corpora of text data, leverage transformer architectures "
            "with attention mechanisms to capture long-range dependencies. Recent developments include "
            "smaller, more efficient models that can run on consumer hardware, making AI accessible "
            "to a broader audience. Quantization techniques like GGUF and GPTQ reduce model size "
            "while maintaining quality, enabling deployment on devices with limited memory."
        ),
        "check": lambda r: 20 < len(r.split()) < 100,
        "weight": 0.8,
    },
    "instruction_following": {
        "name": "지시 따르기 (Instruction Following)",
        "prompt": (
            "List exactly 5 programming languages that start with the letter 'P'. "
            "Format: numbered list, one per line. Nothing else."
        ),
        "check": lambda r: r.count("P") >= 3 or r.count("p") >= 3,
        "weight": 0.8,
    },
    "math": {
        "name": "수학 (Math)",
        "prompt": (
            "What is 247 × 83? Show your work step by step, then give the final answer."
        ),
        "check": lambda r: "20501" in r.replace(",", "").replace(" ", ""),
        "weight": 0.9,
    },
}


# ─────────────────────────────────────────────
# 5. OLLAMA RUNNER & EVALUATOR
# ─────────────────────────────────────────────

class OllamaRunner:
    """Ollama API를 통한 모델 실행 및 평가"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/version")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def pull_model(self, model_name: str) -> float:
        """모델 Pull, 소요 시간 반환"""
        print(f"  📥 모델 다운로드 중: {model_name}...")
        start = time.time()
        try:
            proc = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True, encoding="utf-8", errors="replace", timeout=1800  # 30분 타임아웃
            )
            elapsed = time.time() - start
            if proc.returncode != 0:
                print(f"  ⚠️  Pull 실패: {proc.stderr[:200]}")
                return -1
            print(f"  ✅ 완료 ({elapsed:.1f}s)")
            return elapsed
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Pull 타임아웃 (30분 초과)")
            return -1
        except FileNotFoundError:
            print(f"  ❌ Ollama가 설치되어 있지 않습니다")
            return -1

    def generate(self, model_name: str, prompt: str, timeout: int = 120) -> dict:
        """Ollama API로 텍스트 생성"""
        import urllib.request

        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - start
                return {
                    "response": data.get("response", ""),
                    "total_duration": data.get("total_duration", 0) / 1e9,  # ns → s
                    "load_duration": data.get("load_duration", 0) / 1e9,
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0) / 1e9,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "wall_time": elapsed,
                }
        except Exception as e:
            return {"error": str(e), "wall_time": time.time() - start}

    def list_local_models(self) -> list:
        """로컬에 이미 설치된 모델 목록"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def get_model_info(self, model_name: str) -> dict:
        """모델 상세 정보"""
        import urllib.request
        try:
            payload = json.dumps({"name": model_name}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}


class TaskEvaluator:
    """태스크별 모델 평가"""

    def __init__(self, runner: OllamaRunner):
        self.runner = runner

    def evaluate_model(self, model_name: str, task_keys: list[str] = None) -> list[TaskResult]:
        tasks = task_keys or list(EVALUATION_TASKS.keys())
        results = []

        for key in tasks:
            if key not in EVALUATION_TASKS:
                print(f"  ⚠️  알 수 없는 태스크: {key}")
                continue

            task = EVALUATION_TASKS[key]
            result = TaskResult(task_name=task["name"], prompt=task["prompt"])

            print(f"    🔄 {task['name']}...", end=" ", flush=True)

            response = self.runner.generate(model_name, task["prompt"])

            if "error" in response:
                result.error = response["error"]
                print(f"❌ {result.error[:60]}")
            else:
                result.response = response["response"]
                result.tokens_generated = response.get("eval_count", 0)
                result.time_seconds = response.get("eval_duration", 0) or response["wall_time"]

                if result.time_seconds > 0 and result.tokens_generated > 0:
                    result.tokens_per_second = result.tokens_generated / result.time_seconds

                # 점수 산정
                result.score, result.score_reason = self._score_response(key, result.response)
                print(f"✅ {result.score:.0f}점 | {result.tokens_per_second:.1f} tok/s")

            results.append(result)

        return results

    def _score_response(self, task_key: str, response: str) -> tuple[float, str]:
        task = EVALUATION_TASKS[task_key]
        score = 0.0
        reasons = []

        if not response.strip():
            return 0.0, "빈 응답"

        # 1. 정확성 체크 (50점)
        try:
            if task["check"](response):
                score += 50
                reasons.append("정답 포함")
            else:
                reasons.append("정답 미포함")
        except Exception:
            reasons.append("체크 오류")

        # 2. 응답 품질 (30점)
        word_count = len(response.split())
        if 10 <= word_count <= 500:
            score += 20
            reasons.append("적절한 길이")
        elif word_count > 500:
            score += 10
            reasons.append("다소 긴 응답")
        else:
            score += 5
            reasons.append("너무 짧음")

        # 구조화 여부
        if any(marker in response for marker in ["```", "def ", "class ", "1.", "- ", "Step"]):
            score += 10
            reasons.append("구조적 응답")

        # 3. 완성도 (20점)
        if response.rstrip().endswith((".", "!", "?", "```", ")", "]")):
            score += 10
            reasons.append("완결된 응답")

        if not any(bad in response.lower() for bad in ["i cannot", "i'm sorry", "as an ai"]):
            score += 10
            reasons.append("직접 응답")
        else:
            reasons.append("회피 응답")

        return min(score, 100), " / ".join(reasons)


# ─────────────────────────────────────────────
# 6. REPORT GENERATOR
# ─────────────────────────────────────────────

class ReportGenerator:
    """결과 리포트 생성 (Terminal + HTML + JSON)"""

    @staticmethod
    def print_hardware(hw: HardwareProfile):
        print("\n" + "="*60)
        print("  🖥️  하드웨어 프로필")
        print("="*60)
        print(f"  OS        : {hw.os_name} {hw.os_version}")
        print(f"  CPU       : {hw.cpu_model}")
        print(f"  코어/스레드 : {hw.cpu_cores}C / {hw.cpu_threads}T")
        print(f"  AVX2/F16C : {'✅' if hw.has_avx2 else '❌'} / {'✅' if hw.has_f16c else '❌'}")
        print(f"  RAM       : {hw.ram_total_gb} GB (사용가능 {hw.ram_available_gb} GB)")
        print(f"  Swap      : {hw.swap_gb} GB")

        for i, gpu in enumerate(hw.gpus):
            tag = " (iGPU)" if gpu.is_igpu else ""
            print(f"  GPU {i}     : {gpu.name}{tag}")
            print(f"    VRAM    : {gpu.vram_mb} MB")
            print(f"    Backend : {gpu.compute_backend.upper()}")
            if gpu.driver:
                print(f"    Driver  : {gpu.driver}")

        print(f"  Ollama    : {hw.ollama_version or '❌ 미설치'}")
        print(f"  성능 등급   : {hw.tier.upper()}")
        print(f"  가용 메모리  : ~{hw.usable_memory_gb:.1f} GB")
        print("="*60)

    @staticmethod
    def print_recommendations(models: list[ModelSpec], hw: HardwareProfile):
        print("\n" + "="*60)
        print(f"  🎯 추천 모델 (하드웨어 등급: {hw.tier.upper()})")
        print("="*60)
        for i, m in enumerate(models, 1):
            kr = "🇰🇷" if m.supports_korean else "  "
            print(f"  {i}. {kr} {m.display_name}")
            print(f"     모델: {m.name} | {m.param_billions}B | RAM≥{m.min_ram_gb}GB")
            print(f"     강점: {', '.join(m.strengths)}")
        print("="*60)

    @staticmethod
    def print_benchmark_results(results: list[BenchmarkResult]):
        valid = [r for r in results if not r.error]
        errors = [r for r in results if r.error]

        print("\n" + "="*60)
        print("  📊 벤치마크 결과 (종합 순위)")
        print("="*60)

        for r in sorted(valid, key=lambda x: x.overall_rank):
            print(f"\n  종합 #{r.overall_rank} {r.model_name}")
            print(f"     종합: {r.composite_score:.1f} | 점수: {r.avg_score:.1f}/100 (#{r.score_rank}) | 속도: {r.avg_tps:.1f} tok/s (#{r.speed_rank})")

            for t in r.task_results:
                status = "✅" if t.score >= 50 else "⚠️" if t.score >= 25 else "❌"
                speed = f"{t.tokens_per_second:.1f} tok/s" if t.tokens_per_second else "N/A"
                print(f"     {status} {t.task_name}: {t.score:.0f}점 ({speed})")

        for r in errors:
            print(f"\n  ❌ {r.model_name}: {r.error}")

        # 요약 테이블
        if valid:
            print("\n" + "─"*60)
            print("  📋 순위 요약")
            print("─"*60)
            print(f"  {'모델':<20} {'점수순':>7} {'속도순':>7} {'종합':>7}")
            print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7}")
            for r in sorted(valid, key=lambda x: x.overall_rank):
                print(f"  {r.model_name:<20} #{r.score_rank:<6} #{r.speed_rank:<6} #{r.overall_rank:<6}")

        print("\n" + "="*60)

    @staticmethod
    def generate_html(hw: HardwareProfile, models: list[ModelSpec],
                      results: list[BenchmarkResult], output_path: str):
        """HTML 리포트 생성"""
        sorted_results = sorted(results, key=lambda x: x.overall_rank)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # GPU 정보
        gpu_html = ""
        for i, gpu in enumerate(hw.gpus):
            tag = " (iGPU)" if gpu.is_igpu else ""
            gpu_html += f"""
            <tr><td>GPU {i}</td><td>{gpu.name}{tag}</td></tr>
            <tr><td>VRAM</td><td>{gpu.vram_mb} MB</td></tr>
            <tr><td>Backend</td><td>{gpu.compute_backend.upper()}</td></tr>"""

        # 추천 모델 HTML
        rec_html = ""
        for i, m in enumerate(models, 1):
            kr = "🇰🇷 " if m.supports_korean else ""
            rec_html += f"""
            <tr>
                <td>{i}</td>
                <td>{kr}{m.display_name}</td>
                <td><code>{m.name}</code></td>
                <td>{m.param_billions}B</td>
                <td>{m.min_ram_gb} GB</td>
                <td>{', '.join(m.strengths)}</td>
            </tr>"""

        # 벤치마크 결과 HTML
        bench_html = ""
        for r in sorted_results:
            if r.error:
                bench_html += f'<tr><td colspan="7">❌ {r.model_name}: {r.error}</td></tr>'
                continue
            tasks_detail = ""
            for t in r.task_results:
                icon = "✅" if t.score >= 50 else "⚠️" if t.score >= 25 else "❌"
                tasks_detail += f"<li>{icon} {t.task_name}: {t.score:.0f}점 ({t.tokens_per_second:.1f} tok/s)</li>"

            bench_html += f"""
            <tr>
                <td><strong>#{r.overall_rank}</strong></td>
                <td><strong>{r.model_name}</strong></td>
                <td>{r.avg_score:.1f} <span style="color:var(--muted);font-size:0.85em">#{r.score_rank}</span></td>
                <td>{r.avg_tps:.1f} <span style="color:var(--muted);font-size:0.85em">#{r.speed_rank}</span></td>
                <td>{r.composite_score:.1f}</td>
                <td><ul style="margin:0;padding-left:1.2em">{tasks_detail}</ul></td>
            </tr>"""

        # 순위 요약 테이블 HTML
        rank_summary_html = ""
        for r in sorted_results:
            if r.error:
                continue
            # 하이라이트: 각 순위 1등인 경우
            s_tag = ' style="color:#4ade80;font-weight:700"' if r.score_rank == 1 else ""
            sp_tag = ' style="color:#38bdf8;font-weight:700"' if r.speed_rank == 1 else ""
            o_tag = ' style="color:#fbbf24;font-weight:700"' if r.overall_rank == 1 else ""
            rank_summary_html += f"""
            <tr>
                <td>{r.model_name}</td>
                <td{s_tag}>#{r.score_rank} ({r.avg_score:.1f})</td>
                <td{sp_tag}>#{r.speed_rank} ({r.avg_tps:.1f})</td>
                <td{o_tag}>#{r.overall_rank} ({r.composite_score:.1f})</td>
            </tr>"""

        # 차트 데이터
        chart_labels = json.dumps([r.model_name for r in sorted_results if not r.error])
        chart_scores = json.dumps([r.avg_score for r in sorted_results if not r.error])
        chart_tps = json.dumps([r.avg_tps for r in sorted_results if not r.error])

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Benchmark Report - {timestamp}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --accent2: #818cf8;
           --text: #e2e8f0; --muted: #94a3b8; --green: #4ade80; --red: #f87171; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text);
          line-height:1.6; padding:2rem; max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:1.8rem; margin-bottom:0.5rem; color:var(--accent); }}
  h2 {{ font-size:1.3rem; color:var(--accent2); margin:2rem 0 1rem; }}
  .subtitle {{ color:var(--muted); margin-bottom:2rem; }}
  .card {{ background:var(--card); border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; }}
  th, td {{ padding:0.6rem 0.8rem; text-align:left; border-bottom:1px solid #334155; }}
  th {{ color:var(--accent); font-weight:600; }}
  code {{ background:#334155; padding:0.15em 0.4em; border-radius:4px; font-size:0.9em; }}
  .tier {{ display:inline-block; padding:0.2em 0.8em; border-radius:999px; font-weight:700;
           font-size:0.85rem; }}
  .tier-high {{ background:#065f46; color:#6ee7b7; }}
  .tier-mid {{ background:#1e3a5f; color:#93c5fd; }}
  .tier-low {{ background:#78350f; color:#fcd34d; }}
  .tier-minimal {{ background:#7f1d1d; color:#fca5a5; }}
  .chart-container {{ position:relative; height:350px; margin:1rem 0; }}
  ul {{ list-style: none; padding:0; }}
  ul li::before {{ content: ""; }}
</style>
</head>
<body>

<h1>🚀 LLM Benchmark Report</h1>
<p class="subtitle">{timestamp} | Pipeline v1.0</p>

<div class="card">
  <h2>🖥️ 하드웨어 프로필</h2>
  <table>
    <tr><td style="width:140px">OS</td><td>{hw.os_name} {hw.os_version}</td></tr>
    <tr><td>CPU</td><td>{hw.cpu_model} ({hw.cpu_cores}C/{hw.cpu_threads}T)</td></tr>
    <tr><td>AVX2 / F16C</td><td>{'✅' if hw.has_avx2 else '❌'} / {'✅' if hw.has_f16c else '❌'}</td></tr>
    <tr><td>RAM</td><td>{hw.ram_total_gb} GB (가용 {hw.ram_available_gb} GB)</td></tr>
    <tr><td>Swap</td><td>{hw.swap_gb} GB</td></tr>
    {gpu_html}
    <tr><td>Ollama</td><td>{hw.ollama_version or '미설치'}</td></tr>
    <tr><td>성능 등급</td><td><span class="tier tier-{hw.tier}">{hw.tier.upper()}</span></td></tr>
    <tr><td>가용 메모리</td><td>~{hw.usable_memory_gb:.1f} GB</td></tr>
  </table>
</div>

<div class="card">
  <h2>🎯 추천 모델</h2>
  <table>
    <tr><th>#</th><th>모델</th><th>Ollama ID</th><th>크기</th><th>최소 RAM</th><th>강점</th></tr>
    {rec_html}
  </table>
</div>

<div class="card">
  <h2>📊 벤치마크 결과</h2>
  <div class="chart-container"><canvas id="scoreChart"></canvas></div>
  <div class="chart-container"><canvas id="speedChart"></canvas></div>

  <h2 style="margin-top:1.5rem">📋 순위 요약</h2>
  <table>
    <tr><th>모델</th><th>점수순</th><th>속도순</th><th>종합</th></tr>
    {rank_summary_html}
  </table>

  <h2 style="margin-top:1.5rem">태스크별 상세</h2>
  <table>
    <tr><th>종합</th><th>모델</th><th>avg 점수</th><th>avg tok/s</th><th>종합점</th><th>태스크별 결과</th></tr>
    {bench_html}
  </table>
</div>

<script>
const labels = {chart_labels};
const scores = {chart_scores};
const tps = {chart_tps};

new Chart(document.getElementById('scoreChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{ label: '평균 점수 (0-100)', data: scores,
      backgroundColor: 'rgba(56,189,248,0.6)', borderColor: '#38bdf8', borderWidth: 1 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ title: {{ display:true, text:'모델별 평균 점수', color:'#e2e8f0' }} }},
    scales: {{ y: {{ beginAtZero:true, max:100, ticks:{{color:'#94a3b8'}}, grid:{{color:'#334155'}} }},
              x: {{ ticks:{{color:'#94a3b8'}}, grid:{{color:'#334155'}} }} }}
  }}
}});

new Chart(document.getElementById('speedChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{ label: '평균 속도 (tok/s)', data: tps,
      backgroundColor: 'rgba(129,140,248,0.6)', borderColor: '#818cf8', borderWidth: 1 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ title: {{ display:true, text:'모델별 생성 속도', color:'#e2e8f0' }} }},
    scales: {{ y: {{ beginAtZero:true, ticks:{{color:'#94a3b8'}}, grid:{{color:'#334155'}} }},
              x: {{ ticks:{{color:'#94a3b8'}}, grid:{{color:'#334155'}} }} }}
  }}
}});
</script>
</body></html>"""

        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(html)
        print(f"\n  📄 HTML 리포트 저장: {output_path}")

    @staticmethod
    def generate_json(hw: HardwareProfile, models: list[ModelSpec],
                      results: list[BenchmarkResult], output_path: str):
        """JSON 리포트 생성"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "cpu": hw.cpu_model, "cores": hw.cpu_cores, "threads": hw.cpu_threads,
                "ram_gb": hw.ram_total_gb, "ram_available_gb": hw.ram_available_gb,
                "tier": hw.tier, "usable_memory_gb": hw.usable_memory_gb,
                "gpus": [asdict(g) for g in hw.gpus],
            },
            "recommended_models": [asdict(m) for m in models],
            "benchmarks": [],
        }
        for r in results:
            entry = {
                "model": r.model_name,
                "overall_rank": r.overall_rank,
                "score_rank": r.score_rank,
                "speed_rank": r.speed_rank,
                "composite_score": r.composite_score,
                "avg_tps": r.avg_tps, "avg_score": r.avg_score,
                "tasks": [asdict(t) for t in r.task_results],
            }
            data["benchmarks"].append(entry)

        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  📄 JSON 리포트 저장: {output_path}")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🚀 LLM Hardware Detection & Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python llm_bench.py                          # 전체 파이프라인
  python llm_bench.py --detect-only            # 하드웨어 감지만
  python llm_bench.py --recommend-only         # 감지 + 추천만
  python llm_bench.py --recommend-only --top-n 15  # 상위 15개 추천
  python llm_bench.py --update-catalog         # 최신 모델 카탈로그 갱신
  python llm_bench.py --models "qwen3:8b,deepseek-r1:7b,gemma3:4b"
  python llm_bench.py --tasks reasoning,coding,korean
  python llm_bench.py --top-n 3 --output report.html
  python llm_bench.py --skip-pull              # 이미 설치된 모델만 테스트
  python llm_bench.py --weight-score 0.9 --weight-speed 0.1  # 점수 중시
  python llm_bench.py --weight-score 0.5 --weight-speed 0.5  # 균등 가중
        """
    )
    parser.add_argument("--detect-only", action="store_true", help="하드웨어 감지만 수행")
    parser.add_argument("--recommend-only", action="store_true", help="감지 + 모델 추천까지만")
    parser.add_argument("--models", type=str, help="벤치마크할 모델 (쉼표 구분)")
    parser.add_argument("--tasks", type=str, help="평가 태스크 (쉼표 구분: reasoning,coding,korean,...)")
    parser.add_argument("--top-n", type=int, default=10, help="추천 모델 상위 N개 (기본: 10)")
    parser.add_argument("--skip-pull", action="store_true", help="모델 다운로드 건너뛰기")
    parser.add_argument("--output", type=str, help="HTML 리포트 출력 경로")
    parser.add_argument("--json-output", type=str, help="JSON 리포트 출력 경로")
    parser.add_argument("--update-catalog", action="store_true",
                        help="ollama.com에서 최신 모델 카탈로그 갱신")
    parser.add_argument("--catalog", type=str, default="",
                        help="외부 모델 카탈로그 JSON 경로 (없으면 내장 카탈로그 사용)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--timeout", type=int, default=120, help="모델 응답 타임아웃 (초)")
    parser.add_argument("--weight-score", type=float, default=0.7,
                        help="종합 순위에서 점수 가중치 (0.0~1.0, 기본: 0.7)")
    parser.add_argument("--weight-speed", type=float, default=0.3,
                        help="종합 순위에서 속도 가중치 (0.0~1.0, 기본: 0.3)")

    args = parser.parse_args()

    print("\n🚀 LLM Hardware Detection & Benchmark Pipeline")
    print("="*50)

    # ── Step 1: Hardware Detection ──
    print("\n[1/4] 🔍 하드웨어 감지 중...")
    detector = HardwareDetector()
    hw = detector.detect()
    ReportGenerator.print_hardware(hw)

    if args.detect_only:
        return

    # ── Step 2: Model Recommendation ──
    print("\n[2/4] 🎯 모델 추천 중...")

    # 카탈로그 결정: --update-catalog > --catalog > 내장
    catalog = None
    if args.update_catalog:
        catalog = update_catalog_from_web(
            save_path=args.catalog or CATALOG_FILE
        )
    elif args.catalog:
        catalog = load_catalog_json(args.catalog)
        if not catalog:
            print("  ℹ️  외부 카탈로그가 비어있거나 없음, 내장 카탈로그 사용")
            catalog = MODEL_CATALOG
    else:
        # 기존 JSON 카탈로그가 있으면 자동 로드
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_catalog = os.path.join(script_dir, CATALOG_FILE)
        if os.path.exists(default_catalog):
            catalog = load_catalog_json(default_catalog)
        if not catalog:
            catalog = MODEL_CATALOG
            print(f"  ℹ️  내장 카탈로그 사용 ({len(catalog)}개 모델)")
            print(f"      최신화: --update-catalog 옵션으로 ollama.com에서 갱신 가능")

    recommender = ModelRecommender(hw, catalog)
    recommended = recommender.recommend(top_n=args.top_n)
    ReportGenerator.print_recommendations(recommended, hw)

    if args.recommend_only:
        return

    # ── Step 3: Ollama Check ──
    if not hw.ollama_version:
        print("\n❌ Ollama가 설치되어 있지 않습니다.")
        print("   설치: https://ollama.com/download")
        print("   설치 후 `ollama serve`로 서버를 시작하세요.")
        sys.exit(1)

    runner = OllamaRunner(args.ollama_url)
    if not runner.is_available():
        print(f"\n❌ Ollama 서버에 연결할 수 없습니다 ({args.ollama_url})")
        print("   `ollama serve`로 서버를 시작하세요.")
        sys.exit(1)

    # 테스트할 모델 결정
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = [m.name for m in recommended]

    # 태스크 결정
    task_keys = None
    if args.tasks:
        task_keys = [t.strip() for t in args.tasks.split(",")]

    # ── Step 4: Benchmark ──
    print(f"\n[3/4] 🏋️ 벤치마크 시작 ({len(model_names)}개 모델)")
    evaluator = TaskEvaluator(runner)
    benchmark_results = []

    local_models = runner.list_local_models()

    for idx, model_name in enumerate(model_names, 1):
        print(f"\n{'─'*50}")
        print(f"  모델 {idx}/{len(model_names)}: {model_name}")
        print(f"{'─'*50}")

        result = BenchmarkResult(model_name=model_name)

        # Pull
        if not args.skip_pull and model_name not in local_models:
            pull_time = runner.pull_model(model_name)
            if pull_time < 0:
                result.error = "모델 다운로드 실패"
                benchmark_results.append(result)
                continue
            result.pull_time_seconds = pull_time

        # 모델 정보 + 카탈로그에 없는 모델 자동 감지
        info = runner.get_model_info(model_name)
        if info:
            size_str = info.get("details", {}).get("parameter_size", "")
            family_str = info.get("details", {}).get("family", "")
            quant_str = info.get("details", {}).get("quantization_level", "")
            if size_str:
                print(f"  📋 {size_str} | {family_str} | {quant_str}")

            # 카탈로그에 없는 모델이면 자동으로 스펙 생성
            catalog_names = {m.name for m in catalog} if 'catalog' in dir() else set()
            if model_name not in catalog_names:
                auto_spec = discover_model_spec(runner, model_name)
                if auto_spec:
                    print(f"  💡 카탈로그 미등록 모델 → 자동 감지 완료")

        # Warmup (첫 실행은 로딩 시간 포함되므로)
        print("  🔥 워밍업...")
        runner.generate(model_name, "Hello", timeout=60)

        # 평가
        print("  📝 태스크 평가:")
        result.task_results = evaluator.evaluate_model(model_name, task_keys)

        # 통계
        valid = [t for t in result.task_results if not t.error]
        if valid:
            result.avg_tps = sum(t.tokens_per_second for t in valid) / len(valid)
            result.avg_score = sum(t.score for t in valid) / len(valid)

        benchmark_results.append(result)

    # 순위 산정 (3가지: 점수순, 속도순, 종합)
    w_score = args.weight_score
    w_speed = args.weight_speed
    # 가중치 정규화
    w_total = w_score + w_speed
    if w_total > 0:
        w_score /= w_total
        w_speed /= w_total

    scoreable = [r for r in benchmark_results if not r.error]
    if scoreable:
        max_tps = max(r.avg_tps for r in scoreable) or 1

        # 1) 점수순 순위
        by_score = sorted(scoreable, key=lambda x: x.avg_score, reverse=True)
        for i, r in enumerate(by_score, 1):
            r.score_rank = i

        # 2) 속도순 순위
        by_speed = sorted(scoreable, key=lambda x: x.avg_tps, reverse=True)
        for i, r in enumerate(by_speed, 1):
            r.speed_rank = i

        # 3) 종합 순위 (가중치 적용)
        for r in scoreable:
            r.composite_score = (r.avg_score * w_score) + ((r.avg_tps / max_tps) * 100 * w_speed)
        by_composite = sorted(scoreable, key=lambda x: x.composite_score, reverse=True)
        for i, r in enumerate(by_composite, 1):
            r.overall_rank = i

        # 에러 모델은 마지막
        err_rank = len(benchmark_results)
        for r in benchmark_results:
            if r.error:
                r.overall_rank = err_rank
                r.score_rank = err_rank
                r.speed_rank = err_rank

    # ── Step 5: Report ──
    print(f"\n[4/4] 📊 리포트 생성")
    ReportGenerator.print_benchmark_results(benchmark_results)

    if args.output:
        ReportGenerator.generate_html(hw, recommended, benchmark_results, args.output)
    if args.json_output:
        ReportGenerator.generate_json(hw, recommended, benchmark_results, args.json_output)

    # 최종 요약
    if scoreable:
        best_composite = sorted(scoreable, key=lambda x: x.overall_rank)[0]
        best_score = sorted(scoreable, key=lambda x: x.score_rank)[0]
        best_speed = sorted(scoreable, key=lambda x: x.speed_rank)[0]

        print(f"\n🏆 종합 1위: {best_composite.model_name} (종합 {best_composite.composite_score:.1f})")
        if best_score.model_name != best_composite.model_name:
            print(f"🎯 점수 1위: {best_score.model_name} (점수 {best_score.avg_score:.1f}/100)")
        if best_speed.model_name != best_composite.model_name:
            print(f"⚡ 속도 1위: {best_speed.model_name} (속도 {best_speed.avg_tps:.1f} tok/s)")
        print(f"\n   가중치: 점수 {w_score*100:.0f}% + 속도 {w_speed*100:.0f}%")
        print(f"   변경: --weight-score 0.9 --weight-speed 0.1")

        gpu = hw.primary_gpu
        if gpu and gpu.vendor == "apple":
            print(f"\n💡 Apple Silicon 팁:")
            print(f"   - Ollama가 Metal 가속을 자동으로 활용합니다 (별도 설정 불필요)")
            print(f"   - Unified Memory 덕분에 RAM 용량 = 사용 가능한 모델 크기입니다")
            print(f"   - 16GB → 14B, 32GB → 32B, 64GB → 70B 모델까지 구동 가능합니다")
            print(f"   - 메모리 압박 시: sudo sysctl iogpu.wired_limit_mb=N 으로 조절 가능합니다")
        elif gpu and gpu.is_igpu and gpu.vendor == "amd":
            print(f"\n💡 AMD iGPU 팁:")
            print(f"   - ROCm 또는 Vulkan 백엔드를 활용하려면 최신 드라이버를 설치하세요")
            print(f"   - BIOS에서 iGPU 전용 VRAM을 최대로 설정하세요 (보통 512MB~4GB)")
            print(f"   - Ollama에서 GPU 오프로드: OLLAMA_NUM_GPU=999 ollama serve")
            print(f"   - HSA_OVERRIDE_GFX_VERSION 환경변수로 호환성을 조정할 수 있습니다")

    print(f"\n✅ 파이프라인 완료!\n")


if __name__ == "__main__":
    main()

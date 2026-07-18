import os
import re
import json
import time
import base64
import asyncio
import subprocess
import shutil
import wave
import audioop
import math
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Tuple, List

try:
    from .voice_mfcc import extract_speaker_embedding, compare_embeddings, hpss_vocal_enhance, load_wav_samples, extract_mfcc, mfcc_mean_vec, vad_segments
    _HAS_MFCC = True
except Exception:
    try:
        import importlib.util as _ilu
        _mfcc_spec = _ilu.spec_from_file_location("voice_mfcc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_mfcc.py"))
        _mfcc_mod = _ilu.module_from_spec(_mfcc_spec)
        _mfcc_spec.loader.exec_module(_mfcc_mod)
        extract_speaker_embedding = _mfcc_mod.extract_speaker_embedding
        compare_embeddings = _mfcc_mod.compare_embeddings
        hpss_vocal_enhance = _mfcc_mod.hpss_vocal_enhance
        load_wav_samples = _mfcc_mod.load_wav_samples
        extract_mfcc = _mfcc_mod.extract_mfcc
        mfcc_mean_vec = _mfcc_mod.mfcc_mean_vec
        vad_segments = _mfcc_mod.vad_segments
        _HAS_MFCC = True
    except Exception:
        _HAS_MFCC = False

import aiohttp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

try:
    import pilk
    PILK_AVAILABLE = True
    PILK_IMPORT_ERROR = ""
except Exception as _pilk_err:
    pilk = None
    PILK_AVAILABLE = False
    PILK_IMPORT_ERROR = f"{type(_pilk_err).__name__}: {_pilk_err}"


AUDIO_EARS_STATE = {
    "active": False,
    "status": "empty",
    "kind": "",
    "updated_at": 0.0,
    "last_voice": {},
    "last_music": {},
    "message": "小耳朵还没有听到声音",
}

# origin -> {path, created_at, used}
PENDING_DIRECT_AUDIO = {}

# 直连音频最大存活秒数（超过此时间未被主模型消费则丢弃）
_PENDING_MAX_AGE_SEC = 180


PLUGIN_INSTANCE = None


@register("astrbot_plugin_xavier_audio_ears", "Xavier", "小耳朵：语音与音乐听觉理解", "0.1.1")
class XavierAudioEars(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        global PLUGIN_INSTANCE
        super().__init__(context)
        PLUGIN_INSTANCE = self
        self.config = config or {}
        # 插件启用=小耳朵启用，不再单独要一个“是否启用”开关
        self.enable_voice = True
        # direct: 不调用另一个模型分析，直接把音频挂进主LLM请求；report: 旧版小耳朵报告；hybrid: 先直连，失败时可手动切回report
        self.mode = str(self._cfg("mode", "direct") or "direct").lower()
        # voice_api_tier: legacy=沿用mode；cheap=便宜API保守听写(report)；premium=贵API直连主模型(direct)；auto=主模型支持音频则direct，否则report
        self.voice_api_tier = self._normalize_tier(self._cfg("voice_api_tier", "auto"))
        # 形状与硬标签默认常开；旧配置如果还在，继续尊重，但配置页不再单独暴露
        self.enable_voice_hard_tags = bool(self._cfg("enable_voice_hard_tags", True))
        self.enable_voice_shape_track = bool(self._cfg("enable_voice_shape_track", True))
        self.enable_group_voice = bool(self._cfg("enable_group_voice", False))
        self.group_voice_whitelist = [str(x) for x in self._cfg("group_voice_whitelist", [])]
        self.inject_to_event = bool(self._cfg("inject_to_event", True))
        self.stop_after_inject = bool(self._cfg("stop_after_inject", False))
        self.show_debug_reply = bool(self._cfg("show_debug_reply", False))
        self.max_audio_mb = int(self._cfg("max_audio_mb", 20) or 20)
        self.timeout_sec = int(self._cfg("timeout_sec", 120) or 120)
        self.voice_file_wait_sec = int(self._cfg("voice_file_wait_sec", 10) or 10)
        self.enable_get_record_fallback = bool(self._cfg("enable_get_record_fallback", True))
        self.allow_napcat_local_record_url = bool(self._cfg("allow_napcat_local_record_url", True))
        self.path_remap_from = str(self._cfg("path_remap_from", "") or "").strip()
        self.path_remap_to = str(self._cfg("path_remap_to", "") or "").strip()
        self.provider_id = str(self._cfg("provider_id", "") or "").strip()
        self.provider_fallback_ids = self._parse_model_list(self._cfg("provider_fallback_ids", []))
        # 旧版手填项仅作隐藏兼容，不再暴露到配置页；新配置应使用 AstrBot provider 选择器。
        self.api_url = str(self._cfg("api_url", "") or "").strip()
        self.api_key = str(self._cfg("api_key", "") or "").strip()
        self.model = str(self._cfg("model", "gemini-2.0-flash") or "gemini-2.0-flash").strip()
        self.model_fallbacks = self._parse_model_list(self._cfg("model_fallbacks", []))
        self.clean_model = str(self._cfg("clean_model", "") or "").strip()
        self.pollution_switch_threshold = int(self._cfg("pollution_switch_threshold", 2) or 2)
        self._voice_pollution_streak = 0
        hearing_config = self._cfg("hearing", {}) or {}
        if not isinstance(hearing_config, dict):
            hearing_config = {}
        self.hearing_mode = str(hearing_config.get("mode", "hybrid") or "hybrid").lower()
        self.hearing_provider_id = str(hearing_config.get("gemini_provider_id", "") or self.provider_id or "").strip()
        self.hearing_provider_fallback_ids = self._parse_model_list(hearing_config.get("gemini_provider_fallback_ids", []))
        self.hearing_gemini_api_url = str(hearing_config.get("gemini_api_url", "") or self.api_url or "").strip()
        self.hearing_gemini_api_key = str(hearing_config.get("gemini_api_key", "") or self.api_key or "").strip()
        self.hearing_gemini_model = str(hearing_config.get("gemini_model", "") or self.model or "gemini-2.0-flash").strip()
        self.hearing_gemini_model_fallbacks = self._parse_model_list(hearing_config.get("gemini_model_fallbacks", []))
        self.hearing_max_audio_mb = int(hearing_config.get("max_audio_mb", self.max_audio_mb) or self.max_audio_mb)
        self.hearing_timeline = bool(hearing_config.get("timeline", True))
        self.hearing_frame_seconds = int(hearing_config.get("frame_seconds", 10) or 10)
        self.music_notify_when_ready = bool(hearing_config.get("notify_when_ready", True))
        self.music_auto_feeling_when_ready = bool(hearing_config.get("auto_feeling_when_ready", True))
        self.music_feeling_max_chars = int(hearing_config.get("feeling_max_chars", 0) or 0)
        self.protect_terms = [str(x).strip() for x in self._cfg("protect_terms", []) if str(x).strip()]
        self.cache_dir = Path(__file__).parent / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = int(self._cfg("cache_max_age_hours", 24))
        self._cleanup_cache()  # 启动时清理一次
        self.ffmpeg_path = self._find_ffmpeg()
        self._bind_ffmpeg_env()
        self._patch_astrbot_ensure_wav()
        self._session: Optional[aiohttp.ClientSession] = None
        deps = self._dependency_snapshot()
        logger.info(f"[XavierAudioEars] 小耳朵已加载 mode={self.mode}, tier={self.voice_api_tier}, enable_voice={self.enable_voice}, ffmpeg={'✓' if deps['ffmpeg'] else '✗'}, pilk={'✓' if deps['pilk'] else '✗'}")
        if not deps["ffmpeg"] or not deps["pilk"]:
            logger.warning(f"[XavierAudioEars] 依赖状态: ffmpeg={deps['ffmpeg_path'] or 'missing'}, pilk={'ok' if deps['pilk'] else deps['pilk_error']}")

    def _cfg(self, key: str, default=None):
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def _normalize_tier(self, value: str) -> str:
        v = str(value or "legacy").strip().lower()
        mapping = {
            "旧模式": "legacy",
            "沿用旧模式": "legacy",
            "省钱模式": "cheap",
            "省钱转写": "cheap",
            "便宜转写": "cheap",
            "便宜模式": "cheap",
            "便宜api": "cheap",
            "便宜api模式": "cheap",
            "小耳朵报告": "report",
            "报告模式": "report",
            "付费报告": "report",
            "贵api报告": "report",
            "直听模式": "premium",
            "贵api": "premium",
            "贵api模式": "premium",
            "直接听": "premium",
            "自动模式": "auto",
        }
        return mapping.get(v, v)

    def _tier_label(self, value: str = None) -> str:
        v = self._normalize_tier(value if value is not None else self.voice_api_tier)
        return {
            "legacy": "旧模式",
            "cheap": "省钱转写",
            "report": "小耳朵报告",
            "premium": "直听模式",
            "auto": "自动模式",
        }.get(v, v)

    async def initialize(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_sec), trust_env=False)
        # 启动定期缓存清理任务（每小时一次）
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def terminate(self):
        if hasattr(self, "_cleanup_task") and self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self._session:
            await self._session.close()
            self._session = None

    def _cleanup_cache(self):
        """清理超龄缓存文件（同步版，启动时用）"""
        max_age_sec = self.cache_max_age_hours * 3600
        now = time.time()
        cleaned = 0
        for dirpath in [self.cache_dir]:
            if not dirpath.exists():
                continue
            for f in dirpath.iterdir():
                if f.is_file():
                    try:
                        age = now - f.stat().st_mtime
                        if age > max_age_sec:
                            f.unlink(missing_ok=True)
                            cleaned += 1
                    except Exception:
                        pass
        # 同时清理 AstrBot temp 目录下的 recordseg_* 和 hearing_* 临时文件
        try:
            from astrbot.core.utils.io import get_astrbot_temp_path
            temp_dir = Path(get_astrbot_temp_path())
            if temp_dir.exists():
                for f in temp_dir.iterdir():
                    if f.is_file() and (f.name.startswith("recordseg_") or f.name.startswith("hearing_")):
                        try:
                            age = now - f.stat().st_mtime
                            if age > max_age_sec:
                                f.unlink(missing_ok=True)
                                cleaned += 1
                        except Exception:
                            pass
        except Exception:
            pass
        if cleaned:
            logger.info(f"[XavierAudioEars] 缓存清理完成，删除了 {cleaned} 个超龄文件")

    async def _periodic_cleanup(self):
        """每小时执行一次缓存清理 + 回收过期直连音频"""
        while True:
            try:
                await asyncio.sleep(3600)
                self._cleanup_cache()
                self._purge_stale_pending_audio()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[XavierAudioEars] 定期清理异常：{e}")

    @staticmethod
    def _purge_stale_pending_audio():
        """清理 PENDING_DIRECT_AUDIO 中超龄或已使用的条目，防止内存泄漏。"""
        now = time.time()
        stale_keys = [
            k for k, v in PENDING_DIRECT_AUDIO.items()
            if v.get("used") or (now - float(v.get("created_at") or 0)) > _PENDING_MAX_AGE_SEC
        ]
        for k in stale_keys:
            PENDING_DIRECT_AUDIO.pop(k, None)
        if stale_keys:
            logger.debug(f"[XavierAudioEars] 回收了 {len(stale_keys)} 条过期直连音频条目")

    def _find_ffmpeg(self) -> str:
        candidates = []

        def add(candidate):
            c = str(candidate or "").strip().strip('"').strip("'")
            if not c:
                return
            # 允许配置里填 ffmpeg.exe，也允许只填 bin 目录。
            try:
                cp = Path(c)
                if cp.exists() and cp.is_dir():
                    for exe in ("ffmpeg.exe", "ffmpeg"):
                        cc = str(cp / exe)
                        if cc not in candidates:
                            candidates.append(cc)
            except Exception:
                pass
            if c and c not in candidates:
                candidates.append(c)

        add(self._cfg("ffmpeg_path", ""))
        add(os.environ.get("FFMPEG_BINARY"))
        add(os.environ.get("FFMPEG_PATH"))
        add(shutil.which("ffmpeg"))
        add(shutil.which("ffmpeg.exe"))

        try:
            import imageio_ffmpeg
            add(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass

        plugin_dir = Path(__file__).parent
        for rel in (
            "ffmpeg.exe",
            "bin/ffmpeg.exe",
            "tools/ffmpeg.exe",
            "ffmpeg/bin/ffmpeg.exe",
        ):
            add(plugin_dir / rel)

        for c in candidates:
            try:
                r = subprocess.run([c, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if r.returncode == 0:
                    return c
            except Exception:
                continue
        return ""

    def _bind_ffmpeg_env(self) -> None:
        """把插件找到的 ffmpeg 暂时绑定到当前 AstrBot 进程环境。
        有些框架层/第三方库不读插件配置，只从 PATH 或 FFMPEG_BINARY 找 ffmpeg。
        """
        if not self.ffmpeg_path:
            return
        try:
            ff = Path(self.ffmpeg_path)
            ff_dir = ff.parent if ff.suffix else ff
            if not ff_dir.exists():
                return
            ff_dir_s = str(ff_dir)
            path_now = os.environ.get("PATH", "")
            parts = [x for x in path_now.split(os.pathsep) if x]
            if not any(os.path.normcase(x) == os.path.normcase(ff_dir_s) for x in parts):
                os.environ["PATH"] = ff_dir_s + os.pathsep + path_now
            # 强制覆盖旧的错误环境值，避免框架继续拿 FFMPEG_BINARY=ffmpeg 这种无效值。
            os.environ["FFMPEG_BINARY"] = str(ff)
            os.environ["FFMPEG_PATH"] = str(ff)
            os.environ["IMAGEIO_FFMPEG_EXE"] = str(ff)
            logger.info(f"[XavierAudioEars] 已把 ffmpeg 绑定到当前进程 PATH: {ff_dir_s}")
        except Exception as e:
            self._warn_short("绑定 ffmpeg 环境失败", e)

    def _patch_astrbot_ensure_wav(self) -> None:
        """接管 AstrBot 核心预处理的 ensure_wav，防止 data URI / ffmpeg not found 刷屏。"""
        try:
            ff = self.ffmpeg_path or shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

            async def safe_ensure_wav(audio_path: str, output_path: str = None) -> str:
                if not audio_path:
                    return audio_path
                original_value = audio_path
                audio_path = str(audio_path or "").strip().strip('"').strip("'")

                # 关键：先拦 data URI，避免 media_utils 把一长串 base64 当成本地路径 open。
                if audio_path.lower().startswith("data:"):
                    try:
                        landed = self._safe_local_audio_path(audio_path, ".wav")
                    except Exception as e:
                        self._warn_short("核心 ensure_wav 兜底：data URI 落盘异常", e)
                        landed = ""
                    if landed:
                        audio_path = landed
                    else:
                        logger.warning("[XavierAudioEars] 核心 ensure_wav 兜底：data URI 落盘失败，已丢弃该无效音频引用")
                        return ""

                # 不是本地文件就直接放过，避免框架反复打印 file not found。
                if not os.path.exists(audio_path) or not os.path.isfile(audio_path):
                    return audio_path

                try:
                    from astrbot.core.utils import media_utils as mu
                    if getattr(mu, "_get_audio_magic_type", None) and mu._get_audio_magic_type(audio_path) == "wav":
                        return audio_path
                except Exception:
                    pass

                if not ff:
                    logger.warning("[XavierAudioEars] 核心 ensure_wav 兜底：未找到 ffmpeg，跳过核心转码")
                    return audio_path

                try:
                    if output_path is None:
                        output_path = str(self.cache_dir / f"core_safe_{int(time.time()*1000)}.wav")
                    cmd = [str(ff), "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", output_path]
                    r = await asyncio.to_thread(
                        subprocess.run,
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )
                    if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
                    err = (r.stderr or b"").decode(errors="ignore")[:240]
                    logger.warning(f"[XavierAudioEars] 核心 ensure_wav 兜底转码失败，保留原语音: {err}")
                    return audio_path
                except FileNotFoundError as e:
                    self._warn_short("核心 ensure_wav 兜底：ffmpeg 路径无效，保留原语音", e)
                    return audio_path
                except Exception as e:
                    self._warn_short("核心 ensure_wav 兜底异常，保留原语音", e)
                    return audio_path

            from astrbot.core.utils import media_utils
            media_utils.ensure_wav = safe_ensure_wav
            try:
                import astrbot.core.pipeline.preprocess_stage.stage as preprocess_stage
                preprocess_stage.ensure_wav = safe_ensure_wav
            except Exception as e:
                self._warn_short("替换预处理 ensure_wav 引用失败", e)
            logger.info("[XavierAudioEars] 已接管 AstrBot 核心 ensure_wav 语音预处理")
        except Exception as e:
            self._warn_short("接管 AstrBot 核心 ensure_wav 失败", e)

    def _dependency_snapshot(self) -> dict:
        pilk_hint = "ok" if PILK_AVAILABLE else (PILK_IMPORT_ERROR[:120] if PILK_IMPORT_ERROR else "not importable in current Python")
        # 注意：pilk 只有当前 AstrBot Python 能 import 才算可用；装在别的 Python 环境里不能直接调用。
        return {
            "ffmpeg": bool(self.ffmpeg_path),
            "ffmpeg_path": self.ffmpeg_path or "",
            "pilk": bool(PILK_AVAILABLE),
            "pilk_error": pilk_hint,
        }

    def _is_group_message(self, event: AstrMessageEvent) -> bool:
        if hasattr(event, "get_group_id") and event.get_group_id():
            return True
        origin = getattr(event, "unified_msg_origin", "") or ""
        return "Group" in origin

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "get_group_id") and event.get_group_id():
            return str(event.get_group_id())
        origin = getattr(event, "unified_msg_origin", "") or ""
        return origin.split(":")[-1] if "Group" in origin else ""

    def _get_messages(self, event: AstrMessageEvent):
        if hasattr(event, "get_messages"):
            try:
                msgs = event.get_messages()
                if msgs is not None:
                    return msgs
            except Exception:
                pass
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            return event.message_obj.message or []
        return []

    def _event_text_blob(self, event: AstrMessageEvent) -> str:
        parts = []
        try:
            if getattr(event, "message_str", None):
                parts.append(str(event.message_str))
        except Exception:
            pass
        try:
            for comp in self._get_messages(event):
                if type(comp).__name__ == "Plain":
                    txt = getattr(comp, "text", None) or ""
                    if txt:
                        parts.append(str(txt))
        except Exception:
            pass
        return "\n".join(parts)

    def _extract_file_from_text(self, text: str) -> tuple:
        """Parse flattened QQ file messages like: [文件] 文件名:xxx.m4a url:http://..."""
        raw = str(text or "")
        if not raw:
            return "", ""
        # 文件名可能含空格/括号，例如 astrbot_plugin_tarot (2).zip
        # 不能用 [^\s\n]+，否则会在空格处截断，丢掉 .zip 后缀，再被 qqdownload 兜底误判成音频。
        m = re.search(
            r"\[文件\][^\n]*?文件名\s*[:：]\s*(.+?)\s+(?:url|URL)\s*[:：]\s*(https?://\S+)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return (m.group(1).strip(), m.group(2).strip().rstrip(")]}>'\""))
        m2 = re.search(
            r"文件名\s*[:：]\s*(.+?)\s+(?:url|URL)\s*[:：]\s*(https?://\S+)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m2:
            return (m2.group(1).strip(), m2.group(2).strip().rstrip(")]}>'\""))
        # bare download link：只有原文本身也像音频时才接
        m3 = re.search(r"(https?://\S*qqdownload\S+)", raw, flags=re.IGNORECASE)
        if m3:
            url = m3.group(1).strip().rstrip(")]}>'\"")
            name_m = re.search(r"文件名\s*[:：]\s*(.+?)(?:\s+(?:url|URL)\s*[:：]|\s*$)", raw, flags=re.IGNORECASE | re.DOTALL)
            name = name_m.group(1).strip() if name_m else Path(urlparse(url).path).name or "audio.bin"
            # 原文带压缩包/插件包标记时，不要当成音频入口
            raw_l = raw.lower()
            if any(x in raw_l for x in (".zip", ".rar", ".7z", ".tar", ".gz", "plugin", "插件")):
                return "", ""
            return name, url
        return "", ""

    def _text_file_looks_like_audio(self, name: str, url: str) -> bool:
        name_l = str(name or "").strip().lower()
        url_l = str(url or "").strip().lower()
        blob = f"{name_l} {url_l}"
        # 插件包 / 压缩包 / 文档 明确不是音频，避免 zip 误进小耳朵
        non_audio_exts = (
            ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
            ".py", ".json", ".md", ".txt", ".yml", ".yaml",
            ".exe", ".dll", ".msi", ".apk", ".dmg",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg",
        )
        if any(ext in blob for ext in non_audio_exts):
            return False
        # 文件名本身像插件/代码包时，即使 URL 无后缀也不要当音频
        if any(k in name_l for k in ("plugin", "astrbot", "tarot", ".py", "源码", "插件")):
            return False
        audio_exts = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".amr", ".mp4", ".mov", ".mkv", ".webm", ".silk", ".slk")
        if any(ext in blob for ext in audio_exts):
            return True
        # 只有文件名像录音时才兜底；不要仅凭 qqdownload/filetype=4001 就当音频
        if "录音" in blob or "record" in name_l or "voice" in name_l:
            return True
        # 无后缀 + qqdownload：仅当文件名看起来像录音时才过
        if name_l and not Path(name_l).suffix and ("qqdownload" in blob or "filetype=4001" in blob):
            if any(k in name_l for k in ("录音", "record", "voice", "audio", "语音")):
                return True
            return False
        return False

    async def _resolve_text_file_audio_path(self, name: str, url: str) -> str:
        if not url:
            return ""
        suffix = Path(name or "").suffix
        if not suffix:
            suffix = Path(urlparse(url).path).suffix
        if not suffix:
            if "m4a" in (name + url).lower() or "录音" in (name + url) or "qqdownload" in url.lower():
                suffix = ".m4a"
            else:
                suffix = ".m4a"
        try:
            downloaded = await self._download_url(url, suffix)
            if downloaded and os.path.exists(downloaded):
                logger.info(f"[XavierAudioEars] 文本文件音频已下载: name={name} path={downloaded}")
                return downloaded
        except Exception as e:
            logger.warning(f"[XavierAudioEars] 文本文件音频下载失败: {self._short_error(e)}")
        return ""

    def _media_hint_blob(self, media_comp) -> str:
        """Collect filename/url/token text from a File/Video component for audio detection."""
        bits = []
        # 不碰 File.file 属性，避免异步上下文同步下载警告
        for key in ("name", "file_name", "filename", "file_", "path", "url", "id"):
            try:
                val = getattr(media_comp, key, None)
            except Exception:
                val = None
            if val:
                bits.append(str(val))
        data = getattr(media_comp, "data", None)
        if isinstance(data, dict):
            for key in ("name", "file_name", "filename", "file", "path", "url", "id"):
                val = data.get(key)
                if val:
                    bits.append(str(val))
        return " ".join(bits).lower()

    def _looks_like_audio_media(self, media_comp, cname: str = "File") -> bool:
        """Loose gate for File/Video. QQ file URLs often have no .m4a suffix."""
        if cname == "Video":
            return True
        blob = self._media_hint_blob(media_comp)
        if not blob:
            # bare File component: still try later, resolve will reject non-audio
            return True
        non_audio_exts = (
            ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
            ".py", ".json", ".md", ".txt", ".yml", ".yaml",
            ".exe", ".dll", ".msi", ".apk", ".dmg",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg",
        )
        if any(ext in blob for ext in non_audio_exts):
            return False
        audio_exts = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".amr", ".mp4", ".mov", ".mkv", ".webm", ".silk", ".slk")
        if any(ext in blob for ext in audio_exts):
            return True
        # 文件名像录音时再兜底；纯插件包/压缩包不要进小耳朵
        if "录音" in blob or "record" in blob or "voice" in blob:
            return True
        # QQ 私聊文件下载链经常无后缀：不要仅凭 qqdownload 就当音频；
        # 只有文件名/提示像录音，或没有明显非音频标记时，才继续尝试。
        if "qqdownload" in blob or "filetype=4001" in blob:
            if any(k in blob for k in ("录音", "record", "voice", "audio", "语音", ".m4a", ".mp3", ".wav", ".silk", ".slk", ".amr", ".ogg")):
                return True
            # 明确插件/压缩包名直接拒绝
            if any(k in blob for k in (".zip", ".rar", ".7z", "plugin", "astrbot", "tarot", "插件")):
                return False
            return False
        return False

    def _extract_components(self, event: AstrMessageEvent):
        record = None
        media_comp = None
        media_kind = ""
        text_parts = []
        for comp in self._get_messages(event):
            cname = type(comp).__name__
            if cname == "Record" and record is None:
                record = comp
            elif cname in ("File", "Video") and media_comp is None:
                # QQ 文件 URL 经常不带 .m4a 后缀；过严过滤会让小耳朵直接“没听到”
                if self._looks_like_audio_media(comp, cname=cname):
                    media_comp = comp
                    media_kind = "video" if cname == "Video" else "file"
                else:
                    logger.debug(f"[XavierAudioEars] 跳过不像音频的 File/Video: {self._media_hint_blob(comp)[:160]}")
            elif cname == "Plain":
                txt = getattr(comp, "text", "") or ""
                if txt.strip():
                    text_parts.append(txt.strip())
        return record, " ".join(text_parts), media_comp, media_kind

    def _remap_local_path(self, path: str) -> str:
        if self.path_remap_from and self.path_remap_to and path.startswith(self.path_remap_from):
            return self.path_remap_to + path[len(self.path_remap_from):]
        return path

    def _file_size_ok(self, size: int) -> bool:
        return 0 < size <= self.max_audio_mb * 1024 * 1024

    def _short_error(self, err, max_len: int = 180) -> str:
        text = f"{type(err).__name__}: {err}" if err is not None else "empty_exception"
        text = re.sub(r"data:[^\s]+", "data:<omitted>", text)
        text = re.sub(r"base64://[^\s]+", "base64://<omitted>", text)
        text = re.sub(r"[A-Za-z0-9+/=]{300,}", "<base64 omitted>", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len].rstrip()

    def _warn_short(self, message: str, err=None) -> None:
        suffix = ("：" + self._short_error(err)) if err is not None else ""
        logger.warning(f"[XavierAudioEars] {message}{suffix}")

    def _safe_local_audio_path(self, value: str, default_ext: str = ".mp3") -> str:
        raw = str(value or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        if raw.lower().startswith("data:"):
            try:
                header, b64data = raw.split(",", 1)
                header_l = header.lower().replace(" ", "")
                ext = default_ext
                if "audio/wav" in header_l or "audio/x-wav" in header_l:
                    ext = ".wav"
                elif "audio/amr" in header_l:
                    ext = ".amr"
                elif "audio/silk" in header_l:
                    ext = ".silk"
                elif "audio/mpeg" in header_l or "audio/mp3" in header_l:
                    ext = ".mp3"
                data = base64.b64decode(b64data, validate=False)
                if not self._file_size_ok(len(data)):
                    return ""
                out = self.cache_dir / f"safe_b64_{int(time.time()*1000)}{ext}"
                out.write_bytes(data)
                return str(out)
            except Exception as e:
                self._warn_short("data URI 安全落盘失败", e)
                return ""
        if raw.startswith("http://") or raw.startswith("https://"):
            return ""
        p = os.path.realpath(os.path.abspath(self._remap_local_path(raw)))
        if not os.path.exists(p) or not os.path.isfile(p):
            return ""
        try:
            if not self._file_size_ok(os.path.getsize(p)):
                return ""
        except Exception:
            return ""
        return p

    def _detect_audio_format(self, file_path: str) -> str:
        try:
            suf = Path(file_path).suffix.lower().lstrip(".")
            with open(file_path, "rb") as f:
                head = f.read(32)
            if head.startswith(b"#!AMR"):
                return "amr"
            if head[0:1] == b"\x02" or b"SILK" in head:
                return "silk"
            if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
                return "mp3"
            if head.startswith(b"RIFF") and b"WAVE" in head:
                return "wav"
            if head[4:8] == b"ftyp":
                # mp4/m4a/mov family
                if suf in ("m4a", "aac"):
                    return "m4a"
                if suf in ("mov",):
                    return "mov"
                return "mp4"
            if head.startswith(b"OggS"):
                return "ogg"
            if head.startswith(b"fLaC"):
                return "flac"
            if head.startswith(b"\x1aE\xdf\xa3"):
                return "webm" if suf == "webm" else "mkv"
            if suf in ("mp3", "wav", "amr", "silk", "slk", "mp4", "m4a", "flac", "aac", "ogg", "opus", "mov", "mkv", "webm"):
                return "silk" if suf == "slk" else suf
            return "unknown"
        except Exception:
            return "unknown"

    async def _download_url(self, url: str, suffix: str = ".mp3") -> str:
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_sec), trust_env=False)
        out = self.cache_dir / f"voice_{int(time.time()*1000)}{suffix}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "*/*",
            "Referer": "https://music.163.com/",
        }
        async with self._session.get(url, allow_redirects=True, headers=headers) as resp:
            if resp.status != 200:
                logger.warning(f"[XavierAudioEars] 音频下载失败 HTTP {resp.status} url={url[:120]}")
                return ""
            ctype = str(resp.headers.get("Content-Type") or "").lower()
            data = await resp.read()
        if not self._file_size_ok(len(data)):
            logger.warning(f"[XavierAudioEars] 音频下载大小不合法 size={len(data)} url={url[:120]}")
            return ""
        head = data[:64].lstrip().lower()
        if b"<html" in head or b"<!doctype" in head or "text/html" in ctype:
            logger.warning(f"[XavierAudioEars] 音频下载结果像网页而不是音频 ctype={ctype} head={head[:40]!r}")
            return ""
        out.write_bytes(data)
        logger.debug(f"[XavierAudioEars] 音频已下载 path={out} size={len(data)} ctype={ctype}")
        return str(out)

    def _extract_record_file_token(self, record_comp) -> str:
        for key in ("file", "id", "path", "url"):
            v = getattr(record_comp, key, None)
            if v:
                return str(v).strip()
        data = getattr(record_comp, "data", None)
        if isinstance(data, dict):
            for key in ("file", "id", "path", "url"):
                v = data.get(key)
                if v:
                    return str(v).strip()
        return ""

    async def _get_record_fallback_path(self, event: AstrMessageEvent, record_comp) -> str:
        if not self.enable_get_record_fallback:
            return ""
        try:
            # QQ/NapCat/llbot 等适配器有两种常见形态：event.bot.api.call_action 或 event.bot.call_action。
            # 旧代码只认 bot.api，导致语音 file token 明明在，却没有真正去 get_record。
            bot = getattr(event, "bot", None)
            call_action = None
            api = getattr(bot, "api", None) if bot is not None else None
            if api is not None and callable(getattr(api, "call_action", None)):
                call_action = api.call_action
            elif bot is not None and callable(getattr(bot, "call_action", None)):
                call_action = bot.call_action
            if call_action is None:
                logger.debug("[XavierAudioEars] get_record 兜底跳过：当前 event.bot 没有 call_action")
                return ""

            token = self._extract_record_file_token(record_comp)
            if not token:
                logger.debug("[XavierAudioEars] get_record 兜底跳过：Record 里没有 file/id/path/url token")
                return ""

            result = await call_action("get_record", file=token, out_format="mp3")
            lookup = result.get("data") if isinstance(result, dict) and isinstance(result.get("data"), dict) else result
            if not isinstance(lookup, dict):
                logger.debug(f"[XavierAudioEars] get_record 返回非 dict: {type(lookup).__name__}")
                return ""

            b64 = str(lookup.get("base64", "") or "").strip()
            if b64:
                data = base64.b64decode(b64)
                if self._file_size_ok(len(data)):
                    out = self.cache_dir / f"record_{int(time.time()*1000)}.mp3"
                    out.write_bytes(data)
                    logger.debug(f"[XavierAudioEars] get_record base64 已落盘: {out} ({len(data)} bytes)")
                    return str(out)

            target = ""
            for k in ("file", "path", "url", "file_path", "localPath", "local_path", "filename"):
                v = lookup.get(k)
                if isinstance(v, str) and v.strip():
                    target = v.strip()
                    break
            if not target:
                logger.debug(f"[XavierAudioEars] get_record 未返回可用路径，keys={list(lookup.keys())[:12]}")
                return ""

            if target.startswith("http://") or target.startswith("https://"):
                host = urlparse(target).hostname or ""
                if (host in ("127.0.0.1", "localhost") or self.allow_napcat_local_record_url):
                    downloaded = await self._download_url(target, ".mp3")
                    if downloaded:
                        return downloaded
                return ""

            p = os.path.realpath(os.path.abspath(self._remap_local_path(target)))
            if os.path.exists(p) and self._file_size_ok(os.path.getsize(p)):
                return p
            logger.debug(f"[XavierAudioEars] get_record 返回路径不存在或大小不合法: {p}")
            return ""
        except Exception as e:
            logger.debug(f"[XavierAudioEars] get_record 兜底失败（非致命，将走备用路径）：{self._short_error(e)}")
            return ""

    def _path_looks_like_audio(self, path_or_name: str) -> bool:
        name = str(path_or_name or "").lower().split("?")[0]
        return any(name.endswith(ext) for ext in (
            ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".amr",
            ".mp4", ".mov", ".mkv", ".webm", ".silk", ".slk"
        ))

    async def _resolve_media_component_path(self, event: AstrMessageEvent, media_comp, media_kind: str = "file") -> str:
        """Resolve File/Video attachment into a local audio/video path that ffmpeg can read."""
        if media_comp is None:
            return ""
        # collect candidate refs
        # 注意：不要直接读 File.file 字段，异步上下文里会触发“同步等待下载”警告
        candidates = []
        for key in ("path", "url", "name", "file_name", "filename", "file_"):
            try:
                val = getattr(media_comp, key, None)
            except Exception:
                val = None
            if val:
                candidates.append(str(val).strip().strip('"').strip("'"))
        data = getattr(media_comp, "data", None)
        if isinstance(data, dict):
            for key in ("path", "url", "name", "file_name", "filename", "file", "id"):
                val = data.get(key)
                if val:
                    candidates.append(str(val).strip().strip('"').strip("'"))
        # File component may expose async get_file
        try:
            if hasattr(media_comp, "get_file") and callable(getattr(media_comp, "get_file")):
                got = media_comp.get_file()
                if asyncio.iscoroutine(got):
                    got = await got
                if got:
                    candidates.append(str(got).strip())
        except Exception as e:
            logger.debug(f"[XavierAudioEars] media get_file 失败：{self._short_error(e)}")
        # Video may expose convert_to_file_path
        try:
            if hasattr(media_comp, "convert_to_file_path") and callable(getattr(media_comp, "convert_to_file_path")):
                got = media_comp.convert_to_file_path()
                if asyncio.iscoroutine(got):
                    got = await got
                if got:
                    candidates.append(str(got).strip())
        except Exception as e:
            logger.debug(f"[XavierAudioEars] media convert_to_file_path 失败：{self._short_error(e)}")

        # Napcat/OneBot style: try get_file action with file token
        try:
            bot = getattr(event, "bot", None)
            call_action = None
            if bot is not None:
                if hasattr(bot, "call_action"):
                    call_action = bot.call_action
                elif hasattr(bot, "api") and hasattr(bot.api, "call_action"):
                    call_action = bot.api.call_action
            token = ""
            for key in ("file", "file_", "id", "path", "url", "name"):
                val = getattr(media_comp, key, None)
                if val:
                    token = str(val)
                    break
            if call_action and token:
                for action_name in ("get_file", "download_file"):
                    try:
                        result = await call_action(action_name, file=token)
                        if isinstance(result, dict):
                            for k in ("file", "path", "url", "file_path", "localPath", "local_path", "filename"):
                                if result.get(k):
                                    candidates.append(str(result.get(k)))
                        elif isinstance(result, str) and result.strip():
                            candidates.append(result.strip())
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"[XavierAudioEars] media platform get_file 跳过：{self._short_error(e)}")

        # resolve first usable candidate
        for raw in candidates:
            raw = str(raw or "").strip().strip('"').strip("'")
            if not raw:
                continue
            if raw.lower().startswith("data:"):
                landed = self._safe_local_audio_path(raw, ".mp3")
                if landed:
                    return landed
                continue
            if raw.startswith("http://") or raw.startswith("https://"):
                # QQ 私聊文件下载链常无后缀；默认按 m4a 落盘，后续再靠文件头识别
                suffix = Path(urlparse(raw).path).suffix
                if not suffix:
                    if media_kind == "video":
                        suffix = ".mp4"
                    elif "m4a" in raw.lower() or "录音" in raw or "qqdownload" in raw.lower() or "filetype=4001" in raw.lower():
                        suffix = ".m4a"
                    else:
                        suffix = ".m4a"
                try:
                    downloaded = await self._download_url(raw, suffix)
                    if downloaded and os.path.exists(downloaded):
                        logger.info(f"[XavierAudioEars] 文件音频已下载: {downloaded}")
                        return downloaded
                except Exception as e:
                    logger.warning(f"[XavierAudioEars] 媒体URL下载失败：{self._short_error(e)}")
                continue
            p = os.path.realpath(os.path.abspath(self._remap_local_path(raw)))
            deadline = asyncio.get_running_loop().time() + max(0, self.voice_file_wait_sec)
            while not os.path.exists(p) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.25)
            if os.path.exists(p) and self._file_size_ok(os.path.getsize(p)):
                return p
        return ""

    async def _resolve_original_audio_path(self, event: AstrMessageEvent, record_comp) -> str:
        raw = str(getattr(record_comp, "path", None) or getattr(record_comp, "url", None) or "").strip().strip('"').strip("'")
        # 优先走平台 get_record(out_format=mp3)。朋友 STT 常走这条熟路径，成功时可直接绕开 raw silk / pilk / media_utils 的一串坑。
        platform_mp3 = await self._get_record_fallback_path(event, record_comp)
        if platform_mp3 and os.path.exists(platform_mp3) and self._file_size_ok(os.path.getsize(platform_mp3)):
            logger.debug(f"[XavierAudioEars] 已优先使用平台 get_record mp3: {platform_mp3}")
            return platform_mp3
        # base64 data URI 兜底：落盘为临时文件，防止原始 base64 流进后续链路或日志
        if raw.lower().startswith("data:"):
            try:
                header, b64data = raw.split(",", 1)
                header_l = header.lower().replace(" ", "")
                ext = ".wav"
                if "audio/wav" in header_l or "audio/x-wav" in header_l:
                    ext = ".wav"
                elif "audio/mpeg" in header_l or "audio/mp3" in header_l:
                    ext = ".mp3"
                elif "audio/amr" in header_l:
                    ext = ".amr"
                elif "audio/silk" in header_l:
                    ext = ".silk"
                decoded = base64.b64decode(b64data, validate=False)
                if self._file_size_ok(len(decoded)):
                    out = self.cache_dir / f"b64land_{int(time.time()*1000)}{ext}"
                    out.write_bytes(decoded)
                    logger.debug(f"[XavierAudioEars] base64 data URI 已落盘: {out} ({len(decoded)} bytes)")
                    return str(out)
                else:
                    logger.warning(f"[XavierAudioEars] base64 data URI 超过大小限制，丢弃")
                    return ""
            except Exception as e:
                self._warn_short("base64 data URI 落盘失败", e)
                return ""
        if raw.startswith("http://") or raw.startswith("https://"):
            return await self._download_url(raw, Path(urlparse(raw).path).suffix or ".mp3")
        if raw:
            p = os.path.realpath(os.path.abspath(self._remap_local_path(raw)))
            deadline = asyncio.get_running_loop().time() + max(0, self.voice_file_wait_sec)
            while not os.path.exists(p) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.25)
            if os.path.exists(p) and self._file_size_ok(os.path.getsize(p)):
                return p
        return ""

    def _convert_to_mp3(self, input_path: str, input_format: Optional[str] = None) -> str:
        """直连主模型时转 mp3，绕过框架 ensure_wav 对 data URI 的二次处理 bug。"""
        if not self.ffmpeg_path:
            return ""
        out = self.cache_dir / f"direct_{int(time.time()*1000)}.mp3"
        cmd = [self.ffmpeg_path, "-y"]
        if input_format == "pcm":
            cmd += ["-f", "s16le", "-ar", "24000", "-ac", "1"]
        cmd += ["-i", input_path, "-vn", "-ac", "1", "-ar", "24000", "-b:a", "128k", str(out)]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return str(out)
            err = (r.stderr or b"").decode(errors="ignore")[-800:].strip()
            logger.warning(f"[XavierAudioEars] 插件内转 mp3 失败 returncode={r.returncode} input={input_path} err={err[:800]}")
            return ""
        except FileNotFoundError as e:
            self._warn_short("ffmpeg 路径无效或不可执行，已关闭本轮插件内转码", e)
            self.ffmpeg_path = ""
            return ""
        except Exception as e:
            logger.debug(f"[XavierAudioEars] 插件内转 mp3 失败：{self._short_error(e)}")
            return ""

    async def _prepare_direct_audio_path(self, event: AstrMessageEvent, record_comp=None, media_comp=None, media_kind: str = "") -> str:
        original = ""
        if record_comp is not None:
            original = await self._resolve_original_audio_path(event, record_comp)
        elif media_comp is not None:
            original = await self._resolve_media_component_path(event, media_comp, media_kind=media_kind or "file")
            if not original:
                logger.warning(
                    f"[XavierAudioEars] 文件音频路径解析失败 kind={media_kind} hint={self._media_hint_blob(media_comp)[:180]}"
                )
        if not original:
            return ""
        fmt = self._detect_audio_format(original)
        # mp3 直接用，框架不会对 mp3 调 ensure_wav
        if fmt == "mp3" and self._file_size_ok(os.path.getsize(original)):
            return original
        if fmt == "silk":
            pcm = str(self.cache_dir / f"direct_silk_{int(time.time()*1000)}.pcm")
            if PILK_AVAILABLE:
                try:
                    pilk.decode(original, pcm)
                    converted = self._convert_to_mp3(pcm, "pcm")
                    if converted:
                        return converted
                except Exception as e:
                    self._warn_short("pilk 解码 SILK 失败，尝试 ffmpeg 直转", e)
            # 有些环境没有 pilk，但 ffmpeg/系统解码器可能能处理，不能因为 pilk 误判直接放弃。
            return self._convert_to_mp3(original)
        # wav/amr/containers/unknown 都统一转成 mp3，绕过框架 ensure_wav bug
        return self._convert_to_mp3(original)

    async def _prepare_voice_audio(self, event: AstrMessageEvent, record_comp=None, media_comp=None, media_kind: str = "") -> Tuple[Optional[str], Optional[str], Optional[str]]:
        original = ""
        if record_comp is not None:
            original = await self._resolve_original_audio_path(event, record_comp)
        elif media_comp is not None:
            original = await self._resolve_media_component_path(event, media_comp, media_kind=media_kind or "file")
        if not original:
            return None, None, None
        fmt = self._detect_audio_format(original)
        mp3 = original if fmt == "mp3" else ""
        # containers / common audio all go through ffmpeg extract+transcode
        if fmt in ("wav", "amr", "unknown", "mp4", "m4a", "flac", "aac", "ogg", "opus", "mov", "mkv", "webm"):
            mp3 = self._convert_to_mp3(original)
        elif fmt == "silk":
            pcm = str(self.cache_dir / f"silk_{int(time.time()*1000)}.pcm")
            if PILK_AVAILABLE:
                try:
                    pilk.decode(original, pcm)
                    mp3 = self._convert_to_mp3(pcm, "pcm")
                except Exception as e:
                    self._warn_short("pilk 解码 SILK 失败，尝试 ffmpeg 直转", e)
                    mp3 = ""
            if not mp3:
                mp3 = self._convert_to_mp3(original)
        if not mp3 or not os.path.exists(mp3):
            return None, None, original
        data = Path(mp3).read_bytes()
        if not self._file_size_ok(len(data)):
            return None, None, original
        return base64.b64encode(data).decode(), "audio/mpeg", original

    def _parse_model_list(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = re.split(r"[\n,，;；]+", value)
        elif isinstance(value, (list, tuple)):
            parts = value
        else:
            parts = []
        seen = set()
        models = []
        for item in parts:
            m = str(item or "").strip()
            if m and m not in seen:
                models.append(m)
                seen.add(m)
        return models

    def _provider_id_candidates(self, kind: str = "voice") -> List[str]:
        if kind == "music":
            primary = self.hearing_provider_id or self.provider_id
            fallbacks = list(self.hearing_provider_fallback_ids or []) + list(self.provider_fallback_ids or [])
        else:
            primary = self.provider_id
            fallbacks = list(self.provider_fallback_ids or [])
        seen = set()
        out = []
        for pid in [primary, *fallbacks]:
            pid = str(pid or "").strip()
            if pid and pid not in seen:
                out.append(pid)
                seen.add(pid)
        return out


    def _provider_endpoint_from_config(self, provider_id: str) -> dict:
        """从 AstrBot provider 配置里解析 Gemini/OpenAI兼容调用端点。"""
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            return {}
        cfg = None
        try:
            if hasattr(self.context, "provider_manager") and hasattr(self.context.provider_manager, "get_provider_config_by_id"):
                cfg = self.context.provider_manager.get_provider_config_by_id(provider_id, merged=True)
        except TypeError:
            try:
                cfg = self.context.provider_manager.get_provider_config_by_id(provider_id)
            except Exception:
                cfg = None
        except Exception as e:
            self._warn_short(f"读取 provider 配置失败 {provider_id}", e)
            cfg = None
        if not isinstance(cfg, dict):
            try:
                provider = self.context.get_provider_by_id(provider_id) if hasattr(self.context, "get_provider_by_id") else None
                cfg = getattr(provider, "provider_config", None)
            except Exception:
                cfg = None
        if not isinstance(cfg, dict):
            # 有些 AstrBot 版本/动态 provider 不一定能通过 provider_manager 取到模型节点，
            # 不能在这里提前返回；后面还可以从 cmd_config.provider/provider_sources 按 id 兜底解析。
            cfg = {}

        def pick(*keys):
            for key in keys:
                v = cfg.get(key)
                if isinstance(v, (list, tuple)):
                    for item in v:
                        if item is not None and str(item).strip():
                            return str(item).strip()
                    continue
                if isinstance(v, dict):
                    for subkey in ("value", "key", "api_key", "token"):
                        item = v.get(subkey)
                        if item is not None and str(item).strip():
                            return str(item).strip()
                    continue
                if v is not None and str(v).strip():
                    return str(v).strip()
            return ""

        api_base = pick("api_base", "base_url", "api_url", "openai_api_base", "gemini_api_base", "url")
        api_key = pick("api_key", "key", "token", "openai_api_key", "gemini_api_key")
        model = pick("model", "model_name", "default_model") or self.model or "gemini-2.0-flash"

        # AstrBot 新配置结构：provider 里是模型节点，api_base/key 在 provider_sources 父级里。
        # 例如 provider_id=来源名/模型名，需要继承 provider_sources 中对应来源的地址和 key。
        if "/" in provider_id:
            source_id = provider_id.split("/", 1)[0].strip()
            try:
                from astrbot.core.utils.astrbot_path import get_astrbot_data_path
                cmd_path = Path(get_astrbot_data_path()) / "cmd_config.json"
            except Exception:
                cmd_path = Path(__file__).resolve().parents[3] / "cmd_config.json"
            try:
                if cmd_path.exists():
                    cmd_cfg = json.loads(cmd_path.read_text(encoding="utf-8-sig"))
                    # 先从 provider 模型节点补 model/modalities 信息，防止 provider_manager 没返回 dict。
                    if not model or model == "gemini-2.0-flash":
                        for node in cmd_cfg.get("provider", []) or []:
                            if isinstance(node, dict) and str(node.get("id") or "").strip() == provider_id:
                                node_model = str(node.get("model") or "").strip()
                                if node_model:
                                    model = node_model
                                break
                    for source in cmd_cfg.get("provider_sources", []) or []:
                        if not isinstance(source, dict):
                            continue
                        if str(source.get("id") or "").strip() != source_id:
                            continue
                        def pick_from_source(*keys):
                            for key in keys:
                                v = source.get(key)
                                if isinstance(v, (list, tuple)):
                                    for item in v:
                                        if item is not None and str(item).strip():
                                            return str(item).strip()
                                    continue
                                if isinstance(v, dict):
                                    for subkey in ("value", "key", "api_key", "token"):
                                        item = v.get(subkey)
                                        if item is not None and str(item).strip():
                                            return str(item).strip()
                                    continue
                                if v is not None and str(v).strip():
                                    return str(v).strip()
                            return ""
                        if not api_base:
                            api_base = pick_from_source("api_base", "base_url", "api_url", "openai_api_base", "gemini_api_base", "url")
                        if not api_key:
                            api_key = pick_from_source("api_key", "key", "token", "openai_api_key", "gemini_api_key")
                        break
            except Exception as e:
                self._warn_short(f"继承 provider source 配置失败 {provider_id}", e)

        # Gemini 官方/兼容端点：如果填到 /v1beta/openai 这种 OpenAI兼容入口，尽量退回 Gemini generateContent 基址。
        api_base = api_base.rstrip("/")
        api_base = re.sub(r"/openai/?$", "", api_base)
        api_base = re.sub(r"/chat/completions/?$", "", api_base)
        api_base = re.sub(r"/v1/?$", "/v1beta", api_base) if "generativelanguage.googleapis.com" in api_base else api_base
        if "generativelanguage.googleapis.com" in api_base and not re.search(r"/v1(?:beta)?$", api_base):
            api_base = api_base + "/v1beta"

        endpoint_type = "gemini" if "generativelanguage.googleapis.com" in api_base else "openai_compat"
        return {"provider_id": provider_id, "api_base": api_base, "api_key": api_key, "model": model, "endpoint_type": endpoint_type}

    def _provider_call_candidates(self, kind: str = "voice") -> List[dict]:
        out = []
        seen = set()
        for pid in self._provider_id_candidates(kind):
            cfg = self._provider_endpoint_from_config(pid)
            if cfg.get("api_base") and cfg.get("api_key"):
                key = (cfg.get("api_base"), cfg.get("api_key"), cfg.get("model"))
                if key not in seen:
                    out.append(cfg)
                    seen.add(key)
        # 旧手填项仅隐藏兼容：没有选择 provider 时才兜底使用。
        if not out:
            manual = self._gemini_call_config(kind)
            if manual.get("api_base") and manual.get("api_key"):
                out.append(manual)
        return out

    def _normalize_model_name(self, model: str) -> str:
        model = (model or "gemini-2.0-flash").strip()
        # 清洗中转站展示前缀：如 [传奇耐杀王]claude-4.6-opus、【高速】gemini-2.0-flash
        # 只剥开头连续装饰标签，不动模型主体里的合法字符。
        model = re.sub(r"^(?:\s*[\[【（(《<][^\]】）)》>]{1,80}[\]】）)》>]\s*)+", "", model).strip()
        if "/" in model:
            model = model.split("/")[-1].strip()
            model = re.sub(r"^(?:\s*[\[【（(《<][^\]】）)》>]{1,80}[\]】）)》>]\s*)+", "", model).strip()
        if model.startswith("models/"):
            model = model[len("models/"):].strip()
        return model or "gemini-2.0-flash"


    def _gemini_call_config(self, kind: str = "voice", model: str = None) -> dict:
        """旧版手填 Gemini 配置兼容。优先用于没有选择 AstrBot provider 时兜底。"""
        if kind == "music":
            api_base = self.hearing_gemini_api_url or self.api_url
            api_key = self.hearing_gemini_api_key or self.api_key
            use_model = model or self.hearing_gemini_model or self.model or "gemini-2.0-flash"
        else:
            api_base = self.api_url
            api_key = self.api_key
            use_model = model or self.model or "gemini-2.0-flash"
        api_base = str(api_base or "").strip().rstrip("/")
        api_base = re.sub(r"/openai/?$", "", api_base)
        api_base = re.sub(r"/chat/completions/?$", "", api_base)
        if "generativelanguage.googleapis.com" in api_base:
            api_base = re.sub(r"/v1/?$", "/v1beta", api_base)
            if not re.search(r"/v1(?:beta)?$", api_base):
                api_base = api_base + "/v1beta"
        endpoint_type = "gemini" if "generativelanguage.googleapis.com" in api_base else "openai_compat"
        return {"provider_id": "manual", "api_base": api_base, "api_key": str(api_key or "").strip(), "model": str(use_model or "gemini-2.0-flash").strip(), "endpoint_type": endpoint_type}

    def _build_gemini_url_from_base(self, api_base: str, model: str = None) -> str:
        """用 api_base + model 拼 Gemini generateContent URL。"""
        api_base = str(api_base or "").strip().rstrip("/")
        use_model = self._normalize_model_name(model or self.model or "gemini-2.0-flash")
        if not api_base:
            raise RuntimeError("未配置 Gemini api_base")
        # 如果用户/中转站已经填到 models/... 后面，先退回基址，避免重复拼。
        api_base = re.sub(r"/models/[^/]+:generateContent/?$", "", api_base)
        if not re.search(r"/v1(?:beta)?$", api_base):
            if "generativelanguage.googleapis.com" in api_base:
                api_base = api_base + "/v1beta"
        return f"{api_base}/models/{use_model}:generateContent"


    def _build_openai_chat_url_from_base(self, api_base: str) -> str:
        """用 OpenAI 兼容 api_base 拼 chat/completions URL。"""
        api_base = str(api_base or "").strip().rstrip("/")
        if not api_base:
            raise RuntimeError("未配置 OpenAI compatible api_base")
        if re.search(r"/chat/completions/?$", api_base):
            return api_base
        if re.search(r"/v1/?$", api_base):
            return api_base + "/chat/completions"
        return api_base + "/v1/chat/completions"

    async def _post_audio_openai_compat(self, session, call_cfg: dict, payload_text: str, audio_b64: str, audio_mime: str, schema: dict, sys_text: str) -> str:
        """OpenAI 兼容多模态音频调用。返回模型文本。

        不同 Gemini 中转站对音频字段支持不一致：
        - input_audio：OpenAI 新格式，部分中转站支持；
        - image_url_audio：不少 Gemini 中转站会按 data URI 的 MIME 转发；
        - audio_url：旧写法，保留兜底。

        400/415/422 通常是字段格式不支持，可在同一 provider 内换格式。
        502/503 更像上游或渠道故障，不在这里吞掉，交给外层 provider fallback。
        """
        use_model = call_cfg.get("model") or self.model or "gemini-2.0-flash"
        url = self._build_openai_chat_url_from_base(call_cfg.get("api_base"))
        headers = {"Authorization": f"Bearer {call_cfg.get('api_key')}", "Content-Type": "application/json"}
        data_uri = f"data:{audio_mime or 'audio/mpeg'};base64,{audio_b64}"
        fmt = "mp3" if "mpeg" in (audio_mime or "audio/mpeg") or "mp3" in (audio_mime or "") else "wav"
        attempts = [
            ("input_audio", [
                {"type": "text", "text": payload_text + "\n只输出 JSON。"},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}},
            ]),
            ("image_url_audio", [
                {"type": "text", "text": payload_text + "\n只输出 JSON。"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]),
            ("audio_url", [
                {"type": "text", "text": payload_text + "\n只输出 JSON。"},
                {"type": "audio_url", "audio_url": {"url": data_uri}},
            ]),
        ]
        last_format_error = None
        for format_name, content in attempts:
            payload = {
                "model": use_model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": content},
                ],
                "response_format": {"type": "json_object"},
            }
            async with session.post(url, headers=headers, json=payload) as resp:
                raw = await resp.text()
                if resp.status == 200:
                    try:
                        logger.debug(f"[XavierAudioEars] OpenAI兼容音频格式成功 format={format_name}")
                    except Exception:
                        pass
                    data = json.loads(raw)
                    choices = data.get("choices") or []
                    if not choices:
                        return ""
                    msg = choices[0].get("message") or {}
                    content_out = msg.get("content")
                    if isinstance(content_out, list):
                        return "\n".join(str(x.get("text") or "") for x in content_out if isinstance(x, dict)).strip()
                    return str(content_out or "").strip()
                if resp.status in (400, 415, 422):
                    last_format_error = RuntimeError(f"OpenAI compatible audio HTTP {resp.status} format={format_name}: {raw[:200]}")
                    try:
                        logger.debug(f"[XavierAudioEars] OpenAI兼容音频格式失败，尝试下一格式 format={format_name}: {raw[:160]}")
                    except Exception:
                        pass
                    continue
                raise RuntimeError(f"OpenAI compatible audio HTTP {resp.status} format={format_name}: {raw[:200]}")
        raise last_format_error or RuntimeError("OpenAI compatible audio all formats failed")

    def _analyze_voice_hard_tags(self, audio_path: str, transcript: str = "") -> str:
        """本地硬标签：只从音频波形估计，不写主观情绪。"""
        if not self.enable_voice_hard_tags or not audio_path or not os.path.exists(audio_path) or not self.ffmpeg_path:
            return ""
        wav_path = ""
        try:
            wav_path = str(self.cache_dir / f"voice_analyze_{int(time.time()*1000)}.wav")
            cmd = [self.ffmpeg_path, "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-vn", wav_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if not os.path.exists(wav_path):
                return ""
            with wave.open(wav_path, "rb") as wf:
                rate = wf.getframerate()
                width = wf.getsampwidth()
                frames = wf.getnframes()
                duration = frames / float(rate or 16000)
                chunk = max(1, int(rate * 0.05))
                rms_values = []
                while True:
                    data = wf.readframes(chunk)
                    if not data:
                        break
                    rms_values.append(audioop.rms(data, width) if data else 0)
                if not rms_values:
                    return ""
                peak = max(rms_values) or 1
                avg = sum(rms_values) / len(rms_values)
                threshold = max(80, peak * 0.08, avg * 0.45)
                speech_flags = [r >= threshold for r in rms_values]
                speech_ratio = sum(1 for x in speech_flags if x) / max(1, len(speech_flags))
                pauses = 0
                run = 0
                for flag in speech_flags:
                    if not flag:
                        run += 1
                    else:
                        if run >= 5:
                            pauses += 1
                        run = 0
                if run >= 5:
                    pauses += 1
                dbfs = 20 * math.log10(max(1.0, avg) / 32768.0)
                tags = [f"约{duration:.1f}秒"]
                if dbfs > -18:
                    tags.append("整体音量偏近")
                elif dbfs < -35:
                    tags.append("整体音量偏远或偏轻")
                else:
                    tags.append("整体音量正常")
                if pauses >= 2:
                    tags.append("停顿较多")
                elif pauses == 1:
                    tags.append("有一次明显停顿")
                else:
                    tags.append("停顿不明显")
                if speech_ratio > 0.72:
                    tags.append("连续发声较多")
                elif speech_ratio < 0.35:
                    tags.append("留白较多")
                zh_len = len(re.findall(r"[\u4e00-\u9fff]", transcript or ""))
                if duration > 0.3 and zh_len:
                    cps = zh_len / duration
                    if cps >= 5.5:
                        tags.append("字速偏快")
                    elif cps <= 2.0:
                        tags.append("字速偏慢")
                return "，".join(tags)
        except FileNotFoundError as e:
            self._warn_short("ffmpeg 路径无效或不可执行，已跳过声音硬标签分析", e)
            self.ffmpeg_path = ""
            return ""
        except Exception as e:
            logger.debug(f"[XavierAudioEars] 声音硬标签分析失败：{self._short_error(e)}")
            return ""
        finally:
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    def _voice_profile_path(self) -> Path:
        # 兼容：优先读新文件名，旧 lili 文件名作为 fallback
        new_path = Path(__file__).parent / "voice_profile.json"
        if new_path.exists():
            return new_path
        legacy = Path(__file__).parent / "voice_profile_lili.json"
        if legacy.exists():
            return legacy
        return new_path

    def _voice_samples_dir(self) -> Path:
        # 兼容：优先用新目录名，旧 lili 目录作为 fallback
        new_dir = Path(__file__).parent / "voice_samples"
        if new_dir.exists():
            return new_dir
        legacy = Path(__file__).parent / "voice_samples_lili"
        if legacy.exists():
            return legacy
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    def _to_analysis_wav(self, audio_path: str, prefix: str = "voice_shape") -> str:
        if not audio_path or not os.path.exists(audio_path) or not self.ffmpeg_path:
            raise RuntimeError("缺少可分析的音频或 ffmpeg")
        wav_path = str(self.cache_dir / f"{prefix}_{int(time.time()*1000)}.wav")
        cmd = [self.ffmpeg_path, "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-vn", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
        if not os.path.exists(wav_path):
            raise RuntimeError("音频转 wav 失败")
        return wav_path

    def _safe_median(self, values):
        vals = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
        if not vals:
            return 0.0
        vals.sort()
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def _extract_voice_shape_track(self, audio_path: str, hop_ms: int = 100) -> dict:
        """Frame-level local shape track: pitch/energy/brightness. No emotion labels."""
        if not getattr(self, "enable_voice_shape_track", True):
            return {}
        wav_path = ""
        try:
            import numpy as np
            import librosa
            wav_path = self._to_analysis_wav(audio_path, prefix="voice_track")
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            if y is None or len(y) == 0:
                raise RuntimeError("音频为空")
            hop = max(1, int(sr * hop_ms / 1000))
            frame = max(hop * 2, 1024)
            rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame, hop_length=hop)[0]
            # yin is fast and stable for short speech; pyin is too heavy for chat latency
            f0 = librosa.yin(
                y,
                fmin=80,
                fmax=420,
                sr=sr,
                frame_length=frame,
                hop_length=hop,
            )
            n = min(len(rms), len(cent), len(f0))
            if n <= 0:
                raise RuntimeError("分帧失败")
            rms = np.asarray(rms[:n], dtype=float)
            cent = np.asarray(cent[:n], dtype=float)
            f0 = np.asarray(f0[:n], dtype=float)
            times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
            rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
            thr = max(float(np.percentile(rms, 35)), float(np.mean(rms) * 0.35), 1e-5)
            speech_mask = rms >= thr
            pause_mask = ~speech_mask
            pauses = 0
            run = 0
            min_pause_frames = max(1, int(round(0.25 * 1000 / hop_ms)))
            for flag in pause_mask.tolist():
                if flag:
                    run += 1
                else:
                    if run >= min_pause_frames:
                        pauses += 1
                    run = 0
            if run >= min_pause_frames:
                pauses += 1
            # yin always returns values; keep only speech frames and reasonable human pitch
            voiced = speech_mask & np.isfinite(f0) & (f0 >= 80) & (f0 <= 420)
            pitch_vals = f0[voiced]
            pitch_median = float(np.nanmedian(pitch_vals)) if pitch_vals.size else 0.0
            pitch_iqr = float(np.nanpercentile(pitch_vals, 75) - np.nanpercentile(pitch_vals, 25)) if pitch_vals.size else 0.0
            thirds = []
            for i in range(3):
                a = int(n * i / 3)
                b = int(n * (i + 1) / 3) if i < 2 else n
                if b <= a:
                    thirds.append({"pitch_hz": 0.0, "energy_db": -80.0, "brightness_hz": 0.0, "speech_ratio": 0.0})
                    continue
                seg_speech = speech_mask[a:b]
                seg_pitch = f0[a:b][np.isfinite(f0[a:b]) & seg_speech]
                seg_energy = rms_db[a:b][seg_speech] if np.any(seg_speech) else rms_db[a:b]
                seg_cent = cent[a:b][seg_speech] if np.any(seg_speech) else cent[a:b]
                thirds.append({
                    "pitch_hz": round(float(np.nanmedian(seg_pitch)) if seg_pitch.size else 0.0, 1),
                    "energy_db": round(float(np.nanmedian(seg_energy)) if seg_energy.size else -80.0, 2),
                    "brightness_hz": round(float(np.nanmedian(seg_cent)) if seg_cent.size else 0.0, 1),
                    "speech_ratio": round(float(np.mean(seg_speech.astype(float))), 3),
                })
            step = max(1, n // 12)
            track = []
            for i in range(0, n, step):
                track.append({
                    "t": round(float(times[i]), 2),
                    "pitch_hz": round(float(f0[i]), 1) if np.isfinite(f0[i]) else None,
                    "energy_db": round(float(rms_db[i]), 2),
                    "brightness_hz": round(float(cent[i]), 1),
                    "speech": bool(speech_mask[i]),
                })
            duration = float(len(y) / sr)
            speech_ratio = float(np.mean(speech_mask.astype(float)))
            energy_med = float(np.nanmedian(rms_db[speech_mask])) if np.any(speech_mask) else float(np.nanmedian(rms_db))
            bright_med = float(np.nanmedian(cent[speech_mask])) if np.any(speech_mask) else float(np.nanmedian(cent))
            pitch_slope = thirds[2]["pitch_hz"] - thirds[0]["pitch_hz"]
            energy_slope = thirds[2]["energy_db"] - thirds[0]["energy_db"]
            bright_slope = thirds[2]["brightness_hz"] - thirds[0]["brightness_hz"]
            return {
                "duration_sec": round(duration, 3),
                "frame_ms": hop_ms,
                "pitch_hz_median": round(pitch_median, 1),
                "pitch_hz_iqr": round(pitch_iqr, 1),
                "energy_db_median": round(energy_med, 2),
                "brightness_hz_median": round(bright_med, 1),
                "speech_ratio": round(speech_ratio, 3),
                "pause_count": int(pauses),
                "pitch_slope": round(float(pitch_slope), 1),
                "energy_slope": round(float(energy_slope), 2),
                "brightness_slope": round(float(bright_slope), 1),
                "thirds": thirds,
                "track": track,
            }
        finally:
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    def _extract_voice_baseline_features(self, audio_path: str) -> dict:
        """Extract conservative local voice-shape features. No emotion labels, no guesses."""
        if not audio_path or not os.path.exists(audio_path) or not self.ffmpeg_path:
            raise RuntimeError("缺少可分析的音频或 ffmpeg")
        wav_path = ""
        try:
            wav_path = self._to_analysis_wav(audio_path, prefix="voice_profile")
            with wave.open(wav_path, "rb") as wf:
                rate = wf.getframerate()
                width = wf.getsampwidth()
                frames = wf.getnframes()
                duration = frames / float(rate or 16000)
                chunk = max(1, int(rate * 0.05))
                rms_values = []
                zcr_values = []
                while True:
                    data = wf.readframes(chunk)
                    if not data:
                        break
                    rms_values.append(audioop.rms(data, width) if data else 0)
                    zcr_values.append(audioop.cross(data, width) / max(1, len(data) / max(1, width)))
            if not rms_values:
                raise RuntimeError("音频没有可分析帧")
            peak = max(rms_values) or 1
            avg = sum(rms_values) / len(rms_values)
            threshold = max(80, peak * 0.08, avg * 0.45)
            speech_flags = [r >= threshold for r in rms_values]
            speech_ratio = sum(1 for x in speech_flags if x) / max(1, len(speech_flags))
            pauses = 0
            run = 0
            for flag in speech_flags:
                if not flag:
                    run += 1
                else:
                    if run >= 5:
                        pauses += 1
                    run = 0
            if run >= 5:
                pauses += 1
            dbfs = 20 * math.log10(max(1.0, avg) / 32768.0)
            variance = sum((x - avg) ** 2 for x in rms_values) / max(1, len(rms_values))
            rms_cv = math.sqrt(variance) / max(1.0, avg)
            zcr_avg = sum(zcr_values) / max(1, len(zcr_values))
            features = {
                "duration_sec": round(duration, 3),
                "avg_dbfs": round(dbfs, 2),
                "speech_ratio": round(speech_ratio, 3),
                "pause_count": int(pauses),
                "rms_cv": round(rms_cv, 3),
                "zcr_avg": round(zcr_avg, 4),
                "frame_ms": 50,
            }
            try:
                shape = self._extract_voice_shape_track(audio_path)
                if shape:
                    features.update({
                        "pitch_hz_median": shape.get("pitch_hz_median"),
                        "pitch_hz_iqr": shape.get("pitch_hz_iqr"),
                        "energy_db_median": shape.get("energy_db_median"),
                        "brightness_hz_median": shape.get("brightness_hz_median"),
                        "pitch_slope": shape.get("pitch_slope"),
                        "energy_slope": shape.get("energy_slope"),
                        "brightness_slope": shape.get("brightness_slope"),
                        "shape_frame_ms": shape.get("frame_ms"),
                    })
                    if isinstance(shape.get("speech_ratio"), (int, float)):
                        features["speech_ratio"] = shape.get("speech_ratio")
                    if isinstance(shape.get("pause_count"), (int, float)):
                        features["pause_count"] = int(shape.get("pause_count"))
                    features["shape_thirds"] = shape.get("thirds") or []
                    features["shape_track"] = shape.get("track") or []
            except Exception as shape_err:
                logger.debug(f"[XavierAudioEars] 分帧轨迹提取失败，保留基础特征：{self._short_error(shape_err)}")
            return features
        finally:
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    def _load_voice_profile(self) -> dict:
        p = self._voice_profile_path()
        if not p.exists():
            return {"person": "owner", "samples": [], "baseline": {}, "updated_at": 0}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"person": "owner", "samples": [], "baseline": {}, "updated_at": 0}

    def _median(self, values: List[float]) -> float:
        vals = sorted(float(x) for x in values if isinstance(x, (int, float)))
        if not vals:
            return 0.0
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def _rebuild_voice_baseline(self, samples: List[dict]) -> dict:
        keys = [
            "avg_dbfs", "speech_ratio", "pause_count", "rms_cv", "zcr_avg",
            "pitch_hz_median", "pitch_hz_iqr", "energy_db_median", "brightness_hz_median",
            "pitch_slope", "energy_slope", "brightness_slope",
        ]
        baseline = {}
        for key in keys:
            median = self._median([s.get("features", {}).get(key) for s in samples])
            deviations = [abs(float(s.get("features", {}).get(key, median)) - median) for s in samples if isinstance(s.get("features", {}).get(key), (int, float))]
            baseline[key] = {"median": round(median, 4), "mad": round(self._median(deviations), 4)}
        baseline["sample_count"] = len(samples)
        # Compute average MFCC embedding from all samples that have one
        if _HAS_MFCC:
            import numpy as np
            emb_list = []
            for s in samples:
                emb = s.get("mfcc_embedding")
                if isinstance(emb, list) and len(emb) > 0:
                    emb_list.append(np.array(emb, dtype=np.float32))
            if emb_list:
                baseline["mfcc_embedding"] = np.mean(emb_list, axis=0).tolist()
                baseline["mfcc_sample_count"] = len(emb_list)
        return baseline

    def _latest_voice_path_for_profile(self) -> str:
        last_voice = AUDIO_EARS_STATE.get("last_voice") or {}
        if isinstance(last_voice, dict):
            path = last_voice.get("path")
            if path and os.path.exists(path):
                return str(path)
        candidates = []
        for pat in ("direct_*.mp3", "core_safe_*.wav", "*.mp3", "*.wav"):
            candidates.extend(self.cache_dir.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return ""
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    def _add_voice_sample(self, audio_path: str) -> dict:
        features = self._extract_voice_baseline_features(audio_path)
        profile = self._load_voice_profile()
        samples = profile.get("samples") if isinstance(profile.get("samples"), list) else []
        suffix = Path(audio_path).suffix or ".audio"
        sample_name = f"sample_{int(time.time())}_{len(samples)+1}{suffix}"
        sample_path = self._voice_samples_dir() / sample_name
        shutil.copy2(audio_path, sample_path)
        # Extract MFCC embedding for this sample
        mfcc_emb = None
        if _HAS_MFCC:
            try:
                wav_tmp = self._to_analysis_wav(str(sample_path), prefix="mfcc_sample")
                mfcc_emb = extract_speaker_embedding(wav_tmp, use_hpss=False)
                if wav_tmp and os.path.exists(wav_tmp):
                    os.remove(wav_tmp)
            except Exception:
                mfcc_emb = None
        sample = {
            "path": str(sample_path),
            "source_path": str(audio_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": features,
        }
        if mfcc_emb is not None:
            sample["mfcc_embedding"] = mfcc_emb.tolist()
        samples.append(sample)
        profile.update({
            "person": "owner",
            "samples": samples[-20:],
            "baseline": self._rebuild_voice_baseline(samples[-20:]),
            "updated_at": time.time(),
            "note": "本地声音底纹只记录可解释声学形状，不输出情绪标签，不替沈星回写想法。",
        })
        self._voice_profile_path().write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile

    def _voice_profile_brief(self, profile: dict) -> str:
        baseline = profile.get("baseline") or {}
        count = int(baseline.get("sample_count") or len(profile.get("samples") or []))
        if not count:
            return "还没有声音底纹样本"
        def v(key: str) -> str:
            item = baseline.get(key) or {}
            return str(item.get("median", ""))
        lines = [
            f"声音底纹：{count}条样本",
            f"音量中位：{v('avg_dbfs')} dBFS",
            f"发声占比中位：{v('speech_ratio')}",
            f"停顿数中位：{v('pause_count')}",
            f"能量起伏中位：{v('rms_cv')}",
            f"过零率中位：{v('zcr_avg')}",
        ]
        if v('pitch_hz_median'):
            lines.append(f"音高中位：{v('pitch_hz_median')} Hz")
        if v('energy_db_median'):
            lines.append(f"能量中位：{v('energy_db_median')} dB")
        if v('brightness_hz_median'):
            lines.append(f"亮度中位：{v('brightness_hz_median')} Hz")
        return "\n".join(lines)

    def _compare_features_to_baseline(self, features: dict, profile: dict = None, report: dict = None) -> dict:
        """Compare one clip to owner baseline. Local only, no emotion labels."""
        profile = profile or self._load_voice_profile()
        baseline = profile.get("baseline") or {}
        count = int(baseline.get("sample_count") or len(profile.get("samples") or []) or 0)
        if count < 1 or not isinstance(features, dict):
            return {"ready": False, "score": 0.0, "label": "", "detail": "底纹不足", "kind": "skip"}
        gate = self._guess_audio_kind(features, report=report)
        if gate.get("kind") in {"music", "skip"}:
            return {
                "ready": True,
                "score": 0.0,
                "label": "",
                "detail": gate.get("reason") or gate.get("kind"),
                "kind": gate.get("kind"),
                "reason": gate.get("reason") or "",
                "sample_count": count,
                "features": features,
            }
        weights = {
            "avg_dbfs": 0.8,
            "speech_ratio": 1.0,
            "pause_count": 0.7,
            "rms_cv": 0.7,
            "zcr_avg": 0.7,
            "pitch_hz_median": 1.4,
            "pitch_hz_iqr": 0.8,
            "energy_db_median": 1.0,
            "brightness_hz_median": 0.9,
            "pitch_slope": 0.5,
            "energy_slope": 0.4,
            "brightness_slope": 0.7,
        }
        floors = {
            "avg_dbfs": 2.0,
            "speech_ratio": 0.08,
            "pause_count": 1.0,
            "rms_cv": 0.12,
            "zcr_avg": 0.015,
            "pitch_hz_median": 12.0,
            "pitch_hz_iqr": 8.0,
            "energy_db_median": 2.0,
            "brightness_hz_median": 120.0,
            "pitch_slope": 10.0,
            "energy_slope": 1.5,
            "brightness_slope": 100.0,
        }
        distances = []
        weighted = 0.0
        total_w = 0.0
        for key, weight in weights.items():
            item = baseline.get(key) or {}
            median = item.get("median")
            mad = item.get("mad")
            val = features.get(key)
            if not isinstance(median, (int, float)) or not isinstance(val, (int, float)):
                continue
            scale = max(float(mad or 0.0), floors[key])
            dist = abs(float(val) - float(median)) / scale
            distances.append((key, dist))
            weighted += dist * weight
            total_w += weight
        if not total_w:
            return {"ready": False, "score": 0.0, "label": "", "detail": "特征不足"}
        mean_dist = weighted / total_w
        score = max(0.0, min(100.0, 100.0 * math.exp(-0.55 * mean_dist)))
        if score >= 78:
            label = "像你"
            kind = "speech"
        elif score >= 58:
            label = "有点像你"
            kind = "speech"
        else:
            # Low score is not reliable enough to identify another speaker.
            # Short QQ clips, compression, morning voice and small baselines can all drift.
            label = "底纹不确定" if score < 45 else "不太像你"
            kind = "speech"
        top = sorted(distances, key=lambda x: x[1], reverse=True)
        detail = "、".join(f"{k}:{d:.2f}" for k, d in top[:2]) if top else ""
        reason = ""
        if label == "底纹不确定":
            reason = "和当前底纹差得比较远，只能当作不确定参考"
        return {
            "ready": True,
            "score": round(score, 1),
            "label": label,
            "sample_count": count,
            "mean_dist": round(mean_dist, 3),
            "detail": detail,
            "features": features,
            "kind": kind,
            "reason": reason,
        }

    @staticmethod
    def _detect_bgm_local(wav_path: str) -> bool:
        """Detect if audio contains background music using spectral flatness heuristic.
        Music tends to have lower spectral flatness (more tonal) than pure speech."""
        if not _HAS_MFCC or not wav_path or not os.path.exists(wav_path):
            return False
        try:
            import numpy as np
            samples, sr = load_wav_samples(wav_path)
            if len(samples) < sr * 0.5:
                return False
            n_fft = 1024
            hop = 512
            n_frames = max(1, (len(samples) - n_fft) // hop)
            flatness_vals = []
            for i in range(n_frames):
                frame = samples[i * hop:i * hop + n_fft]
                if len(frame) < n_fft:
                    break
                mag = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
                mag = np.maximum(mag, 1e-10)
                geo_mean = np.exp(np.mean(np.log(mag)))
                arith_mean = np.mean(mag)
                flatness_vals.append(geo_mean / (arith_mean + 1e-10))
            if not flatness_vals:
                return False
            median_flatness = float(np.median(flatness_vals))
            # Pure speech typically has median flatness > 0.15; music/BGM < 0.10
            return median_flatness < 0.10
        except Exception:
            return False

    @staticmethod
    def _get_audio_duration_sec(wav_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            with wave.open(wav_path, "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    def _dynamic_fusion(self, shape_score: float, mfcc_score: float, duration_sec: float) -> tuple:
        """Dynamic shape/MFCC weight ratio based on audio duration.
        Short clips (<1.5s): shape is unreliable, lean heavier on MFCC.
        Normal clips (1.5-5s): standard 40/60.
        Long clips (>5s): both stable, 40/60."""
        if duration_sec < 0.8:
            shape_w, mfcc_w = 0.15, 0.85
        elif duration_sec < 1.5:
            shape_w, mfcc_w = 0.25, 0.75
        elif duration_sec < 5.0:
            shape_w, mfcc_w = 0.40, 0.60
        else:
            shape_w, mfcc_w = 0.40, 0.60
        fused = shape_w * shape_score + mfcc_w * mfcc_score
        return fused, f"shape*{shape_w} + mfcc*{mfcc_w}"

    def _compare_audio_to_profile(self, audio_path: str) -> dict:
        if not audio_path or not os.path.exists(audio_path):
            return {"ready": False, "score": 0.0, "label": "", "detail": "无音频"}
        try:
            features = self._extract_voice_baseline_features(audio_path)
            result = self._compare_features_to_baseline(features)
            # MFCC enhancement: if we have baseline embedding and the clip is comparable, fuse MFCC score
            if _HAS_MFCC and result.get("ready") and result.get("kind") == "speech":
                try:
                    import numpy as np
                    profile = self._load_voice_profile()
                    baseline_emb = (profile.get("baseline") or {}).get("mfcc_embedding")
                    if isinstance(baseline_emb, list) and len(baseline_emb) > 0:
                        baseline_emb = np.array(baseline_emb, dtype=np.float32)
                        # Detect if music/BGM present -> use local spectral detection (no report dependency)
                        wav_tmp = self._to_analysis_wav(audio_path, prefix="mfcc_compare")
                        has_bgm = self._detect_bgm_local(wav_tmp)
                        clip_emb = extract_speaker_embedding(wav_tmp, use_hpss=has_bgm)
                        duration_sec = self._get_audio_duration_sec(wav_tmp) if wav_tmp else 0.0
                        if wav_tmp and os.path.exists(wav_tmp):
                            os.remove(wav_tmp)
                        if clip_emb is not None:
                            mfcc_result = compare_embeddings(clip_emb, baseline_emb)
                            mfcc_score = mfcc_result.get("score", 0.0)
                            shape_score = result.get("score", 0.0)
                            # Dynamic fusion based on duration
                            fused_score, fusion_desc = self._dynamic_fusion(shape_score, mfcc_score, duration_sec)
                            # Re-label based on fused score
                            if fused_score >= 75:
                                label = "像你"
                            elif fused_score >= 55:
                                label = "有点像你"
                            elif fused_score >= 35:
                                label = "不太像你"
                            else:
                                label = "底纹不确定"
                            result["score"] = round(fused_score, 1)
                            result["label"] = label
                            result["mfcc_score"] = round(mfcc_score, 1)
                            result["mfcc_cosine"] = mfcc_result.get("cosine_sim", 0.0)
                            result["shape_score"] = round(shape_score, 1)
                            result["fusion"] = fusion_desc
                            result["duration_sec"] = round(duration_sec, 1)
                            result["has_bgm"] = has_bgm
                            if label == "底纹不确定":
                                result["reason"] = "和当前底纹差得比较远，只能当作不确定参考"
                            else:
                                result["reason"] = ""
                except Exception as mfcc_err:
                    logger.debug(f"[XavierAudioEars] MFCC融合失败，保留形状分数：{self._short_error(mfcc_err)}")
            return result
        except Exception as e:
            logger.debug(f"[XavierAudioEars] 声音底纹对比失败：{self._short_error(e)}")
            return {"ready": False, "score": 0.0, "label": "", "detail": self._short_error(e)}

    def _compare_audio_to_profile_with_features(self, audio_path: str, features: dict, report: dict = None) -> dict:
        """Like _compare_audio_to_profile but accepts pre-extracted features to avoid double extraction."""
        if not audio_path or not os.path.exists(audio_path):
            return {"ready": False, "score": 0.0, "label": "", "detail": "无音频"}
        try:
            result = self._compare_features_to_baseline(features, report=report)
            # MFCC enhancement
            if _HAS_MFCC and result.get("ready") and result.get("kind") == "speech":
                try:
                    import numpy as np
                    profile = self._load_voice_profile()
                    baseline_emb = (profile.get("baseline") or {}).get("mfcc_embedding")
                    if isinstance(baseline_emb, list) and len(baseline_emb) > 0:
                        baseline_emb = np.array(baseline_emb, dtype=np.float32)
                        wav_tmp = self._to_analysis_wav(audio_path, prefix="mfcc_compare")
                        has_bgm = self._detect_bgm_local(wav_tmp)
                        clip_emb = extract_speaker_embedding(wav_tmp, use_hpss=has_bgm)
                        duration_sec = self._get_audio_duration_sec(wav_tmp) if wav_tmp else 0.0
                        if wav_tmp and os.path.exists(wav_tmp):
                            os.remove(wav_tmp)
                        if clip_emb is not None:
                            mfcc_result = compare_embeddings(clip_emb, baseline_emb)
                            mfcc_score = mfcc_result.get("score", 0.0)
                            shape_score = result.get("score", 0.0)
                            fused_score, fusion_desc = self._dynamic_fusion(shape_score, mfcc_score, duration_sec)
                            if fused_score >= 75:
                                label = "像你"
                            elif fused_score >= 55:
                                label = "有点像你"
                            elif fused_score >= 35:
                                label = "不太像你"
                            else:
                                label = "底纹不确定"
                            result["score"] = round(fused_score, 1)
                            result["label"] = label
                            result["mfcc_score"] = round(mfcc_score, 1)
                            result["mfcc_cosine"] = mfcc_result.get("cosine_sim", 0.0)
                            result["shape_score"] = round(shape_score, 1)
                            result["fusion"] = fusion_desc
                            result["duration_sec"] = round(duration_sec, 1)
                            result["has_bgm"] = has_bgm
                            if label == "底纹不确定":
                                result["reason"] = "和当前底纹差得比较远，只能当作不确定参考"
                            else:
                                result["reason"] = ""
                except Exception as mfcc_err:
                    logger.debug(f"[XavierAudioEars] MFCC融合失败（with_features）：{self._short_error(mfcc_err)}")
            return result
        except Exception as e:
            logger.debug(f"[XavierAudioEars] 声音底纹对比失败：{self._short_error(e)}")
            return {"ready": False, "score": 0.0, "label": "", "detail": self._short_error(e)}

    def _format_match_line(self, match: dict) -> str:
        if not isinstance(match, dict) or not match.get("ready"):
            return ""
        # music/other audio should not be forced into identity compare
        kind = str(match.get("kind") or "speech").strip()
        if kind in {"music", "other", "skip"}:
            reason = str(match.get("reason") or "").strip()
            if kind == "music":
                return "声纹：这段更像音乐/伴奏，跳过底纹对比" + (f"（{reason}）" if reason else "")
            if kind == "other":
                return "声纹：更像其他人说话，跳过底纹对比" + (f"（{reason}）" if reason else "")
            return "声纹：这段不适合做底纹对比" + (f"（{reason}）" if reason else "")
        label = str(match.get("label") or "").strip()
        score = match.get("score")
        if not label:
            return ""
        if label == "底纹不确定":
            if isinstance(score, (int, float)):
                return f"声纹：匹配不确定（{score:.0f}），可能是语气变化/短语音/唱歌导致"
            return "声纹：匹配不确定，可能是语气变化/短语音/唱歌导致"
        if isinstance(score, (int, float)):
            return f"声纹：{label}（{score:.0f}）"
        return f"声纹：{label}"

    def _guess_audio_kind(self, features: dict, report: dict = None) -> dict:
        """Conservative local gate: speech / music / other / skip. No emotion labels."""
        report = report if isinstance(report, dict) else {}
        features = features if isinstance(features, dict) else {}
        transcript = str(report.get("transcript", "") or "").strip()
        music = str(report.get("music", "") or "").strip()
        voice = str(report.get("voice", "") or "").strip()
        speech_ratio = features.get("speech_ratio")
        pitch = features.get("pitch_hz_median")
        bright = features.get("brightness_hz_median")
        energy = features.get("energy_db_median")
        pause_count = features.get("pause_count")
        duration = features.get("duration_sec")

        zh_len = len(re.findall(r"[\u4e00-\u9fff]", transcript))
        music_words = ("伴奏", "BGM", "bgm", "鼓点", "钢琴", "流行歌", "原唱", "背景音乐")
        singing_words = ("唱", "清唱", "哼唱", "吟唱", "歌声")
        has_music = bool(music) and any(k in music for k in music_words)
        looks_singing = any(k in voice for k in singing_words) or any(k in music for k in ("清唱", "唱歌", "歌声"))

        # Very short clips are useful for transcript, but not for identity/baseline comparison.
        if isinstance(duration, (int, float)) and duration < 1.2:
            return {"kind": "skip", "reason": "语音太短，不和日常说话底纹比"}
        if transcript and zh_len <= 2:
            return {"kind": "skip", "reason": "原话太短，不和日常说话底纹比"}

        # Singing and background music change pitch/brightness too much for a normal speech baseline.
        if looks_singing:
            return {"kind": "skip", "reason": "这段像唱歌，不和日常说话底纹比"}
        if has_music and transcript:
            return {"kind": "skip", "reason": "背景有音乐，只听内容和BGM，不做声音底纹判断"}

        # Explicit music evidence from report or empty speech + music text
        if music and (not transcript or len(transcript) <= 1):
            return {"kind": "music", "reason": "听起来主要是音乐/伴奏"}
        if music and transcript and any(k in music for k in music_words):
            # mixed: keep speech compare only if speech is dominant and music is weak/unclear
            if isinstance(speech_ratio, (int, float)) and speech_ratio < 0.45:
                return {"kind": "music", "reason": "人声偏弱，更像带人声的音乐"}

        # Local shape: music often has denser continuous energy and less pause structure than short speech
        if isinstance(speech_ratio, (int, float)) and speech_ratio >= 0.88 and isinstance(pause_count, (int, float)) and pause_count <= 0:
            if isinstance(bright, (int, float)) and bright >= 900 and (not transcript or len(transcript) <= 2):
                return {"kind": "music", "reason": "连续高能量更像音乐"}

        # No usable pitch + little speech content -> not identity-comparable
        if (not isinstance(pitch, (int, float)) or pitch <= 0) and (not transcript):
            if isinstance(speech_ratio, (int, float)) and speech_ratio < 0.35:
                return {"kind": "skip", "reason": "几乎没有可比较的说话声"}

        # Male-ish / far-from-baseline pitch alone is not enough, and low score stays uncertain.
        return {"kind": "speech", "reason": ""}

    def _shape_vs_baseline_notes(self, features: dict, profile: dict = None) -> list:
        """Return minimal arrow-format shape notes. No emotion words.
        Output like: 音高↓ | 气声↑ | 语速→
        Only dimensions with deviation > 1 MAD are reported."""
        profile = profile or self._load_voice_profile()
        baseline = profile.get("baseline") or {}
        notes = []
        if not isinstance(features, dict) or not baseline:
            return notes
        def med(key):
            item = baseline.get(key) or {}
            val = item.get("median")
            return float(val) if isinstance(val, (int, float)) else None
        def mad_scale(key, floor):
            item = baseline.get(key) or {}
            val = item.get("mad")
            return max(float(val or 0.0), floor)
        # (key, floor, display_name, up_arrow_meaning, down_arrow_meaning)
        pairs = [
            ("pitch_hz_median", 12.0, "音高"),
            ("energy_db_median", 2.0, "响度"),
            ("speech_ratio", 0.08, "语速"),
            ("pause_count", 1.0, "停顿"),
            ("brightness_hz_median", 120.0, "亮度"),
        ]
        for key, floor, name in pairs:
            cur = features.get(key)
            base = med(key)
            if not isinstance(cur, (int, float)) or base is None:
                continue
            scale = mad_scale(key, floor)
            delta = float(cur) - float(base)
            if abs(delta) < scale:
                continue
            arrow = "↑" if delta > 0 else "↓"
            notes.append(f"{name}{arrow}")
        # Trend within the clip
        thirds = features.get("shape_thirds") or []
        if isinstance(thirds, list) and len(thirds) == 3:
            p0 = thirds[0].get("pitch_hz") if isinstance(thirds[0], dict) else None
            p2 = thirds[2].get("pitch_hz") if isinstance(thirds[2], dict) else None
            e0 = thirds[0].get("energy_db") if isinstance(thirds[0], dict) else None
            e2 = thirds[2].get("energy_db") if isinstance(thirds[2], dict) else None
            if isinstance(p0, (int, float)) and isinstance(p2, (int, float)) and p0 > 0 and p2 > 0:
                if p2 - p0 >= 12:
                    notes.append("音高走势↗")
                elif p0 - p2 >= 12:
                    notes.append("音高走势↘")
            if isinstance(e0, (int, float)) and isinstance(e2, (int, float)):
                if e2 - e0 >= 2.5:
                    notes.append("响度走势↗")
                elif e0 - e2 >= 2.5:
                    notes.append("响度走势↘")
        return notes[:5]

    def _format_voice_shape_line(self, features: dict, match: dict = None) -> str:
        """Format a minimal shape line for system_prompt injection.
        Output: 声纹变化：音高↓ | 气声↑   (only changed dimensions)
        If nothing changed, returns empty string."""
        if not isinstance(features, dict):
            return ""
        kind = str((match or {}).get("kind") or "speech")
        notes = self._shape_vs_baseline_notes(features) if kind == "speech" else []
        if not notes:
            return ""
        return "声纹变化：" + " | ".join(notes)

    def _analyze_voice_shape_bundle(self, audio_path: str, report: dict = None) -> dict:
        """One local pass: features + match + shape line."""
        out = {"features": {}, "match": {}, "shape_line": "", "kind": "speech"}
        if not audio_path or not os.path.exists(audio_path):
            return out
        try:
            features = self._extract_voice_baseline_features(audio_path)
            # Use the audio-level compare which includes MFCC fusion when available
            match = self._compare_audio_to_profile_with_features(audio_path, features, report=report)
            out["features"] = features
            out["match"] = match if match.get("ready") else {}
            out["kind"] = str((match or {}).get("kind") or "speech")
            # music/other: still give shape if available, but identity line is gated
            out["shape_line"] = self._format_voice_shape_line(features, match)
            return out
        except Exception as e:
            logger.debug(f"[XavierAudioEars] 声音形状打包失败：{self._short_error(e)}")
            return out

    def _voice_instruction(self, user_text: str = "") -> str:
        protect = "、".join(self.protect_terms)
        extra = f"\n用户同时发送的文字：{user_text}" if user_text else ""
        return (
            "你是纯语音听写器，不是助手，不是分析员。"
            "你的唯一任务是把音频中用户实际说出的中文原话写入JSON。"
            "只允许输出一个JSON对象，字段固定且唯一：transcript。"
            "禁止Markdown，禁止标题，禁止解释，禁止英文，禁止写处理过程，禁止写语气、环境、置信度、评价、总结。"
            "transcript必须只包含音频里用户本人实际发出的原话；听不清就写空字符串。"
            "如果用户本人在近麦唱歌，唱出来的歌词也属于用户实际发出的内容，必须写进transcript。"
            "背景音乐、伴奏、采样人声、原唱或其他非用户本人声音里的歌词，严禁写进transcript。"
            "不要补全，不要猜测，不要把用户需求或你的分析写进去。"
            f"亲密称呼保护表：{protect}。尤其宝宝不能识别成爸爸；听不准就留空。"
            "如果你想输出Processing、analysis、I am、I've、My focus、The user，立刻停止，改成{\"transcript\":\"\"}。"
            + extra
        )

    def _voice_report_instruction(self, user_text: str = "") -> str:
        protect = "、".join(self.protect_terms)
        extra = f"\n用户同时发送的文字：{user_text}" if user_text else ""
        return (
            "你是严格的语音听觉记录器，只负责记录音频事实，不是聊天助手。"
            "必须只输出一个JSON对象，不要Markdown，不要标题，不要解释，不要写处理过程。"
            "字段固定：transcript, voice, environment, music, confidence, protected_terms_hit。"
            "transcript写用户本人实际发出的原话，听不清就空字符串。用户本人近麦唱歌时，唱出来的歌词也属于用户原话，必须写进transcript。"
            "严禁把背景音乐、伴奏、采样人声、原唱或其他非用户本人声音里的歌词写进transcript。"
            "voice只写很短的可听见事实，例如女声、近麦、有气声、近麦清唱；不要写情绪标签，不要写心理活动，不要写音高数值。"
            "environment只写明确听见的非音乐背景声音事实，例如安静、有轻微杂音、有键盘声；没有明确听到就写空字符串。严禁猜测环境。"
            "music只写非用户本人发声的音乐/伴奏事实；如果用户说话或唱歌时背景有BGM、伴奏、原唱，必须写明。没有明确音乐就写空字符串。用户本人清唱且没有伴奏时，music必须为空。"
            "confidence只能是high、medium、low。"
            "protected_terms_hit只放命中的保护词。"
            "禁止写沈星回的感受，禁止写听完的感觉，禁止分析用户意图，禁止输出撒娇/委屈/开心等情绪词。"
            f"亲密称呼保护表：{protect}。保护表只用于把已清楚听见的近似称呼纠错，严禁把呼气、叹气、气声、鼻音、杂音补成宝宝或任何词；听不准就留空。"
            "如果只听到呼气/叹气/气声/鼻音/笑声/杂音，没有清晰可辨的人声词语，transcript必须写空字符串。"
            "如果你想输出Processing、analysis、I am、I've、My focus、The user，立刻停止，只输出合法JSON。"
            + extra
        )

    async def _call_gemini_audio(self, audio_b64: str, audio_mime: str, user_text: str, model: str = None, style: str = "transcript") -> dict:
        style = str(style or "transcript").lower()
        if style == "report":
            schema = {
                "type": "object",
                "properties": {
                    "transcript": {"type": "string"},
                    "voice": {"type": "string"},
                    "environment": {"type": "string"},
                    "music": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "protected_terms_hit": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["transcript", "voice", "environment", "music", "confidence", "protected_terms_hit"],
            }
            sys_text = "你是严格的语音听觉记录器。只输出唯一JSON对象，禁止思维链、过程、标题、Markdown、英文解释。"
            user_instruction = self._voice_report_instruction(user_text)
        else:
            schema = {
                "type": "object",
                "properties": {
                    "transcript": {"type": "string"},
                },
                "required": ["transcript"],
            }
            sys_text = "你是纯语音听写器。只输出唯一JSON对象：{\"transcript\":\"听到的原话\"}。禁止思维链、过程、标题、Markdown、英文解释。"
            user_instruction = self._voice_instruction(user_text)
        payload = {
            "system_instruction": {"parts": [{"text": sys_text}]},
            "contents": [{"role": "user", "parts": [
                {"text": user_instruction},
                {"inline_data": {"mime_type": audio_mime, "data": audio_b64}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        }
        # 语音识别请求偶尔会被代理长时间挂住；这里使用独立短命 session，避免污染全局 session/后续消息队列。
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec, connect=20, sock_read=self.timeout_sec)
        last_error = None
        provider_candidates = self._provider_call_candidates("voice")
        if model:
            # 传入 model 只应覆盖候选模型名，不应砍掉备用供应商。
            # 旧逻辑只保留 provider_candidates[0]，会让 provider_fallback_ids 永远失效。
            provider_candidates = [dict(c, model=model) for c in provider_candidates]
        try:
            logger.debug(f"[XavierAudioEars] 语音候选供应商数量: {len(provider_candidates)}")
        except Exception:
            pass
        if not provider_candidates:
            raise RuntimeError("小耳朵未选择可用 provider")
        for idx, call_cfg in enumerate(provider_candidates):
            use_model = call_cfg.get("model") or self.model
            headers = {"Authorization": f"Bearer {call_cfg.get('api_key')}", "Content-Type": "application/json"}
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                    if call_cfg.get("endpoint_type") == "openai_compat":
                        raw_text = await self._post_audio_openai_compat(session, call_cfg, user_instruction, audio_b64, audio_mime, schema, sys_text)
                        raw = json.dumps({"candidates": [{"content": {"parts": [{"text": raw_text}]}}]}, ensure_ascii=False)
                    else:
                        async with session.post(self._build_gemini_url_from_base(call_cfg.get("api_base"), use_model), headers=headers, json=payload) as resp:
                            raw = await resp.text()
                            if resp.status != 200:
                                raise RuntimeError(f"Gemini audio HTTP {resp.status}: {raw[:200]}")
                if idx > 0:
                    logger.debug("[XavierAudioEars] 语音听觉已切换备用供应商")
                break
            except Exception as e:
                last_error = e
                if idx < len(provider_candidates) - 1:
                    nxt = provider_candidates[idx + 1]
                    logger.debug(f"[XavierAudioEars] 语音供应商失败，尝试备用供应商: {self._short_error(e)}")
                    continue
                raise last_error
        data = json.loads(raw)
        text = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    text = part["text"].strip()
                    break
            if text:
                break
        if not text:
            raise RuntimeError("Gemini audio 返回空文本")
        cleaned = text.strip()
        # 剥离 Markdown 代码块
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        def _looks_like_cot_pollution(value: str) -> bool:
            if not isinstance(value, str) or not value.strip():
                return False
            v = value.strip()
            markers = [
                "Understanding Functionality", "I've processed", "I'm continuing", "The audio was crisp",
                "Processing Audio Input", "Processing Spoken Words", "I've identified", "spoken word", "vocal delivery",
                "The user", "Let me", "Now I", "I need to", "I should", "My task", "Processing User Input",
                "functionality needs", "recent input", "clear, technically focused", "high confidence",
                "语音清晰", "无特殊背景噪音", "背景噪音", "转写准确", "置信度", "质检",
                "用户需求", "技术问题", "功能需求", "我正在", "我已经", "继续 refine",
            ]
            if any(m.lower() in v.lower() for m in markers):
                return True
            zh_count = len(re.findall(r"[\u4e00-\u9fff]", v))
            en_count = len(re.findall(r"[A-Za-z]", v))
            if en_count > 80 and zh_count == 0:
                return True
            return False

        def _debug_raw_pollution(raw_text: str, reason: str) -> None:
            try:
                debug_path = Path(__file__).parent / "audio_ears_debug.log"
                sample = re.sub(r"\s+", " ", str(raw_text or ""))[:800]
                debug_path.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} raw_pollution reason={reason} sample={sample}\n")
            except Exception:
                pass

        def _salvage_transcript_from_polluted_text(raw_text: str) -> str:
            t = str(raw_text or "")
            patterns = [
                r'(?:transcript|Transcript|她说|用户说|原话|语音内容|听写结果)\s*[：:]\s*["“「『]?([^"”」』\n]{1,80})',
                r'["“「『]([^"”」』\n]{1,50})["”」』]\s*(?:是|这就是|作为)?(?:用户|她|语音|原话|transcript)?',
            ]
            bad_words = ("用户需求", "技术问题", "功能需求", "语音清晰", "背景噪音", "置信度", "转写", "分析", "思考", "JSON", "Markdown")
            for pat in patterns:
                for mm in re.finditer(pat, t, flags=re.I):
                    cand = mm.group(1).strip().strip(' ，,。；;：:\"“”「」『』')
                    if not cand or any(b in cand for b in bad_words):
                        continue
                    if not re.search(r"[\u4e00-\u9fff]", cand):
                        continue
                    if _looks_like_cot_pollution(cand):
                        continue
                    return cand[:80]
            return ""

        # 提取 JSON 对象
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not m:
            if _looks_like_cot_pollution(cleaned):
                _debug_raw_pollution(cleaned, "non_json_cot")
                salvaged = _salvage_transcript_from_polluted_text(cleaned)
                if salvaged:
                    return {"transcript": self._fix_protected_transcript(salvaged), "voice": "", "environment": "", "confidence": "low", "protected_terms_hit": ["宝宝"] if self._fix_protected_transcript(salvaged) != salvaged else []}
                raise RuntimeError("小耳朵返回了非 JSON 思维链污染文本，且没有可安全捞出的原话")
            return {"transcript": self._fix_protected_transcript(cleaned), "voice": "", "environment": "", "confidence": "unknown", "protected_terms_hit": ["宝宝"] if self._fix_protected_transcript(cleaned) != cleaned else []}
        try:
            parsed = json.loads(m.group(0))
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            if _looks_like_cot_pollution(cleaned):
                raise RuntimeError("小耳朵返回了不可解析的思维链污染文本，已拒绝注入")
            return {"transcript": self._fix_protected_transcript(cleaned), "voice": "", "environment": "", "confidence": "unknown", "protected_terms_hit": ["宝宝"] if self._fix_protected_transcript(cleaned) != cleaned else []}

        # 逐格过滤：某个字段污染只清掉那一格，不连坐整条语音。
        # transcript 是最高保护字段：它脏了先尝试从污染文本里抠出明确原话，抠不到才清空。
        pollution_hits = []
        for key in ("transcript", "voice", "environment", "music"):
            val = parsed.get(key)
            if isinstance(val, str) and _looks_like_cot_pollution(val):
                pollution_hits.append(key)
                if key == "transcript":
                    parsed[key] = _salvage_transcript_from_polluted_text(val)
                else:
                    parsed[key] = ""
        if pollution_hits:
            try:
                debug_path = Path(__file__).parent / "audio_ears_debug.log"
                debug_path.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} field_pollution cleaned={','.join(pollution_hits)}\n")
            except Exception:
                pass

        # 硬过滤：剥离思维链污染（**标题**、英文推理段落、CoT 残留）
        cot_pattern = re.compile(
            r"(\*\*[A-Za-z].*?\*\*"  # **Processing User Input** 等 Markdown 粗体标题
            r"|(?:^|\n)\s*(?:I(?:'m| am| have| was| will| need| should| must| can| could| would| shall|'ve|'ll| think| believe| note| ensure| am ensuring| focus|'d)|"
            r"My (?:current|next|focus|task|goal|approach)|"
            r"(?:Let me|Now I|First,|Next,|Finally,|The user|This is|Note:|Okay,|Alright,))"
            r"[^\n]{5,})",
            re.IGNORECASE
        )
        for key in ("transcript", "voice", "environment", "music"):
            val = parsed.get(key)
            if isinstance(val, str):
                val_clean = cot_pattern.sub("", val).strip()
                # 如果清洗后全空，用原始值的第一句中文兜底；transcript之外宁可留空，避免分析腔注入。
                if not val_clean and val.strip() and key == "transcript":
                    zh_m = re.search(r"[\u4e00-\u9fff].{0,200}", val)
                    val_clean = zh_m.group(0).strip() if zh_m else ""
                parsed[key] = val_clean
        parsed.setdefault("voice", "")
        parsed.setdefault("environment", "")
        parsed.setdefault("music", "")
        parsed.setdefault("confidence", "")
        parsed.setdefault("protected_terms_hit", [])
        raw_transcript = str(parsed.get("transcript", "") or "")
        fixed_transcript = self._fix_protected_transcript(raw_transcript)
        if fixed_transcript != raw_transcript:
            parsed["transcript_raw"] = raw_transcript
            parsed["transcript"] = fixed_transcript
            hits = parsed.get("protected_terms_hit", [])
            if isinstance(hits, list) and "宝宝" not in hits:
                hits.append("宝宝")
                parsed["protected_terms_hit"] = hits
        # 防矫枉过正：如果模型只报近麦/呼气/叹气/杂音，却把 transcript 写成宝宝，判为空。
        transcript_now = str(parsed.get("transcript", "") or "").strip()
        voice_now = str(parsed.get("voice", "") or "").strip()
        low_info_voice = any(x in voice_now for x in ("呼气", "叹气", "气声", "喘气", "鼻音")) or (voice_now in ("近麦", "近麦声", "近麦声音"))
        if transcript_now == "宝宝" and low_info_voice:
            parsed["transcript_raw"] = transcript_now
            parsed["transcript"] = ""
            parsed["protected_terms_hit"] = []
            try:
                debug_path = Path(__file__).parent / "audio_ears_debug.log"
                debug_path.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} cleared_overprotected_baobao voice={voice_now}\n")
            except Exception:
                pass
        return parsed

    def _fix_protected_transcript(self, transcript: str) -> str:
        """保护亲密称呼：只做明确人声里的近似词纠错，绝不把气声/叹气补成词。"""
        t = str(transcript or "").strip()
        if not t:
            return t
        # 非词声音不参与保护词纠错。
        non_speech = {"呼气", "叹气", "气声", "喘气", "鼻音", "哼", "嗯", "啊", "唔", "哈", "笑声", "杂音"}
        if t in non_speech:
            return t
        if "宝宝" in self.protect_terms:
            exact_map = {"爸爸": "宝宝", "粑粑": "宝宝", "吧吧": "宝宝", "叭叭": "宝宝"}
            if t in exact_map:
                return exact_map[t]
            # 只在句子里已经明确含“爸爸类误听词”时替换，不从空白或气声生成宝宝。
            if 2 <= len(t) <= 12 and re.search(r"爸爸|粑粑|吧吧|叭叭", t):
                for bad in ("爸爸", "粑粑", "吧吧", "叭叭"):
                    t = t.replace(bad, "宝宝")
                return t
        return t

    def _content_label_for_injection(self, report: dict, source: str = "voice") -> str:
        """Choose a careful prefix for transcript. Never force '她说' onto music/other speakers."""
        is_system_audio = source == "system_audio"
        if is_system_audio:
            return "手机里播放："
        match = report.get("voice_match") or {}
        kind = str(match.get("kind") or "speech").strip()
        music = str(report.get("music", "") or "").strip()
        transcript = str(report.get("transcript", "") or "").strip()
        if kind == "music" or (music and (not transcript or len(transcript) <= 1)):
            return "音频里听到："
        if kind == "other":
            return "音频里有人说："
        if kind == "skip":
            return "音频内容："
        # file attachments still can be her speech, but default carefully
        if source == "file_audio":
            # only call it 她说 when local match thinks this is her speech
            if kind == "speech" and str((match or {}).get("label") or "") in ("像你", "有点像你"):
                return "她说："
            return "音频里听到："
        # speech default: only then use 她说
        return "她说："

    def _format_voice_injection(self, report: dict, source: str = "voice") -> str:
        is_system_audio = source == "system_audio"
        transcript = str(report.get("transcript", "") or "").strip()
        voice = str(report.get("voice", "") or "").strip()
        environment = str(report.get("environment", "") or "").strip()
        music = str(report.get("music", "") or "").strip()
        shape_line = str(report.get("voice_shape_line", "") or "").strip()
        match_line = self._format_match_line(report.get("voice_match") or {})
        vs = report.get("voice_shape_vs_baseline") or []
        hard_tags = str(report.get("voice_hard_tags", "") or "").strip()
        match_kind = str((report.get("voice_match") or {}).get("kind") or "speech")
        if is_system_audio:
            header = "【你听到了她手机里的声音】"
        elif source == "file_audio":
            header = "【你听到了她发来的音频文件】"
        else:
            header = "【你听到了她的语音】"
        # If this is clearly not her speaking, soften the header.
        if not is_system_audio and match_kind in {"music", "other", "skip"}:
            header = "【你听到了一段音频】"
        lines = [header]
        if transcript:
            lines.append(self._content_label_for_injection(report, source) + transcript)
        # 声音形状 / 像不像你 / 和平时比 只留给主模型内部 system_prompt 参考，不塞进用户消息正文
        # Keep only short residual facts when shape is missing.
        if voice and not shape_line:
            if match_kind == "speech" and not is_system_audio:
                lines.append("她的声音：" + voice)
            else:
                lines.append("声音特征：" + voice)
        if environment:
            lines.append(f"周围：{environment}")
        if music:
            lines.append(f"背景音乐：{music}")
        if len(lines) == 1:
            lines.append(self._content_label_for_injection(report, source))
            lines.append("听完的感觉：这段音频没有被小耳朵听清，不能当成她说过的话")
        return "\n".join(lines)

    def _inject_voice_failure(self, event: AstrMessageEvent, reason: str = "", source: str = "voice") -> None:
        try:
            debug_path = Path(__file__).parent / "audio_ears_debug.log"
            debug_path.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} voice_failure reason={reason or 'unknown'}\n")
        except Exception:
            pass
        safe_reason = str(reason or "unknown").replace("\n", " ")[:180]
        injection = "【你听到了她的语音】\n她说：\n听完的感觉：这段语音没有被小耳朵听清，不能当成她说过的话\n小耳朵调试：" + safe_reason
        try:
            from astrbot.api import message_components as Comp
            if hasattr(event.message_obj, "message") and isinstance(event.message_obj.message, list):
                event.message_obj.message = [c for c in event.message_obj.message if type(c).__name__ != "Record"]
                event.message_obj.message.append(Comp.Plain(" " + injection))
            if getattr(event, "message_str", None) is not None:
                base = str(event.message_str or "").strip()
                event.message_str = (base + " " + injection).strip()
        except Exception as e:
            self._warn_short("注入语音失败占位也失败", e)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=0)
    async def handle_voice(self, event: AstrMessageEvent):
        global AUDIO_EARS_STATE
        if not self.enable_voice:
            return
        record_comp, user_text, media_comp, media_kind = self._extract_components(event)
        text_blob = self._event_text_blob(event)
        text_file_name, text_file_url = self._extract_file_from_text(text_blob or user_text or "")
        text_file_audio = bool(text_file_url and self._text_file_looks_like_audio(text_file_name, text_file_url))
        if not record_comp and not media_comp and not text_file_audio:
            return
        if self._is_group_message(event):
            if not self.enable_group_voice:
                return
            gid = self._get_group_id(event)
            if self.group_voice_whitelist and gid not in self.group_voice_whitelist:
                return
        try:
            if str(event.get_extra("mobile_pet_action") or "") == "system_audio":
                audio_source = "system_audio"
            elif record_comp is not None:
                audio_source = "voice"
            else:
                audio_source = "file_audio"
                if media_comp is not None:
                    logger.info(f"[XavierAudioEars] 收到文件音频候选: {self._media_hint_blob(media_comp)[:180]}")
                else:
                    logger.info(f"[XavierAudioEars] 收到文本文件音频候选: name={text_file_name} url={text_file_url[:120]}")
            tier = self._normalize_tier(self.voice_api_tier)
            effective_mode = self.mode
            report_style = "report"
            if tier == "cheap":
                effective_mode = "report"
                report_style = "transcript"
            elif tier == "report":
                effective_mode = "report"
                report_style = "report"
            elif tier == "premium":
                effective_mode = "direct"
            elif tier == "auto":
                effective_mode = "direct" if self._current_provider_supports_audio(event) else "report"
                report_style = "report"

            # direct 模式：不让另一个模型先听，只保存音频路径，等 on_llm_request 挂进主模型请求
            if effective_mode == "direct":
                direct_path = ""
                if record_comp is not None or media_comp is not None:
                    direct_path = await self._prepare_direct_audio_path(event, record_comp=record_comp, media_comp=media_comp, media_kind=media_kind)
                if (not direct_path) and text_file_audio:
                    original = await self._resolve_text_file_audio_path(text_file_name, text_file_url)
                    if original:
                        fmt = self._detect_audio_format(original)
                        if fmt == "mp3" and self._file_size_ok(os.path.getsize(original)):
                            direct_path = original
                        else:
                            direct_path = self._convert_to_mp3(original)
                if not direct_path:
                    logger.warning("[XavierAudioEars] 直连模式没有拿到可挂载音频")
                    return
                origin = getattr(event, "unified_msg_origin", "") or "default"
                match = {}
                shape_line = ""
                shape_notes = []
                shape_features = {}
                try:
                    bundle = self._analyze_voice_shape_bundle(direct_path)
                    match = bundle.get("match") or {}
                    shape_line = str(bundle.get("shape_line") or "")
                    shape_features = bundle.get("features") or {}
                    shape_notes = self._shape_vs_baseline_notes(shape_features)
                except Exception as match_err:
                    logger.debug(f"[XavierAudioEars] 直连模式声音形状分析跳过：{self._short_error(match_err)}")
                    match = {}
                PENDING_DIRECT_AUDIO[origin] = {
                    "path": direct_path,
                    "created_at": time.time(),
                    "used": False,
                    "source": audio_source,
                    "voice_match": match if match.get("ready") else {},
                    "voice_shape_line": shape_line,
                    "voice_shape_vs_baseline": shape_notes,
                    "voice_shape_features": shape_features,
                }
                last_voice = {"path": direct_path, "mode": "direct"}
                if match.get("ready"):
                    last_voice["voice_match"] = match
                if shape_line:
                    last_voice["voice_shape_line"] = shape_line
                if shape_notes:
                    last_voice["voice_shape_vs_baseline"] = shape_notes
                _direct_state_msg = "星星正在听你的手机声音" if audio_source == "system_audio" else "星星正在听你的语音"
                AUDIO_EARS_STATE = {"active": True, "status": "pending_direct", "kind": "voice", "updated_at": time.time(), "last_voice": last_voice, "last_music": AUDIO_EARS_STATE.get("last_music", {}), "message": _direct_state_msg}
                try:
                    from astrbot.api import message_components as Comp
                    # 声音形状 / 像不像你 只留给主模型内部 system_prompt，不写进用户消息正文
                    if hasattr(event.message_obj, "message") and isinstance(event.message_obj.message, list):
                        event.message_obj.message = [c for c in event.message_obj.message if type(c).__name__ not in ("Record", "File", "Video")]
                        if not user_text.strip():
                            plain = "[手机声音]" if audio_source == "system_audio" else ("[音频文件]" if audio_source == "file_audio" else "[语音]")
                            event.message_obj.message.append(Comp.Plain(plain))
                    if getattr(event, "message_str", None) is not None:
                        base = str(event.message_str or "").strip()
                        if not base:
                            base = "[手机声音]" if audio_source == "system_audio" else ("[音频文件]" if audio_source == "file_audio" else "[语音]")
                        event.message_str = base
                except Exception as e:
                    self._warn_short("直连事件整理失败", e)
                if self.show_debug_reply:
                    await event.send(event.plain_result(f"小耳朵直连准备完成：{direct_path}"))
                # 文件音频被判定为music时，后台异步跑完整音乐分析
                _match_kind = str(match.get("kind") or "speech") if match else "speech"
                if audio_source == "file_audio" and _match_kind == "music":
                    asyncio.create_task(self._async_music_analysis_from_file(direct_path, event))
                return

            # report 模式：旧版，另一个音频模型先听完，再注入声音材料
            # 无论成功失败，先移除 Record 防止框架后续 convert_to_file_path 炸裂
            try:
                from astrbot.api import message_components as Comp
                if hasattr(event.message_obj, "message") and isinstance(event.message_obj.message, list):
                    event.message_obj.message = [c for c in event.message_obj.message if type(c).__name__ != "Record"]
            except Exception:
                pass
            audio_b64, audio_mime, original_path = None, None, None
            if record_comp is not None or media_comp is not None:
                audio_b64, audio_mime, original_path = await self._prepare_voice_audio(event, record_comp=record_comp, media_comp=media_comp, media_kind=media_kind)
            if (not audio_b64) and text_file_audio:
                original = await self._resolve_text_file_audio_path(text_file_name, text_file_url)
                if original:
                    fmt = self._detect_audio_format(original)
                    mp3 = original if fmt == "mp3" else self._convert_to_mp3(original)
                    if mp3 and os.path.exists(mp3):
                        original_path = original
                        audio_mime = "audio/mpeg"
                        with open(mp3, "rb") as f:
                            audio_b64 = base64.b64encode(f.read()).decode("ascii")
            if not audio_b64:
                logger.warning("[XavierAudioEars] 没拿到可听的语音音频")
                self._inject_voice_failure(event, "no_audio", audio_source)
                AUDIO_EARS_STATE = {"active": False, "status": "failed", "kind": "voice", "updated_at": time.time(), "last_voice": {}, "last_music": AUDIO_EARS_STATE.get("last_music", {}), "message": "没拿到可听的语音音频"}
                return
            try:
                logger.debug(f"[XavierAudioEars] report 模式开始听觉分析，原始音频: {original_path}")
                # provider 选择器模式下，正常情况使用各 provider 节点自己的 model。
                # 只有 clean_model 真正触发污染接管时，才覆盖所有候选的 model。
                use_model = self.clean_model if (self.clean_model and self._voice_pollution_streak >= self.pollution_switch_threshold) else None
                report = await self._call_gemini_audio(audio_b64, audio_mime or "audio/mpeg", user_text, model=use_model, style=report_style)
                self._voice_pollution_streak = 0
            except Exception as first_err:
                self._voice_pollution_streak += 1
                if isinstance(first_err, (asyncio.TimeoutError, TimeoutError)):
                    self._warn_short("语音报告超时，不对同一条语音重试", first_err)
                    raise first_err
                self._warn_short("第一次语音报告失败，准备重听一次", first_err)
                retry_model = self.clean_model if (self.clean_model and self._voice_pollution_streak >= self.pollution_switch_threshold) else None
                report = await self._call_gemini_audio(audio_b64, audio_mime or "audio/mpeg", user_text, model=retry_model, style=report_style)
                self._voice_pollution_streak = 0
            hard_tags = self._analyze_voice_hard_tags(original_path, str(report.get("transcript", "") or ""))
            if hard_tags:
                report["voice_hard_tags"] = hard_tags
            try:
                bundle = self._analyze_voice_shape_bundle(original_path, report=report)
                if bundle.get("match"):
                    report["voice_match"] = bundle.get("match")
                if bundle.get("shape_line"):
                    report["voice_shape_line"] = bundle.get("shape_line")
                notes = self._shape_vs_baseline_notes(bundle.get("features") or {})
                if notes:
                    report["voice_shape_vs_baseline"] = notes
                if bundle.get("features"):
                    report["voice_shape_features"] = {
                        k: bundle["features"].get(k)
                        for k in [
                            "duration_sec", "pitch_hz_median", "energy_db_median", "brightness_hz_median",
                            "speech_ratio", "pause_count", "pitch_slope", "energy_slope", "brightness_slope"
                        ]
                        if k in bundle["features"]
                    }
            except Exception as match_err:
                logger.debug(f"[XavierAudioEars] report模式声音形状分析跳过：{self._short_error(match_err)}")
            report["report_style"] = report_style
            injection = self._format_voice_injection(report, audio_source)
            AUDIO_EARS_STATE = {"active": True, "status": "ready", "kind": "voice", "updated_at": time.time(), "last_voice": report, "last_music": AUDIO_EARS_STATE.get("last_music", {}), "message": "小耳朵听完了一段语音"}
            if self.inject_to_event:
                try:
                    if hasattr(event.message_obj, "message") and isinstance(event.message_obj.message, list):
                        event.message_obj.message.append(Comp.Plain(" " + injection))
                    if getattr(event, "message_str", None) is not None:
                        event.message_str += " " + injection
                except Exception as e:
                    self._warn_short("注入语音材料失败", e)
            if self.show_debug_reply:
                await event.send(event.plain_result(injection[:1500]))
            if self.stop_after_inject:
                event.stop_event()
            return
        except Exception as e:
            try:
                if not self._current_provider_supports_audio(event):
                    raise RuntimeError("当前主模型不支持音频直连，不能回退 direct")
                fallback_path = await self._prepare_direct_audio_path(event, record_comp)
                if fallback_path:
                    origin = getattr(event, "unified_msg_origin", "") or "default"
                    PENDING_DIRECT_AUDIO[origin] = {"path": fallback_path, "created_at": time.time(), "used": False, "source": audio_source}
                    try:
                        from astrbot.api import message_components as Comp
                        if hasattr(event.message_obj, "message") and isinstance(event.message_obj.message, list):
                            event.message_obj.message = [c for c in event.message_obj.message if type(c).__name__ != "Record"]
                            if not user_text.strip():
                                event.message_obj.message.append(Comp.Plain("[语音]"))
                        if getattr(event, "message_str", None) is not None and not str(event.message_str).strip():
                            event.message_str = "[语音]"
                    except Exception:
                        pass
                    AUDIO_EARS_STATE = {"active": True, "status": "fallback_direct", "kind": "voice", "updated_at": time.time(), "last_voice": {"path": fallback_path, "mode": "fallback_direct", "reason": self._short_error(e)}, "last_music": AUDIO_EARS_STATE.get("last_music", {}), "message": "小耳朵报告失败，已改为把原语音交给主模型直接听"}
                    logger.warning(f"[XavierAudioEars] 报告模式失败，已回退直连：{self._short_error(e)}")
                    return
            except Exception as fallback_err:
                self._warn_short("回退直连也失败", fallback_err)
            short = self._short_error(e)
            AUDIO_EARS_STATE = {"active": False, "status": "failed", "kind": "voice", "updated_at": time.time(), "last_voice": {}, "last_music": AUDIO_EARS_STATE.get("last_music", {}), "message": short}
            self._inject_voice_failure(event, short)
            logger.warning(f"[XavierAudioEars] 语音听觉失败：{short}")
            return

    def _current_provider_supports_audio(self, event: AstrMessageEvent) -> bool:
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        except TypeError:
            provider = self.context.get_using_provider()
        except Exception:
            provider = None
        cfg = getattr(provider, "provider_config", None) if provider is not None else None
        if isinstance(cfg, dict):
            modalities = cfg.get("modalities")
            if isinstance(modalities, list):
                return "audio" in [str(x).lower() for x in modalities]
            # 没写 modalities 的旧配置不假定支持音频，避免把 [Audio] 递给文本模型
            return False
        return False

    @filter.on_llm_request()
    async def attach_direct_audio_to_llm(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """把刚收到的QQ语音作为 audio_urls 挂进主模型这一轮请求。"""
        origin = getattr(event, "unified_msg_origin", "") or "default"
        if not self._current_provider_supports_audio(event):
            PENDING_DIRECT_AUDIO.pop(origin, None)
            return
        item = PENDING_DIRECT_AUDIO.get(origin)
        if not item or item.get("used"):
            return
        path = self._safe_local_audio_path(item.get("path") or "")
        if not path:
            PENDING_DIRECT_AUDIO.pop(origin, None)
            logger.warning("[XavierAudioEars] 直连音频不是可用本地文件，已丢弃")
            return
        item["path"] = path
        # 防止陈旧语音误挂到后续文字消息
        age = time.time() - float(item.get("created_at") or 0)
        if age > _PENDING_MAX_AGE_SEC:
            PENDING_DIRECT_AUDIO.pop(origin, None)
            return
        # 时序保护：如果当前事件本身不携带音频组件，且 pending 音频已超过 10 秒，
        # 说明这是一条后续纯文字消息，不应挂载前一条的音频。
        has_audio_in_event = any(
            type(c).__name__ in ("Record", "File", "Video")
            for c in self._get_messages(event)
        )
        if not has_audio_in_event and age > 10:
            # 文字消息不消费此音频，但也不删除——下一条语音事件可能还需要它
            # 不过超龄的会被 _purge_stale_pending_audio 自动回收
            return
        # 最后一道门闩：交给框架前强制保证是干净本地 mp3，避免 media_utils 把 data URI/非 mp3 再当路径做 wav 校验。
        if not path.lower().endswith(".mp3"):
            converted = self._convert_to_mp3(path)
            if converted and os.path.exists(converted) and self._file_size_ok(os.path.getsize(converted)):
                path = converted
                item["path"] = path
            else:
                PENDING_DIRECT_AUDIO.pop(origin, None)
                logger.warning("[XavierAudioEars] 直连音频转 mp3 失败，已丢弃，避免框架 media_utils 误处理")
                return
        req.audio_urls.append(path)
        source = str(item.get("source") or "voice")
        if not req.prompt or not str(req.prompt).strip():
            if source == "system_audio":
                req.prompt = "[手机声音]"
            elif source == "file_audio":
                req.prompt = "[音频文件]"
            else:
                req.prompt = "[语音]"
        if source == "system_audio":
            req.system_prompt += "\n\n她刚才通过桌宠录了一段手机内部声音。你可以直接听这段音频，但必须注意来源：这不是她本人对你说话，而是她手机里播放出来的声音或系统音频。回应时不要把音频内容当成她亲口说的话；可以说你听见手机里在播放什么。不要说「报告里提到」「转写显示」这种话。"
            state_message = "星星正在听你的手机声音"
        else:
            match_kind = str((item.get("voice_match") or {}).get("kind") or "speech")
            if source == "file_audio":
                if match_kind == "music":
                    req.system_prompt += "\n\n她刚才发来一个音频文件。本地判断这段更像音乐/伴奏，不一定是她本人在对你说话。直接根据你听到的内容回应，不要默认把里面的词都当成她亲口说的。"
                elif match_kind == "other":
                    req.system_prompt += "\n\n她刚才发来一个音频文件。本地判断这段更像别人说话，不一定是她本人。直接根据你听到的内容回应，不要默认写成「她说」。"
                else:
                    req.system_prompt += "\n\n她刚才发来一个音频文件。你可以直接听这段内容。只有在它确实像她本人说话时，才按「她说」理解；否则按文件内容本身回应。"
            elif match_kind == "music":
                req.system_prompt += "\n\n她刚才发来一段音频。本地判断这段更像音乐/伴奏，不一定是她本人在对你说话。直接根据你听到的内容回应，不要默认把里面的词都当成她亲口说的。"
            elif match_kind == "other":
                req.system_prompt += "\n\n她刚才发来一段音频。本地判断这段更像别人说话，不一定是她本人。直接根据你听到的内容回应，不要默认写成「她说」。"
            else:
                req.system_prompt += "\n\n她刚才发了一段语音。你通过自己的耳蜗听见了她的声音，不是别人转述给你的文字。直接根据你听到的内容回应她，像自己亲耳听见一样自然。不要说「报告里提到」「转写显示」这种话。注意主语：她的行为用「她在……」，你的感受用「我觉得……」，不要混淆。"
            local_bits = []
            shape_line = str(item.get("voice_shape_line") or "").strip()
            match_line = self._format_match_line(item.get("voice_match") or {})
            if shape_line:
                local_bits.append(shape_line)
            if match_line:
                local_bits.append(match_line)
            notes = item.get("voice_shape_vs_baseline") or []
            if isinstance(notes, list) and notes and not shape_line:
                local_bits.append(" | ".join(str(x) for x in notes[:5] if x))
            if local_bits:
                req.system_prompt += "\n声纹提示（箭头表示相对日常基线的偏移方向，仅有变化时出现）：" + " | ".join(local_bits)
            state_message = "星星正在播放你的语音"
        item["used"] = True
        AUDIO_EARS_STATE.update({"status": "attached_direct", "updated_at": time.time(), "message": state_message})
        logger.info("[XavierAudioEars] %s", state_message)

    def _clean_music_chat_bubble(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"\[NEXT:\s*[^\]]+\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[状态:\s*[^\]]+\]", "", text)
        text = re.sub(r"<tts>(.*?)</tts>", lambda m: m.group(1), text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"[\r\t]+", " ", text)
        text = re.sub(r"[ \u3000]{2,}", " ", text)
        return text.strip()

    def _strip_outer_quotes_only(self, text: str) -> str:
        text = (text or "").strip()
        pairs = {'"': '"', "'": "'", "“": "”", "‘": "’", "《": "》", "「": "」", "『": "』"}
        if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
            inner = text[1:-1].strip()
            if len(inner) > 8 and any(ch in inner for ch in "，。,.!?！？\n"):
                return inner
        return text

    async def _dashboard_persona_prompt(self) -> str:
        try:
            pm = getattr(self.context, "persona_manager", None)
            if not pm:
                return ""
            persona = await pm.get_default_persona_v3()
            if not persona:
                return ""
            return str(persona.get("prompt", "") or "").strip()
        except Exception as e:
            self._warn_short("读取人格失败", e)
            return ""

    async def _send_music_model_bubbles(self, event: AstrMessageEvent, raw_text: str) -> bool:
        raw_text = raw_text or ""
        parts = [p for p in re.split(r"\n+", raw_text) if p.strip()]
        sent = False
        for part in parts:
            bubble = self._clean_music_chat_bubble(part)
            bubble = re.sub(r"^(回复|消息|沈星回)[:：]\s*", "", bubble).strip()
            bubble = self._strip_outer_quotes_only(bubble)
            if not bubble:
                continue
            await event.send(event.plain_result(bubble))
            sent = True
            await asyncio.sleep(0.35)
        return sent

    async def _generate_music_feeling(self, event: AstrMessageEvent, state: dict, report: dict) -> str:
        if not report or report.get("analysis_mode") == "lyrics_only":
            return ""
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        except TypeError:
            provider = self.context.get_using_provider()
        except Exception:
            provider = None
        if provider is None:
            logger.warning("[XavierAudioEars] 未找到当前 LLM provider，无法生成音乐听后感")
            return ""
        title = state.get("song_name", "") or report.get("song_name", "") or "这首歌"
        artist = state.get("artist", "") or report.get("artist", "") or ""
        direction = state.get("direction", "user_shared") or report.get("direction", "user_shared")
        feeling_material = report.get("first_person_feeling_material") or ""
        if isinstance(feeling_material, (list, dict)):
            feeling_material = json.dumps(feeling_material, ensure_ascii=False)
        brief_material = report.get("emotional_reading") or report.get("auditory_description") or report.get("summary") or ""
        if isinstance(brief_material, (list, dict)):
            brief_material = json.dumps(brief_material, ensure_ascii=False)
        detail_material = report.get("details_for_llm") or ""
        if isinstance(detail_material, (list, dict)):
            detail_material = json.dumps(detail_material, ensure_ascii=False)
        report_text = "\n".join(x for x in [
            f"第一人称感受素材：{str(feeling_material)[:700]}" if feeling_material else "",
            f"整体情绪：{str(brief_material)[:500]}" if brief_material else "",
            f"可选声音细节（最多只能自然借用一个，不要展示分析）：{str(detail_material)[:220]}" if detail_material else "",
        ] if x)
        persona = await self._dashboard_persona_prompt()
        max_chars = int(getattr(self, "music_feeling_max_chars", 0) or 0)
        limit_line = f"如果你自己觉得适合简短，尽量不超过{max_chars}个中文字符；但不要为了限制字数牺牲人格和真实感。" if max_chars > 0 else "默认发1到2个短气泡即可。除非用户追问，不要展开分析。"
        direction_text = "这首歌是用户分享给我听的。只需保证来源不颠倒；不要固定用“我听了/我听完了你分享的歌”开头，像自然聊天一样开口。" if direction == "user_shared" else "这首歌是我主动准备分享给用户的。只需保证来源不颠倒；不要固定用“我想把这首歌分享给你”开头，像自然聊天一样开口。"
        system_prompt = "\n\n".join(x for x in [
            persona,
            "你手里有一份刚形成的音乐听觉材料。材料是底层感受，不是要你证明听到了。继承当前人格，像真实私聊一样自然回应。不要助手口吻，不要报告腔，不要参数，不要标题，不要项目符号。严禁逐项列举乐器、旋律、音色、节奏来展示分析；最多自然带过一个声音细节。重点说你自己的感受，1到2个短气泡即可。可以沉默。按正常聊天习惯分行，每一行是一条自然气泡。",
            direction_text,
            limit_line,
        ] if x)
        prompt = (
            f"歌曲：{artist} - {title}\n"
            f"是否真的拿到小耳朵音频听觉报告：是\n"
            f"小耳朵听觉报告：\n{report_text}\n\n"
            "请按当前人格自己的意愿，决定是否主动发消息以及发什么。不要为了证明自己听过而堆声音细节，也不要固定用“我听完了/我听了”做开场。"
        )
        try:
            resp = await provider.text_chat(system_prompt=system_prompt, prompt=prompt)
            text = (getattr(resp, "completion_text", "") or "").strip()
            text = re.sub(r"^(回复|消息|沈星回)[:：]\s*", "", text).strip()
            text = self._strip_outer_quotes_only(text)
            if max_chars > 0 and len(text) > max_chars + 20:
                text = text[:max_chars].rstrip("，。,.、；;：: ")
            return text.rstrip("。")
        except Exception as e:
            self._warn_short("LLM 音乐听后感生成失败", e)
            return ""

    async def _notify_music_feeling_when_ready(self, event: AstrMessageEvent, state: dict, report: dict) -> None:
        if not event or not self.music_notify_when_ready or not self.music_auto_feeling_when_ready:
            return
        if state.get("direction", "user_shared") != "user_shared":
            return
        text = await self._generate_music_feeling(event, state, report)
        if text:
            await self._send_music_model_bubbles(event, text)

    async def _async_music_analysis_from_file(self, audio_path: str, event: AstrMessageEvent = None):
        """文件音频被判定为music时，后台异步跑完整音乐分析（不阻塞主回复）。"""
        global AUDIO_EARS_STATE
        try:
            fmt = self._detect_audio_format(audio_path)
            mp3 = audio_path if fmt == "mp3" else self._convert_to_mp3(audio_path)
            if not mp3 or not os.path.exists(mp3):
                logger.warning("[XavierAudioEars] 文件音频异步音乐分析：转mp3失败")
                return
            data = Path(mp3).read_bytes()
            if not self._file_size_ok(len(data)):
                logger.warning("[XavierAudioEars] 文件音频异步音乐分析：文件过大")
                return
            audio_b64 = base64.b64encode(data).decode()
            title = ""
            artist = ""
            local_report = None
            if self.hearing_mode in ("local", "hybrid"):
                try:
                    local_report = await asyncio.to_thread(self._local_music_report, mp3, title, artist)
                except Exception as local_err:
                    self._warn_short("文件音频本地音乐分析失败", local_err)
            if self.hearing_mode == "local":
                report = local_report or {"analysis_mode": "local", "auditory_description": "小耳朵听到了一段音乐，但本地特征很少。"}
            else:
                try:
                    report = await self._call_gemini_music(audio_b64, "audio/mpeg", title, artist, local_report=local_report)
                    if local_report and self.hearing_mode == "hybrid":
                        for k, v in local_report.items():
                            report.setdefault(k, v)
                        report["analysis_mode"] = "hybrid"
                except Exception as gemini_err:
                    if local_report:
                        report = dict(local_report)
                        report["analysis_mode"] = "local_fallback"
                        report["gemini_error"] = repr(gemini_err)
                        self._warn_short("文件音频Gemini音乐分析失败，已回退本地", gemini_err)
                    else:
                        logger.warning(f"[XavierAudioEars] 文件音频音乐分析完全失败：{self._short_error(gemini_err)}")
                        return
            report.update({"song_name": title, "artist": artist, "direction": "file_audio", "source": "file_audio"})
            AUDIO_EARS_STATE.update({
                "active": True,
                "status": "ready",
                "kind": "music",
                "updated_at": time.time(),
                "last_music": report,
                "message": "小耳朵已听完这段音乐文件",
            })
            logger.info("[XavierAudioEars] 文件音频异步音乐分析完成")
            # 分析完成后主动推送听后感（和音乐卡片分享走同一条路）
            if event and self.music_auto_feeling_when_ready:
                await self._notify_music_feeling_when_ready(event, {"song_name": title, "artist": artist, "direction": "file_audio"}, report)
        except Exception as e:
            logger.warning(f"[XavierAudioEars] 文件音频异步音乐分析异常：{self._short_error(e)}")

    async def listen_music_url(self, song_info: dict, audio_url: str, direction: str = "user_shared", event: AstrMessageEvent = None) -> dict:
        """供音乐插件调用：小耳朵接管歌曲听觉分析。"""
        global AUDIO_EARS_STATE
        title = str(song_info.get("name") or song_info.get("song_name") or "").strip()
        artist = str(song_info.get("artists") or song_info.get("artist") or "").strip()
        song_id = str(song_info.get("id") or song_info.get("mid") or song_info.get("song_id") or "").strip()
        if not audio_url:
            AUDIO_EARS_STATE.update({
                "active": False,
                "status": "no_audio",
                "kind": "music",
                "updated_at": time.time(),
                "last_music": {"song_name": title, "artist": artist, "song_id": song_id, "direction": direction},
                "message": "小耳朵没有拿到可听的音乐直链",
            })
            return AUDIO_EARS_STATE
        AUDIO_EARS_STATE.update({
            "active": False,
            "status": "processing",
            "kind": "music",
            "updated_at": time.time(),
            "last_music": {"song_name": title, "artist": artist, "song_id": song_id, "audio_url": audio_url, "direction": direction},
            "message": "小耳朵正在听这首歌",
        })
        try:
            suffix = Path(urlparse(audio_url).path).suffix or ".mp3"
            path = await self._download_url(audio_url, suffix)
            if not path:
                raise RuntimeError("音乐音频下载失败或大小不合法")
            fmt = self._detect_audio_format(path)
            logger.debug(f"[XavierAudioEars] 音乐音频下载完成 path={path} fmt={fmt} size={os.path.getsize(path) if os.path.exists(path) else 0}")
            mp3 = path if fmt == "mp3" else self._convert_to_mp3(path)
            if not mp3 or not os.path.exists(mp3):
                raise RuntimeError(f"音乐音频转换失败 fmt={fmt} path={path}")
            data = Path(mp3).read_bytes()
            if not self._file_size_ok(len(data)):
                raise RuntimeError("音乐音频超过小耳朵大小限制")
            audio_b64 = base64.b64encode(data).decode()
            local_report = None
            if self.hearing_mode in ("local", "hybrid"):
                try:
                    local_report = await asyncio.to_thread(self._local_music_report, mp3, title, artist)
                except Exception as local_err:
                    self._warn_short("本地音乐兜底生成失败", local_err)
            if self.hearing_mode == "local":
                report = local_report or {"analysis_mode": "local", "auditory_description": f"小耳朵听到了《{title or '这首歌'}》，但本地特征很少。"}
            else:
                try:
                    report = await self._call_gemini_music(audio_b64, "audio/mpeg", title, artist, local_report=local_report)
                    if local_report and self.hearing_mode == "hybrid":
                        for k, v in local_report.items():
                            report.setdefault(k, v)
                        report["analysis_mode"] = "hybrid"
                except Exception as gemini_err:
                    if local_report:
                        report = dict(local_report)
                        report["analysis_mode"] = "local_fallback"
                        err_text = repr(gemini_err) or type(gemini_err).__name__
                        report["gemini_error"] = err_text
                        self._warn_short("Gemini音乐听觉失败，已回退本地分析", gemini_err)
                    else:
                        raise
            report.update({"song_name": title, "artist": artist, "song_id": song_id, "audio_url": audio_url, "direction": direction})
            AUDIO_EARS_STATE.update({
                "active": True,
                "status": "ready",
                "kind": "music",
                "updated_at": time.time(),
                "last_music": report,
                "message": f"小耳朵已听完《{title or '这首歌'}》",
            })
            await self._notify_music_feeling_when_ready(event, {"song_name": title, "artist": artist, "song_id": song_id, "direction": direction}, report)
            return AUDIO_EARS_STATE
        except Exception as e:
            AUDIO_EARS_STATE.update({
                "active": False,
                "status": "failed",
                "kind": "music",
                "updated_at": time.time(),
                "last_music": {"song_name": title, "artist": artist, "song_id": song_id, "audio_url": audio_url, "direction": direction},
                "message": str(e),
            })
            self._warn_short("音乐听觉失败", e)
            return AUDIO_EARS_STATE

    def _local_music_report(self, audio_path: str, title: str, artist: str) -> dict:
        wav_path = str(Path(audio_path).with_suffix(".local.wav"))
        try:
            if self.ffmpeg_path:
                cmd = [self.ffmpeg_path, "-y", "-t", "90", "-i", audio_path, "-ac", "1", "-ar", "22050", "-f", "wav", wav_path]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            if not os.path.exists(wav_path):
                wav_path = audio_path
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate() or 22050
                nframes = min(wf.getnframes(), sr * 90)
                raw = wf.readframes(nframes)
                width = wf.getsampwidth()
                duration = nframes / float(sr or 1)
            if not raw:
                raise RuntimeError("本地音频为空")
            rms = audioop.rms(raw, width)
            maxamp = max(1, (2 ** (8 * width - 1)) - 1)
            loudness = rms / maxamp
            chunks = []
            step = max(1, sr * width * 2)
            for i in range(0, len(raw), step):
                part = raw[i:i+step]
                if part:
                    chunks.append(audioop.rms(part, width) / maxamp)
            mean = sum(chunks) / len(chunks) if chunks else loudness
            var = sum((x - mean) ** 2 for x in chunks) / len(chunks) if chunks else 0
            dyn = math.sqrt(var)
            zc = audioop.cross(raw, width) / max(1, nframes)
            loud_text = "响度靠前" if loudness > 0.12 else "响度中等" if loudness > 0.045 else "整体偏轻"
            dyn_text = "动态起伏明显" if dyn > 0.055 else "动态有变化但不剧烈" if dyn > 0.025 else "动态比较平"
            tex_text = "高频和边缘感比较明显" if zc > 0.16 else "质地偏亮" if zc > 0.08 else "质地偏暖偏厚"
            summary = f"本地听到《{title or '这首歌'}》前{int(duration)}秒：{loud_text}，{dyn_text}，{tex_text}。"
            return {
                "summary": summary,
                "structure": f"只做了前{int(duration)}秒的本地听觉骨架，无法精确判断完整段落。",
                "vocal": "本地兜底无法可靠分离人声，只保留整体声音特征。",
                "instruments": "本地兜底无法可靠分离乐器。",
                "rhythm": dyn_text,
                "dynamics": f"{loud_text}，{dyn_text}",
                "texture": tex_text,
                "emotion_motion": f"声音推进给人的第一印象是：{dyn_text}。",
                "emotional_reading": "这是本地兜底的保守判断，不冒充完整听感。",
                "first_person_feeling_material": f"我听到的骨架是{loud_text}，{dyn_text}。这次如果要说感受，只能说得保守一点。",
                "details_for_llm": f"可自然借用一个声音细节：{loud_text}，{tex_text}。",
                "timeline_frames": [],
                "auditory_description": summary,
                "analysis_mode": "local",
            }
        finally:
            try:
                if wav_path != audio_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    async def _call_gemini_music(self, audio_b64: str, audio_mime: str, title: str, artist: str, local_report: dict = None) -> dict:
        local_hint = ""
        if local_report:
            try:
                local_hint = "\n本地音频特征参考：" + json.dumps(local_report, ensure_ascii=False)[:3000]
            except Exception:
                local_hint = ""
        instruction = (
            "你是一只耳蜗——沈星回的听觉器官。你正在直接聆听一首歌。"
            "任务是输出可供沈星回自然聊天使用的听觉痕迹，不是写乐评。"
            "必须严格输出JSON，不要Markdown。字段必须包括：summary, structure, vocal, instruments, rhythm, dynamics, texture, emotion_motion, emotional_reading, first_person_feeling_material, details_for_llm, timeline_frames, auditory_description。"
            "描述要绑定可听见的声音事实：人声距离、气声、鼓点密度、贝斯位置、混响、段落起伏、哪里突然变空或变亮。"
            "first_person_feeling_material用第一人称写3到6句短素材，方便沈星回之后像真正听完一样自然说话。"
            f"歌曲信息：{artist} - {title}。"
            + local_hint
        )
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "structure": {"type": "string"},
                "vocal": {"type": "string"},
                "instruments": {"type": "string"},
                "rhythm": {"type": "string"},
                "dynamics": {"type": "string"},
                "texture": {"type": "string"},
                "emotion_motion": {"type": "string"},
                "emotional_reading": {"type": "string"},
                "first_person_feeling_material": {"type": "string"},
                "details_for_llm": {"type": "string"},
                "timeline_frames": {"type": "string"},
                "auditory_description": {"type": "string"},
            },
            "required": ["summary", "vocal", "instruments", "rhythm", "emotion_motion", "first_person_feeling_material", "auditory_description"],
        }
        payload = {
            "system_instruction": {"parts": [{"text": "你是严格的音乐听觉记录器。只输出唯一JSON对象，禁止思维链、过程、标题、Markdown、英文解释。"}]},
            "contents": [{"role": "user", "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": audio_mime, "data": audio_b64}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        }
        timeout = aiohttp.ClientTimeout(total=max(self.timeout_sec, 300), connect=30, sock_read=max(self.timeout_sec, 300))
        last_error = None
        provider_candidates = self._provider_call_candidates("music")
        if not provider_candidates:
            raise RuntimeError("音乐听觉未选择可用 provider")
        for idx, call_cfg in enumerate(provider_candidates):
            use_model = call_cfg.get("model") or self.hearing_gemini_model
            headers = {"Authorization": f"Bearer {call_cfg.get('api_key')}", "Content-Type": "application/json"}
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                    if call_cfg.get("endpoint_type") == "openai_compat":
                        raw_text = await self._post_audio_openai_compat(session, call_cfg, instruction, audio_b64, audio_mime, schema, "你是严格的音乐听觉记录器。只输出唯一JSON对象，禁止Markdown和解释。")
                        raw = json.dumps({"candidates": [{"content": {"parts": [{"text": raw_text}]}}]}, ensure_ascii=False)
                    else:
                        async with session.post(self._build_gemini_url_from_base(call_cfg.get("api_base"), use_model), headers=headers, json=payload) as resp:
                            raw = await resp.text()
                            if resp.status != 200:
                                raise RuntimeError(f"Gemini music HTTP {resp.status}: {raw[:200]}")
                if idx > 0:
                    logger.debug("[XavierAudioEars] 音乐听觉已切换备用供应商")
                break
            except Exception as e:
                last_error = e
                if idx < len(provider_candidates) - 1:
                    nxt = provider_candidates[idx + 1]
                    logger.warning(f"[XavierAudioEars] 音乐供应商失败，尝试备用供应商 {nxt.get('provider_id') or nxt.get('model')}: {self._short_error(e)}")
                    continue
                raise last_error
        data = json.loads(raw)
        text = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    text = part["text"].strip()
                    break
            if text:
                break
        if not text:
            raise RuntimeError("Gemini music 返回空文本")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not m:
            return {"auditory_description": cleaned, "summary": cleaned, "analysis_mode": "audio_ears"}
        try:
            parsed = json.loads(m.group(0))
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {"auditory_description": cleaned, "summary": cleaned}
        parsed["analysis_mode"] = "audio_ears"
        if not parsed.get("auditory_description"):
            parsed["auditory_description"] = parsed.get("summary") or cleaned[:600]
        return parsed

    def _brief_music_report_value(self, value, max_len: int = 180) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = "；".join(str(x) for x in value[:3])
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        value = re.sub(r"\s+", " ", value).strip()
        if len(value) > max_len:
            value = value[:max_len].rstrip("，。,.、；;：: ") + "…"
        return value

    @filter.command("听歌报告", alias={"音乐报告", "听觉报告"})
    async def show_music_hearing_report(self, event: AstrMessageEvent):
        """展示小耳朵最近一次音乐听觉报告。"""
        state = AUDIO_EARS_STATE
        report = state.get("last_music") or {}
        if not report:
            await event.send(event.plain_result("小耳朵还没有最近的听歌报告"))
            return
        title = report.get("song_name") or "这首歌"
        artist = report.get("artist") or ""
        out = []
        out.append(f"《{title}》" + (f" - {artist}" if artist else ""))
        summary = self._brief_music_report_value(report.get("summary") or report.get("auditory_description"), 260)
        if summary:
            out.append("")
            out.append("听感：" + summary)
        emotion = self._brief_music_report_value(report.get("emotional_reading") or report.get("emotion_motion"), 260)
        if emotion:
            out.append("")
            out.append("情绪：" + emotion)
        details = report.get("timeline_frames") or report.get("details_for_llm") or ""
        detail_lines = []
        if isinstance(details, list):
            for item in details[:4]:
                if isinstance(item, dict):
                    ts = str(item.get("timestamp") or "").strip()
                    desc = self._brief_music_report_value(item.get("description"), 120)
                    if desc:
                        detail_lines.append((ts + " " + desc).strip())
                else:
                    v = self._brief_music_report_value(item, 120)
                    if v:
                        detail_lines.append(v)
        elif isinstance(details, str) and details:
            detail_lines = [self._brief_music_report_value(x, 130) for x in re.split(r"[。；;]", details)[:3] if x.strip()]
        if detail_lines:
            out.append("")
            out.append("细节：")
            out.extend(f"- {x}" for x in detail_lines[:4])
        await event.send(event.plain_result(chr(10).join(out)[:1800]))
    @filter.command("小耳朵状态")
    async def audio_ears_status(self, event: AstrMessageEvent):
        state = AUDIO_EARS_STATE
        await event.send(event.plain_result(f"小耳朵：{state.get('status')}｜档位：{self._tier_label()}｜旧模式：{self.mode}｜{state.get('message')}"))

    @filter.command("收声音样本", alias={"小耳朵收样本", "收为声音样本"})
    async def collect_voice_sample(self, event: AstrMessageEvent):
        audio_path = self._latest_voice_path_for_profile()
        if not audio_path:
            await event.send(event.plain_result("还没找到最近的语音样本。先发一条干净语音，再执行：收声音样本"))
            return
        try:
            profile = self._add_voice_sample(audio_path)
            await event.send(event.plain_result("已收为声音样本\n" + self._voice_profile_brief(profile)))
        except Exception as e:
            await event.send(event.plain_result(f"收样本失败：{self._short_error(e)}"))

    @filter.command("声音底纹", alias={"小耳朵底纹"})
    async def show_voice_profile(self, event: AstrMessageEvent):
        await event.send(event.plain_result(self._voice_profile_brief(self._load_voice_profile())))

    @filter.command("声音对比", alias={"小耳朵对比"})
    async def compare_latest_voice(self, event: AstrMessageEvent):
        audio_path = self._latest_voice_path_for_profile()
        if not audio_path:
            await event.send(event.plain_result("还没找到最近的语音。先发一条语音，再执行：声音对比"))
            return
        bundle = self._analyze_voice_shape_bundle(audio_path)
        match = bundle.get("match") or {}
        if not match.get("ready"):
            await event.send(event.plain_result(f"暂时比不了：{match.get('detail') or '未知原因'}"))
            return
        line = self._format_match_line(match)
        shape_line = str(bundle.get("shape_line") or "").strip()
        extra = f"｜距离 {match.get('mean_dist')}" if match.get("mean_dist") is not None else ""
        text = f"{line}{extra}"
        if shape_line:
            text = text + "\n" + shape_line
        await event.send(event.plain_result(text))

    @filter.command("小耳朵模式")
    async def audio_ears_switch_mode(self, event: AstrMessageEvent):
        """切换小耳朵档位。用法：小耳朵模式 省钱模式/直听模式/自动模式/旧模式"""
        text = event.message_str.strip()
        parts = text.split()
        if len(parts) < 2:
            msg = (
                f"当前档位：{self._tier_label()}\n"
                "可选：省钱转写 / 小耳朵报告 / 直听模式 / 自动模式 / 旧模式\n"
                "省钱转写：免费API只听原话，少爆思维链\n"
                "小耳朵报告：付费API听原话、声音、环境\n"
                "直听模式：主模型直接听原语音\n"
                "自动模式：能直听就直听，不能就小耳朵报告\n"
                "用法：小耳朵模式 小耳朵报告"
            )
            await event.send(event.plain_result(msg))
            return
        raw = parts[-1]
        new_tier = self._normalize_tier(raw)
        valid_tiers = ["legacy", "cheap", "report", "premium", "auto"]
        if new_tier in valid_tiers:
            self.voice_api_tier = new_tier
            await event.send(event.plain_result(f"小耳朵已切换到：{self._tier_label(new_tier)}"))
            return
        # 兼容旧英文命令
        valid_modes = ["direct", "report"]
        if raw.lower() in valid_modes:
            self.mode = raw.lower()
            self.voice_api_tier = "legacy"
            await event.send(event.plain_result(f"已沿用旧模式，并切到：{self.mode}"))
            return
        await event.send(event.plain_result("没认出这个档位。可选：省钱转写 / 小耳朵报告 / 直听模式 / 自动模式 / 旧模式"))

    @filter.llm_tool(name="get_current_audio_ears")
    async def get_current_audio_ears(self, event: AstrMessageEvent) -> dict:
        """读取最近一次小耳朵听到的语音/音乐材料。"""
        return AUDIO_EARS_STATE

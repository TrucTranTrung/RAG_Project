import uuid
import torch
import soundfile as sf
import numpy as np
import io
from models import LFinference
from fastapi import FastAPI, Form, Response, Request
from fastapi.responses import JSONResponse
import time
import os
import logging
import socket
import base64

# --- imports relevant to tracing/logging ---
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ----- Metrics (Prometheus) -----
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    ProcessCollector,
    PLATFORM_COLLECTOR
)

# ---------------- Logging setup (JSON) ----------------
try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None  # fallback to plain logging if not installed

# read config from env (falls back to defaults)
JAEGER_HOST = os.getenv("JAEGER_AGENT_HOST", "jaeger")
JAEGER_PORT = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
SERVICE_NAME_STR = os.getenv("OTEL_SERVICE_NAME", "tts-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "v1.0")
ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() in ("1", "true", "yes")

logger = logging.getLogger(SERVICE_NAME_STR)
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    handler = logging.StreamHandler()
    if jsonlogger:
        fmt = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(fmt)
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Logging enrichment: include trace_id in logs
def enrich_log(record):
    try:
        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        trace_id = format(ctx.trace_id, '032x') if ctx and ctx.trace_id else None
    except Exception:
        trace_id = None
    record.trace_id = trace_id
    record.service = SERVICE_NAME_STR
    record.service_version = SERVICE_VERSION
    record.env = ENV
    record.host = socket.gethostname()
    return True

class EnrichFilter(logging.Filter):
    def filter(self, record):
        enrich_log(record)
        return True

logger.addFilter(EnrichFilter())

# --- Init tracing using env values ---
set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE_NAME_STR}))
)
tracer = get_tracer_provider().get_tracer("TextToSpeech", "0.1.1")

jaeger_exporter = JaegerExporter(
    agent_host_name=JAEGER_HOST,
    agent_port=JAEGER_PORT,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
get_tracer_provider().add_span_processor(span_processor)

# ---------- Init prometheus metrics (improved names + labels) ----------
REGISTRY = CollectorRegistry()

# register process/platform collectors when available
if ProcessCollector is not None:
    try:
        ProcessCollector(registry=REGISTRY)
    except Exception:
        pass
if PLATFORM_COLLECTOR is not None:
    try:
        PLATFORM_COLLECTOR(registry=REGISTRY)
    except Exception:
        pass


REQUEST_COUNTER = Counter(
    "tts_requests_total", 
    "Total number of TTS requests", 
    ["service", "route", "status"], 
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    "tts_request_duration_seconds", 
    "Request duration seconds", 
    ["service", "route"], 
    registry=REGISTRY, 
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10)
)

# ---------- App & Device ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = FastAPI()

# instrument app for http spans and requests propagation
try:
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
except Exception:
    # if instrumentation fails, continue — manual spans still work
    logger.warning("OTel instrumentation failed to initialize (continuing without auto-instrument)")

# Initialize pynvml if available
# if PYNVML_AVAILABLE:
#     try:
#         pynvml.nvmlInit()
#         GPU_COUNT = pynvml.nvmlDeviceGetCount()
#     except Exception:
#         PYNVML_AVAILABLE = False
#         GPU_COUNT = 0
# else:
#     GPU_COUNT = 0

# Background task to update GPU gauges periodically
# async def _gpu_updater_task(interval_s: float = 5.0):
#     if not PYNVML_AVAILABLE:
#         return
#     while True:
#         try:
#             for i in range(GPU_COUNT):
#                 try:
#                     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#                     # utilization percent
#                     util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # int percent

#                     # memory info (bytes)
#                     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#                     mem_used = int(mem_info.used)
#                     mem_total = int(mem_info.total)

#                     # set prometheus gauges (use string label for gpu_index)
#                     GPU_UTIL_GAUGE.labels(service=SERVICE_NAME_STR, gpu_index=str(i)).set(float(util))
#                     GPU_MEM_USED_GAUGE.labels(service=SERVICE_NAME_STR, gpu_index=str(i)).set(mem_used)
#                     GPU_MEM_TOTAL_GAUGE.labels(gpu_index=str(i)).set(mem_total)
#                     mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
#                     GPU_MEM_UTIL_PERCENT.labels(service=SERVICE_NAME_STR, gpu_index=str(i)).set(mem_percent)

#                 except Exception:
#                     # don't crash updater for a single GPU error
#                     logger.debug(f"gpu metric update failed for index {i}", exc_info=True)
#         except Exception:
#             # ignore update errors
#             pass
#         await asyncio.sleep(interval_s)

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    # Tạo một request_id riêng cho mỗi request
    request_id = str(uuid.uuid4())
    start = time.time()

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("request_id", request_id)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.path", request.url.path)

    response = await call_next(request)

    if span and span.is_recording():
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("process_time_ms", (time.time() - start) * 1000)

    return response

# @app.on_event("startup")
# async def startup_event():
#     # start background gpu update (non-blocking)
#     if PYNVML_AVAILABLE and GPU_COUNT > 0:
#         asyncio.create_task(_gpu_updater_task(5.0))
#     logger.info("service_startup", extra={"service": SERVICE_NAME_STR, "version": SERVICE_VERSION, "device": device})

# Expose /metrics for Prometheus
@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)



@app.post("/transcribe")
async def transcribe_audio(text_input: str = Form(...)):
    request_id = str(uuid.uuid4())
    REQUEST_START = time.time()
    route = "/transcribe"
    try:
        sentences = text_input.split('.')
        wavs = []
        s_prev = None

        logger.info("transcribe_received", extra={
            "request_id": request_id,
            "input_length": len(text_input),
            "segments": len([s for s in sentences if s.strip() != ""])
        })

        with tracer.start_as_current_span("transcribe_request") as req_span:
            # thêm một số tag hữu dụng lên span request
            req_span.set_attribute("request_id", request_id)
            req_span.set_attribute("http.route", route)
            req_span.set_attribute("input_length", len(text_input))
            req_span.set_attribute("device", device)
            req_span.set_attribute("service.version", SERVICE_VERSION)

            valid_sentences = [s for s in sentences if s.strip() != ""]
            req_span.set_attribute("segments_count", len(valid_sentences))

            for idx, text in enumerate(valid_sentences):
                text = text + '.'
                t0 = time.time()


                # add span for this particular inference segment
                with tracer.start_as_current_span(f"inference_segment.{idx}") as seg_span:
                    seg_span.set_attribute("segment_index", idx)
                    seg_span.set_attribute("segment_length", len(text))
                    seg_span.set_attribute("model", "LFinference")
                    # thêm event để hiện trong phần Logs/Events
                    seg_span.add_event("segment.inference.start", {"index": idx})

                    # inference
                    noise = torch.randn(1,1,256).to(device)
                    wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
                    wavs.append(wav)

                    seg_span.add_event("segment.inference.end", {"index": idx, "duration_s": time.time() - t0})
                    total = time.time() - REQUEST_START
                    REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route="/transcribe").observe(total)

            REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route="/transcribe", status="200").inc()
            audio = np.concatenate(wavs) if wavs else np.array([], dtype=np.float32)

            # Lưu vào buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=24000, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            # print("Generated audio length (samples): ", len(audio_base64))
            logger.info("transcribe_completed", extra={"request_id": request_id, "duration_s": total})
            # logger.info("Audio debug",
            #     extra={
            #         "request_id": request_id,
            #         "wavs_count": len(wavs),
            #         "samples_len": len(audio),
            #         "wav_bytes_len": len(wav_bytes),
            #         "first_12_bytes": wav_bytes[:12]
            #     }
            # )

            return JSONResponse(
                content={
                    "request_id": request_id,
                    "audio_base64": audio_base64
                }
            )

    except Exception as e:
        total = time.time() - REQUEST_START
        REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route="/transcribe").observe(total)
        REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route="/transcribe", status="500").inc()

        # record exception in current span and set status
        span = trace.get_current_span()
        if span is not None:
            span.record_exception(e)
            try:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception:
                span.set_attribute("error", True)

        logger.exception("transcribe_failed", extra={"request_id": request_id, "error": str(e)})
        return JSONResponse(status_code=500, content={"error": str(e)})


import uuid
import torch
import soundfile as sf
import numpy as np
import base64
import io
from models import LFinference
from fastapi import FastAPI, Form, Response
from fastapi.responses import JSONResponse
import time
import os
import logging
import socket
# ---------------- Tracing (OpenTelemetry -> Jaeger) ----------------
from opentelemetry import trace
from opentelemetry.trace import get_current_span
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# ---------------- Logging setup (JSON) ----------------
try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None  # fallback to plain logging if not installed

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

def enrich_log(record):
    # add common fields to record
    try:
        span = get_current_span()
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


# ----- Metrics (Prometheus) -----
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry

# ----- Optional GPU metrics -----
try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# ---------- Init tracing ----------
service_name = "Text-To-Speech-Service"
trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
)
jaeger_agent_host =  "jaeger" 
jaeger_agent_port = 6831
jaeger_exporter = JaegerExporter(agent_host_name=jaeger_agent_host, agent_port=jaeger_agent_port)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

# ---------- Init prometheus metrics ----------
REGISTRY = CollectorRegistry() 
REQUEST_COUNTER = Counter("Text-To-Speech_total", "Total /transcribe requests", ["status"], registry=REGISTRY)
REQUEST_LATENCY = Histogram("Text-To-Speech_duration_seconds", "Request latency seconds", registry=REGISTRY)
INFER_GCUM_DURATION = Histogram("Text-To-Speech_inference_duration_seconds", "Inference duration per segment", registry=REGISTRY)
GPU_UTIL_GAUGE = Gauge("Text-To-Speech_gpu_utilization_percent", "GPU utilization percent", ["gpu_index"], registry=REGISTRY)
GPU_MEM_USED_GAUGE = Gauge("Text-To-Speech_gpu_memory_used_bytes", "GPU memory used bytes", ["gpu_index"], registry=REGISTRY)


# ---------- App & Device ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = FastAPI()

# Initialize pynvml if available
if PYNVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        GPU_COUNT = pynvml.nvmlDeviceGetCount()
    except Exception:
        PYNVML_AVAILABLE = False
        GPU_COUNT = 0
else:
    GPU_COUNT = 0


# Background task to update GPU gauges periodically
import asyncio
async def _gpu_updater_task(interval_s: float = 5.0):
    if not PYNVML_AVAILABLE:
        return
    while True:
        try:
            for i in range(GPU_COUNT):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # percent
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                GPU_UTIL_GAUGE.labels(gpu_index=str(i)).set(float(util))
                GPU_MEM_USED_GAUGE.labels(gpu_index=str(i)).set(int(mem_info.used))
        except Exception:
            # ignore update errors
            pass
        await asyncio.sleep(interval_s)


@app.on_event("startup")
async def startup_event():
    # start background gpu update (non-blocking)
    if PYNVML_AVAILABLE and GPU_COUNT > 0:
        asyncio.create_task(_gpu_updater_task(5.0))
    logger.info("service_startup", extra={"service": SERVICE_NAME_STR, "version": SERVICE_VERSION, "device": device})

# Expose /metrics for Prometheus
@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/transcribe")
async def transcribe_audio(text_input: str = Form(...)):
    status = "ok"
    request_id = str(uuid.uuid4())
    REQUEST_START = time.time()
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
            req_span.set_attribute("input_length", len(text_input))
            for idx, text in enumerate(sentences):
                if text.strip() == "": continue
                text += '.' # add it back
                t0 = time.time()
                # add span for this particular inference segment
                with tracer.start_as_current_span("inference_segment") as seg_span:
                    seg_span.set_attribute("segment_index", idx)
                    seg_span.set_attribute("segment_length", len(text))
                    logger.info("segment_inference_start", extra={"request_id": request_id, "segment_index": idx, "segment_length": len(text)})
                    noise = torch.randn(1,1,256).to(device)
                    wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
                    wavs.append(wav)
                t1 = time.time()
                INFER_GCUM_DURATION.observe(t1 - t0)
                wavs.append(wav)
            audio = np.concatenate(wavs)
            
            # Lưu vào buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=24000, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            REQUEST_LATENCY.observe(time.time() - REQUEST_START)
            REQUEST_COUNTER.labels(status="200").inc()
            logger.info("transcribe_completed", extra={"request_id": request_id, "duration_s": time.time() - REQUEST_START})
            return JSONResponse(content={
                "output_sound": audio_base64
            })

    except Exception as e:
        REQUEST_LATENCY.observe(time.time() - REQUEST_START)
        REQUEST_COUNTER.labels(status="500").inc()
        # record exception in current span
        span = trace.get_current_span()
        if span is not None:
            span.record_exception(e)
            span.set_attribute("error", True)
        logger.exception("transcribe_failed", extra={"request_id": request_id, "error": str(e)})
        return JSONResponse(status_code=500, content={"error": str(e)})


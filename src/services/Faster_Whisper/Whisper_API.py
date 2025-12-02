import uuid
import torch
import tempfile
import shutil
import os
import time
import logging
import socket

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response

from faster_whisper import WhisperModel

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    ProcessCollector,
    PLATFORM_COLLECTOR
)

# ---------------- Environment ----------------
JAEGER_COLLECTOR_URL = os.getenv("JAEGER_COLLECTOR_URL", "http://jaeger:14268/api/traces")
SERVICE_NAME_STR = os.getenv("OTEL_SERVICE_NAME", "whisper-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "v1.0")
ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()  # DEBUG để log chi tiết
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() in ("1", "true", "yes")

# ---------------- Logging ----------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(SERVICE_NAME_STR)
logger.setLevel(LOG_LEVEL)

class EnrichFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context() if span else None
        record.trace_id = format(ctx.trace_id, '032x') if ctx and ctx.trace_id else None
        record.service = SERVICE_NAME_STR
        record.service_version = SERVICE_VERSION
        record.env = ENV
        record.host = socket.gethostname()
        return True

logger.addFilter(EnrichFilter())

# ---------------- Tracing ----------------
resource = Resource.create({
    "service.name": SERVICE_NAME_STR,
    "service.version": SERVICE_VERSION,
    "env": ENV
})

provider = TracerProvider(resource=resource)

# Jaeger exporter
jaeger_exporter = JaegerExporter(
    collector_endpoint=JAEGER_COLLECTOR_URL
)
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

# Console exporter (debug span ngay trên console)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer(SERVICE_NAME_STR, SERVICE_VERSION)

# ---------------- Prometheus ----------------
REGISTRY = CollectorRegistry()
try:
    ProcessCollector(registry=REGISTRY)
except Exception:
    pass
try:
    PLATFORM_COLLECTOR(registry=REGISTRY)
except Exception:
    pass

STT_REQUEST_COUNTER = Counter(
    "stt_requests_total",
    "Total number of STT requests",
    ["service", "route", "status"],
    registry=REGISTRY
)

STT_REQUEST_LATENCY = Histogram(
    "stt_request_duration_seconds",
    "Request duration seconds",
    ["service", "route"],
    registry=REGISTRY,
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10)
)

# ---------------- FastAPI ----------------
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "medium"
model = WhisperModel(model_size, device=device, compute_type="int8_float16")

# ---------------- Middleware tracing ----------------
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()

    with tracer.start_as_current_span("http_request") as span:
        span.set_attribute("request_id", request_id)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.path", request.url.path)

        response = await call_next(request)

        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("process_time_ms", (time.time() - start) * 1000)

        # Flush span ngay lập tức
        provider.force_flush()
        logger.debug(f"Span flushed for request_id: {request_id}")

        return response

# ---------------- Metrics endpoint ----------------
@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# ---------------- STT endpoint ----------------
@app.post("/STT/")
async def transcribe_audio(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    route = "/STT/"
    start_time = time.time()
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            shutil.copyfileobj(file.file, tmp_file)

        with tracer.start_as_current_span("stt_request") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("file_name", file.filename)
            span.set_attribute("file_suffix", suffix)
            span.set_attribute("device", device)
            span.set_attribute("model_size", model_size)

            segments, info = model.transcribe(temp_path, beam_size=5)
            output_text = " ".join([seg.text for seg in segments])

            logger.info(f"STT trace for request {request_id}")

            # flush span ngay lập tức
            provider.force_flush()
            logger.debug(f"Span flushed for STT request_id: {request_id}")

        duration = time.time() - start_time
        STT_REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route=route).observe(duration)
        STT_REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route=route, status="200").inc()

        logger.info(f"STT completed for request {request_id}, duration: {duration:.3f}s")
        return JSONResponse(content={
            "language": info.language,
            "language_probability": info.language_probability,
            "output_text": output_text.strip()
        })

    except Exception as e:
        duration = time.time() - start_time
        STT_REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route=route).observe(duration)
        STT_REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route=route, status="500").inc()

        span = trace.get_current_span()
        if span:
            span.record_exception(e)
            try:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception:
                span.set_attribute("error", True)
                
                
        logger.exception(f"STT failed for request {request_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
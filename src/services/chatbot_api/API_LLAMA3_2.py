import uuid
import os
import time
import base64
import logging
import socket
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Union
from db import query_similar_vectors_from_pgvector, get_pgvector_store
from model import get_entities_as_string_GEMINI
from utils import get_top_k_contexts
from prompt import prompt_template


# --- OpenTelemetry / Jaeger imports ---
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor


# --- Prometheus imports ---
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
    
# --- Config from env ---
JAEGER_HOST = os.getenv("JAEGER_AGENT_HOST", "jaeger")
JAEGER_PORT = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
SERVICE_NAME_STR = os.getenv("OTEL_SERVICE_NAME", "rag-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "v1.0")
ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() in ("1", "true", "yes")

# --- Logger setup (json if available) ---
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

# enrich logs with trace_id, service, host, etc.
from opentelemetry.trace import get_current_span

def enrich_log(record):
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

# --- Init tracing (Jaeger) ---
if ENABLE_TRACING:
    try:
        set_tracer_provider(
            TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE_NAME_STR}))
        )
        tracer = get_tracer_provider().get_tracer("RAGService", SERVICE_VERSION)

        jaeger_exporter = JaegerExporter(
            agent_host_name=JAEGER_HOST,
            agent_port=JAEGER_PORT,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        get_tracer_provider().add_span_processor(span_processor)
        logger.info("Tracing initialized (Jaeger)", extra={"jaeger_host": JAEGER_HOST, "jaeger_port": JAEGER_PORT})
    except Exception as e:
        tracer = trace.get_tracer(__name__)
        logger.warning("Failed to initialize tracing; continuing without auto-jaeger", extra={"error": str(e)})
else:
    tracer = trace.get_tracer(__name__)
    logger.info("Tracing disabled by env")

# --- Prometheus registry & metrics ---
REGISTRY = CollectorRegistry()
# register process/platform collectors if available
try:
    if ProcessCollector is not None:
        ProcessCollector(registry=REGISTRY)
except Exception:
    pass
try:
    if PLATFORM_COLLECTOR is not None:
        PLATFORM_COLLECTOR(registry=REGISTRY)
except Exception:
    pass

REQUEST_COUNTER = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["service", "route", "status"],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "RAG request duration seconds",
    ["service", "route"],
    registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)
)

# load env
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    logger.info("FastAPI and requests instrumented for OTel (if enabled)")
except Exception as e:
    logger.warning("OTel instrumentation failed to initialize (continuing without auto-instrument)", extra={"error": str(e)})

# Expose Prometheus metrics
@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# middleware to create span and collect metrics
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    # Tạo request_id riêng cho mỗi request
    request_id = str(uuid.uuid4())
    start = time.time()

    # Ghi vào span
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("request_id", request_id)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.path", request.url.path)

    response = await call_next(request)

    # Ghi thêm thông tin sau khi request xong
    if span and span.is_recording():
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("process_time_ms", (time.time() - start) * 1000)

    return response


collection_name = os.getenv("COLLECTION_NAME", "my_default_collection")
vector_store = get_pgvector_store(collection_name=collection_name)
        
@app.post("/answer")
async def rag_api(question: str = Form(None), audio: Union[UploadFile, str] = File(None)):
    request_id = str(uuid.uuid4())
    REQUEST_START = time.time()
    route = "/answer"

    if isinstance(audio, str) and audio == "":
        audio = None

    if not question and not audio:
        return JSONResponse(status_code=400, content={"error": "No question or audio provided"})

    logger.info("rag_received", extra={
        "request_id": request_id,
        "has_question": bool(question),
        "has_audio": bool(audio)
    })

    try:
        with tracer.start_as_current_span("rag_request") as req_span:
            req_span.set_attribute("request_id", request_id)
            req_span.set_attribute("http.route", route)
            req_span.set_attribute("has_question", bool(question))
            req_span.set_attribute("has_audio", bool(audio))
            req_span.set_attribute("service.version", SERVICE_VERSION)

            # --- Chỉ question, không audio ---
            if question and not audio:
                valid_text = question
                req_span.set_attribute("query_type", "text_only")

            # --- Chỉ audio, không question ---
            elif audio and not question:
                req_span.set_attribute("query_type", "audio_only")
                # Gọi STT
                logger.info("Calling STT service", extra={"request_id": request_id})
                response_stt = requests.post("http://whisper-api:8000/STT/", files={"file": audio.file})
                response_stt.raise_for_status()
                data = response_stt.json()
                valid_text = data["output_text"]
                # print(valid_text)
                req_span.set_attribute("stt_text_length", len(valid_text))

            # --- Cả question và audio ---
            else:
                req_span.set_attribute("query_type", "both_text_audio")
                logger.info("Calling STT service for audio", extra={"request_id": request_id})
                response_stt = requests.post("http://whisper-api:8000/STT/", files={"file": audio.file})
                response_stt.raise_for_status()
                data = response_stt.json()
                output_stt = data["output_text"]
                valid_text = output_stt + " " + question
                req_span.set_attribute("stt_text_length", len(output_stt))
                req_span.set_attribute("question_length", len(question))

            # --- Query PGVector ---
            output_database = query_similar_vectors_from_pgvector(valid_text, vector_store, top_k=5)
            similarities = []
            documents = []
            for document, score in output_database:
                documents.append(document.page_content)
                similarities.append(score)

            reranked_indices = get_top_k_contexts(documents, valid_text, similarities, k=3)
            req_span.set_attribute("contexts_count", len(reranked_indices))

            output_text = get_entities_as_string_GEMINI(prompt_template, information=reranked_indices, question=valid_text)

            # --- Nếu cần TTS ---
            if audio:
                logger.info("Calling TTS service", extra={"request_id": request_id})
                # print(output_text)
                tts_response = requests.post("http://tts-api:8001/transcribe/", data={"text_input": output_text})
                tts_response.raise_for_status()
                tts_data = tts_response.json()
                final_audio_base64 = tts_data.get("audio_base64")
                # print("TTS audio length (base64): ", len(final_audio_base64))
                total = time.time() - REQUEST_START
                REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route=route).observe(total)
                REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route=route, status="200").inc()
                logger.info("rag_completed", extra={"request_id": request_id, "duration_s": total})
                return JSONResponse(
                    content={
                        "request_id": request_id,
                        "audio_base64": final_audio_base64
                    }
                )

            # --- Trả text nếu không audio ---
            total = time.time() - REQUEST_START
            REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route=route).observe(total)
            REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route=route, status="200").inc()
            logger.info("rag_completed", extra={"request_id": request_id, "duration_s": total})
            return {"type": "text", "content": output_text}

    except Exception as e:
        total = time.time() - REQUEST_START
        REQUEST_LATENCY.labels(service=SERVICE_NAME_STR, route=route).observe(total)
        REQUEST_COUNTER.labels(service=SERVICE_NAME_STR, route=route, status="500").inc()

        span = trace.get_current_span()
        if span:
            span.record_exception(e)
            try:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception:
                span.set_attribute("error", True)

        logger.exception("rag_failed", extra={"request_id": request_id, "error": str(e)})
        return JSONResponse(status_code=500, content={"error": str(e)})

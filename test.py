from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

SERVICE_NAME_STR = "whisper-api"

provider = TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE_NAME_STR}))
jaeger_exporter = JaegerExporter(agent_host_name="jaeger", agent_port=6831)
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("hello_world"):
    print("Span sent!")

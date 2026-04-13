"""
PostSale MVP — Fase 2: API REST
================================
Backend completo con FastAPI + SQLite para el motor de churn.

Endpoints:
  POST /clientes          — registra un cliente nuevo
  POST /analizar/{id}     — analiza el riesgo de un cliente
  GET  /clientes          — lista todos los clientes con su riesgo actual
  GET  /clientes/{id}     — ficha completa de un cliente con historial
  GET  /dashboard         — resumen ejecutivo para el gestor

Uso:
    1. $env:GROQ_API_KEY="gsk_tu-clave"
    2. python -m pip install fastapi uvicorn sqlmodel groq pydantic
    3. uvicorn api:app --reload

Autor: PostSale Engineering
Versión: 0.3.0 (Fase 2 — API REST)
"""

import asyncio
import json
import resend
import logging
import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional

from groq import AsyncGroq
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postsale")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN CENTRAL
# ---------------------------------------------------------------------------

CONFIG = {
    "modelo": "llama-3.3-70b-versatile",
    "temperatura": 0,
    "max_tokens": 600,
    "max_reintentos": 3,
    "espera_base_segundos": 2,
    "score_minimo_confianza": 6,
}

DATABASE_URL = "sqlite:///postsale.db"

# ---------------------------------------------------------------------------
# MODELOS DE BASE DE DATOS (SQLModel = Pydantic + SQLite en uno)
# ---------------------------------------------------------------------------


class PlanTipo(str, Enum):
    STARTER = "Starter"
    PROFESSIONAL = "Professional"
    ENTERPRISE = "Enterprise"


class NivelRiesgo(str, Enum):
    BAJO = "Bajo"
    MEDIO = "Medio"
    ALTO = "Alto"
    CRITICO = "Crítico"


class ClienteDB(SQLModel, table=True):
    """
    Tabla 'clientedb' en SQLite.
    Guarda los datos básicos de cada cliente B2B.
    """
    __tablename__ = "clientes"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    nombre: str
    plan_actual: PlanTipo
    fecha_registro: str = SQLField(
        default_factory=lambda: datetime.now().isoformat()
    )


class AnalisisDB(SQLModel, table=True):
    """
    Tabla 'analisisdb' en SQLite.
    Guarda cada análisis realizado — permite ver la evolución del riesgo.
    """
    __tablename__ = "analisis"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    cliente_id: int = SQLField(foreign_key="clientes.id")
    nivel_riesgo: str
    probabilidad_churn_porcentaje: int
    razon_principal: str
    accion_recomendada: str
    score_confianza: int
    intentos_realizados: int
    tiempo_procesamiento_segundos: float
    requiere_revision_manual: bool
    fecha_analisis: str = SQLField(
        default_factory=lambda: datetime.now().isoformat()
    )
    interacciones_json: str  # guardamos las señales usadas para este análisis


# ---------------------------------------------------------------------------
# MODELOS DE REQUEST / RESPONSE (lo que recibe y devuelve la API)
# ---------------------------------------------------------------------------


class TipoInteraccion(str, Enum):
    EMAIL = "email"
    TICKET = "ticket"
    INACTIVIDAD_LOGIN = "inactividad_login"


class InteraccionInput(BaseModel):
    """Una señal de interacción del cliente — lo que llega en el request."""
    tipo_interaccion: TipoInteraccion
    texto_mensaje: str
    dias_desde_ultima_conexion: int = Field(ge=0)


class ClienteInput(BaseModel):
    """Datos para registrar un cliente nuevo."""
    nombre: str
    plan_actual: PlanTipo


class AnalisisInput(BaseModel):
    """Señales que se envían para analizar el riesgo de un cliente."""
    interacciones: list[InteraccionInput]


class AnalisisResponse(BaseModel):
    """Respuesta completa del análisis de riesgo."""
    cliente_id: int
    cliente_nombre: str
    nivel_riesgo: str
    probabilidad_churn_porcentaje: int
    razon_principal: str
    accion_recomendada_para_el_gestor: str
    score_confianza: int
    requiere_revision_manual: bool
    intentos_realizados: int
    tiempo_procesamiento_segundos: float
    fecha_analisis: str


class ClienteResumen(BaseModel):
    """Vista resumida de un cliente para la lista del dashboard."""
    id: int
    nombre: str
    plan_actual: str
    ultimo_nivel_riesgo: Optional[str]
    ultima_probabilidad_churn: Optional[int]
    ultima_accion_recomendada: Optional[str]
    fecha_ultimo_analisis: Optional[str]
    total_analisis: int


class DashboardResponse(BaseModel):
    """Resumen ejecutivo para el gestor de cuenta."""
    total_clientes: int
    criticos: int
    alto_riesgo: int
    medio_riesgo: int
    bajo_riesgo: int
    sin_analisis: int
    requieren_revision_manual: int
    clientes_urgentes: list[ClienteResumen]


# ---------------------------------------------------------------------------
# MOTOR DE IA (igual que Fase 1, reutilizado)
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = """Eres PostSale-AI, un motor experto en análisis de fricción \
operativa y predicción de churn B2B. Tu única función es analizar señales de \
clientes corporativos y determinar su nivel de riesgo de cancelación.

REGLAS ESTRICTAS:
1. Respondés EXCLUSIVAMENTE con un objeto JSON válido. Sin texto adicional, \
sin explicaciones, sin markdown, sin backticks.
2. El campo nivel_riesgo SOLO puede ser: Bajo, Medio, Alto o Crítico.
3. probabilidad_churn_porcentaje debe ser un entero entre 0 y 100.
4. No inferís riesgo alto por educación o formalidad en el tono. Analizás hechos.
5. Si hay múltiples señales contradictorias, pesás las negativas con más fuerza.

EJEMPLOS DE CALIBRACIÓN:

Entrada: cliente activo, elogia el producto, pide nuevas funciones.
Salida correcta: {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":8,\
"razon_principal":"Engagement activo y satisfacción explícita",\
"accion_recomendada_para_el_gestor":"Responder consulta sobre nuevas funciones \
y agendar demo de roadmap Q2"}

Entrada: 15 días sin login, ticket confuso sin resolución.
Salida correcta: {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":75,\
"razon_principal":"Inactividad prolongada combinada con fricción no resuelta",\
"accion_recomendada_para_el_gestor":"Llamar hoy, ofrecer sesión 1:1 de \
onboarding y resolver el ticket en menos de 24 horas"}

Entrada: tercer error técnico crítico en una semana, cliente menciona evaluar \
competencia, contrato vence en 45 días.
Salida correcta: {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":92,\
"razon_principal":"Falla técnica reiterada + intención explícita de cancelar \
+ ventana contractual inmediata","accion_recomendada_para_el_gestor":\
"Escalar a gerencia hoy, asignar ingeniero dedicado al error, ofrecer \
extensión de contrato con descuento como puente"}"""


def construir_prompt_usuario(
    nombre: str,
    plan: str,
    interacciones: list[InteraccionInput],
) -> str:
    """Arma el prompt con el perfil y señales del cliente."""
    interacciones_texto = "\n".join(
        f"  - [{i.tipo_interaccion.value.upper()}] "
        f"Días sin login: {i.dias_desde_ultima_conexion} | "
        f"Mensaje: \"{i.texto_mensaje}\""
        for i in interacciones
    )
    return f"""PERFIL DEL CLIENTE:
- Empresa: {nombre}
- Plan: {plan}

SEÑALES RECIENTES:
{interacciones_texto}

Analizá estas señales y respondé SOLO con el JSON."""


def validar_respuesta_ia(contenido_raw: str) -> tuple[Optional[dict], int, str]:
    """Doble validación: parseo JSON + coherencia semántica."""
    try:
        datos = json.loads(contenido_raw.strip())
        niveles_validos = {"Bajo", "Medio", "Alto", "Crítico"}
        assert datos.get("nivel_riesgo") in niveles_validos
        assert isinstance(datos.get("probabilidad_churn_porcentaje"), int)
        assert 0 <= datos["probabilidad_churn_porcentaje"] <= 100
        assert len(datos.get("razon_principal", "")) >= 10
        assert len(datos.get("accion_recomendada_para_el_gestor", "")) >= 20
    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        return None, 0, f"Validación fallida: {e}"

    score = 10
    errores = []

    if datos["nivel_riesgo"] == "Crítico" and datos["probabilidad_churn_porcentaje"] < 70:
        errores.append("Crítico con prob < 70%")
        score -= 3
    if datos["nivel_riesgo"] == "Bajo" and datos["probabilidad_churn_porcentaje"] > 30:
        errores.append("Bajo con prob > 30%")
        score -= 3

    return datos, max(0, score), " | ".join(errores)


async def analizar_con_ia(
    nombre: str,
    plan: str,
    interacciones: list[InteraccionInput],
) -> dict:
    """
    Llama a Groq con reintentos y backoff exponencial.
    Devuelve el resultado del análisis como diccionario.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY no configurada. "
            "Ejecutá: $env:GROQ_API_KEY='gsk_tu-clave'"
        )

    groq_client = AsyncGroq(api_key=api_key)
    prompt = construir_prompt_usuario(nombre, plan, interacciones)
    inicio = time.time()
    ultimo_error = ""
    mejor_datos = None
    mejor_score = 0

    for intento in range(1, CONFIG["max_reintentos"] + 1):
        log.info(f"[{nombre}] Intento {intento}/{CONFIG['max_reintentos']}...")
        try:
            response = await groq_client.chat.completions.create(
                model=CONFIG["modelo"],
                temperature=CONFIG["temperatura"],
                max_tokens=CONFIG["max_tokens"],
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            datos, score, error = validar_respuesta_ia(raw)

            if datos and score >= CONFIG["score_minimo_confianza"] and not error:
                log.info(f"[{nombre}] OK — score: {score}/10")
                return {
                    **datos,
                    "score_confianza": score,
                    "intentos_realizados": intento,
                    "tiempo_procesamiento_segundos": round(time.time() - inicio, 2),
                    "requiere_revision_manual": False,
                }

            if datos and score > mejor_score:
                mejor_datos = datos
                mejor_score = score

            ultimo_error = error or "Score insuficiente"
            log.warning(f"[{nombre}] Intento {intento} — {ultimo_error}")

        except Exception as e:
            ultimo_error = str(e)
            log.warning(f"[{nombre}] Error API intento {intento}: {e}")

        if intento < CONFIG["max_reintentos"]:
            espera = CONFIG["espera_base_segundos"] ** intento
            await asyncio.sleep(espera)

    # Devolvemos el mejor resultado parcial si existe
    base = mejor_datos or {
        "nivel_riesgo": "Medio",
        "probabilidad_churn_porcentaje": 50,
        "razon_principal": "No se pudo determinar con certeza",
        "accion_recomendada_para_el_gestor": "Revisión manual requerida",
    }
    return {
        **base,
        "score_confianza": mejor_score,
        "intentos_realizados": CONFIG["max_reintentos"],
        "tiempo_procesamiento_segundos": round(time.time() - inicio, 2),
        "requiere_revision_manual": True,
    }


# ---------------------------------------------------------------------------
# BASE DE DATOS
# ---------------------------------------------------------------------------

engine = create_engine(DATABASE_URL, echo=False)


def crear_tablas():
    """Crea las tablas en SQLite si no existen."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Generador de sesión de base de datos."""
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# APP FASTAPI
# ---------------------------------------------------------------------------

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PostSale API",
    description="Motor de predicción de churn B2B con IA",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Crea las tablas al arrancar el servidor."""
    crear_tablas()
    log.info("PostSale API iniciada — base de datos lista.")


# --- POST /clientes — registrar cliente nuevo ---

@app.post("/clientes", response_model=dict, status_code=201)
def registrar_cliente(
    datos: ClienteInput,
    session: Session = Depends(get_session),
):
    """Registra un cliente nuevo en la base de datos."""
    cliente = ClienteDB(nombre=datos.nombre, plan_actual=datos.plan_actual)
    session.add(cliente)
    session.commit()
    session.refresh(cliente)
    log.info(f"Cliente registrado: {cliente.nombre} [ID: {cliente.id}]")
    return {
        "mensaje": "Cliente registrado exitosamente",
        "cliente_id": cliente.id,
        "nombre": cliente.nombre,
        "plan": cliente.plan_actual,
    }


# --- POST /analizar/{cliente_id} — analizar riesgo ---

@app.post("/analizar/{cliente_id}", response_model=AnalisisResponse)
async def analizar_cliente(
    cliente_id: int,
    datos: AnalisisInput,
    session: Session = Depends(get_session),
):
    """
    Analiza el riesgo de churn de un cliente.
    Guarda el resultado en la base de datos para historial.
    """
    cliente = session.get(ClienteDB, cliente_id)
    if not cliente:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente con ID {cliente_id} no encontrado",
        )

    if not datos.interacciones:
        raise HTTPException(
            status_code=400,
            detail="Debe enviar al menos una interacción para analizar",
        )

    # Llamar al motor de IA
    resultado = await analizar_con_ia(
        nombre=cliente.nombre,
        plan=cliente.plan_actual.value,
        interacciones=datos.interacciones,
    )

    # Guardar en base de datos
    analisis_db = AnalisisDB(
        cliente_id=cliente_id,
        nivel_riesgo=resultado["nivel_riesgo"],
        probabilidad_churn_porcentaje=resultado["probabilidad_churn_porcentaje"],
        razon_principal=resultado["razon_principal"],
        accion_recomendada=resultado["accion_recomendada_para_el_gestor"],
        score_confianza=resultado["score_confianza"],
        intentos_realizados=resultado["intentos_realizados"],
        tiempo_procesamiento_segundos=resultado["tiempo_procesamiento_segundos"],
        requiere_revision_manual=resultado["requiere_revision_manual"],
        interacciones_json=json.dumps(
            [i.dict() for i in datos.interacciones], ensure_ascii=False
        ),
    )
    session.add(analisis_db)
    session.commit()
    session.refresh(analisis_db)
    
    # Enviar alerta si el riesgo es Alto o Crítico
    nivel = resultado["nivel_riesgo"]
    if nivel in {"Alto", "Crítico"}:
        resend_key = os.getenv("RESEND_API_KEY")
        if resend_key:
            resend.api_key = resend_key
            emoji = "🔴" if nivel == "Crítico" else "🟠"
            try:
                resend.Emails.send({
                    "from": "PostSale <onboarding@resend.dev>",
                    "to": ["santivinas1@gmail.com"],
                    "subject": f"{emoji} PostSale — {nivel}: {cliente.nombre}",
                    "html": f"""
                    <div style='font-family:Arial,sans-serif;max-width:560px;margin:0 auto'>
                      <h2 style='color:#1a1a18'>{emoji} Alerta {nivel} — {cliente.nombre}</h2>
                      <p style='color:#6b6b67'>Plan: {cliente.plan_actual.value}</p>
                      <hr style='border:none;border-top:1px solid #e5e5e2'>
                      <p><strong>Probabilidad de cancelación:</strong> {resultado['probabilidad_churn_porcentaje']}%</p>
                      <p><strong>Causa:</strong> {resultado['razon_principal']}</p>
                      <div style='background:#fef2f2;border-left:3px solid #dc2626;padding:14px;border-radius:4px;margin-top:16px'>
                        <strong>Acción recomendada:</strong><br><br>
                        {resultado['accion_recomendada_para_el_gestor']}
                      </div>
                      <p style='color:#9b9b97;font-size:12px;margin-top:24px'>Confianza IA: {resultado['score_confianza']}/10 · PostSale</p>
                    </div>""",
                })
                log.info(f"Alerta email enviada para {cliente.nombre} ({nivel})")
            except Exception as e:
                log.warning(f"Error enviando email: {e}")

    return AnalisisResponse(
        cliente_id=cliente_id,
        cliente_nombre=cliente.nombre,
        nivel_riesgo=resultado["nivel_riesgo"],
        probabilidad_churn_porcentaje=resultado["probabilidad_churn_porcentaje"],
        razon_principal=resultado["razon_principal"],
        accion_recomendada_para_el_gestor=resultado["accion_recomendada_para_el_gestor"],
        score_confianza=resultado["score_confianza"],
        requiere_revision_manual=resultado["requiere_revision_manual"],
        intentos_realizados=resultado["intentos_realizados"],
        tiempo_procesamiento_segundos=resultado["tiempo_procesamiento_segundos"],
        fecha_analisis=analisis_db.fecha_analisis,
    )


# --- GET /clientes — listar todos los clientes ---

@app.get("/clientes", response_model=list[ClienteResumen])
def listar_clientes(session: Session = Depends(get_session)):
    """
    Lista todos los clientes con su último análisis de riesgo.
    Ordenados de mayor a menor riesgo.
    """
    clientes = session.exec(select(ClienteDB)).all()
    resultado = []

    orden_riesgo = {"Crítico": 0, "Alto": 1, "Medio": 2, "Bajo": 3, None: 4}

    for cliente in clientes:
        analisis_list = session.exec(
            select(AnalisisDB)
            .where(AnalisisDB.cliente_id == cliente.id)
            .order_by(AnalisisDB.id.desc())
        ).all()

        ultimo = analisis_list[0] if analisis_list else None

        resultado.append(ClienteResumen(
            id=cliente.id,
            nombre=cliente.nombre,
            plan_actual=cliente.plan_actual.value,
            ultimo_nivel_riesgo=ultimo.nivel_riesgo if ultimo else None,
            ultima_probabilidad_churn=ultimo.probabilidad_churn_porcentaje if ultimo else None,
            ultima_accion_recomendada=ultimo.accion_recomendada if ultimo else None,
            fecha_ultimo_analisis=ultimo.fecha_analisis if ultimo else None,
            total_analisis=len(analisis_list),
        ))

    resultado.sort(key=lambda c: orden_riesgo.get(c.ultimo_nivel_riesgo, 4))
    return resultado


# --- GET /clientes/{id} — ficha completa con historial ---

@app.get("/clientes/{cliente_id}", response_model=dict)
def ficha_cliente(
    cliente_id: int,
    session: Session = Depends(get_session),
):
    """Devuelve la ficha completa de un cliente con su historial de análisis."""
    cliente = session.get(ClienteDB, cliente_id)
    if not cliente:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente con ID {cliente_id} no encontrado",
        )

    analisis_list = session.exec(
        select(AnalisisDB)
        .where(AnalisisDB.cliente_id == cliente_id)
        .order_by(AnalisisDB.id.desc())
    ).all()

    historial = [
        {
            "fecha": a.fecha_analisis,
            "nivel_riesgo": a.nivel_riesgo,
            "probabilidad_churn_porcentaje": a.probabilidad_churn_porcentaje,
            "razon_principal": a.razon_principal,
            "accion_recomendada": a.accion_recomendada,
            "score_confianza": a.score_confianza,
            "requiere_revision_manual": a.requiere_revision_manual,
        }
        for a in analisis_list
    ]

    # Calcular tendencia (compara último vs anterior)
    tendencia = "sin_datos"
    if len(historial) >= 2:
        orden = {"Bajo": 0, "Medio": 1, "Alto": 2, "Crítico": 3}
        actual = orden.get(historial[0]["nivel_riesgo"], 0)
        anterior = orden.get(historial[1]["nivel_riesgo"], 0)
        if actual > anterior:
            tendencia = "empeorando"
        elif actual < anterior:
            tendencia = "mejorando"
        else:
            tendencia = "estable"

    return {
        "cliente": {
            "id": cliente.id,
            "nombre": cliente.nombre,
            "plan_actual": cliente.plan_actual.value,
            "fecha_registro": cliente.fecha_registro,
        },
        "tendencia": tendencia,
        "total_analisis": len(historial),
        "historial": historial,
    }


# --- GET /dashboard — resumen ejecutivo ---

@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(session: Session = Depends(get_session)):
    """
    Resumen ejecutivo para el gestor de cuenta.
    Muestra conteos por nivel de riesgo y los clientes más urgentes.
    """
    clientes = session.exec(select(ClienteDB)).all()
    conteos = {"Crítico": 0, "Alto": 0, "Medio": 0, "Bajo": 0}
    sin_analisis = 0
    revision_manual = 0
    clientes_resumen = []
    orden_riesgo = {"Crítico": 0, "Alto": 1, "Medio": 2, "Bajo": 3, None: 4}

    for cliente in clientes:
        analisis_list = session.exec(
            select(AnalisisDB)
            .where(AnalisisDB.cliente_id == cliente.id)
            .order_by(AnalisisDB.id.desc())
        ).all()

        ultimo = analisis_list[0] if analisis_list else None

        if ultimo is None:
            sin_analisis += 1
        else:
            nivel = ultimo.nivel_riesgo
            if nivel in conteos:
                conteos[nivel] += 1
            if ultimo.requiere_revision_manual:
                revision_manual += 1

        clientes_resumen.append(ClienteResumen(
            id=cliente.id,
            nombre=cliente.nombre,
            plan_actual=cliente.plan_actual.value,
            ultimo_nivel_riesgo=ultimo.nivel_riesgo if ultimo else None,
            ultima_probabilidad_churn=ultimo.probabilidad_churn_porcentaje if ultimo else None,
            ultima_accion_recomendada=ultimo.accion_recomendada if ultimo else None,
            fecha_ultimo_analisis=ultimo.fecha_analisis if ultimo else None,
            total_analisis=len(analisis_list),
        ))

    clientes_resumen.sort(
        key=lambda c: orden_riesgo.get(c.ultimo_nivel_riesgo, 4)
    )

    return DashboardResponse(
        total_clientes=len(clientes),
        criticos=conteos["Crítico"],
        alto_riesgo=conteos["Alto"],
        medio_riesgo=conteos["Medio"],
        bajo_riesgo=conteos["Bajo"],
        sin_analisis=sin_analisis,
        requieren_revision_manual=revision_manual,
        clientes_urgentes=clientes_resumen[:5],
    )


# --- GET / — health check ---

@app.get("/")
def health_check():
    """Verifica que la API está funcionando."""
    return {
        "estado": "ok",
        "servicio": "PostSale API",
        "version": "0.3.0",
        "docs": "/docs",
    }
"""
PostSale MVP — Fase 1: IA Confiable
=====================================
Motor de predicción de churn B2B con:
  - Reintentos automáticos con backoff exponencial
  - Doble validación del JSON de respuesta
  - Sistema de confianza (score 1-10) por análisis
  - Temperatura 0 + prompt con ejemplos concretos

Uso:
    1. $env:GROQ_API_KEY="gsk_tu-clave"
    2. python -m pip install groq pydantic
    3. python postsale_mvp.py

Autor: PostSale Engineering
Versión: 0.2.0 (Fase 1 — IA Confiable)
"""

import asyncio
import json
import logging
import os
import sys
import time
from enum import Enum
from typing import Optional

from groq import AsyncGroq
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# LOGGING — muestra qué hace el sistema en cada paso
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
    "temperatura": 0,           # 0 = máxima determinismo, sin creatividad
    "max_tokens": 600,
    "max_reintentos": 3,        # veces que reintenta si falla
    "espera_base_segundos": 2,  # backoff: 2s, 4s, 8s entre reintentos
    "score_minimo_confianza": 6,  # por debajo de esto alerta revisión manual
}


# ---------------------------------------------------------------------------
# MODELOS DE DATOS (Pydantic)
# ---------------------------------------------------------------------------

class PlanTipo(str, Enum):
    STARTER = "Starter"
    PROFESSIONAL = "Professional"
    ENTERPRISE = "Enterprise"


class TipoInteraccion(str, Enum):
    EMAIL = "email"
    TICKET = "ticket"
    INACTIVIDAD_LOGIN = "inactividad_login"


class Cliente(BaseModel):
    """Representa un cliente B2B dentro del sistema PostSale."""
    id: str
    nombre: str
    plan_actual: PlanTipo


class Interaccion(BaseModel):
    """Representa una señal de interacción del cliente con el producto."""
    tipo_interaccion: TipoInteraccion
    texto_mensaje: str
    dias_desde_ultima_conexion: int = Field(ge=0)


class AnalisisChurn(BaseModel):
    """
    Resultado del análisis de riesgo validado por Pydantic.
    Si algún campo falta o tiene tipo incorrecto, Pydantic lanza error.
    """
    nivel_riesgo: str
    probabilidad_churn_porcentaje: int = Field(ge=0, le=100)
    razon_principal: str
    accion_recomendada_para_el_gestor: str

    def es_nivel_valido(self) -> bool:
        """Verifica que el nivel de riesgo sea uno de los valores esperados."""
        return self.nivel_riesgo in {"Bajo", "Medio", "Alto", "Crítico"}


class ResultadoFinal(BaseModel):
    """
    Envuelve el análisis con metadatos de confiabilidad.
    Esto es lo que el sistema entrega al gestor de cuenta.
    """
    cliente_id: str
    cliente_nombre: str
    analisis: Optional[AnalisisChurn]
    score_confianza: int = Field(ge=0, le=10)
    requiere_revision_manual: bool
    intentos_realizados: int
    tiempo_procesamiento_segundos: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# CLIENTE GROQ
# ---------------------------------------------------------------------------

def get_cliente_groq() -> AsyncGroq:
    """
    Inicializa el cliente de Groq.

    Raises:
        SystemExit: Si GROQ_API_KEY no está configurada.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.error(
            "No se encontró GROQ_API_KEY. "
            "Ejecutá: $env:GROQ_API_KEY='gsk_tu-clave'"
        )
        sys.exit(1)
    return AsyncGroq(api_key=api_key)


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL PROMPT (con ejemplos concretos para calibrar la IA)
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
    cliente: Cliente,
    interacciones: list[Interaccion],
) -> str:
    """Arma el mensaje del usuario con el perfil y las señales del cliente."""
    interacciones_texto = "\n".join(
        f"  - [{i.tipo_interaccion.value.upper()}] "
        f"Días sin login: {i.dias_desde_ultima_conexion} | "
        f"Mensaje: \"{i.texto_mensaje}\""
        for i in interacciones
    )

    return f"""PERFIL DEL CLIENTE:
- ID: {cliente.id}
- Empresa: {cliente.nombre}
- Plan: {cliente.plan_actual.value}

SEÑALES RECIENTES:
{interacciones_texto}

Analizá estas señales y respondé SOLO con el JSON."""


# ---------------------------------------------------------------------------
# VALIDACIÓN DOBLE DEL JSON
# ---------------------------------------------------------------------------

def validar_respuesta_ia(contenido_raw: str) -> tuple[Optional[AnalisisChurn], int, str]:
    """
    Valida en dos capas el JSON devuelto por la IA.

    Capa 1: ¿Es JSON válido y tiene los campos correctos? (Pydantic)
    Capa 2: ¿Los valores son semánticamente correctos? (lógica de negocio)

    Returns:
        Tupla de (analisis, score_confianza, mensaje_error)
    """
    # --- Capa 1: Parseo y validación de tipos ---
    try:
        datos = json.loads(contenido_raw.strip())
        analisis = AnalisisChurn(**datos)
    except json.JSONDecodeError as e:
        return None, 0, f"JSON inválido: {e}"
    except ValidationError as e:
        return None, 0, f"Campos faltantes o tipos incorrectos: {e}"

    # --- Capa 2: Validación semántica ---
    errores_semanticos = []
    score = 10

    if not analisis.es_nivel_valido():
        errores_semanticos.append(
            f"nivel_riesgo inválido: '{analisis.nivel_riesgo}'"
        )
        score -= 5

    if analisis.nivel_riesgo == "Crítico" and analisis.probabilidad_churn_porcentaje < 70:
        errores_semanticos.append(
            "Inconsistencia: nivel Crítico con probabilidad menor al 70%"
        )
        score -= 3

    if analisis.nivel_riesgo == "Bajo" and analisis.probabilidad_churn_porcentaje > 30:
        errores_semanticos.append(
            "Inconsistencia: nivel Bajo con probabilidad mayor al 30%"
        )
        score -= 3

    if len(analisis.razon_principal) < 10:
        errores_semanticos.append("razon_principal demasiado corta")
        score -= 1

    if len(analisis.accion_recomendada_para_el_gestor) < 20:
        errores_semanticos.append("accion_recomendada demasiado vaga")
        score -= 1

    score = max(0, score)

    if errores_semanticos:
        return analisis, score, " | ".join(errores_semanticos)

    return analisis, score, ""


# ---------------------------------------------------------------------------
# LLAMADA A LA IA CON REINTENTOS Y BACKOFF EXPONENCIAL
# ---------------------------------------------------------------------------

async def llamar_ia_con_reintentos(
    cliente: Cliente,
    interacciones: list[Interaccion],
    groq_client: AsyncGroq,
) -> ResultadoFinal:
    """
    Llama a la IA con reintentos automáticos.

    Si la respuesta falla validación, reintenta hasta max_reintentos veces.
    Cada reintento espera el doble que el anterior (backoff exponencial).
    """
    inicio = time.time()
    prompt_usuario = construir_prompt_usuario(cliente, interacciones)
    ultimo_error = "Sin intentos realizados"
    mejor_analisis = None
    mejor_score = 0

    for intento in range(1, CONFIG["max_reintentos"] + 1):
        log.info(
            f"[{cliente.nombre}] Intento {intento}/{CONFIG['max_reintentos']}..."
        )

        try:
            response = await groq_client.chat.completions.create(
                model=CONFIG["modelo"],
                temperature=CONFIG["temperatura"],
                max_tokens=CONFIG["max_tokens"],
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": prompt_usuario},
                ],
            )

            contenido_raw = response.choices[0].message.content.strip()
            analisis, score, error_validacion = validar_respuesta_ia(contenido_raw)

            if analisis and score >= CONFIG["score_minimo_confianza"] and not error_validacion:
                log.info(
                    f"[{cliente.nombre}] OK en intento {intento} "
                    f"— score confianza: {score}/10"
                )
                return ResultadoFinal(
                    cliente_id=cliente.id,
                    cliente_nombre=cliente.nombre,
                    analisis=analisis,
                    score_confianza=score,
                    requiere_revision_manual=False,
                    intentos_realizados=intento,
                    tiempo_procesamiento_segundos=round(time.time() - inicio, 2),
                )

            if analisis and score > mejor_score:
                mejor_analisis = analisis
                mejor_score = score

            ultimo_error = error_validacion or "Score insuficiente"
            log.warning(
                f"[{cliente.nombre}] Intento {intento} falló validación: "
                f"{ultimo_error} (score: {score}/10)"
            )

        except Exception as e:
            ultimo_error = str(e)
            log.warning(f"[{cliente.nombre}] Intento {intento} error de API: {e}")

        if intento < CONFIG["max_reintentos"]:
            espera = CONFIG["espera_base_segundos"] ** intento
            log.info(f"[{cliente.nombre}] Esperando {espera}s antes de reintentar...")
            await asyncio.sleep(espera)

    tiempo_total = round(time.time() - inicio, 2)

    if mejor_analisis:
        log.warning(
            f"[{cliente.nombre}] Agotados reintentos. "
            f"Devolviendo mejor resultado parcial (score: {mejor_score}/10). "
            f"REQUIERE REVISIÓN MANUAL."
        )
        return ResultadoFinal(
            cliente_id=cliente.id,
            cliente_nombre=cliente.nombre,
            analisis=mejor_analisis,
            score_confianza=mejor_score,
            requiere_revision_manual=True,
            intentos_realizados=CONFIG["max_reintentos"],
            tiempo_procesamiento_segundos=tiempo_total,
            error=f"Resultado parcial tras {CONFIG['max_reintentos']} intentos: {ultimo_error}",
        )

    log.error(f"[{cliente.nombre}] Todos los intentos fallaron. Sin resultado.")
    return ResultadoFinal(
        cliente_id=cliente.id,
        cliente_nombre=cliente.nombre,
        analisis=None,
        score_confianza=0,
        requiere_revision_manual=True,
        intentos_realizados=CONFIG["max_reintentos"],
        tiempo_procesamiento_segundos=tiempo_total,
        error=ultimo_error,
    )


# ---------------------------------------------------------------------------
# MOCK DATA — 3 clientes de ejemplo
# ---------------------------------------------------------------------------

def generar_clientes_mock() -> list[tuple[Cliente, list[Interaccion]]]:
    """Genera 3 perfiles de riesgo distintos para pruebas."""

    cliente_feliz = Cliente(
        id="CLT-001", nombre="InnovateTech S.A.",
        plan_actual=PlanTipo.PROFESSIONAL,
    )
    interacciones_feliz = [
        Interaccion(
            tipo_interaccion=TipoInteraccion.EMAIL,
            texto_mensaje=(
                "Buenos días, quiero felicitar al equipo por las mejoras en el "
                "dashboard. Nuestro equipo usa el módulo de reportes diariamente "
                "y mejoró nuestra productividad un 30%. ¿Cuándo sale la versión 2.0?"
            ),
            dias_desde_ultima_conexion=0,
        ),
        Interaccion(
            tipo_interaccion=TipoInteraccion.TICKET,
            texto_mensaje=(
                "Consulta: ¿Es posible exportar reportes en PDF además de Excel? "
                "Sería útil para presentaciones al directorio."
            ),
            dias_desde_ultima_conexion=0,
        ),
    ]

    cliente_inactivo = Cliente(
        id="CLT-002", nombre="Distribuidora Mercurio Ltda.",
        plan_actual=PlanTipo.STARTER,
    )
    interacciones_inactivo = [
        Interaccion(
            tipo_interaccion=TipoInteraccion.INACTIVIDAD_LOGIN,
            texto_mensaje=(
                "El usuario principal no ha iniciado sesión en los últimos 15 días. "
                "No hay actividad en ningún módulo."
            ),
            dias_desde_ultima_conexion=15,
        ),
        Interaccion(
            tipo_interaccion=TipoInteraccion.TICKET,
            texto_mensaje=(
                "Hola, intentamos importar datos de clientes pero no sé si quedó "
                "bien. No sabemos qué botón apretar."
            ),
            dias_desde_ultima_conexion=15,
        ),
    ]

    cliente_critico = Cliente(
        id="CLT-003", nombre="GlobalOps Enterprise",
        plan_actual=PlanTipo.ENTERPRISE,
    )
    interacciones_critico = [
        Interaccion(
            tipo_interaccion=TipoInteraccion.TICKET,
            texto_mensaje=(
                "URGENTE — Tercer ticket esta semana. La integración con nuestro "
                "ERP falla con error 500 en /api/sync. Bloqueó el cierre contable. "
                "Perdimos 2 días de trabajo."
            ),
            dias_desde_ultima_conexion=1,
        ),
        Interaccion(
            tipo_interaccion=TipoInteraccion.EMAIL,
            texto_mensaje=(
                "Les informamos formalmente que si esto no se resuelve antes del "
                "viernes evaluaremos otras alternativas. Nuestro contrato vence en "
                "45 días y no tenemos intención de renovar dado el nivel de servicio."
            ),
            dias_desde_ultima_conexion=1,
        ),
        Interaccion(
            tipo_interaccion=TipoInteraccion.TICKET,
            texto_mensaje=(
                "El error persiste. Necesitamos solución HOY o escalaremos con "
                "nuestro abogado para revisar el SLA del contrato."
            ),
            dias_desde_ultima_conexion=1,
        ),
    ]

    return [
        (cliente_feliz, interacciones_feliz),
        (cliente_inactivo, interacciones_inactivo),
        (cliente_critico, interacciones_critico),
    ]


# ---------------------------------------------------------------------------
# PRESENTACIÓN DE RESULTADOS
# ---------------------------------------------------------------------------

def mostrar_resultado(resultado: ResultadoFinal) -> None:
    """Imprime el resultado de un cliente de forma clara y accionable."""
    separador = "─" * 65
    print(f"\n{separador}")
    print(f"  Cliente  : {resultado.cliente_nombre}  [{resultado.cliente_id}]")
    print(f"  Intentos : {resultado.intentos_realizados} | "
          f"Tiempo: {resultado.tiempo_procesamiento_segundos}s | "
          f"Confianza: {resultado.score_confianza}/10")
    print(separador)

    if resultado.analisis is None:
        print("  ⚠️  Sin resultado — todos los intentos fallaron.")
        print(f"  Error: {resultado.error}")
        return

    emoji_map = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🟠", "Crítico": "🔴"}
    emoji = emoji_map.get(resultado.analisis.nivel_riesgo, "⚪")

    print(f"  {emoji} Nivel de Riesgo  : {resultado.analisis.nivel_riesgo}")
    print(f"  📊 Prob. Churn     : {resultado.analisis.probabilidad_churn_porcentaje}%")
    print(f"  🔍 Razón Principal : {resultado.analisis.razon_principal}")

    if resultado.requiere_revision_manual:
        print("  ⚠️  REQUIERE REVISIÓN MANUAL — confianza baja")

    print(f"  💬 Acción para CSM :")
    accion = resultado.analisis.accion_recomendada_para_el_gestor
    palabras = accion.split()
    linea = "     "
    for palabra in palabras:
        if len(linea) + len(palabra) > 65:
            print(linea)
            linea = "     " + palabra + " "
        else:
            linea += palabra + " "
    if linea.strip():
        print(linea)


# ---------------------------------------------------------------------------
# ENTRYPOINT PRINCIPAL
# ---------------------------------------------------------------------------

async def main() -> None:
    """Orquesta el análisis de todos los clientes con máxima confiabilidad."""
    print("=" * 65)
    print("  🚀  PostSale Fase 1 — Motor Confiable de Predicción de Churn")
    print("=" * 65)
    print(f"  Modelo     : {CONFIG['modelo']}")
    print(f"  Temperatura: {CONFIG['temperatura']} (máximo determinismo)")
    print(f"  Reintentos : hasta {CONFIG['max_reintentos']} por cliente")
    print(f"  Confianza  : mínimo {CONFIG['score_minimo_confianza']}/10 para aceptar resultado")
    print("=" * 65)

    groq_client = get_cliente_groq()
    clientes_mock = generar_clientes_mock()

    tareas = [
        llamar_ia_con_reintentos(cliente, interacciones, groq_client)
        for cliente, interacciones in clientes_mock
    ]
    resultados = await asyncio.gather(*tareas)

    for resultado in resultados:
        mostrar_resultado(resultado)

    print(f"\n{'=' * 65}")
    criticos = [r for r in resultados if r.analisis and r.analisis.nivel_riesgo == "Crítico"]
    altos = [r for r in resultados if r.analisis and r.analisis.nivel_riesgo == "Alto"]
    revision = [r for r in resultados if r.requiere_revision_manual]

    print(f"  📋 RESUMEN EJECUTIVO")
    print(f"  Total clientes analizados : {len(resultados)}")
    print(f"  🔴 Críticos               : {len(criticos)}")
    print(f"  🟠 Riesgo Alto            : {len(altos)}")
    print(f"  ⚠️  Requieren revisión     : {len(revision)}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
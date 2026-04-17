"""
PostSale MVP — Fase 2: API REST + Sistema de Tareas
=====================================================
Backend completo con FastAPI + SQLite para el motor de churn.

Endpoints base:
  POST /clientes          — registra un cliente nuevo
  POST /analizar/{id}     — analiza el riesgo de un cliente
  GET  /clientes          — lista todos los clientes con su riesgo actual
  GET  /clientes/{id}     — ficha completa de un cliente con historial
  GET  /dashboard         — resumen ejecutivo para el gestor

Endpoints de tareas:
  GET    /tareas              — lista tareas (filtrable por estado)
  PATCH  /tareas/{id}         — actualiza una tarea
  GET    /tareas/resumen      — estadísticas de tareas
  DELETE /tareas/{id}         — descarta una tarea

Uso:
    1. $env:GROQ_API_KEY="gsk_tu-clave"
    2. python -m pip install fastapi uvicorn sqlmodel groq pydantic
    3. uvicorn api:app --reload

Autor: PostSale Engineering
Versión: 0.7.0 (Sistema de Tareas)
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
from rag_postsale import analizar_con_rag, inicializar_base_vectorial

coleccion_rag = inicializar_base_vectorial()

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
# MODELOS DE BASE DE DATOS
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


class TareaEstado(str, Enum):
    PENDIENTE = "pendiente"
    EN_CURSO = "en_curso"
    COMPLETADA = "completada"
    DESCARTADA = "descartada"


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


class TareaDB(SQLModel, table=True):
    """
    Tabla de tareas para gestores de cuenta.
    Cada análisis de riesgo Alto o Crítico genera una tarea automática.
    El gestor registra qué hizo y el resultado real.
    """
    __tablename__ = "tareas"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    cliente_id: int = SQLField(foreign_key="clientes.id")
    analisis_id: int = SQLField(foreign_key="analisis.id")
    nivel_riesgo: str
    accion_sugerida: str           # lo que sugirió la IA
    accion_tomada: Optional[str] = None   # lo que hizo el gestor
    resultado_real: Optional[str] = None  # qué pasó realmente
    estado: str = "pendiente"      # pendiente | en_curso | completada | descartada
    fecha_creacion: str = SQLField(
        default_factory=lambda: datetime.now().isoformat()
    )
    fecha_cierre: Optional[str] = None
    gestor: Optional[str] = None   # nombre del gestor asignado
    notas: Optional[str] = None    # notas adicionales del gestor


# ---------------------------------------------------------------------------
# MODELOS DE REQUEST / RESPONSE
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


class TareaInput(BaseModel):
    """Datos para actualizar una tarea."""
    accion_tomada: Optional[str] = None
    resultado_real: Optional[str] = None
    estado: Optional[str] = None
    gestor: Optional[str] = None
    notas: Optional[str] = None


class TareaResponse(BaseModel):
    """Respuesta de una tarea."""
    id: int
    cliente_id: int
    cliente_nombre: str
    analisis_id: int
    nivel_riesgo: str
    accion_sugerida: str
    accion_tomada: Optional[str]
    resultado_real: Optional[str]
    estado: str
    fecha_creacion: str
    fecha_cierre: Optional[str]
    gestor: Optional[str]
    notas: Optional[str]


# ---------------------------------------------------------------------------
# PROMPT DEL SISTEMA
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = """Eres PostSale-AI, el motor más avanzado del mundo en \
detección de fricción operativa y predicción de churn B2B. Tu especialidad \
es detectar señales que los CRMs tradicionales ignoran: el cliente educado \
que internamente ya decidió irse, el técnico frustrado cuyo jefe cree que \
todo va bien, el contrato que no se va a renovar aunque nadie lo haya dicho.

REGLAS ABSOLUTAS:
1. Respondés EXCLUSIVAMENTE con JSON válido. Sin texto, sin markdown, sin \
backticks, sin explicaciones.
2. nivel_riesgo SOLO puede ser: Bajo, Medio, Alto o Crítico.
3. probabilidad_churn_porcentaje: entero entre 0 y 100.
4. El tono educado NO reduce el riesgo. Analizás hechos y patrones, no cortesía.
5. Señales negativas múltiples se potencian entre sí, no se promedian.
6. Una señal de migración activa (pedir exportación, preguntar por APIs, \
comparar competidores) es siempre Alto o Crítico sin importar el resto.
7. Ventana contractual menor a 60 días + cualquier fricción = Crítico.
8. Silencio prolongado (>10 días sin login) en usuario activo previo = Alto mínimo.

═══════════════════════════════════════════════════════
BIBLIOTECA DE 50 ESCENARIOS DE CALIBRACIÓN
═══════════════════════════════════════════════════════

━━━ CATEGORÍA 1: MUERTE SILENCIOSA (Silent Churn) ━━━

[SC-01] Cliente deja de usar módulo premium pero sigue logueándose.
Señal: "Ya no vemos necesario el módulo de reportes avanzados por ahora."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":72,
"razon_principal":"Abandono de módulo premium indica búsqueda de alternativa externa",
"accion_recomendada_para_el_gestor":"Llamar esta semana para entender qué herramienta reemplazó el módulo. Ofrecer capacitación gratuita y caso de uso específico para su industria."}

[SC-02] Usuario técnico principal deja de loguear, solo entra el administrativo.
Señal: Sin tickets, sin actividad técnica, 18 días sin login del usuario power.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":78,
"razon_principal":"El usuario técnico clave abandonó la herramienta — señal de evaluación interna de alternativas",
"accion_recomendada_para_el_gestor":"Contactar directamente al usuario técnico (no al administrativo) para entender su experiencia real. El administrativo puede no saber que hay un problema."}

[SC-03] Reducción gradual del volumen de datos procesados sin explicación.
Señal: El cliente procesaba 10.000 registros/mes, ahora procesa 800.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":81,
"razon_principal":"Reducción del 92% en uso operativo indica migración parcial a otra herramienta",
"accion_recomendada_para_el_gestor":"Solicitar reunión urgente. Preguntar directamente si están usando otra herramienta en paralelo. Ofrecer revisión de integración gratuita."}

[SC-04] Cliente activo que de repente deja de abrir los emails de la plataforma.
Señal: 0% open rate en últimas 6 comunicaciones, antes abría el 80%.
→ {"nivel_riesgo":"Medio","probabilidad_churn_porcentaje":48,
"razon_principal":"Desconexión comunicacional súbita tras engagement previo alto",
"accion_recomendada_para_el_gestor":"Intentar contacto por canal alternativo (llamada o WhatsApp). Cambiar el tipo de comunicación — probablemente recibe demasiado volumen."}

[SC-05] Cliente deja de asistir a las reuniones de seguimiento sin cancelarlas.
Señal: Tres reuniones mensuales seguidas sin asistencia, sin aviso previo.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":74,
"razon_principal":"Desaparición deliberada — evitar confrontación es señal de decisión tomada internamente",
"accion_recomendada_para_el_gestor":"No enviar otra invitación de reunión. Enviar email corto y directo preguntando si hay algo que no está funcionando. El silencio activo es más peligroso que la queja."}

━━━ CATEGORÍA 2: DISONANCIA TÉCNICO-ADMINISTRATIVA ━━━

[DA-01] Gerente feliz, equipo técnico con tickets repetidos sin resolución.
Señal: Email del gerente: "Todo bien". Tickets del técnico: errores de integración semana 3.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":76,
"razon_principal":"Riesgo de boicot interno — el técnico que usa la herramienta está frustrado aunque el pagador crea que todo va bien",
"accion_recomendada_para_el_gestor":"Resolver el ticket técnico HOY. Luego informar al gerente sobre la resolución para que sepa que hubo un problema real que se atendió."}

[DA-02] El técnico pide documentación de la API de exportación.
Señal: "¿Pueden compartirme la documentación completa de la API de exportación de datos?"
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":88,
"razon_principal":"Solicitud de documentación de exportación es señal directa de evaluación de migración",
"accion_recomendada_para_el_gestor":"Escalar inmediatamente. Antes de compartir la documentación, agendar llamada para entender qué necesidad específica tiene. Puede ser una integración legítima o una migración en curso."}

[DA-03] Usuario nuevo asignado tras cambio de personal interno sin onboarding.
Señal: "Hola, soy María, reemplazo a Juan. ¿Pueden explicarme cómo funciona esto?"
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":65,
"razon_principal":"Cambio de usuario clave sin transición — ventana de vulnerabilidad alta donde el nuevo usuario puede recomendar cambiar la herramienta",
"accion_recomendada_para_el_gestor":"Contactar a María en las próximas 24 horas. Ofrecer sesión de onboarding personalizada gratuita. El primer mes del nuevo usuario define la retención del contrato."}

[DA-04] El equipo técnico aprende la herramienta de la competencia.
Señal: El técnico pregunta: "¿Tienen integración nativa con Salesforce? Nuestro equipo está evaluando opciones."
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":91,
"razon_principal":"Evaluación activa de competidores confirmada — proceso de compra alternativo en curso",
"accion_recomendada_para_el_gestor":"Reunión ejecutiva esta semana. Presentar roadmap de integración con Salesforce. Si no existe, ofrecer integración vía Zapier como puente inmediato."}

[DA-05] Administrativo aprueba facturas pero técnico no usa la herramienta.
Señal: Factura pagada puntualmente pero 0 actividad en los últimos 25 días.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":69,
"razon_principal":"Pago automático oculta abandono real — el contrato sigue pero la herramienta está muerta operativamente",
"accion_recomendada_para_el_gestor":"No confundir pago puntual con salud del cliente. Contactar al usuario técnico directamente. Si no responde en 48h, escalar al decisor económico."}

━━━ CATEGORÍA 3: VENTANA CONTRACTUAL CRÍTICA ━━━

[VC-01] Contrato vence en 30 días y el cliente no inició conversación de renovación.
Señal: 30 días para vencimiento, silencio total del cliente.
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":85,
"razon_principal":"Silencio en ventana contractual final — en B2B la no-renovación activa es la señal más tardía posible",
"accion_recomendada_para_el_gestor":"Llamada ejecutiva hoy. No email. Preguntar directamente sobre la renovación y qué necesitan ver para firmar. Tener propuesta de valor lista."}

[VC-02] Cliente pide desglose de ROI justo antes del vencimiento.
Señal: "¿Pueden preparar un informe de todo lo que usamos este año?"
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":71,
"razon_principal":"Solicitud de ROI pre-renovación indica que el valor no es obvio — están justificando internamente si renovar",
"accion_recomendada_para_el_gestor":"Preparar el informe en menos de 24 horas. Incluir métricas de ahorro de tiempo, casos resueltos y comparación con el costo de la alternativa manual."}

[VC-03] Cliente pide reducir el plan justo antes de renovar.
Señal: "Estamos evaluando bajar al plan Starter para el próximo año."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":68,
"razon_principal":"Downsell activo — primer paso hacia la cancelación total en 1-2 ciclos",
"accion_recomendada_para_el_gestor":"No aceptar el downsell sin reunión previa. Entender qué funcionalidades del plan actual no están usando y por qué. Ofrecer descuento en el plan actual antes de bajar."}

[VC-04] Contrato vence en 45 días y acaban de abrir un ticket crítico.
Señal: Error bloqueante + 45 días para renovación.
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":93,
"razon_principal":"Falla técnica en ventana contractual — el cliente usará el error como justificación para no renovar",
"accion_recomendada_para_el_gestor":"Resolver el error en menos de 4 horas. Luego llamar personalmente para confirmar la resolución. Un error bien resuelto en ventana contractual puede fortalecer la renovación."}

[VC-05] Cliente pregunta por condiciones de cancelación anticipada.
Señal: "¿Cuáles son las condiciones si decidimos terminar el contrato antes de tiempo?"
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":95,
"razon_principal":"Consulta de exit es la señal más directa posible de intención de cancelar",
"accion_recomendada_para_el_gestor":"No responder por email. Llamar en la próxima hora. Entender qué pasó. Escalar a gerencia. Esta conversación debe tenerla el responsable de cuenta senior, no un agente."}

━━━ CATEGORÍA 4: SEÑALES DE MIGRACIÓN ACTIVA ━━━

[MA-01] Cliente descarga masivamente sus datos históricos.
Señal: Exportación completa de todos los registros de los últimos 3 años en un día.
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":96,
"razon_principal":"Exportación masiva de histórico es señal inequívoca de migración en curso",
"accion_recomendada_para_el_gestor":"Llamada ejecutiva inmediata. En este punto la retención es difícil pero posible si hay un problema específico resoluble. Preguntar directamente qué pasó."}

[MA-02] Cliente pregunta por integraciones con herramientas de la competencia.
Señal: "¿Tienen conector con [competidor]? Necesitamos sincronizar datos entre los dos."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":79,
"razon_principal":"Uso simultáneo con competidor indica evaluación en paralelo o migración gradual",
"accion_recomendada_para_el_gestor":"Preguntar el contexto de la integración antes de responder. Si están en evaluación paralela, acelerar la conversación de valor. No construir integraciones que faciliten la salida."}

[MA-03] Nuevos usuarios añadidos con perfil de evaluación técnica.
Señal: Agregaron 3 usuarios nuevos con rol 'Admin' en una semana sin aviso.
→ {"nivel_riesgo":"Medio","probabilidad_churn_porcentaje":44,
"razon_principal":"Usuarios nuevos con rol admin pueden indicar auditoría interna o evaluación de migración",
"accion_recomendada_para_el_gestor":"Contactar para dar bienvenida a los nuevos usuarios. Usar la excusa del onboarding para entender el contexto real de por qué se añadieron."}

[MA-04] Cliente pregunta por tiempo de implementación de una alternativa.
Señal: En una reunión mencionan: "¿Cuánto tiempo llevaría migrar todos nuestros datos si decidiéramos cambiar?"
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":89,
"razon_principal":"Pregunta directa sobre tiempo de migración confirma evaluación activa de salida",
"accion_recomendada_para_el_gestor":"Reunión ejecutiva urgente. Presentar propuesta de mejora concreta. Si hay un problema específico, comprometerse a resolverlo con fecha y responsable nombrado."}

[MA-05] Cliente deja de agregar nuevos usuarios pese a crecimiento de su empresa.
Señal: La empresa creció un 40% en headcount pero no añadió licencias nuevas.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":73,
"razon_principal":"Crecimiento sin expansión de licencias indica que los nuevos empleados usan otra herramienta",
"accion_recomendada_para_el_gestor":"Felicitar por el crecimiento de la empresa. Preguntar cómo están incorporando a los nuevos empleados al flujo de trabajo. La respuesta revelará si hay una herramienta alternativa en uso."}

━━━ CATEGORÍA 5: FRICCIÓN TÉCNICA REITERADA ━━━

[FT-01] Mismo error reportado tres veces sin resolución definitiva.
Señal: Tickets #234, #251, #278 sobre el mismo bug en exportación CSV.
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":87,
"razon_principal":"Error recurrente sin resolución definitiva destruye la confianza técnica del cliente",
"accion_recomendada_para_el_gestor":"Asignar ingeniero senior dedicado exclusivamente a este bug. Comunicar al cliente el nombre del ingeniero asignado y fecha comprometida de resolución. El seguimiento personal cambia la percepción."}

[FT-02] Tiempo de respuesta de soporte percibido como lento.
Señal: "Llevamos 3 días esperando respuesta al ticket #445. Esto nos está frenando."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":67,
"razon_principal":"Fricción de soporte activa que bloquea operaciones — cada día sin respuesta multiplica el riesgo",
"accion_recomendada_para_el_gestor":"Responder al ticket en la próxima hora aunque sea para confirmar que se está trabajando. Dar un tiempo comprometido de resolución. El silencio de soporte es el mayor destructor de confianza."}

[FT-03] Errores técnicos durante una demo o presentación del cliente.
Señal: "La herramienta se cayó justo cuando estábamos mostrando el sistema a nuestros inversores."
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":91,
"razon_principal":"Falla pública en momento crítico — daño reputacional al cliente genera riesgo de cancelación inmediata",
"accion_recomendada_para_el_gestor":"Disculpa formal del CEO o gerente senior en menos de 2 horas. Ofrecer mes gratuito y sesión de demo asistida para cuando necesiten volver a presentar. El gesto tiene que ser proporcional al daño."}

[FT-04] Integraciones que fallan en períodos críticos del cliente.
Señal: "La sincronización con nuestro ERP falla siempre a fin de mes cuando más la necesitamos."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":76,
"razon_principal":"Falla predecible en momento crítico del negocio — el cliente ya sabe cuándo va a fallar, lo que es peor que un error random",
"accion_recomendada_para_el_gestor":"Reunión técnica urgente para reproducir el problema antes de fin de mes. Implementar monitoreo específico para ese período. Comunicar el plan de acción con fechas."}

[FT-05] Performance degradada que afecta el workflow diario.
Señal: "El sistema está muy lento desde hace dos semanas. Cargamos los reportes y tardamos 4 minutos."
→ {"nivel_riesgo":"Medio","probabilidad_churn_porcentaje":55,
"razon_principal":"Degradación de performance que afecta productividad diaria — fricción acumulativa que erosiona la satisfacción",
"accion_recomendada_para_el_gestor":"Reconocer el problema explícitamente. Dar un plazo concreto de mejora. Si la mejora tarda más de una semana, compensar con extensión de contrato o crédito."}

━━━ CATEGORÍA 6: EXPANSIÓN VS CONTRACCIÓN DE USO ━━━

[EU-01] Cliente que usa solo el 20% de las funcionalidades disponibles.
Señal: Solo usa el módulo básico, nunca activó las funciones avanzadas incluidas en su plan.
→ {"nivel_riesgo":"Medio","probabilidad_churn_porcentaje":51,
"razon_principal":"Bajo aprovechamiento del plan indica que el cliente no percibe el valor completo — riesgo de downsell o cancelación en renovación",
"accion_recomendada_para_el_gestor":"Enviar caso de uso específico para su industria que demuestre el valor de las funciones no usadas. Ofrecer sesión de 30 minutos para mostrar 3 funciones que le ahorrarían tiempo esta semana."}

[EU-02] Cliente que expande uso activamente y pide nuevas funciones.
Señal: "¿Cuándo sale la integración con Slack? La necesitamos para el Q2."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":8,
"razon_principal":"Engagement activo con roadmap — cliente construyendo procesos sobre la plataforma",
"accion_recomendada_para_el_gestor":"Incluir a este cliente en el beta de la integración con Slack. Los clientes que participan en betas tienen tasa de renovación 3x mayor."}

[EU-03] Cliente que cancela usuarios pero mantiene el plan.
Señal: Redujo de 12 usuarios a 4 usuarios activos en el último mes.
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":71,
"razon_principal":"Reducción de 67% en usuarios activos indica contracción del uso — primer paso hacia cancelación",
"accion_recomendada_para_el_gestor":"Preguntar directamente por qué se redujeron los usuarios. Puede ser restructuración interna o migración. La respuesta define la estrategia de retención."}

[EU-04] Cliente que usa la herramienta para un caso de uso no previsto y lo hace bien.
Señal: "Encontramos una forma de usar el módulo de reportes para gestionar nuestro inventario."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":6,
"razon_principal":"Uso creativo no previsto indica alto nivel de apropiación de la herramienta",
"accion_recomendada_para_el_gestor":"Documentar el caso de uso. Preguntar si estarían dispuestos a ser caso de éxito. Los clientes que inventan usos propios son los mejores embajadores."}

[EU-05] Cliente en plan básico que consistentemente supera los límites.
Señal: Tercer mes consecutivo rozando el límite de registros del plan Starter.
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":12,
"razon_principal":"Uso intensivo del plan básico — oportunidad de upgrade antes de que los límites generen fricción",
"accion_recomendada_para_el_gestor":"Proactivamente ofrecer upgrade antes de que lleguen al límite. Calcular el costo adicional vs el costo de perder datos o funcionalidad. El upgrade proactivo tiene 60% más de conversión que el reactivo."}

━━━ CATEGORÍA 7: CAMBIOS ORGANIZACIONALES INTERNOS ━━━

[CO-01] Cambio de CEO o director que tomó la decisión de compra original.
Señal: "Hola, soy el nuevo Director de Operaciones. Me estoy poniendo al día con todas las herramientas."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":66,
"razon_principal":"Cambio de decisor económico — el nuevo director no tiene apego a la herramienta y puede querer imponer sus propias soluciones",
"accion_recomendada_para_el_gestor":"Reunión ejecutiva con el nuevo director en los próximos 7 días. Presentar el valor generado desde el inicio con métricas concretas. Construir la relación desde cero."}

[CO-02] Fusión o adquisición del cliente.
Señal: "Nos están adquiriendo. Por ahora todo sigue igual pero habrá cambios."
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":82,
"razon_principal":"M&A genera consolidación de herramientas — la empresa adquirente probablemente ya tiene su propia solución",
"accion_recomendada_para_el_gestor":"Contactar al equipo de integración de la empresa adquirente directamente. Posicionar PostSale como la herramienta a mantener. Tener datos de ROI listos para la decisión de consolidación."}

[CO-03] Recorte de presupuesto anunciado internamente.
Señal: "Estamos revisando todos los gastos de software para el próximo trimestre."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":73,
"razon_principal":"Revisión de gastos de software — herramientas sin ROI claramente documentado son las primeras en cortarse",
"accion_recomendada_para_el_gestor":"Enviar informe de ROI no solicitado antes de que llegue la reunión de presupuesto. El cliente necesita un argumento para defenderlo internamente."}

[CO-04] El campeón interno que defendía la herramienta se fue de la empresa.
Señal: "Juan renunció. Era quien más usaba y defendía el sistema."
→ {"nivel_riesgo":"Crítico","probabilidad_churn_porcentaje":84,
"razon_principal":"Pérdida del campeón interno — sin un defensor activo la herramienta queda expuesta a la siguiente revisión de presupuesto",
"accion_recomendada_para_el_gestor":"Identificar y cultivar un nuevo campeón en los próximos 30 días. Ofrecer capacitación gratuita al sucesor. Sin campeón interno no hay renovación."}

[CO-05] Cliente en proceso de transformación digital con consultor externo.
Señal: "Contratamos a una consultora para revisar toda nuestra stack tecnológica."
→ {"nivel_riesgo":"Alto","probabilidad_churn_porcentaje":69,
"razon_principal":"Consultor externo evaluando la stack — terceros sin relación con la herramienta frecuentemente recomiendan cambios para justificar sus honorarios",
"accion_recomendada_para_el_gestor":"Solicitar participar en el proceso de evaluación. Ofrecer documentación técnica y de ROI para el consultor. Si no se puede participar, asegurarse de que el campeón interno tenga argumentos sólidos."}

━━━ CATEGORÍA 8: SEÑALES POSITIVAS GENUINAS ━━━

[SP-01] Cliente refiere activamente a otros potenciales clientes.
Señal: "Le mencioné su herramienta a dos colegas de otra empresa. Los van a contactar."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":3,
"razon_principal":"Referido activo — el cliente no solo está satisfecho sino que pone su reputación detrás del producto",
"accion_recomendada_para_el_gestor":"Agradecer personalmente y ofrecer beneficio por referido. Este cliente es candidato a caso de éxito y potencial embajador de marca."}

[SP-02] Cliente publica caso de éxito o menciona la herramienta públicamente.
Señal: "Publicamos un post en LinkedIn sobre cómo mejoramos nuestra retención usando PostSale."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":2,
"razon_principal":"Advocacy público — el cliente construyó su reputación sobre el uso de la herramienta",
"accion_recomendada_para_el_gestor":"Amplificar el contenido desde los canales propios. Este cliente es prácticamente irretirable a corto plazo. Enfocar energía en asegurar la renovación con anticipación."}

[SP-03] Cliente que paga anticipadamente o pide facturación anual.
Señal: "¿Pueden facturarnos el año completo? Preferimos pagar todo ahora."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":5,
"razon_principal":"Pago anticipado anual indica confianza total y compromiso a largo plazo",
"accion_recomendada_para_el_gestor":"Agradecer y ofrecer descuento por pago anual. Agendar QBR trimestral para asegurar que el valor percibido se mantenga alto durante todo el año."}

[SP-04] Cliente que participa activamente en el beta de nuevas funciones.
Señal: "Sí, queremos ser beta testers de la nueva integración con Zapier."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":7,
"razon_principal":"Participación en beta indica alto compromiso y co-construcción del producto",
"accion_recomendada_para_el_gestor":"Asignar contacto técnico dedicado para el beta. Los beta testers que tienen buena experiencia se convierten en los mejores casos de éxito."}

[SP-05] Cliente que aumenta el plan voluntariamente.
Señal: "Queremos pasar al plan Enterprise. Nuestro equipo creció y necesitamos más capacidad."
→ {"nivel_riesgo":"Bajo","probabilidad_churn_porcentaje":4,
"razon_principal":"Upgrade voluntario indica que el cliente percibe valor claro y está construyendo su operación sobre la plataforma",
"accion_recomendada_para_el_gestor":"Facilitar el upgrade inmediatamente. Ofrecer sesión de onboarding para las nuevas funcionalidades del plan Enterprise. Un cliente que hace upgrade tiene probabilidad de renovación del 94%."}

═══════════════════════════════════════════════════════
INSTRUCCIONES DE ANÁLISIS
═══════════════════════════════════════════════════════

Al analizar un cliente real, aplicá estos principios:

POTENCIADORES DE RIESGO (cada uno suma):
+ Ventana contractual < 60 días: +20% probabilidad
+ Señal de migración activa (exportación, comparación): +25% probabilidad
+ Error técnico sin resolver > 48h: +15% probabilidad
+ Cambio de decisor o campeón interno: +20% probabilidad
+ Inactividad > 10 días en usuario previamente activo: +15% probabilidad
+ Ticket repetido (mismo problema > 2 veces): +20% probabilidad

REDUCTORES DE RIESGO (cada uno resta):
- Referido activo o caso de éxito publicado: -20% probabilidad
- Upgrade voluntario reciente: -15% probabilidad
- Participación en beta o roadmap: -10% probabilidad
- Pago anticipado anual: -15% probabilidad
- NPS alto documentado: -10% probabilidad

REGLA DE ORO: cuando hay señales contradictorias (cliente educado + señal de migración),
siempre prevalece la señal de comportamiento sobre el tono del mensaje.

RESPONDÉ SOLO CON EL JSON. Sin texto adicional."""


# ---------------------------------------------------------------------------
# MOTOR DE IA
# ---------------------------------------------------------------------------

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
    version="0.7.0",
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
    Si el riesgo es Alto o Crítico, crea una tarea automática.
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

    # Llamar al motor de IA con RAG
    resultado = await analizar_con_rag(
        nombre=cliente.nombre,
        plan=cliente.plan_actual.value,
        interacciones=datos.interacciones,
        coleccion=coleccion_rag,
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

    # Crear tarea automática si el riesgo es Alto o Crítico
    if resultado["nivel_riesgo"] in {"Alto", "Crítico"}:
        tarea = TareaDB(
            cliente_id=cliente_id,
            analisis_id=analisis_db.id,
            nivel_riesgo=resultado["nivel_riesgo"],
            accion_sugerida=resultado["accion_recomendada_para_el_gestor"],
            estado="pendiente",
        )
        session.add(tarea)
        session.commit()
        log.info(
            f"Tarea creada para {cliente.nombre} "
            f"({resultado['nivel_riesgo']}) — ID: {tarea.id}"
        )

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


# --- GET /tareas/resumen — estadísticas de tareas ---
# IMPORTANTE: este endpoint debe ir ANTES de /tareas/{tarea_id}
# para que FastAPI no interprete "resumen" como un tarea_id entero.

@app.get("/tareas/resumen", response_model=dict)
def resumen_tareas(session: Session = Depends(get_session)):
    """
    Resumen de tareas para el dashboard del gestor.
    Muestra cuántas tareas hay por estado y nivel de riesgo.
    """
    tareas = session.exec(select(TareaDB)).all()

    pendientes = [t for t in tareas if t.estado == "pendiente"]
    en_curso = [t for t in tareas if t.estado == "en_curso"]
    completadas = [t for t in tareas if t.estado == "completada"]
    descartadas = [t for t in tareas if t.estado == "descartada"]

    criticas_pendientes = [
        t for t in pendientes if t.nivel_riesgo == "Crítico"
    ]

    return {
        "total": len(tareas),
        "pendientes": len(pendientes),
        "en_curso": len(en_curso),
        "completadas": len(completadas),
        "descartadas": len(descartadas),
        "criticas_pendientes": len(criticas_pendientes),
        "tasa_resolucion": round(
            len(completadas) / len(tareas) * 100 if tareas else 0
        ),
    }


# --- GET /tareas — listar tareas ---

@app.get("/tareas", response_model=list[TareaResponse])
def listar_tareas(
    estado: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """
    Lista todas las tareas del sistema.
    Filtrable por estado: pendiente, en_curso, completada, descartada.
    Ordenadas por urgencia (Crítico primero) y fecha de creación.
    """
    query = select(TareaDB)
    if estado:
        query = query.where(TareaDB.estado == estado)

    tareas = session.exec(query).all()

    orden_riesgo = {"Crítico": 0, "Alto": 1, "Medio": 2, "Bajo": 3}
    tareas_ordenadas = sorted(
        tareas,
        key=lambda t: (orden_riesgo.get(t.nivel_riesgo, 4), t.fecha_creacion),
    )

    resultado = []
    for tarea in tareas_ordenadas:
        cliente = session.get(ClienteDB, tarea.cliente_id)
        resultado.append(TareaResponse(
            id=tarea.id,
            cliente_id=tarea.cliente_id,
            cliente_nombre=cliente.nombre if cliente else "—",
            analisis_id=tarea.analisis_id,
            nivel_riesgo=tarea.nivel_riesgo,
            accion_sugerida=tarea.accion_sugerida,
            accion_tomada=tarea.accion_tomada,
            resultado_real=tarea.resultado_real,
            estado=tarea.estado,
            fecha_creacion=tarea.fecha_creacion,
            fecha_cierre=tarea.fecha_cierre,
            gestor=tarea.gestor,
            notas=tarea.notas,
        ))

    return resultado


# --- PATCH /tareas/{tarea_id} — actualizar tarea ---

@app.patch("/tareas/{tarea_id}", response_model=TareaResponse)
def actualizar_tarea(
    tarea_id: int,
    datos: TareaInput,
    session: Session = Depends(get_session),
):
    """
    Actualiza una tarea con lo que hizo el gestor y el resultado real.
    Si el estado pasa a completada o descartada, registra la fecha de cierre.
    """
    tarea = session.get(TareaDB, tarea_id)
    if not tarea:
        raise HTTPException(
            status_code=404,
            detail=f"Tarea {tarea_id} no encontrada",
        )

    if datos.accion_tomada is not None:
        tarea.accion_tomada = datos.accion_tomada
    if datos.resultado_real is not None:
        tarea.resultado_real = datos.resultado_real
    if datos.estado is not None:
        tarea.estado = datos.estado
        if datos.estado in {"completada", "descartada"}:
            tarea.fecha_cierre = datetime.now().isoformat()
    if datos.gestor is not None:
        tarea.gestor = datos.gestor
    if datos.notas is not None:
        tarea.notas = datos.notas

    session.add(tarea)
    session.commit()
    session.refresh(tarea)

    cliente = session.get(ClienteDB, tarea.cliente_id)
    log.info(
        f"Tarea {tarea_id} actualizada — "
        f"Estado: {tarea.estado} — Cliente: {cliente.nombre if cliente else '?'}"
    )

    return TareaResponse(
        id=tarea.id,
        cliente_id=tarea.cliente_id,
        cliente_nombre=cliente.nombre if cliente else "—",
        analisis_id=tarea.analisis_id,
        nivel_riesgo=tarea.nivel_riesgo,
        accion_sugerida=tarea.accion_sugerida,
        accion_tomada=tarea.accion_tomada,
        resultado_real=tarea.resultado_real,
        estado=tarea.estado,
        fecha_creacion=tarea.fecha_creacion,
        fecha_cierre=tarea.fecha_cierre,
        gestor=tarea.gestor,
        notas=tarea.notas,
    )


# --- DELETE /tareas/{tarea_id} — descartar tarea ---

@app.delete("/tareas/{tarea_id}", response_model=dict)
def descartar_tarea(
    tarea_id: int,
    session: Session = Depends(get_session),
):
    """Descarta una tarea sin eliminarla — queda en el historial."""
    tarea = session.get(TareaDB, tarea_id)
    if not tarea:
        raise HTTPException(
            status_code=404,
            detail=f"Tarea {tarea_id} no encontrada",
        )

    tarea.estado = "descartada"
    tarea.fecha_cierre = datetime.now().isoformat()
    session.add(tarea)
    session.commit()

    return {"mensaje": f"Tarea {tarea_id} descartada", "id": tarea_id}


# --- GET / — health check ---

@app.get("/")
def health_check():
    """Verifica que la API está funcionando."""
    return {
        "estado": "ok",
        "servicio": "PostSale API",
        "version": "0.7.0",
        "docs": "/docs",
    }
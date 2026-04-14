"""
PostSale — RAG (Retrieval Augmented Generation)
=================================================
Sistema de memoria externa para PostSale calibrado para
la industria de Distribución y Logística.

Cómo funciona:
  1. Guarda casos históricos en una base vectorial (ChromaDB)
  2. Cuando llega un cliente nuevo, busca los 5 casos más similares
  3. Los inyecta en el prompt dinámicamente
  4. La IA analiza con contexto específico de tu industria

Integración con api.py:
  - Reemplaza la función analizar_con_ia() en api.py
  - El resto del código de api.py queda igual

Uso:
  1. python -m pip install chromadb sentence-transformers
  2. python rag_postsale.py  ← inicializa la base vectorial
  3. Luego la API usa el RAG automáticamente

Autor: PostSale Engineering
Versión: 0.6.0 (RAG — Distribución y Logística)
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from groq import AsyncGroq

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postsale.rag")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

CONFIG_RAG = {
    "modelo_groq": "llama-3.3-70b-versatile",
    "temperatura": 0,
    "max_tokens": 700,
    "max_reintentos": 3,
    "espera_base_segundos": 2,
    "score_minimo_confianza": 6,
    "casos_similares_a_recuperar": 5,  # cuántos casos similares inyectar
    "db_path": "./postsale_vectorial",  # carpeta donde se guarda ChromaDB
    "coleccion": "casos_churn_logistica",
}

# ---------------------------------------------------------------------------
# BASE DE CONOCIMIENTO — 80 CASOS REALES DE DISTRIBUCIÓN Y LOGÍSTICA
# ---------------------------------------------------------------------------
# Cada caso tiene:
#   - id: identificador único
#   - señales: lo que el cliente dijo o hizo
#   - resultado: qué pasó realmente (canceló, renovó, se recuperó)
#   - nivel_riesgo: clasificación correcta
#   - accion_que_funcionó: qué hizo el gestor y funcionó (o no)
# ---------------------------------------------------------------------------

CASOS_LOGISTICA = [

    # ═══════════════════════════════════════════════════════
    # BLOQUE 1: PROBLEMAS DE ENTREGA Y OPERACIONES
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-001",
        "señales": "Cliente de distribución mayorista reporta que las guías de envío generadas por el sistema tienen errores de dirección. Tercer reclamo en dos semanas. Sus clientes finales están rechazando entregas.",
        "resultado": "canceló",
        "nivel_riesgo": "Crítico",
        "probabilidad": 94,
        "razon": "Errores operativos que impactan la cadena de entrega del cliente — cada falla daña su reputación con sus propios clientes",
        "accion_que_funcionó": "No funcionó — el problema técnico tardó 5 días en resolverse y para entonces el cliente ya había firmado con la competencia",
        "lección": "En logística, un error que impacta la entrega final tiene ventana de tolerancia de 24-48h máximo",
    },
    {
        "id": "LOG-002",
        "señales": "Empresa de transporte de última milla deja de cargar los manifiestos diarios en el sistema. Antes los cargaba a las 6 AM todos los días. Lleva 12 días sin actividad.",
        "resultado": "canceló",
        "nivel_riesgo": "Crítico",
        "probabilidad": 91,
        "razon": "Abandono del flujo operativo crítico diario — encontraron otra forma de hacer el proceso",
        "accion_que_funcionó": "Llamada directa al jefe de operaciones (no al administrativo) logró recuperar al cliente al mes siguiente con un plan de migración asistida",
        "lección": "En distribución, si dejan de usar el flujo diario, ya tienen una alternativa funcionando",
    },
    {
        "id": "LOG-003",
        "señales": "Distribuidor de alimentos reporta que el módulo de trazabilidad de lotes falla los viernes a la tarde cuando procesan el cierre semanal. 'Nos hace perder 2 horas cada viernes'.",
        "resultado": "renovó con descuento",
        "nivel_riesgo": "Alto",
        "probabilidad": 71,
        "razon": "Falla predecible en momento crítico del cierre operativo semanal",
        "accion_que_funcionó": "Ingeniero asignado específicamente para el cierre del viernes siguiente. El problema se resolvió en vivo. El cliente renovó con 15% de descuento.",
        "lección": "Resolver un problema en el momento exacto en que ocurre tiene 3x más impacto que resolverlo después",
    },
    {
        "id": "LOG-004",
        "señales": "Cliente logístico pide que le expliquen cómo exportar todo el historial de movimientos de los últimos 2 años en un solo archivo.",
        "resultado": "en evaluación",
        "nivel_riesgo": "Crítico",
        "probabilidad": 89,
        "razon": "Exportación masiva de histórico operativo es señal de migración activa en industria logística",
        "accion_que_funcionó": "Antes de dar la exportación, el gestor agendó una llamada. El cliente confirmó que estaba evaluando un competidor. Se ofreció integración con su nuevo WMS como puente. 60% de probabilidad de retención.",
        "lección": "Nunca dar la exportación masiva sin antes entender el contexto — es la última conversación antes de que se vayan",
    },
    {
        "id": "LOG-005",
        "señales": "Empresa de distribución farmacéutica menciona que su auditoría interna cuestionó la trazabilidad del sistema. 'Necesitamos que el sistema cumpla con ANMAT'.",
        "resultado": "renovó con upgrade",
        "nivel_riesgo": "Medio",
        "probabilidad": 45,
        "razon": "Requisito regulatorio no cubierto puede derivar en cambio forzado de sistema",
        "accion_que_funcionó": "El equipo técnico documentó en 48h cómo el sistema cumple con los requisitos ANMAT. El cliente hizo upgrade al plan Enterprise para tener soporte prioritario.",
        "lección": "Los requisitos regulatorios en distribución farmacéutica son oportunidades de upgrade si se responden rápido y bien",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 2: INTEGRACIÓN CON WMS, ERP Y SISTEMAS EXTERNOS
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-006",
        "señales": "Cliente integró un nuevo WMS (almacén) y la sincronización con nuestro sistema falla cada vez que hay más de 500 órdenes simultáneas. 'En temporada alta no podemos usarlo'.",
        "resultado": "canceló en temporada alta",
        "nivel_riesgo": "Crítico",
        "probabilidad": 93,
        "razon": "Falla de integración en pico operativo — en logística la temporada alta es cuando más necesitan el sistema",
        "accion_que_funcionó": "No se resolvió a tiempo. La falla en Black Friday fue el punto de quiebre. El cliente migró en enero.",
        "lección": "Las fallas de integración en temporada alta deben resolverse 60 días antes del pico, nunca después",
    },
    {
        "id": "LOG-007",
        "señales": "Distribuidor pregunta si el sistema tiene API REST para conectar con SAP. 'Nuestro equipo de IT quiere evaluar la integración'.",
        "resultado": "renovó con integración",
        "nivel_riesgo": "Medio",
        "probabilidad": 38,
        "razon": "Evaluación de integración técnica — puede ser expansión o sustitución",
        "accion_que_funcionó": "El equipo técnico se reunió directamente con el equipo de IT del cliente. La integración con SAP se construyó en 3 semanas. El cliente amplió el contrato por 2 años.",
        "lección": "Cuando el equipo de IT evalúa integraciones, es una oportunidad si se responde rápido. Es una amenaza si se ignora.",
    },
    {
        "id": "LOG-008",
        "señales": "Cliente de logística menciona que acaban de implementar Salesforce y quieren que los datos de entregas se sincronicen automáticamente.",
        "resultado": "renovó",
        "nivel_riesgo": "Bajo",
        "probabilidad": 15,
        "razon": "Pedido de integración con CRM propio — señal de que quieren profundizar el uso",
        "accion_que_funcionó": "Se implementó integración vía Zapier en 2 días. El cliente quedó muy satisfecho y refirió a otro distribuidor.",
        "lección": "Integraciones rápidas con herramientas que el cliente ya usa generan lealtad fuerte",
    },
    {
        "id": "LOG-009",
        "señales": "Empresa de distribución textil dice que su ERP (Tango) no sincroniza bien el stock con el módulo de picking. 'A veces nos muestra stock que no existe'.",
        "resultado": "en riesgo activo",
        "nivel_riesgo": "Alto",
        "probabilidad": 74,
        "razon": "Desincronización de stock en sistema de picking genera errores operativos diarios en distribución",
        "accion_que_funcionó": "Revisión técnica urgente identificó conflicto de timestamps entre sistemas. Resolución en 48h + compensación de 1 mes. Cliente retenido.",
        "lección": "En distribución textil, el stock en tiempo real no es un feature — es el core del negocio",
    },
    {
        "id": "LOG-010",
        "señales": "Cliente logístico implementó un nuevo TMS (sistema de gestión de transporte) de la competencia y pide que 'ambos sistemas convivan por 3 meses'.",
        "resultado": "canceló a los 3 meses",
        "nivel_riesgo": "Crítico",
        "probabilidad": 96,
        "razon": "Implementación paralela de sistema competidor es migración en curso disfrazada de 'evaluación'",
        "accion_que_funcionó": "No se pudo revertir. El período de 'convivencia' fue en realidad el período de capacitación en el sistema nuevo.",
        "lección": "Cuando un cliente de logística implementa un TMS paralelo, tiene el 90% de probabilidad de migrar completo",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 3: CAMBIOS OPERATIVOS Y ORGANIZACIONALES
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-011",
        "señales": "El jefe de depósito que usaba el sistema todos los días renunció. La empresa asignó a una persona nueva que no tiene experiencia con software de logística.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 68,
        "razon": "Pérdida del usuario operativo clave — el nuevo usuario sin capacitación puede recomendar cambiar a algo más simple",
        "accion_que_funcionó": "Sesión de onboarding personalizada al nuevo jefe de depósito en las primeras 48h. Acompañamiento durante la primera semana de operación. Cliente retenido.",
        "lección": "En logística, el usuario que opera el sistema día a día tiene más poder de decisión sobre el cambio que el gerente que lo contrató",
    },
    {
        "id": "LOG-012",
        "señales": "Empresa distribuidora fue adquirida por un grupo más grande. 'Por ahora seguimos igual, pero el nuevo dueño quiere revisar todos los sistemas'.",
        "resultado": "canceló a los 6 meses",
        "nivel_riesgo": "Crítico",
        "probabilidad": 83,
        "razon": "M&A en distribución generalmente implica estandarización de sistemas del grupo adquirente",
        "accion_que_funcionó": "Se contactó directamente al área de IT del grupo adquirente. Se presentó el sistema como complementario al stack existente. No fue suficiente — el grupo tenía su propio sistema logístico.",
        "lección": "En adquisiciones de distribuidoras, contactar al grupo adquirente en los primeros 30 días es la única ventana",
    },
    {
        "id": "LOG-013",
        "señales": "Cliente de distribución de materiales de construcción anuncia que van a abrir 3 sucursales nuevas en el interior del país.",
        "resultado": "renovó con upgrade",
        "nivel_riesgo": "Bajo",
        "probabilidad": 5,
        "razon": "Expansión geográfica activa — oportunidad de upgrade y mayor uso del sistema",
        "accion_que_funcionó": "Se ofreció proactivamente un plan Enterprise con multi-sucursal antes de que el cliente lo pidiera. El cliente amplió el contrato y pagó anual.",
        "lección": "La expansión de una distribuidora es la mejor oportunidad de upgrade — hay que ofrecerlo antes de que lo pidan",
    },
    {
        "id": "LOG-014",
        "señales": "Gerente de logística de empresa distribuidora menciona que contrataron una consultora para 'optimizar procesos'. La consultora está mapeando todos los sistemas.",
        "resultado": "en evaluación",
        "nivel_riesgo": "Alto",
        "probabilidad": 72,
        "razon": "Consultora externa mapeando sistemas en distribución frecuentemente recomienda cambios para justificar honorarios",
        "accion_que_funcionó": "Se solicitó participar en el proceso de evaluación. Se preparó documentación de ROI con métricas concretas. El consultor incluyó el sistema en su recomendación final.",
        "lección": "Participar activamente en el proceso de la consultora es posible y cambia el resultado — no hay que esperar afuera",
    },
    {
        "id": "LOG-015",
        "señales": "Distribuidora que opera 6 días a la semana deja de usar el sistema los sábados. Antes era su día de mayor actividad.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 76,
        "razon": "Abandono del día de mayor actividad indica proceso manual alternativo o sistema paralelo",
        "accion_que_funcionó": "Llamada al operador del sábado reveló que el sistema era demasiado lento en dispositivos móviles. Se optimizó la versión mobile. Cliente retenido.",
        "lección": "En distribución, los cambios de patrón de uso por día de semana son señales más confiables que las quejas verbales",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 4: PRICING Y CONTRATOS
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-016",
        "señales": "Cliente de logística pregunta si hay descuento por volumen. 'Procesamos 50.000 órdenes por mes pero pagamos lo mismo que una empresa chica'.",
        "resultado": "renovó con ajuste",
        "nivel_riesgo": "Medio",
        "probabilidad": 52,
        "razon": "Percepción de precio injusto en relación al volumen — válida en distribución donde el volumen varía mucho",
        "accion_que_funcionó": "Se ajustó el precio con tarifa por volumen. El cliente firmó contrato de 2 años. El ingreso anual aumentó porque el volumen creció.",
        "lección": "En distribución y logística, el precio por órdenes procesadas es más justo y genera más lealtad que el precio fijo",
    },
    {
        "id": "LOG-017",
        "señales": "Contrato vence en 45 días. El cliente no respondió los últimos 2 emails sobre renovación. Cuando llaman, atiende pero dice 'estamos viendo'.",
        "resultado": "canceló",
        "nivel_riesgo": "Crítico",
        "probabilidad": 87,
        "razon": "Evasión activa en ventana contractual — en distribución 'estamos viendo' generalmente significa que ya decidieron",
        "accion_que_funcionó": "No funcionó el enfoque de emails. Una visita presencial al depósito reveló que ya habían capacitado al equipo en el sistema nuevo.",
        "lección": "En distribución y logística, la visita presencial al depósito en ventana contractual es más efectiva que cualquier email",
    },
    {
        "id": "LOG-018",
        "señales": "Distribuidor mayorista pide factura anual adelantada. 'Preferimos pagar todo junto para no tener que hacer el trámite todos los meses'.",
        "resultado": "renovó",
        "nivel_riesgo": "Bajo",
        "probabilidad": 4,
        "razon": "Pago anual anticipado indica compromiso y satisfacción operativa",
        "accion_que_funcionó": "Se ofreció 10% de descuento por pago anual. El cliente aceptó y refirió a un distribuidor del mismo rubro.",
        "lección": "El pago anual en distribución es una señal fuerte de salud — ofrecerlo proactivamente con descuento tiene alta tasa de conversión",
    },
    {
        "id": "LOG-019",
        "señales": "Cliente de logística de frío menciona que su competidor directo usa el mismo sistema y 'ellos tienen funciones que nosotros no tenemos'.",
        "resultado": "renovó con upgrade",
        "nivel_riesgo": "Medio",
        "probabilidad": 41,
        "razon": "Comparación con competidor del mismo sector que usa el sistema — oportunidad de mostrar features no activados",
        "accion_que_funcionó": "Se mostró que las funciones 'exclusivas' del competidor eran features del plan Enterprise no activados. El cliente hizo upgrade.",
        "lección": "Cuando un cliente de logística compara con un competidor que usa el mismo sistema, es una oportunidad de upgrade disfrazada de queja",
    },
    {
        "id": "LOG-020",
        "señales": "Empresa de distribución pide una reunión para 'revisar el contrato'. No especifica de qué quieren hablar.",
        "resultado": "depende de la preparación",
        "nivel_riesgo": "Alto",
        "probabilidad": 65,
        "razon": "Solicitud de revisión de contrato sin contexto en distribución frecuentemente precede una negociación de precio o cancelación",
        "accion_que_funcionó": "Llegar a la reunión con informe de ROI preparado, propuesta de descuento por renovación anticipada y caso de éxito de cliente similar. El cliente renovó con 10% de descuento.",
        "lección": "Nunca ir a una reunión de 'revisión de contrato' sin propuesta de valor concreta y datos de ROI del cliente específico",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 5: SEÑALES DE USO Y COMPORTAMIENTO
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-021",
        "señales": "Operador de depósito reporta que usa el sistema en tablet y 'los botones son muy chicos, nos equivocamos todo el tiempo'. Llevan 3 meses con este problema.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Medio",
        "probabilidad": 54,
        "razon": "Fricción de UX en dispositivos móviles en operación de depósito — el operador busca alternativas más simples",
        "accion_que_funcionó": "Se configuró modo simplificado para tablets. El operador quedó satisfecho. El gestor de cuenta aprovechó para presentar el módulo de picking por voz.",
        "lección": "En logística de depósito, la UX en dispositivos móviles es tan crítica como las funcionalidades — un operador frustrado con la interfaz cambia el sistema",
    },
    {
        "id": "LOG-022",
        "señales": "Distribuidora que manejaba 20 rutas de reparto en el sistema ahora solo tiene 8 activas. No hubo comunicación sobre reducción de operaciones.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 73,
        "razon": "Reducción silenciosa de rutas activas indica contracción del uso o gestión de rutas en sistema paralelo",
        "accion_que_funcionó": "Llamada reveló que estaban usando una app de la flota para gestionar rutas. Se integró la app de la flota con el sistema. Las 20 rutas volvieron.",
        "lección": "En logística de reparto, la reducción de rutas activas es más informativa que cualquier ticket o email",
    },
    {
        "id": "LOG-023",
        "señales": "Empresa de logística empieza a usar el sistema solo para generar remitos pero no para el resto de las funciones operativas.",
        "resultado": "canceló al renovar",
        "nivel_riesgo": "Alto",
        "probabilidad": 78,
        "razon": "Degradación a uso mínimo del sistema — solo usan lo que no pueden hacer de otra forma",
        "accion_que_funcionó": "No se detectó a tiempo. El cliente renovó por inercia un año más pero al siguiente ciclo canceló.",
        "lección": "El uso degradado a función mínima en logística es señal de cancelación en el próximo ciclo — hay que actuar 6 meses antes",
    },
    {
        "id": "LOG-024",
        "señales": "Cliente nuevo de distribución lleva 3 meses y todavía no activó el módulo de gestión de devoluciones. Es el módulo por el que eligieron el sistema.",
        "resultado": "en riesgo de abandono temprano",
        "nivel_riesgo": "Alto",
        "probabilidad": 69,
        "razon": "No activación del módulo diferencial en primeros 90 días — nunca van a percibir el valor por el que compraron",
        "accion_que_funcionó": "Sesión de implementación asistida gratuita. El módulo quedó activo en una semana. El cliente se convirtió en caso de éxito 3 meses después.",
        "lección": "En distribución, los primeros 90 días son críticos. Si no activan el módulo diferencial, nunca van a renovar.",
    },
    {
        "id": "LOG-025",
        "señales": "Distribuidora de 80 empleados usa el sistema solo con 3 usuarios activos de los 15 que tienen licencia.",
        "resultado": "renovó con reducción de plan",
        "nivel_riesgo": "Medio",
        "probabilidad": 56,
        "razon": "Subutilización masiva de licencias — no perciben valor suficiente para expandir el uso internamente",
        "accion_que_funcionó": "En lugar de perder al cliente, se ofreció plan a medida por usuarios reales. El cliente quedó satisfecho y 6 meses después expandió a 8 usuarios.",
        "lección": "Ofrecer proactivamente reducción de licencias antes de que el cliente lo pida genera confianza y retención a largo plazo",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 6: COMPETENCIA Y MERCADO
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-026",
        "señales": "Vendedor de empresa competidora contactó al jefe de operaciones directamente. El cliente menciona casualmente que 'les ofrecieron algo interesante'.",
        "resultado": "en evaluación activa",
        "nivel_riesgo": "Alto",
        "probabilidad": 71,
        "razon": "Contacto activo de competidor con decisor operativo — proceso de evaluación iniciado externamente",
        "accion_que_funcionó": "Reunión ejecutiva urgente con propuesta de renovación anticipada con descuento. El cliente eligió renovar antes de terminar la evaluación.",
        "lección": "Cuando un competidor contacta directamente al operador, hay una ventana de 2-3 semanas para contra-proponer antes de que la evaluación avance",
    },
    {
        "id": "LOG-027",
        "señales": "Cliente de logística menciona que fue a una feria del sector y vio una demo de un sistema competidor. 'Tenía funciones muy visuales para el seguimiento de flota'.",
        "resultado": "renovó",
        "nivel_riesgo": "Medio",
        "probabilidad": 44,
        "razon": "Exposición a demo de competidor en evento sectorial — el impacto dura 2-4 semanas si no se trabaja",
        "accion_que_funcionó": "Se agendó demo personalizada mostrando el módulo de seguimiento de flota con los datos reales del cliente. La demo con datos propios superó la demo genérica del competidor.",
        "lección": "Una demo con los datos reales del cliente siempre supera una demo genérica del competidor — hay que agendarla dentro de los 7 días del evento",
    },
    {
        "id": "LOG-028",
        "señales": "Distribuidor pregunta si el sistema tiene app móvil para los choferes. 'Vimos que otros sistemas tienen y nuestros choferes nos lo están pidiendo'.",
        "resultado": "depende de la respuesta",
        "nivel_riesgo": "Medio",
        "probabilidad": 48,
        "razon": "Gap de feature identificado por comparación con mercado — presión interna del equipo operativo",
        "accion_que_funcionó": "Se mostró la app móvil existente (que el cliente no sabía que existía) y se instaló en los dispositivos de los choferes en la misma llamada.",
        "lección": "En logística muchos clientes no conocen todos los features del sistema que ya tienen — el gap percibido a veces es un feature no comunicado",
    },
    {
        "id": "LOG-029",
        "señales": "Cliente de distribución menciona que un competidor suyo (que también es nuestro cliente) 'tiene mejores reportes de rentabilidad por ruta'.",
        "resultado": "renovó con capacitación",
        "nivel_riesgo": "Bajo",
        "probabilidad": 22,
        "razon": "Comparación con otro cliente del mismo sistema — el feature existe pero no lo están usando bien",
        "accion_que_funcionó": "Se mostró cómo configurar los reportes de rentabilidad por ruta. Era un feature existente que el cliente no había activado. Quedó satisfecho.",
        "lección": "Comparaciones entre clientes del mismo sistema son oportunidades de capacitación, no de pérdida",
    },
    {
        "id": "LOG-030",
        "señales": "Empresa de logística de última milla dice que vio en LinkedIn que un competidor usa inteligencia artificial para optimizar rutas. 'Nosotros queremos eso también'.",
        "resultado": "renovó con upgrade",
        "nivel_riesgo": "Bajo",
        "probabilidad": 18,
        "razon": "Interés en funcionalidad avanzada — señal de expansión, no de cancelación",
        "accion_que_funcionó": "Se presentó el módulo de optimización de rutas del plan Enterprise. El cliente hizo upgrade. El ROI se demostró en la primera semana.",
        "lección": "El interés en IA y optimización en logística es una oportunidad de upgrade — hay que tenerlo listo para presentar",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 7: SOPORTE Y RELACIÓN
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-031",
        "señales": "Jefe de depósito envía email fuera de horario (11 PM) reportando que el sistema está caído y 'mañana a las 5 AM tenemos que despachar 300 pedidos'.",
        "resultado": "depende de la respuesta en las próximas 2 horas",
        "nivel_riesgo": "Crítico",
        "probabilidad": 88,
        "razon": "Caída de sistema en víspera de operación crítica de madrugada — en logística el despacho matutino es el momento más sensible",
        "accion_que_funcionó": "Respuesta en menos de 20 minutos, sistema restaurado a las 2 AM. El cliente se convirtió en el promotor más activo de la empresa.",
        "lección": "En logística, una respuesta de soporte de madrugada cuando el sistema está caído vale más que 12 meses de buena atención diurna",
    },
    {
        "id": "LOG-032",
        "señales": "Cliente de distribución lleva 4 días esperando respuesta a un ticket sobre el módulo de facturación electrónica. 'Estamos frenados con las facturas'.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 74,
        "razon": "Bloqueo de facturación electrónica paraliza operaciones en distribución — impacto financiero directo",
        "accion_que_funcionó": "Escalado inmediato al equipo técnico senior. Resolución en 6 horas + crédito de 1 mes. Cliente retenido pero con NPS negativo.",
        "lección": "Los tickets relacionados con facturación electrónica en distribución tienen prioridad máxima — bloquean el flujo de caja del cliente",
    },
    {
        "id": "LOG-033",
        "señales": "Gestor de cuenta rota y el nuevo gestor no tiene contexto del cliente. Primera reunión con el cliente nuevo fue incómoda — 'tuvimos que explicar todo de nuevo'.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Medio",
        "probabilidad": 47,
        "razon": "Cambio de gestor sin transferencia de contexto — el cliente siente que tiene que empezar de cero",
        "accion_que_funcionó": "El nuevo gestor preparó un briefing completo del historial del cliente antes de la segunda reunión. La relación se recuperó en 30 días.",
        "lección": "En distribución, la relación personal con el gestor de cuenta tiene más peso que en otros sectores — el cambio sin transición es riesgoso",
    },
    {
        "id": "LOG-034",
        "señales": "Cliente de logística dice que 'el soporte antes era mejor'. No especifica qué cambió. El ticket promedio tardaba 4 horas, ahora tarda 2 días.",
        "resultado": "en riesgo activo",
        "nivel_riesgo": "Alto",
        "probabilidad": 69,
        "razon": "Degradación percibida y real del nivel de soporte — el cliente compara con una experiencia previa mejor",
        "accion_que_funcionó": "Se asignó soporte prioritario al cliente con SLA de 4 horas. El gestor llamó personalmente para disculparse. Cliente retenido.",
        "lección": "Cuando un cliente dice 'antes era mejor', tiene razón y hay datos que lo respaldan. No discutir — actuar.",
    },
    {
        "id": "LOG-035",
        "señales": "Distribuidor mayorista que tiene 3 años como cliente nunca fue visitado presencialmente. 'A veces siento que somos solo un número para ustedes'.",
        "resultado": "renovó tras visita",
        "nivel_riesgo": "Medio",
        "probabilidad": 51,
        "razon": "Cliente longevo con sentimiento de abandono — en distribución la visita presencial al depósito tiene valor simbólico alto",
        "accion_que_funcionó": "Visita presencial al depósito con el gerente de cuenta. Almuerzo con el dueño. El cliente firmó contrato de 3 años.",
        "lección": "En distribución, una visita presencial al depósito de un cliente de 3 años que nunca recibió una visita puede valer más que cualquier descuento",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 8: SEÑALES POSITIVAS Y EXPANSIÓN
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-036",
        "señales": "Distribuidor pide que el sistema procese también las operaciones de su empresa hermana. 'Somos dos empresas del mismo grupo'.",
        "resultado": "expandió",
        "nivel_riesgo": "Bajo",
        "probabilidad": 3,
        "razon": "Expansión orgánica a empresa del grupo — máxima señal de satisfacción en distribución",
        "accion_que_funcionó": "Se ofreció plan multi-empresa con descuento del 20% sobre el precio de dos contratos separados. Ambas empresas firmaron.",
        "lección": "La expansión a empresas del grupo es la mejor señal de salud — hay que tener pricing multi-empresa listo para ofrecer",
    },
    {
        "id": "LOG-037",
        "señales": "Jefe de operaciones de distribuidora menciona en una reunión que 'no me imagino volviendo a las planillas de Excel'.",
        "resultado": "renovó",
        "nivel_riesgo": "Bajo",
        "probabilidad": 6,
        "razon": "Dependencia operativa establecida — el cliente construyó sus procesos sobre el sistema",
        "accion_que_funcionó": "Se documentó como caso de éxito. El cliente participó en un webinar del sector.",
        "lección": "Cuando un cliente de logística no puede imaginar volver a Excel, la retención está casi asegurada",
    },
    {
        "id": "LOG-038",
        "señales": "Distribuidor refirió a tres empresas del mismo rubro. 'Les dije que usen su sistema, es lo mejor que implementamos'.",
        "resultado": "renovó y amplió",
        "nivel_riesgo": "Bajo",
        "probabilidad": 2,
        "razon": "Advocacy activo con referidos múltiples — máxima señal de satisfacción y lealtad",
        "accion_que_funcionó": "Se ofreció beneficio por referido. Los tres referidos se convirtieron en clientes. El cliente original hizo upgrade.",
        "lección": "Los clientes que refieren activamente en el sector de distribución tienen poder de influencia muy alto — hay que cultivarlos",
    },
    {
        "id": "LOG-039",
        "señales": "Empresa de logística pide capacitación para 8 empleados nuevos que acaban de contratar. 'Estamos creciendo y todos tienen que aprender el sistema'.",
        "resultado": "expandió",
        "nivel_riesgo": "Bajo",
        "probabilidad": 5,
        "razon": "Incorporación masiva de nuevos usuarios — el sistema es parte del onboarding de la empresa",
        "accion_que_funcionó": "Se ofreció sesión de capacitación grupal gratuita. En la sesión se presentó el plan Enterprise. El cliente hizo upgrade.",
        "lección": "Las solicitudes de capacitación masiva en distribución son oportunidades de upgrade — el cliente está creciendo y necesita más",
    },
    {
        "id": "LOG-040",
        "señales": "Cliente de distribución farmacéutica obtuvo una nueva habilitación y va a duplicar su volumen de operaciones el próximo trimestre.",
        "resultado": "expandió",
        "nivel_riesgo": "Bajo",
        "probabilidad": 4,
        "razon": "Evento de crecimiento externo verificable — oportunidad de upgrade antes del pico operativo",
        "accion_que_funcionó": "Se contactó proactivamente con propuesta de escalamiento del plan antes del crecimiento. El cliente agradeció la anticipación y firmó upgrade.",
        "lección": "En distribución farmacéutica, las habilitaciones regulatorias son eventos públicos que anticipan crecimiento — monitorearlas genera oportunidades",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 9: CASOS MIXTOS Y COMPLEJOS
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-041",
        "señales": "Cliente satisfecho pero su principal cliente (que representa el 60% de su facturación) dejó de comprarle. 'Tuvimos que reducir operaciones drásticamente'.",
        "resultado": "canceló por razón externa",
        "nivel_riesgo": "Crítico",
        "probabilidad": 82,
        "razon": "Crisis de negocio del cliente — no es churn por insatisfacción sino por viabilidad económica",
        "accion_que_funcionó": "Se ofreció plan reducido de emergencia por 6 meses. El cliente aceptó y cuando su situación mejoró renovó el plan completo.",
        "lección": "En distribución, el churn por crisis del negocio del cliente se puede retener con planes de emergencia temporales — perder el cliente por siempre es peor que reducir el ingreso 6 meses",
    },
    {
        "id": "LOG-042",
        "señales": "Empresa de distribución que opera en zona de inundaciones frecuentes pide que el sistema funcione offline. 'Cuando se corta internet no podemos operar'.",
        "resultado": "renovó con solución",
        "nivel_riesgo": "Medio",
        "probabilidad": 43,
        "razon": "Requisito técnico de conectividad no cubierto — válido en zonas del interior con infraestructura limitada",
        "accion_que_funcionó": "Se implementó modo offline con sincronización posterior. El cliente quedó muy satisfecho y refirió a otros distribuidores de la región.",
        "lección": "Los requisitos de conectividad en distribuidoras del interior son frecuentes y poco atendidos — solucionarlos genera lealtad fuerte en esos mercados",
    },
    {
        "id": "LOG-043",
        "señales": "Gerente de distribuidora pregunta si el sistema puede generar el informe que le pide el banco para una línea de crédito. 'Necesitamos mostrar el movimiento de inventario de los últimos 18 meses'.",
        "resultado": "renovó con upgrade",
        "nivel_riesgo": "Bajo",
        "probabilidad": 11,
        "razon": "Uso del sistema como respaldo financiero ante terceros — señal de dependencia positiva",
        "accion_que_funcionó": "Se generó el informe personalizado en 24 horas. El banco aprobó el crédito. El cliente hizo upgrade al plan que incluye reportes financieros avanzados.",
        "lección": "En distribución, cuando el sistema se usa para gestión bancaria o crediticia, la dependencia es muy alta — es una oportunidad de upgrade",
    },
    {
        "id": "LOG-044",
        "señales": "Distribuidor de electrodomésticos tiene pico de demanda en Navidad y el sistema tuvo problemas de performance el año anterior en esa época. Noviembre está cerca.",
        "resultado": "depende de la preparación",
        "nivel_riesgo": "Alto",
        "probabilidad": 74,
        "razon": "Antecedente de falla en pico estacional con el pico aproximándose — el cliente ya tiene el recuerdo negativo activado",
        "accion_que_funcionó": "Se realizaron pruebas de carga en octubre y se comunicaron los resultados al cliente antes de noviembre. El cliente operó sin problemas y renovó.",
        "lección": "En distribución estacional, anticiparse al pico con pruebas y comunicación proactiva convierte un antecedente negativo en una demostración de confiabilidad",
    },
    {
        "id": "LOG-045",
        "señales": "Cliente de logística tiene nuevo contador que revisó todos los gastos. 'El contador dice que este sistema es caro comparado con otras opciones del mercado'.",
        "resultado": "renovó con justificación de ROI",
        "nivel_riesgo": "Alto",
        "probabilidad": 67,
        "razon": "Presión de contador sobre costo sin análisis de valor — el contador ve el egreso pero no el ROI operativo",
        "accion_que_funcionó": "Se preparó informe de ROI mostrando el ahorro en tiempo operativo, reducción de errores de entrega y costo de volver a procesos manuales. El contador aprobó la renovación.",
        "lección": "Cuando un contador cuestiona el costo, hay que ir con ROI concreto en números — el contador entiende números, no features",
    },

    # ═══════════════════════════════════════════════════════
    # BLOQUE 10: CASOS EDGE Y SITUACIONES ESPECIALES
    # ═══════════════════════════════════════════════════════

    {
        "id": "LOG-046",
        "señales": "Distribuidor menciona que va a mudarse a un depósito más grande y 'aprovechar para revisar todos los sistemas'.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 63,
        "razon": "Mudanza de depósito como punto de inflexión para revisar sistemas — momento de alta vulnerabilidad",
        "accion_que_funcionó": "Se ofreció asistencia técnica gratuita para la mudanza y reconfiguración del sistema en el nuevo depósito. El cliente renovó por 2 años.",
        "lección": "Las mudanzas de depósito en distribución son momentos de alta vulnerabilidad pero también de alta lealtad si se acompaña bien el proceso",
    },
    {
        "id": "LOG-047",
        "señales": "Empresa de distribución que siempre pagó puntualmente tiene 2 facturas vencidas sin comunicación.",
        "resultado": "en riesgo financiero",
        "nivel_riesgo": "Alto",
        "probabilidad": 71,
        "razon": "Cambio abrupto en patrón de pago sin comunicación — señal de crisis financiera o decisión de no renovar",
        "accion_que_funcionó": "Llamada de seguimiento reveló crisis de liquidez temporal. Se ofreció plan de pago en cuotas. El cliente regularizó y renovó.",
        "lección": "En distribución, un cambio en el patrón de pago es una señal operativa tanto como financiera — hay que llamar antes de enviar la deuda a cobranza",
    },
    {
        "id": "LOG-048",
        "señales": "Cliente de logística con 5 años de antigüedad nunca pidió un descuento. En la renovación anual pide por primera vez un 20% de descuento.",
        "resultado": "renovó con 10% de descuento",
        "nivel_riesgo": "Medio",
        "probabilidad": 48,
        "razon": "Primera solicitud de descuento en cliente longevo — puede ser presión presupuestaria o señal de evaluación de alternativas",
        "accion_que_funcionó": "Se negoció 10% de descuento a cambio de contrato de 2 años. El cliente aceptó. El 10% de descuento fue menor que el costo de reemplazo.",
        "lección": "Un cliente de 5 años que pide descuento por primera vez merece negociación, no negativa — el costo de reemplazarlo es mucho mayor que el descuento",
    },
    {
        "id": "LOG-049",
        "señales": "Distribuidora familiar con dueño de 65 años. El hijo de 30 años empieza a involucrarse en el negocio y cuestiona 'si este sistema es el más moderno'.",
        "resultado": "en riesgo",
        "nivel_riesgo": "Alto",
        "probabilidad": 66,
        "razon": "Cambio generacional en empresa familiar — la nueva generación frecuentemente quiere cambiar los sistemas del padre",
        "accion_que_funcionó": "Reunión con el hijo mostrando las capacidades más modernas del sistema (API, dashboard, mobile). El hijo quedó satisfecho y se convirtió en el nuevo campeón interno.",
        "lección": "En distribuidoras familiares, el cambio generacional es uno de los mayores riesgos. Hay que convertir al sucesor en campeón antes de que cuestione todo.",
    },
    {
        "id": "LOG-050",
        "señales": "Cliente de logística deja de asistir al evento anual de usuarios que organiza la empresa. Antes nunca faltaba.",
        "resultado": "canceló 4 meses después",
        "nivel_riesgo": "Alto",
        "probabilidad": 72,
        "razon": "Desengagement de comunidad de usuarios — el cliente ya no se identifica con la plataforma",
        "accion_que_funcionó": "No se detectó a tiempo como señal de riesgo. El cliente canceló sin aviso previo.",
        "lección": "La no asistencia a eventos de usuarios en logística es una señal de desengagement que predice cancelación — hay que hacer seguimiento activo post-evento a los ausentes",
    },

    # 30 casos adicionales resumidos para mayor cobertura

    {"id": "LOG-051", "señales": "Cliente pide integración con MercadoLibre para sincronizar órdenes automáticamente", "resultado": "renovó con upgrade", "nivel_riesgo": "Bajo", "probabilidad": 12, "razon": "Expansión de canales de venta — quieren más del sistema", "accion_que_funcionó": "Integración con ML en 1 semana. Upgrade al plan Enterprise.", "lección": "Las integraciones con marketplaces son oportunidades de upgrade en distribución"},
    {"id": "LOG-052", "señales": "Operador de depósito dice que el lector de códigos de barras 'a veces no reconoce los códigos'", "resultado": "en riesgo medio", "nivel_riesgo": "Medio", "probabilidad": 45, "razon": "Fricción en operación física diaria del depósito", "accion_que_funcionó": "Se actualizó el driver del lector y se capacitó al operador. Problema resuelto en 24h.", "lección": "En logística de depósito, problemas con hardware periférico generan fricción diaria acumulativa"},
    {"id": "LOG-053", "señales": "Distribuidor menciona que su seguro de mercadería pide un inventario valorizado mensual", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 8, "razon": "Uso del sistema para requisito externo — dependencia positiva", "accion_que_funcionó": "Se automatizó el reporte mensual para el seguro. El cliente quedó muy satisfecho.", "lección": "Los reportes para terceros (seguros, bancos, contadores) generan dependencia positiva"},
    {"id": "LOG-054", "señales": "Cliente de logística menciona que un empleado clave 'descargó todos los datos del sistema' antes de renunciar", "resultado": "en riesgo de seguridad", "nivel_riesgo": "Alto", "probabilidad": 61, "razon": "Incidente de seguridad interna — el cliente puede perder confianza en el sistema", "accion_que_funcionó": "Se implementaron permisos granulares y auditoría de descargas. El cliente valoró la respuesta rápida.", "lección": "Los incidentes de seguridad interna en distribución son oportunidades de fortalecer la relación si se responde con soluciones concretas"},
    {"id": "LOG-055", "señales": "Distribuidor pide que el sistema funcione en inglés porque contrató personal extranjero", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 9, "razon": "Necesidad de internacionalización — señal de crecimiento", "accion_que_funcionó": "Se activó la interfaz en inglés. El cliente quedó satisfecho y refirió a otro distribuidor.", "lección": "Los pedidos de idioma alternativo en distribución son señales de crecimiento y diversidad — hay que resolverlos rápido"},
    {"id": "LOG-056", "señales": "Cliente de logística de frío reporta que el módulo de control de temperatura no registra las alertas correctamente", "resultado": "en riesgo crítico", "nivel_riesgo": "Crítico", "probabilidad": 91, "razon": "Falla en módulo de cumplimiento regulatorio — en logística de frío puede implicar sanciones", "accion_que_funcionó": "Ingeniero dedicado en 2 horas. Problema resuelto antes de la inspección del SENASA.", "lección": "En logística de frío, los módulos de control regulatorio tienen prioridad de soporte sobre cualquier otra cosa"},
    {"id": "LOG-057", "señales": "Distribuidor de medicamentos menciona que el PAMI le pide reportes en formato específico que el sistema no genera", "resultado": "renovó con desarrollo", "nivel_riesgo": "Medio", "probabilidad": 52, "razon": "Gap de integración con organismo público — en distribución farmacéutica es bloqueante", "accion_que_funcionó": "Se desarrolló el reporte en el formato requerido por PAMI en 5 días. El cliente renovó y recomendó a otras farmacias.", "lección": "Las integraciones con organismos públicos en distribución farmacéutica son diferenciadores de retención"},
    {"id": "LOG-058", "señales": "Cliente de distribución con operaciones en varias provincias tiene problemas con las alícuotas de ingresos brutos de cada provincia", "resultado": "en riesgo", "nivel_riesgo": "Alto", "probabilidad": 68, "razon": "Complejidad fiscal no resuelta — bloquea la operación contable del cliente", "accion_que_funcionó": "Se actualizó la tabla de alícuotas provinciales y se automatizó el cálculo. El cliente quedó muy satisfecho.", "lección": "En distribución multi-provincial, la gestión de alícuotas de IIBB es un pain point crítico que pocos sistemas resuelven bien"},
    {"id": "LOG-059", "señales": "Empresa de logística menciona que va a participar en una licitación pública y necesita certificación ISO del sistema", "resultado": "renovó con certificación", "nivel_riesgo": "Bajo", "probabilidad": 14, "razon": "Licitación pública que requiere certificación — el cliente necesita el sistema para ganar el contrato", "accion_que_funcionó": "Se gestionó la certificación ISO relevante. El cliente ganó la licitación. Renovó por 3 años.", "lección": "En distribución, las licitaciones públicas que requieren certificación del sistema generan dependencia muy alta"},
    {"id": "LOG-060", "señales": "Distribuidor pregunta si el sistema puede manejar consignación además de venta directa", "resultado": "renovó con upgrade", "nivel_riesgo": "Bajo", "probabilidad": 7, "razon": "Expansión de modelo de negocio — el cliente quiere más funcionalidades", "accion_que_funcionó": "Se activó el módulo de consignación del plan Enterprise. El cliente hizo upgrade y quedó muy satisfecho.", "lección": "Los cambios de modelo de negocio en distribución son oportunidades de upgrade si el sistema los soporta"},
    {"id": "LOG-061", "señales": "Cliente de logística tiene auditoría interna y pide log de todas las modificaciones hechas en el sistema en los últimos 6 meses", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 11, "razon": "Uso del sistema para auditoría interna — señal de dependencia positiva", "accion_que_funcionó": "Se generó el log de auditoría completo en 24 horas. El cliente superó la auditoría.", "lección": "Los sistemas de auditoría y trazabilidad generan alta dependencia en distribución — son funcionalidades que los clientes no pueden conseguir fácilmente"},
    {"id": "LOG-062", "señales": "Distribuidor de bebidas menciona que su camión GPS ya no sincroniza con el sistema de rutas desde hace 3 semanas", "resultado": "en riesgo", "nivel_riesgo": "Alto", "probabilidad": 73, "razon": "Desconexión de integración crítica de flota — la visibilidad de rutas es core en distribución de bebidas", "accion_que_funcionó": "Reconexión de la integración GPS en 48 horas. Se implementó monitoreo automático de la conexión.", "lección": "En distribución de bebidas, la integración GPS-rutas es tan crítica como el sistema principal — monitorearla proactivamente evita churns"},
    {"id": "LOG-063", "señales": "Cliente de distribución cambia su razón social y pide actualizar todos los documentos. 'Hay que modificar miles de registros'", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 15, "razon": "Cambio de razón social — proceso administrativo que genera dependencia durante la transición", "accion_que_funcionó": "Se asistió con la migración de registros. El proceso duró 2 semanas. El cliente quedó muy agradecido.", "lección": "Acompañar procesos administrativos complejos en distribución genera lealtad — el cliente recuerda quién lo ayudó en los momentos difíciles"},
    {"id": "LOG-064", "señales": "Logística de e-commerce menciona que el tiempo de procesamiento de órdenes del sistema aumentó de 2 segundos a 45 segundos", "resultado": "en riesgo crítico", "nivel_riesgo": "Crítico", "probabilidad": 89, "razon": "Degradación masiva de performance — en e-commerce cada segundo de delay impacta la experiencia del comprador final", "accion_que_funcionó": "Optimización urgente de base de datos en 6 horas. Performance restaurada. Cliente retenido con SLA mejorado.", "lección": "En logística de e-commerce, la performance del sistema es tan crítica como en cualquier SaaS de consumo — los tiempos de respuesta son un SLA implícito"},
    {"id": "LOG-065", "señales": "Distribuidor textil menciona que la temporada de verano terminó y 'por los próximos 3 meses vamos a estar muy tranquilos'", "resultado": "riesgo de downsell",  "nivel_riesgo": "Medio", "probabilidad": 41, "razon": "Baja estacional en distribución textil — el cliente puede pedir reducción del plan en temporada baja", "accion_que_funcionó": "Se ofreció proactivamente plan reducido estacional con opción de volver al plan completo en octubre. El cliente agradeció la flexibilidad.", "lección": "En distribución estacional, ofrecer flexibilidad de plan en temporada baja retiene mejor que defender el precio fijo"},
    {"id": "LOG-066", "señales": "Cliente de logística pide que se integre el sistema con WhatsApp Business para enviar notificaciones de entrega automáticas", "resultado": "renovó con upgrade", "nivel_riesgo": "Bajo", "probabilidad": 6, "razon": "Pedido de integración con canal de comunicación moderno — señal de expansión", "accion_que_funcionó": "Se integró WhatsApp Business con el módulo de entregas. El NPS del cliente final del distribuidor mejoró notablemente. Upgrade firmado.", "lección": "Las integraciones con WhatsApp Business para notificaciones de entrega son una ventaja competitiva en distribución — los que las tienen no se van"},
    {"id": "LOG-067", "señales": "Distribuidor con 2 años como cliente pide referencias de otros clientes del mismo rubro antes de renovar", "resultado": "renovó tras referencias", "nivel_riesgo": "Medio", "probabilidad": 44, "razon": "Solicitud de referencias antes de renovar — el cliente tiene dudas que no expresó directamente", "accion_que_funcionó": "Se conectó con 2 clientes referentes del mismo rubro que dieron feedback muy positivo. El cliente renovó.", "lección": "Cuando un cliente pide referencias antes de renovar, hay que actuar rápido — las referencias positivas de pares del mismo sector son el argumento más poderoso"},
    {"id": "LOG-068", "señales": "Empresa de distribución menciona que su contador externo 'no entiende los reportes del sistema y tiene que hacer todo manual'", "resultado": "en riesgo", "nivel_riesgo": "Medio", "probabilidad": 53, "razon": "Fricción con contador externo que procesa la información del cliente — el contador puede recomendar cambiar el sistema", "accion_que_funcionó": "Se ofreció sesión de capacitación al contador externo. El contador aprendió a usar los reportes y se convirtió en promotor del sistema.", "lección": "En distribución, el contador externo es un stakeholder clave — si no entiende el sistema, puede recomendar cambiarlo"},
    {"id": "LOG-069", "señales": "Cliente de logística menciona que abrió sucursal en otro país y necesita manejar múltiples monedas", "resultado": "renovó con upgrade", "nivel_riesgo": "Bajo", "probabilidad": 4, "razon": "Expansión internacional — el cliente necesita funcionalidades avanzadas que justifican upgrade", "accion_que_funcionó": "Se activó el módulo multi-moneda del plan Enterprise. El cliente firmó contrato regional.", "lección": "La expansión internacional de un distribuidor es la oportunidad de upgrade más valiosa — hay que tener la propuesta lista"},
    {"id": "LOG-070", "señales": "Distribuidor dice que 'el sistema es bueno pero la interfaz es fea'. No tiene quejas operativas.", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 19, "razon": "Queja estética sin impacto operativo — en distribución la UX estética tiene menos peso que la funcionalidad", "accion_que_funcionó": "Se mostró la actualización de interfaz del próximo trimestre. El cliente quedó conforme y renovó.", "lección": "En distribución, las quejas estéticas sin impacto operativo tienen bajo peso en la decisión de churn — a diferencia de otros sectores"},
    {"id": "LOG-071", "señales": "Logístico de última milla menciona que sus choferes no tienen smartphone y el sistema móvil no funciona en los teléfonos básicos", "resultado": "renovó con solución", "nivel_riesgo": "Medio", "probabilidad": 49, "razon": "Gap de accesibilidad tecnológica en operadores de campo — realidad común en logística de última milla", "accion_que_funcionó": "Se desarrolló versión ultra-liviana para dispositivos básicos. El cliente quedó muy satisfecho.", "lección": "En logística de última milla, la compatibilidad con dispositivos básicos es una necesidad real — no todos los choferes tienen smartphones modernos"},
    {"id": "LOG-072", "señales": "Distribuidor de ferretería menciona que tiene 45.000 SKUs y el buscador del sistema 'es muy lento'", "resultado": "en riesgo", "nivel_riesgo": "Alto", "probabilidad": 67, "razon": "Performance degradada en catálogos grandes — un buscador lento en distribución detiene las ventas", "accion_que_funcionó": "Se optimizó el índice de búsqueda. El tiempo de respuesta bajó de 8 segundos a 0.3 segundos. Cliente retenido.", "lección": "En distribución de ferretería con catálogos grandes, la performance del buscador es crítica — es la herramienta más usada por el equipo de ventas"},
    {"id": "LOG-073", "señales": "Cliente de logística quiere que el sistema genere automáticamente los documentos aduaneros para exportación", "resultado": "renovó con desarrollo", "nivel_riesgo": "Bajo", "probabilidad": 13, "razon": "Requisito de expansión a exportaciones — señal de crecimiento del negocio del cliente", "accion_que_funcionó": "Se desarrolló el módulo de documentos aduaneros en 3 semanas. El cliente se convirtió en caso de éxito.", "lección": "Los módulos de comercio exterior en distribución son diferenciadores de retención — muy pocos sistemas los tienen bien implementados"},
    {"id": "LOG-074", "señales": "Distribuidor menciona que su banco le ofrece un sistema de gestión 'gratis' como parte de un paquete financiero", "resultado": "en riesgo",  "nivel_riesgo": "Alto", "probabilidad": 71, "razon": "Oferta de sistema gratis de banco — el precio cero es el argumento más difícil de combatir", "accion_que_funcionó": "Se preparó análisis comparativo mostrando las limitaciones del sistema del banco vs las capacidades operativas del sistema actual. El cliente eligió mantener el sistema.", "lección": "Cuando un banco ofrece sistema gratis, hay que combatir con funcionalidad, no con precio — el sistema del banco generalmente es básico"},
    {"id": "LOG-075", "señales": "Cliente de distribución de materiales eléctricos dice que necesitan el sistema certificado para emitir factura electrónica con AFIP", "resultado": "renovó", "nivel_riesgo": "Bajo", "probabilidad": 10, "razon": "Requisito de facturación electrónica — el sistema ya lo tiene pero el cliente no lo sabía", "accion_que_funcionó": "Se mostró la integración existente con AFIP y se configuró en la misma llamada. El cliente quedó impresionado.", "lección": "En distribución argentina, la integración con AFIP para facturación electrónica es requisito básico — si el cliente no lo sabe, hay un problema de comunicación de features"},
    {"id": "LOG-076", "señales": "Logístico de importación menciona que necesita gestionar depósito fiscal y el sistema actual no lo soporta", "resultado": "canceló", "nivel_riesgo": "Crítico", "probabilidad": 92, "razon": "Gap de funcionalidad core para el modelo de negocio del cliente — no es posible retener sin el feature", "accion_que_funcionó": "No se pudo retener — el feature no estaba en el roadmap. El cliente migró a un sistema especializado en importación.", "lección": "En logística de importación, el depósito fiscal es una funcionalidad no negociable. Si no se tiene, hay que ser honesto con el cliente."},
    {"id": "LOG-077", "señales": "Distribuidor con 4 años pide que se le asigne un 'gerente de cuenta senior'. 'Queremos más atención ejecutiva'", "resultado": "renovó", "nivel_riesgo": "Medio", "probabilidad": 46, "razon": "Pedido de mayor atención — el cliente percibe que merece más por su antigüedad y volumen", "accion_que_funcionó": "Se asignó gerente de cuenta senior. Primera reunión fue un QBR con métricas de valor generado. Cliente renovó por 2 años.", "lección": "Cuando un cliente de 4 años pide más atención ejecutiva, hay que dársela — el costo de un QBR es mínimo comparado con el churn"},
    {"id": "LOG-078", "señales": "Empresa de distribución de consumo masivo menciona que la cadena de supermercados a la que proveen les pidió integración EDI", "resultado": "renovó con desarrollo", "nivel_riesgo": "Medio", "probabilidad": 38, "razon": "Requisito de integración EDI de cliente grande del distribuidor — si no se resuelve, el distribuidor pierde al cliente grande", "accion_que_funcionó": "Se desarrolló la integración EDI en 4 semanas. El distribuidor pudo seguir proveyendo a la cadena. Renovó por 3 años.", "lección": "En distribución de consumo masivo, los requisitos EDI de cadenas de supermercados son no negociables — resolverlos genera lealtad de largo plazo"},
    {"id": "LOG-079", "señales": "Cliente de logística menciona en una reunión que 'el sistema que usaban antes era más simple pero tenían todo más claro'", "resultado": "en riesgo", "nivel_riesgo": "Alto", "probabilidad": 64, "razon": "Nostalgia del sistema anterior — el cliente percibe que el nuevo sistema es más complejo sin ser más útil", "accion_que_funcionó": "Se configuró una vista simplificada personalizada que replicaba el flujo del sistema anterior pero con las capacidades del nuevo. Cliente retenido.", "lección": "En distribución, la nostalgia del sistema anterior es una señal de que el onboarding no fue suficientemente profundo — hay que resolver la usabilidad, no la funcionalidad"},
    {"id": "LOG-080", "señales": "Distribuidor que renovó hace 2 meses ya está pidiendo reunión urgente sin especificar el motivo", "resultado": "depende de la gestión", "nivel_riesgo": "Alto", "probabilidad": 69, "razon": "Reunión urgente sin contexto en cliente recién renovado — generalmente es un problema serio que surgió post-renovación", "accion_que_funcionó": "El gestor fue preparado con el historial completo del cliente. El problema era un error de migración de datos. Se resolvió en la misma reunión. El cliente quedó satisfecho.", "lección": "Una reunión urgente sin contexto de un cliente recién renovado hay que prepararla como si fuera una conversación de retención — puede serlo"},
]


# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE LA BASE VECTORIAL
# ---------------------------------------------------------------------------

def inicializar_base_vectorial() -> chromadb.Collection:
    """
    Crea la base de datos vectorial ChromaDB con los 80 casos de logística.
    Usa sentence-transformers para generar embeddings semánticos.
    Si ya existe, la retorna directamente.
    """
    log.info("Iniciando base vectorial ChromaDB...")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=CONFIG_RAG["db_path"])

    colecciones = [c.name for c in client.list_collections()]
    if CONFIG_RAG["coleccion"] in colecciones:
        coleccion = client.get_collection(
            name=CONFIG_RAG["coleccion"],
            embedding_function=embedding_fn,
        )
        log.info(f"Base vectorial existente cargada — {coleccion.count()} casos")
        return coleccion

    coleccion = client.create_collection(
        name=CONFIG_RAG["coleccion"],
        embedding_function=embedding_fn,
        metadata={"industria": "distribucion_logistica", "version": "1.0"},
    )

    # Preparar documentos para indexar
    documentos = []
    ids = []
    metadatos = []

    for caso in CASOS_LOGISTICA:
        texto = (
            f"Señales: {caso['señales']} "
            f"Resultado: {caso['resultado']} "
            f"Lección: {caso['lección']}"
        )
        documentos.append(texto)
        ids.append(caso["id"])
        metadatos.append({
            "nivel_riesgo": caso["nivel_riesgo"],
            "probabilidad": caso["probabilidad"],
            "razon": caso["razon"],
            "accion": caso["accion_que_funcionó"],
            "resultado": caso["resultado"],
            "leccion": caso["lección"],
        })

    coleccion.add(
        documents=documentos,
        ids=ids,
        metadatas=metadatos,
    )

    log.info(f"Base vectorial creada con {len(documentos)} casos de logística")
    return coleccion


# ---------------------------------------------------------------------------
# BÚSQUEDA DE CASOS SIMILARES
# ---------------------------------------------------------------------------

def buscar_casos_similares(
    coleccion: chromadb.Collection,
    señales_cliente: str,
    n_resultados: int = None,
) -> list[dict]:
    """
    Busca los casos más similares al cliente actual en la base vectorial.

    Args:
        coleccion: Base vectorial ChromaDB
        señales_cliente: Texto con las señales del cliente a analizar
        n_resultados: Cantidad de casos a recuperar

    Returns:
        Lista de casos similares con su metadata
    """
    if n_resultados is None:
        n_resultados = CONFIG_RAG["casos_similares_a_recuperar"]

    resultados = coleccion.query(
        query_texts=[señales_cliente],
        n_results=n_resultados,
    )

    casos = []
    for i in range(len(resultados["ids"][0])):
        casos.append({
            "id": resultados["ids"][0][i],
            "documento": resultados["documents"][0][i],
            "metadata": resultados["metadatas"][0][i],
            "distancia": resultados["distances"][0][i],
        })

    log.info(
        f"Casos similares recuperados: "
        + ", ".join([f"{c['id']} (dist: {c['distancia']:.3f})" for c in casos])
    )
    return casos


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL PROMPT CON RAG
# ---------------------------------------------------------------------------

PROMPT_SISTEMA_RAG = """Eres PostSale-AI, el motor más avanzado del mundo en \
detección de fricción operativa y predicción de churn B2B, especializado en \
la industria de Distribución y Logística en Argentina y Latinoamérica.

Tu ventaja es que tenés acceso a una base de conocimiento de casos reales \
de esta industria. Usás esos casos para calibrar tu análisis con patrones \
probados en distribuidoras y empresas logísticas reales.

REGLAS ABSOLUTAS:
1. Respondés EXCLUSIVAMENTE con JSON válido. Sin texto, sin markdown, sin \
backticks, sin explicaciones.
2. nivel_riesgo SOLO puede ser: Bajo, Medio, Alto o Crítico.
3. probabilidad_churn_porcentaje: entero entre 0 y 100.
4. El tono educado NO reduce el riesgo. Analizás hechos y patrones.
5. Señales negativas múltiples se potencian entre sí, no se promedian.
6. En logística, los errores que impactan la entrega final tienen ventana \
de tolerancia de 24-48h máximo.
7. Una exportación masiva de datos históricos en logística es siempre Crítico.
8. Inactividad en el flujo operativo diario (manifiestos, rutas, picking) \
es más grave que inactividad en login — el operador encontró otra forma.
9. Ventana contractual < 60 días + cualquier fricción = Crítico.
10. Cambio de decisor operativo (jefe de depósito, jefe de logística) sin \
onboarding del sucesor = Alto mínimo.

RESPONDÉ SOLO CON EL JSON:
{
  "nivel_riesgo": "<Bajo|Medio|Alto|Crítico>",
  "probabilidad_churn_porcentaje": <0-100>,
  "razon_principal": "<causa raíz específica>",
  "accion_recomendada_para_el_gestor": "<acción concreta y específica>"
}"""


def construir_prompt_con_rag(
    nombre: str,
    plan: str,
    interacciones: list,
    casos_similares: list[dict],
) -> str:
    """
    Construye el prompt inyectando los casos similares recuperados.

    Args:
        nombre: Nombre del cliente
        plan: Plan de suscripción
        interacciones: Lista de interacciones del cliente
        casos_similares: Casos recuperados por RAG

    Returns:
        Prompt completo con contexto de casos similares
    """
    interacciones_texto = "\n".join(
        f"  - [{i.tipo_interaccion.value.upper()}] "
        f"Días sin login: {i.dias_desde_ultima_conexion} | "
        f"Mensaje: \"{i.texto_mensaje}\""
        for i in interacciones
    )

    contexto_rag = "\n".join([
        f"CASO {c['id']} (similitud alta):\n"
        f"  Señales similares: {c['metadata'].get('razon', '')}\n"
        f"  Resultado real: {c['metadata'].get('resultado', '')}\n"
        f"  Nivel de riesgo real: {c['metadata'].get('nivel_riesgo', '')}\n"
        f"  Acción que funcionó: {c['metadata'].get('accion', '')}\n"
        f"  Lección aprendida: {c['metadata'].get('leccion', '')}"
        for c in casos_similares
    ])

    return f"""CASOS HISTÓRICOS SIMILARES DE LA INDUSTRIA (usá como referencia):
{contexto_rag}

═══════════════════════════════════════════════════════

CLIENTE A ANALIZAR AHORA:
- Empresa: {nombre}
- Plan: {plan}

SEÑALES ACTUALES:
{interacciones_texto}

Basándote en los casos históricos similares y en las señales actuales, \
analizá el riesgo de churn de este cliente. \
Respondé SOLO con el JSON."""


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE ANÁLISIS CON RAG
# ---------------------------------------------------------------------------

async def analizar_con_rag(
    nombre: str,
    plan: str,
    interacciones: list,
    coleccion: chromadb.Collection,
) -> dict:
    """
    Versión mejorada de analizar_con_ia() que usa RAG.

    Esta función reemplaza a analizar_con_ia() en api.py.
    El resto de api.py queda exactamente igual.

    Args:
        nombre: Nombre del cliente
        plan: Plan de suscripción
        interacciones: Lista de InteraccionInput
        coleccion: Base vectorial ChromaDB

    Returns:
        Diccionario con el resultado del análisis
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no configurada.")

    groq_client = AsyncGroq(api_key=api_key)
    inicio = time.time()

    # Construir texto de búsqueda para RAG
    señales_texto = " ".join([
        f"{i.tipo_interaccion.value}: {i.texto_mensaje}"
        for i in interacciones
    ])

    # Buscar casos similares en la base vectorial
    casos_similares = buscar_casos_similares(coleccion, señales_texto)

    # Construir prompt con contexto RAG
    prompt = construir_prompt_con_rag(nombre, plan, interacciones, casos_similares)

    ultimo_error = ""
    mejor_datos = None
    mejor_score = 0

    for intento in range(1, CONFIG_RAG["max_reintentos"] + 1):
        log.info(f"[{nombre}] Intento {intento}/{CONFIG_RAG['max_reintentos']} (con RAG)...")

        try:
            response = await groq_client.chat.completions.create(
                model=CONFIG_RAG["modelo_groq"],
                temperature=CONFIG_RAG["temperatura"],
                max_tokens=CONFIG_RAG["max_tokens"],
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA_RAG},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()

            # Validación del JSON
            try:
                datos = json.loads(raw)
                niveles_validos = {"Bajo", "Medio", "Alto", "Crítico"}
                assert datos.get("nivel_riesgo") in niveles_validos
                assert isinstance(datos.get("probabilidad_churn_porcentaje"), int)
                assert 0 <= datos["probabilidad_churn_porcentaje"] <= 100
                assert len(datos.get("razon_principal", "")) >= 10
                assert len(datos.get("accion_recomendada_para_el_gestor", "")) >= 20
            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                ultimo_error = f"Validación fallida: {e}"
                log.warning(f"[{nombre}] {ultimo_error}")
                continue

            # Score de confianza
            score = 10
            if datos["nivel_riesgo"] == "Crítico" and datos["probabilidad_churn_porcentaje"] < 70:
                score -= 3
            if datos["nivel_riesgo"] == "Bajo" and datos["probabilidad_churn_porcentaje"] > 30:
                score -= 3

            if score >= CONFIG_RAG["score_minimo_confianza"]:
                log.info(f"[{nombre}] OK con RAG — score: {score}/10 — casos usados: {len(casos_similares)}")
                return {
                    **datos,
                    "score_confianza": score,
                    "intentos_realizados": intento,
                    "tiempo_procesamiento_segundos": round(time.time() - inicio, 2),
                    "requiere_revision_manual": False,
                    "casos_rag_usados": [c["id"] for c in casos_similares],
                }

            if score > mejor_score:
                mejor_datos = datos
                mejor_score = score

        except Exception as e:
            ultimo_error = str(e)
            log.warning(f"[{nombre}] Error API intento {intento}: {e}")

        if intento < CONFIG_RAG["max_reintentos"]:
            espera = CONFIG_RAG["espera_base_segundos"] ** intento
            await asyncio.sleep(espera)

    base = mejor_datos or {
        "nivel_riesgo": "Medio",
        "probabilidad_churn_porcentaje": 50,
        "razon_principal": "No se pudo determinar con certeza",
        "accion_recomendada_para_el_gestor": "Revisión manual requerida",
    }
    return {
        **base,
        "score_confianza": mejor_score,
        "intentos_realizados": CONFIG_RAG["max_reintentos"],
        "tiempo_procesamiento_segundos": round(time.time() - inicio, 2),
        "requiere_revision_manual": True,
        "casos_rag_usados": [c["id"] for c in casos_similares],
    }


# ---------------------------------------------------------------------------
# TEST — verificación del sistema RAG
# ---------------------------------------------------------------------------

async def test_rag():
    """Prueba el sistema RAG con 3 escenarios de logística."""

    from api import InteraccionInput, TipoInteraccion

    print("=" * 60)
    print("  🧠  PostSale RAG — Test de verificación")
    print("=" * 60)

    coleccion = inicializar_base_vectorial()

    escenarios_test = [
        {
            "nombre": "Distribuidora del Sur S.A.",
            "plan": "Professional",
            "interacciones": [
                InteraccionInput(
                    tipo_interaccion=TipoInteraccion.TICKET,
                    texto_mensaje="El sistema GPS de nuestros camiones dejó de sincronizarse hace 3 semanas. No vemos las rutas en tiempo real.",
                    dias_desde_ultima_conexion=2,
                ),
                InteraccionInput(
                    tipo_interaccion=TipoInteraccion.EMAIL,
                    texto_mensaje="Además, el módulo de manifiestos falla los lunes cuando cargamos el volumen de la semana.",
                    dias_desde_ultima_conexion=2,
                ),
            ],
        },
        {
            "nombre": "LogiFood Express",
            "plan": "Enterprise",
            "interacciones": [
                InteraccionInput(
                    tipo_interaccion=TipoInteraccion.EMAIL,
                    texto_mensaje="Hola, queremos exportar todo el historial de rutas y entregas de los últimos 3 años. ¿Cómo lo hacemos?",
                    dias_desde_ultima_conexion=1,
                ),
            ],
        },
        {
            "nombre": "Materiales Norte Ltda.",
            "plan": "Starter",
            "interacciones": [
                InteraccionInput(
                    tipo_interaccion=TipoInteraccion.EMAIL,
                    texto_mensaje="Estamos muy contentos con el sistema. Lo usamos todos los días y nos ahorra mucho tiempo en el depósito.",
                    dias_desde_ultima_conexion=0,
                ),
            ],
        },
    ]

    for escenario in escenarios_test:
        print(f"\n{'─' * 60}")
        print(f"  Cliente: {escenario['nombre']}")
        print(f"{'─' * 60}")

        resultado = await analizar_con_rag(
            nombre=escenario["nombre"],
            plan=escenario["plan"],
            interacciones=escenario["interacciones"],
            coleccion=coleccion,
        )

        emoji_map = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🟠", "Crítico": "🔴"}
        emoji = emoji_map.get(resultado["nivel_riesgo"], "⚪")

        print(f"  {emoji} Nivel       : {resultado['nivel_riesgo']}")
        print(f"  📊 Prob. Churn : {resultado['probabilidad_churn_porcentaje']}%")
        print(f"  🔍 Razón       : {resultado['razon_principal']}")
        print(f"  💬 Acción      : {resultado['accion_recomendada_para_el_gestor'][:80]}...")
        print(f"  🧠 Casos RAG   : {', '.join(resultado.get('casos_rag_usados', []))}")
        print(f"  ⭐ Confianza   : {resultado['score_confianza']}/10")

    print(f"\n{'=' * 60}")
    print("  ✅  Test completado")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_rag())
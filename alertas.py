"""
PostSale MVP — Fase 4: Alertas Automáticas
===========================================
Sistema de alertas por email con análisis automático cada 24 horas.

Funciones:
  - Analiza todos los clientes automáticamente cada 24 horas
  - Envía email cuando un cliente sube a Alto o Crítico
  - Resumen diario con los clientes más urgentes
  - Email con diseño HTML profesional

Uso:
    1. $env:GROQ_API_KEY="gsk_tu-clave"
    2. $env:RESEND_API_KEY="re_tu-clave"
    3. python alertas.py

Autor: PostSale Engineering
Versión: 0.4.0 (Fase 4 — Alertas Automáticas)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime

import resend
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from groq import AsyncGroq
from sqlmodel import Session, select

# Importamos todo lo que ya construimos en la Fase 2
from api import (
    AnalisisDB,
    ClienteDB,
    CONFIG,
    PROMPT_SISTEMA,
    InteraccionInput,
    TipoInteraccion,
    analizar_con_ia,
    construir_prompt_usuario,
    crear_tablas,
    engine,
    validar_respuesta_ia,
)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postsale.alertas")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE ALERTAS
# ---------------------------------------------------------------------------

ALERTA_CONFIG = {
    "email_destino": "santivinas1@gmail.com",
    "email_origen": "PostSale <onboarding@resend.dev>",
    "niveles_alerta": {"Alto", "Crítico"},
    "hora_resumen_diario": "08:00",  # HH:MM — hora del resumen matutino
    "intervalo_analisis_horas": 24,
}


# ---------------------------------------------------------------------------
# TEMPLATES DE EMAIL HTML
# ---------------------------------------------------------------------------

def email_alerta_html(
    cliente_nombre: str,
    plan: str,
    nivel_riesgo: str,
    probabilidad: int,
    razon: str,
    accion: str,
    score: int,
) -> str:
    """Genera el HTML del email de alerta individual."""

    color_map = {
        "Crítico": "#dc2626",
        "Alto": "#ea580c",
        "Medio": "#ca8a04",
        "Bajo": "#16a34a",
    }
    bg_map = {
        "Crítico": "#fef2f2",
        "Alto": "#fff7ed",
        "Medio": "#fefce8",
        "Bajo": "#f0fdf4",
    }
    color = color_map.get(nivel_riesgo, "#6b6b67")
    bg = bg_map.get(nivel_riesgo, "#f7f7f5")
    emoji = {"Crítico": "🔴", "Alto": "🟠", "Medio": "🟡", "Bajo": "🟢"}.get(nivel_riesgo, "⚪")

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width"></head>
<body style="margin:0;padding:0;background:#f7f7f5;font-family:'Inter',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0">
<tr><td align="center" style="padding:40px 20px">
<table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;border:1px solid #e5e5e2;overflow:hidden">

  <!-- HEADER -->
  <tr><td style="padding:24px 32px;border-bottom:1px solid #e5e5e2">
    <table width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td>
        <div style="display:inline-flex;align-items:center;gap:8px">
          <div style="width:24px;height:24px;background:#1a1a18;border-radius:6px;display:inline-block;vertical-align:middle"></div>
          <span style="font-size:16px;font-weight:700;color:#1a1a18;vertical-align:middle;margin-left:8px">PostSale</span>
        </div>
      </td>
      <td align="right">
        <span style="font-size:11px;font-weight:600;color:{color};background:{bg};padding:4px 10px;border-radius:20px;border:1px solid {color}30">
          {emoji} ALERTA {nivel_riesgo.upper()}
        </span>
      </td>
    </tr>
    </table>
  </td></tr>

  <!-- BODY -->
  <tr><td style="padding:28px 32px">

    <p style="font-size:13px;color:#6b6b67;margin:0 0 20px">
      {datetime.now().strftime('%d %b %Y — %H:%M')}
    </p>

    <h1 style="font-size:22px;font-weight:700;color:#1a1a18;margin:0 0 6px;letter-spacing:-0.4px">
      {cliente_nombre}
    </h1>
    <p style="font-size:13px;color:#9b9b97;margin:0 0 24px">Plan {plan}</p>

    <!-- METRIC ROW -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px">
    <tr>
      <td width="50%" style="padding-right:8px">
        <div style="background:#f7f7f5;border:1px solid #e5e5e2;border-radius:8px;padding:14px 16px">
          <div style="font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">Nivel de riesgo</div>
          <div style="font-size:20px;font-weight:700;color:{color}">{nivel_riesgo}</div>
        </div>
      </td>
      <td width="50%" style="padding-left:8px">
        <div style="background:#f7f7f5;border:1px solid #e5e5e2;border-radius:8px;padding:14px 16px">
          <div style="font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">Prob. cancelación</div>
          <div style="font-size:20px;font-weight:700;color:{color}">{probabilidad}%</div>
        </div>
      </td>
    </tr>
    </table>

    <!-- RAZON -->
    <div style="margin-bottom:20px">
      <div style="font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">Causa detectada</div>
      <p style="font-size:14px;color:#1a1a18;margin:0;line-height:1.6;background:#f7f7f5;padding:12px 16px;border-radius:8px;border:1px solid #e5e5e2">{razon}</p>
    </div>

    <!-- ACCION -->
    <div style="margin-bottom:24px">
      <div style="font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">Acción recomendada para el gestor</div>
      <p style="font-size:14px;color:#1a1a18;margin:0;line-height:1.6;background:{bg};padding:14px 16px;border-radius:8px;border:1px solid {color}30;border-left:3px solid {color}">{accion}</p>
    </div>

    <!-- SCORE -->
    <div style="font-size:12px;color:#9b9b97">
      Confianza IA: {score}/10 &nbsp;·&nbsp; Generado por PostSale AI
    </div>

  </td></tr>

  <!-- FOOTER -->
  <tr><td style="padding:16px 32px;border-top:1px solid #e5e5e2;background:#f7f7f5">
    <p style="font-size:11px;color:#9b9b97;margin:0">
      PostSale — Motor de predicción de churn B2B &nbsp;·&nbsp;
      Esta alerta fue generada automáticamente.
    </p>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def email_resumen_diario_html(clientes_urgentes: list[dict], total: int) -> str:
    """Genera el HTML del resumen diario matutino."""

    filas = ""
    for c in clientes_urgentes:
        color = "#dc2626" if c["nivel"] == "Crítico" else "#ea580c" if c["nivel"] == "Alto" else "#ca8a04"
        filas += f"""
        <tr>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e5e2">
            <div style="font-size:13px;font-weight:600;color:#1a1a18">{c['nombre']}</div>
            <div style="font-size:11px;color:#9b9b97">{c['plan']}</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e5e2">
            <span style="font-size:11px;font-weight:600;color:{color};background:{color}15;padding:3px 9px;border-radius:20px">{c['nivel']}</span>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e5e2;font-size:13px;font-weight:600;color:{color}">{c['prob']}%</td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e5e2;font-size:12px;color:#6b6b67;max-width:200px">{c['accion'][:80]}...</td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f7f7f5;font-family:'Inter',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0">
<tr><td align="center" style="padding:40px 20px">
<table width="620" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;border:1px solid #e5e5e2;overflow:hidden">

  <!-- HEADER -->
  <tr><td style="padding:24px 32px;border-bottom:1px solid #e5e5e2">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td>
        <span style="font-size:16px;font-weight:700;color:#1a1a18">PostSale</span>
        <span style="font-size:13px;color:#9b9b97;margin-left:10px">Resumen diario</span>
      </td>
      <td align="right">
        <span style="font-size:12px;color:#6b6b67">{datetime.now().strftime('%d %b %Y')}</span>
      </td>
    </tr></table>
  </td></tr>

  <!-- INTRO -->
  <tr><td style="padding:24px 32px 16px">
    <h2 style="font-size:18px;font-weight:700;color:#1a1a18;margin:0 0 8px">Buenos días 👋</h2>
    <p style="font-size:14px;color:#6b6b67;margin:0">
      Tenés <strong style="color:#1a1a18">{len(clientes_urgentes)} clientes</strong> que requieren atención hoy
      de un total de <strong style="color:#1a1a18">{total}</strong> analizados.
    </p>
  </td></tr>

  <!-- TABLA -->
  <tr><td style="padding:0 32px 24px">
    <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e5e5e2;border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#f7f7f5">
          <th style="padding:10px 16px;text-align:left;font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase">Cliente</th>
          <th style="padding:10px 16px;text-align:left;font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase">Riesgo</th>
          <th style="padding:10px 16px;text-align:left;font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase">Prob.</th>
          <th style="padding:10px 16px;text-align:left;font-size:10px;font-weight:600;color:#9b9b97;letter-spacing:0.06em;text-transform:uppercase">Acción</th>
        </tr>
      </thead>
      <tbody>{filas}</tbody>
    </table>
  </td></tr>

  <!-- FOOTER -->
  <tr><td style="padding:16px 32px;border-top:1px solid #e5e5e2;background:#f7f7f5">
    <p style="font-size:11px;color:#9b9b97;margin:0">
      PostSale — Resumen automático diario &nbsp;·&nbsp; {datetime.now().strftime('%H:%M')}
    </p>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# FUNCIONES DE ENVÍO DE EMAIL
# ---------------------------------------------------------------------------

def configurar_resend() -> bool:
    """Configura la API key de Resend. Retorna True si está configurada."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        log.error(
            "No se encontró RESEND_API_KEY. "
            "Ejecutá: $env:RESEND_API_KEY='re_tu-clave'"
        )
        return False
    resend.api_key = api_key
    return True


def enviar_alerta_email(
    cliente_nombre: str,
    plan: str,
    nivel_riesgo: str,
    probabilidad: int,
    razon: str,
    accion: str,
    score: int,
) -> bool:
    """Envía un email de alerta para un cliente en riesgo alto o crítico."""
    try:
        emoji = {"Crítico": "🔴", "Alto": "🟠"}.get(nivel_riesgo, "⚠️")
        params = {
            "from": ALERTA_CONFIG["email_origen"],
            "to": [ALERTA_CONFIG["email_destino"]],
            "subject": f"{emoji} PostSale — {nivel_riesgo}: {cliente_nombre}",
            "html": email_alerta_html(
                cliente_nombre, plan, nivel_riesgo,
                probabilidad, razon, accion, score,
            ),
        }
        resend.Emails.send(params)
        log.info(f"Alerta enviada para {cliente_nombre} ({nivel_riesgo})")
        return True
    except Exception as e:
        log.error(f"Error enviando alerta para {cliente_nombre}: {e}")
        return False


def enviar_resumen_diario(clientes_urgentes: list[dict], total: int) -> bool:
    """Envía el resumen diario matutino."""
    if not clientes_urgentes:
        log.info("Sin clientes urgentes hoy — no se envía resumen.")
        return True
    try:
        params = {
            "from": ALERTA_CONFIG["email_origen"],
            "to": [ALERTA_CONFIG["email_destino"]],
            "subject": f"📋 PostSale — Resumen del {datetime.now().strftime('%d %b')} — {len(clientes_urgentes)} clientes urgentes",
            "html": email_resumen_diario_html(clientes_urgentes, total),
        }
        resend.Emails.send(params)
        log.info(f"Resumen diario enviado — {len(clientes_urgentes)} urgentes de {total} total")
        return True
    except Exception as e:
        log.error(f"Error enviando resumen diario: {e}")
        return False


# ---------------------------------------------------------------------------
# ANÁLISIS AUTOMÁTICO
# ---------------------------------------------------------------------------

async def analizar_todos_los_clientes() -> None:
    """
    Analiza todos los clientes registrados en la base de datos.
    Envía alertas individuales para los que suban a Alto o Crítico.
    Luego envía el resumen diario con todos los urgentes.
    """
    log.info("=" * 50)
    log.info("Iniciando análisis automático de todos los clientes...")

    if not configurar_resend():
        return

    with Session(engine) as session:
        clientes = session.exec(select(ClienteDB)).all()

    if not clientes:
        log.info("No hay clientes registrados. Finalizando.")
        return

    log.info(f"Clientes a analizar: {len(clientes)}")
    clientes_urgentes = []

    for cliente in clientes:
        log.info(f"Analizando: {cliente.nombre}...")

        # Interacción de inactividad por defecto si no hay datos frescos
        # En producción esto vendría del CRM o logs reales
        interacciones_default = [
            InteraccionInput(
                tipo_interaccion=TipoInteraccion.INACTIVIDAD_LOGIN,
                texto_mensaje=(
                    "Análisis automático de rutina. "
                    "Sin nuevas interacciones registradas en las últimas 24 horas."
                ),
                dias_desde_ultima_conexion=1,
            )
        ]

        try:
            resultado = await analizar_con_ia(
                nombre=cliente.nombre,
                plan=cliente.plan_actual.value,
                interacciones=interacciones_default,
            )

            # Guardar en base de datos
            with Session(engine) as session:
                analisis_db = AnalisisDB(
                    cliente_id=cliente.id,
                    nivel_riesgo=resultado["nivel_riesgo"],
                    probabilidad_churn_porcentaje=resultado["probabilidad_churn_porcentaje"],
                    razon_principal=resultado["razon_principal"],
                    accion_recomendada=resultado["accion_recomendada_para_el_gestor"],
                    score_confianza=resultado["score_confianza"],
                    intentos_realizados=resultado["intentos_realizados"],
                    tiempo_procesamiento_segundos=resultado["tiempo_procesamiento_segundos"],
                    requiere_revision_manual=resultado["requiere_revision_manual"],
                    interacciones_json=json.dumps(
                        [i.dict() for i in interacciones_default],
                        ensure_ascii=False,
                    ),
                )
                session.add(analisis_db)
                session.commit()

            nivel = resultado["nivel_riesgo"]

            # Enviar alerta individual si es Alto o Crítico
            if nivel in ALERTA_CONFIG["niveles_alerta"]:
                enviar_alerta_email(
                    cliente_nombre=cliente.nombre,
                    plan=cliente.plan_actual.value,
                    nivel_riesgo=nivel,
                    probabilidad=resultado["probabilidad_churn_porcentaje"],
                    razon=resultado["razon_principal"],
                    accion=resultado["accion_recomendada_para_el_gestor"],
                    score=resultado["score_confianza"],
                )
                clientes_urgentes.append({
                    "nombre": cliente.nombre,
                    "plan": cliente.plan_actual.value,
                    "nivel": nivel,
                    "prob": resultado["probabilidad_churn_porcentaje"],
                    "accion": resultado["accion_recomendada_para_el_gestor"],
                })

            # Pequeña pausa entre clientes para no saturar la API
            await asyncio.sleep(1)

        except Exception as e:
            log.error(f"Error analizando {cliente.nombre}: {e}")

    # Enviar resumen diario con todos los urgentes
    enviar_resumen_diario(clientes_urgentes, len(clientes))
    log.info("Análisis automático completado.")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# SCHEDULER — programa el análisis cada 24 horas
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Arranca el scheduler que corre el análisis automático.
    También ejecuta un análisis inmediato al iniciar para probar.
    """
    print("=" * 55)
    print("  📬  PostSale Fase 4 — Alertas Automáticas")
    print("=" * 55)
    print(f"  Email destino : {ALERTA_CONFIG['email_destino']}")
    print(f"  Resumen diario: {ALERTA_CONFIG['hora_resumen_diario']}")
    print(f"  Intervalo     : cada {ALERTA_CONFIG['intervalo_analisis_horas']}h")
    print("=" * 55)

    crear_tablas()

    if not configurar_resend():
        return

    scheduler = AsyncIOScheduler()

    # Resumen diario a las 8:00 AM
    hora, minuto = ALERTA_CONFIG["hora_resumen_diario"].split(":")
    scheduler.add_job(
        analizar_todos_los_clientes,
        "cron",
        hour=int(hora),
        minute=int(minuto),
        id="resumen_diario",
    )

    scheduler.start()
    log.info("Scheduler iniciado. Ejecutando análisis inicial...")

    # Análisis inmediato al arrancar
    await analizar_todos_los_clientes()

    log.info(
        f"Próximo análisis automático: "
        f"{ALERTA_CONFIG['hora_resumen_diario']} de mañana."
    )

    # Mantener el proceso vivo
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        log.info("PostSale Alertas detenido.")


if __name__ == "__main__":
    asyncio.run(main())